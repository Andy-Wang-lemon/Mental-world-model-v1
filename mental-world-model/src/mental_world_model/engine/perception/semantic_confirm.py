"""二执路径：弱规则召回 + LLM 语义确认。

契约：
- 规则不过 → 不近，不调 LLM
- 规则过且无 LLM → 近（兼容单测 / 无客户端路径）
- 规则过且有 LLM → 二次确认；失败/缺答默认不近（宁可不并、不 boost）
"""

from __future__ import annotations

import threading

from mental_world_model.engine.extraction.validators import normalize_match_text
from mental_world_model.engine.perception.config import SEMANTIC_CONFIRM_MAX_PAIRS
from mental_world_model.engine.perception.prompts import (
    SEMANTIC_CONFIRM_SCHEMA,
    build_semantic_confirm_messages,
)
from mental_world_model.engine.perception.validators import (
    text_fingerprint,
    texts_semantically_near,
)
from mental_world_model.engine.scoring.parallel_exec import run_parallel
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context


def _pair_cache_key(a: str, b: str) -> tuple[str, str]:
    fa = text_fingerprint(a) or normalize_match_text(a)
    fb = text_fingerprint(b) or normalize_match_text(b)
    return (fa, fb) if fa <= fb else (fb, fa)


class PerspectiveNearCache:
    """跨代表去重 / 去同视角 / persist 共用，避免重复打 LLM。"""

    def __init__(self) -> None:
        self._decided: dict[tuple[str, str], bool] = {}
        self._lock = threading.Lock()

    def get(self, a: str, b: str) -> bool | None:
        with self._lock:
            return self._decided.get(_pair_cache_key(a, b))

    def set(self, a: str, b: str, value: bool) -> None:
        with self._lock:
            self._decided[_pair_cache_key(a, b)] = value


def perspectives_same(
    a: str,
    b: str,
    *,
    llm: LlmClient | None = None,
    cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> bool:
    """单对判定：弱规则门闸 →（可选）LLM 确认。"""
    ta = str(a or "").strip()
    tb = str(b or "").strip()
    if not ta or not tb:
        return False
    if (text_fingerprint(ta) or normalize_match_text(ta)) == (
        text_fingerprint(tb) or normalize_match_text(tb)
    ):
        if cache is not None:
            cache.set(ta, tb, True)
        return True

    if cache is not None:
        hit = cache.get(ta, tb)
        if hit is not None:
            return hit

    if not texts_semantically_near(ta, tb):
        if cache is not None:
            cache.set(ta, tb, False)
        return False

    if llm is None:
        if cache is not None:
            cache.set(ta, tb, True)
        return True

    results = confirm_perspective_pairs(
        [("p0", ta, tb)],
        llm,
        focus_person=focus_person,
        cache=cache,
    )
    return bool(results.get("p0", False))


def confirm_perspective_pairs(
    pairs: list[tuple[str, str, str]],
    llm: LlmClient,
    *,
    focus_person: str = "",
    cache: PerspectiveNearCache | None = None,
    max_pairs: int = SEMANTIC_CONFIRM_MAX_PAIRS,
) -> dict[str, bool]:
    """批量 LLM 确认。pairs=[(pair_id, text_a, text_b), ...]。

    调用方应只传入「规则已过闸」的配对；本函数不再二次规则过滤。
    ``max_pairs`` 是每批上限，多批问完，不截断。
    失败或缺 pair_id → False。
    """
    out: dict[str, bool] = {}
    if not pairs:
        return out

    chunk_size = max(1, int(max_pairs))
    pending: list[tuple[str, str, str]] = []
    for pid, ta, tb in pairs:
        ta_s, tb_s = str(ta or "").strip(), str(tb or "").strip()
        if not ta_s or not tb_s:
            out[pid] = False
            continue
        if cache is not None:
            hit = cache.get(ta_s, tb_s)
            if hit is not None:
                out[pid] = hit
                continue
        pending.append((pid, ta_s, tb_s))

    if not pending:
        return out

    chunks = [
        pending[start : start + chunk_size]
        for start in range(0, len(pending), chunk_size)
    ]
    if len(chunks) == 1:
        out.update(
            _confirm_perspective_chunk(
                chunks[0],
                llm,
                focus_person=focus_person,
                cache=cache,
            )
        )
        return out

    batch_results = run_parallel(
        {
            str(index): (
                lambda c=chunk: _confirm_perspective_chunk(
                    c,
                    llm,
                    focus_person=focus_person,
                    cache=cache,
                )
            )
            for index, chunk in enumerate(chunks)
        }
    )
    for index in range(len(chunks)):
        out.update(batch_results[str(index)])
    return out


def _confirm_perspective_chunk(
    pending: list[tuple[str, str, str]],
    llm: LlmClient,
    *,
    focus_person: str,
    cache: PerspectiveNearCache | None,
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not pending:
        return out
    try:
        with llm_call_context(stage="perception.semantic_confirm"):
            response = llm.complete(
                build_semantic_confirm_messages(
                    pending, focus_person=focus_person
                ),
                schema=SEMANTIC_CONFIRM_SCHEMA,
            )
    except Exception:  # noqa: BLE001
        for pid, ta_s, tb_s in pending:
            out[pid] = False
            if cache is not None:
                cache.set(ta_s, tb_s, False)
        return out

    raw = response if isinstance(response, dict) else {}
    by_id: dict[str, bool] = {}
    items = raw.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("pair_id") or "").strip()
            if not pid:
                continue
            by_id[pid] = bool(item.get("same_perspective"))

    for pid, ta_s, tb_s in pending:
        decided = by_id.get(pid, False)
        out[pid] = decided
        if cache is not None:
            cache.set(ta_s, tb_s, decided)
    return out
