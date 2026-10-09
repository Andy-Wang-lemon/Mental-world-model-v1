"""带调用统计的 LLM 客户端装饰器。"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import (
    LlmCallRecord,
    current_llm_stage,
    current_llm_tracker,
)

_progress_lock = threading.Lock()
_progress_seq = 0
_last_milestone: str | None = None

# 里程碑阶段：默认只在「换阶段」时打一行，不逐次 LLM 刷屏。
# MWM_LLM_PROGRESS=all 恢复逐次；=0 关闭。
_STAGE_MILESTONES: dict[str, str] = {
    "perception.stimulus_action_split": "感知·刺激/行动拆分",
    "perception.objective_event": "感知·客观所缘",
    "perception.grasping_pair": "感知·我法二执",
    "perception.grasping_pair_action": "感知·行动扩支二执",
    "perception.combo_meta": "感知·组合激活",
    "perception.combo_meta_action": "感知·行动扩支组合",
    "perception.subjective_alambana": "感知·主观所缘",
    "perception.semantic_confirm": "感知·语义确认",
    "cognition.action_likelihood": "认知·行动贴合",
    "cognition.branch_xiang": "认知·想",
    "cognition.behavior": "认知·行为预测",
    "continuity.event_meta": "连续性·事件元数据",
    "continuity.compress_bias": "连续性·认知压缩",
    "behavior.predict": "行为预测",
    "extraction.suoyuan": "提取·所缘",
    "extraction.xiang_evidence": "提取·想证据",
    "extraction.xiang_si": "提取·想思",
    "extraction.router": "提取·路由",
}


def is_tracing_enabled() -> bool:
    raw = os.environ.get("MWM_LLM_TRACE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def progress_mode() -> str:
    """进度模式：off | milestones（默认）| all。"""
    raw = os.environ.get("MWM_LLM_PROGRESS", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"all", "full", "2", "verbose"}:
        return "all"
    return "milestones"


def is_progress_enabled() -> bool:
    """默认开启进度；设 MWM_LLM_PROGRESS=0 关闭。"""
    return progress_mode() != "off"


def _next_progress_seq() -> int:
    global _progress_seq
    with _progress_lock:
        _progress_seq += 1
        return _progress_seq


def _log_progress(message: str) -> None:
    print(f"[MWM] {message}", file=sys.stderr, flush=True)


def _milestone_label(stage: str) -> str:
    if stage.startswith("scoring.") or stage.startswith("reconciliation."):
        return "心所评分"
    if stage in _STAGE_MILESTONES:
        return _STAGE_MILESTONES[stage]
    if stage.startswith("perception."):
        return "感知"
    if stage.startswith("cognition."):
        return "认知"
    if stage.startswith("continuity."):
        return "连续性"
    if stage.startswith("extraction."):
        return "提取"
    return stage or "LLM"


def reset_progress_for_tests() -> None:
    """测试用：重置进度序号与里程碑记忆。"""
    global _progress_seq, _last_milestone
    with _progress_lock:
        _progress_seq = 0
        _last_milestone = None


class TracingLlmClient:
    """包装任意 ``LlmClient``，记录每次 ``complete()`` 的阶段与耗时。"""

    def __init__(self, inner: LlmClient) -> None:
        self._inner = inner

    @property
    def inner(self) -> LlmClient:
        return self._inner

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        global _last_milestone
        stage = current_llm_stage()
        tracker = current_llm_tracker()
        started = time.perf_counter()
        ok = True
        error: str | None = None
        mode = progress_mode()
        seq = _next_progress_seq() if mode != "off" else 0
        milestone = _milestone_label(stage)
        log_start = False
        if mode == "all" and seq:
            log_start = True
        elif mode == "milestones" and seq:
            with _progress_lock:
                if milestone != _last_milestone:
                    _last_milestone = milestone
                    log_start = True
        if log_start:
            total = tracker.total_calls_so_far() if tracker is not None else 0
            if mode == "all":
                _log_progress(f"LLM #{seq} 开始 stage={stage}（已完成 {total}）")
            else:
                _log_progress(f"{milestone}…（累计完成 {total}）")
        try:
            return self._inner.complete(messages, schema=schema)
        except Exception as exc:  # noqa: BLE001
            ok = False
            error = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            if tracker is not None:
                tracker.record(
                    LlmCallRecord(
                        stage=stage,
                        duration_ms=duration_ms,
                        ok=ok,
                        error=error,
                    )
                )
            if mode == "all" and seq:
                status = "ok" if ok else f"FAIL:{error[:80] if error else '?'}"
                total = tracker.total_calls_so_far() if tracker is not None else seq
                _log_progress(
                    f"LLM #{seq} {status} stage={stage} "
                    f"{duration_ms:.0f}ms（累计完成 {total}）"
                )
            elif mode == "milestones" and seq and not ok:
                _log_progress(
                    f"{milestone} 失败 stage={stage}: "
                    f"{(error or '?')[:80]}"
                )


def ensure_tracing_client(client: LlmClient) -> LlmClient:
    """若 tracing 开启且尚未包装，则返回 ``TracingLlmClient``。

    链上已有 tracing（如 ``StagePrefix(Tracing(inner))``）时不再外包一层，
    避免重复计数、并让前缀客户端保持在 tracing 外侧。
    """
    if not is_tracing_enabled():
        return client
    cursor: object | None = client
    seen: set[int] = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, TracingLlmClient):
            return client
        cursor = getattr(cursor, "inner", None)
    return TracingLlmClient(client)
