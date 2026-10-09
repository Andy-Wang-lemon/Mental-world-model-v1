"""认知连续性：H 登记为完整命题，按 mass 入库，近义合并加重。"""

from __future__ import annotations

from mental_world_model.engine.continuity.config import (
    COGNITIVE_REINFORCE_ALPHA,
)
from mental_world_model.engine.continuity.models import BiasFrame, utc_now_iso
from mental_world_model.engine.extraction.validators import normalize_match_text
from mental_world_model.engine.perception.semantic_confirm import (
    PerspectiveNearCache,
    confirm_perspective_pairs,
    perspectives_same,
)
from mental_world_model.engine.perception.validators import (
    text_fingerprint,
    texts_semantically_near,
)
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.confidence import as_unit_float, clamp_unit_interval

H_COMPRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["source_id", "text"],
            },
        }
    },
    "required": ["propositions"],
}


def _clip_clause(text: str, limit: int = 80) -> str:
    cleaned = str(text or "").strip().replace("\n", "")
    if not cleaned:
        return ""
    for sep in ("。", "；", "！", "."):
        idx = cleaned.find(sep)
        if 8 <= idx <= limit:
            return cleaned[: idx + 1]
    return cleaned[:limit]


def format_h_frame_text(
    alambana: str,
    self_grasping: list[str],
    dharma_grasping: list[str],
    xiang: str = "",
) -> str:
    """无 LLM 时的完整命题：具体事用所缘/事件，从此的倾向用二执。想不当事件。"""
    del xiang
    event = _clip_clause(alambana)
    tendency_bits = [
        str(item).strip()
        for item in [*self_grasping, *dharma_grasping]
        if str(item).strip()
    ]
    tendency = _clip_clause("；".join(tendency_bits[:2]), 120)
    if event and tendency:
        return f"{event}从此{tendency}"
    return event or tendency


def frames_from_final_branches(belief_items: list[dict]) -> list[BiasFrame]:
    """最终支 → 一条完整 H 命题（无 LLM 回退）。"""
    now = utc_now_iso()
    frames: list[BiasFrame] = []
    for index, item in enumerate(belief_items or [], start=1):
        if not isinstance(item, dict):
            continue
        text = format_h_frame_text(
            str(item.get("alambana") or ""),
            list(item.get("self_grasping") or []),
            list(item.get("dharma_grasping") or []),
            xiang=str(item.get("xiang") or ""),
        )
        if not text.strip():
            continue
        frames.append(
            BiasFrame(
                id=f"bf_{index}",
                text=text,
                mass=as_unit_float(item.get("posterior"), 0.0),
                support=1,
                last_seen=now,
            )
        )
    return frames


def build_h_compress_messages(
    yuanwen: str,
    belief_items: list[dict],
) -> list[dict[str, str]]:
    payload = []
    for item in belief_items:
        if not isinstance(item, dict):
            continue
        payload.append(
            {
                "source_id": str(item.get("id") or ""),
                "alambana": str(item.get("alambana") or ""),
                "xiang": str(item.get("xiang") or ""),
                "self_grasping": list(item.get("self_grasping") or []),
                "dharma_grasping": list(item.get("dharma_grasping") or []),
            }
        )
    system = (
        "你把本事件的认知结果压成历史认知偏置 H。\n"
        "每条必须是一句完整命题，同时包含：具体发生了什么，以及从此怎样理解这类对象/这类事。\n"
        "不要整段搬所缘布景，不要只写没有事件的性格句，不要心所名，不要行为预测。"
    )
    user = (
        f"当前原文：\n{yuanwen}\n\n"
        f"各支材料：\n{payload}\n\n"
        '返回 JSON：{"propositions":[{"source_id":"O_C1","text":""}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compress_belief_to_frames(
    yuanwen: str,
    belief_items: list[dict],
    llm: LlmClient | None = None,
) -> list[BiasFrame]:
    """事件结束：压成「具体事 + 从此的倾向」。失败则本地压缩。"""
    items = [item for item in (belief_items or []) if isinstance(item, dict)]
    if not items:
        return []
    if llm is None:
        return frames_from_final_branches(items)
    try:
        with llm_call_context(stage="continuity.h_compress"):
            raw = llm.complete(
                build_h_compress_messages(yuanwen, items),
                schema=H_COMPRESS_SCHEMA,
            )
    except Exception:  # noqa: BLE001
        return frames_from_final_branches(items)
    data = raw if isinstance(raw, dict) else {}
    rows = data.get("propositions")
    if not isinstance(rows, list) or not rows:
        return frames_from_final_branches(items)
    by_id = {str(item.get("id") or ""): item for item in items}
    now = utc_now_iso()
    frames: list[BiasFrame] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        source_id = str(row.get("source_id") or "").strip()
        src = by_id.get(source_id)
        if src is None:
            continue
        frames.append(
            BiasFrame(
                id=f"bf_{index}",
                text=text,
                mass=as_unit_float(src.get("posterior"), 0.0),
                support=1,
                last_seen=now,
            )
        )
    return frames or frames_from_final_branches(items)


def _grasping_texts_for_ids(
    ids: list[str],
    hypotheses: list,
) -> list[str]:
    """按 id 取执取正文；保持 ids 顺序，跳过缺失。"""
    by_id = {str(getattr(h, "id", "") or ""): h for h in hypotheses}
    out: list[str] = []
    for raw in ids:
        key = str(raw or "").strip()
        if not key:
            continue
        hyp = by_id.get(key)
        text = str(getattr(hyp, "text", "") or "").strip() if hyp is not None else ""
        if text:
            out.append(text)
    return out


def build_belief_items_from_branches(
    candidates: list,
    branches: list,
    *,
    self_grasping: list | None = None,
    dharma_grasping: list | None = None,
) -> list[dict]:
    """构造 H 输入：我法二执 + 主观所缘 + 后验 p' + 想（不含 activation）。"""
    self_list = list(self_grasping or [])
    dharma_list = list(dharma_grasping or [])
    xiang_by_aid: dict[str, str] = {}
    for branch in branches:
        aid = getattr(getattr(branch, "alambana", None), "id", None)
        xiang = getattr(branch, "xiang", None)
        if aid and xiang and getattr(xiang, "text", ""):
            xiang_by_aid[str(aid)] = str(xiang.text).strip()

    items: list[dict] = []
    for cand in candidates:
        getattr(cand, "sync_grasping_aliases", lambda: None)()
        used_self = list(
            getattr(cand, "used_self_ids", None)
            or getattr(cand, "self_grasping_ids", None)
            or []
        )
        used_dharma = list(
            getattr(cand, "used_dharma_ids", None)
            or getattr(cand, "dharma_grasping_ids", None)
            or []
        )
        posterior = float(getattr(cand, "posterior_probability", 0.0) or 0.0)
        items.append(
            {
                "id": cand.id,
                "self_grasping": _grasping_texts_for_ids(used_self, self_list),
                "dharma_grasping": _grasping_texts_for_ids(used_dharma, dharma_list),
                "alambana": str(getattr(cand, "text", "") or "").strip(),
                "posterior": posterior,
                "xiang": xiang_by_aid.get(str(cand.id), ""),
            }
        )
    return items


def _copy_frame(frame: BiasFrame) -> BiasFrame:
    return BiasFrame(
        id=frame.id,
        text=frame.text,
        mass=frame.mass,
        support=frame.support,
        last_seen=frame.last_seen,
    )


def merge_cognitive_frames(
    existing: list[BiasFrame],
    incoming: list[BiasFrame],
    *,
    reinforce_alpha: float = COGNITIVE_REINFORCE_ALPHA,
    llm: LlmClient | None = None,
    focus_person: str = "",
) -> list[BiasFrame]:
    """近义合成一条并加重；不同维度全部保留。

    过闸配对走批量 LLM，超过每批上限则分批问完，不截断。
    """
    now = utc_now_iso()
    frames = [_copy_frame(f) for f in existing]
    cache = PerspectiveNearCache()

    for neu in incoming:
        if not neu.text.strip():
            continue
        if llm is not None:
            pending: list[tuple[str, str, str]] = []
            for index, frame in enumerate(frames):
                if cache.get(frame.text, neu.text) is not None:
                    continue
                fa = text_fingerprint(frame.text) or normalize_match_text(frame.text)
                fb = text_fingerprint(neu.text) or normalize_match_text(neu.text)
                if fa and fa == fb:
                    cache.set(frame.text, neu.text, True)
                    continue
                if not texts_semantically_near(frame.text, neu.text):
                    cache.set(frame.text, neu.text, False)
                    continue
                pending.append((f"h{index}", frame.text, neu.text))
            if pending:
                confirm_perspective_pairs(
                    pending,
                    llm,
                    focus_person=focus_person,
                    cache=cache,
                )
        matched = next(
            (
                f
                for f in frames
                if perspectives_same(
                    f.text,
                    neu.text,
                    llm=llm,
                    cache=cache,
                    focus_person=focus_person,
                )
            ),
            None,
        )
        if matched is not None:
            matched.mass = clamp_unit_interval(
                matched.mass
                + reinforce_alpha * (1.0 - matched.mass) * neu.mass
            )
            matched.support += 1
            matched.last_seen = now
            if len(neu.text) > len(matched.text):
                matched.text = neu.text
            continue

        frames.append(
            BiasFrame(
                id=neu.id or f"bf_{len(frames) + 1}",
                text=neu.text,
                mass=as_unit_float(neu.mass, 0.4),
                support=1,
                last_seen=now,
            )
        )

    frames.sort(key=lambda f: f.mass, reverse=True)
    for index, frame in enumerate(frames, start=1):
        frame.id = f"bf_{index}"
    return frames


def update_cognitive_bias(
    existing: list[BiasFrame],
    incoming: list[BiasFrame],
    *,
    llm: LlmClient | None = None,
    focus_person: str = "",
) -> list[BiasFrame]:
    """不衰减；只近义加重或追加。"""
    return merge_cognitive_frames(
        existing,
        incoming,
        llm=llm,
        focus_person=focus_person,
    )


def lexical_h_relevance(stimulus: str, frame_text: str) -> float:
    """当前刺激与 H 正文的字符重叠（作用域 / 绑定兜底）。"""
    na = normalize_match_text(stimulus)
    nb = normalize_match_text(frame_text)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 4 and shorter in longer:
        return 1.0
    set_a, set_b = set(na), set(nb)
    union = len(set_a | set_b)
    if union <= 0:
        return 0.0
    return len(set_a & set_b) / union
