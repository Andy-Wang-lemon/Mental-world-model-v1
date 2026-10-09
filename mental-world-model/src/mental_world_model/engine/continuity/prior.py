"""H 作为长期认知先验：作用域资格 → 调制哪条 Fresh 执。"""

from __future__ import annotations

from mental_world_model.engine.continuity.cognitive import lexical_h_relevance
from mental_world_model.engine.continuity.config import (
    COGNITIVE_TOP_K_INJECT,
    H_MODULATION_GAIN,
    H_SCOPE_LEXICAL_FLOOR,
)
from mental_world_model.engine.continuity.models import BiasFrame
from mental_world_model.engine.perception.models import GraspingHypothesis
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.confidence import clamp_unit_interval

H_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string"},
                    "triggered": {"type": "boolean"},
                },
                "required": ["frame_id", "triggered"],
            },
        }
    },
    "required": ["items"],
}

H_BIND_SCHEMA = {
    "type": "object",
    "properties": {
        "bindings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string"},
                    "grasping_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["frame_id", "grasping_ids"],
            },
        }
    },
    "required": ["bindings"],
}


def _as_bool(value: object, default: bool = False) -> bool:
    """LLM 常把 false 写成字符串；bool("false") 为真，必须单独解析。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0", ""}:
            return False
    return default


def frames_from_injection(raw: list | None) -> list[BiasFrame]:
    frames: list[BiasFrame] = []
    for item in raw or []:
        if isinstance(item, BiasFrame):
            frames.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        fid = str(item.get("id") or item.get("frame_id") or "").strip()
        if not text or not fid:
            continue
        frames.append(
            BiasFrame(
                id=fid,
                text=text,
                mass=float(item.get("mass") or 0.0),
                support=int(item.get("support") or 0),
            )
        )
    return frames


def working_set_by_mass(
    frames: list[BiasFrame],
    *,
    k: int = COGNITIVE_TOP_K_INJECT,
) -> list[BiasFrame]:
    if k <= 0 or not frames:
        return []
    ranked = sorted(frames, key=lambda f: float(f.mass), reverse=True)
    return ranked[:k]


def build_h_scope_messages(yuanwen: str, frames: list[BiasFrame]) -> list[dict[str, str]]:
    payload = [{"frame_id": f.id, "text": f.text} for f in frames]
    system = (
        "你判断历史认知偏置 H 的作用域是否被当前刺激打中。\n"
        "H 是「从此怎样理解某类对象/某类结构」的沉淀，不是场面回忆。\n"
        "triggered=true：当前刺激碰到了这条 H 所管的对象或那类结构变化"
        "（位置、边界、知情、资格、归属、对错等），不论布景是否像当年。\n"
        "triggered=false：当前刺激没有碰到这条 H 的作用域。\n"
        "不要改写 H，不要输出心所名或行为预测。"
    )
    user = (
        f"当前刺激：\n{yuanwen}\n\n"
        f"H 列表：\n{payload}\n\n"
        '返回 JSON：{"items":[{"frame_id":"bf_1","triggered":false}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def trigger_h_scope(
    yuanwen: str,
    frames: list[BiasFrame],
    llm: LlmClient | None = None,
) -> list[BiasFrame]:
    """第一级：本场能不能用。不决定绑到哪条执。"""
    if not frames:
        return []
    lexical_hit = [
        f
        for f in frames
        if lexical_h_relevance(yuanwen, f.text) >= H_SCOPE_LEXICAL_FLOOR
    ]
    if llm is None:
        return lexical_hit
    try:
        with llm_call_context(stage="continuity.h_scope"):
            raw = llm.complete(
                build_h_scope_messages(yuanwen, frames),
                schema=H_SCOPE_SCHEMA,
            )
    except Exception:  # noqa: BLE001
        return lexical_hit
    data = raw if isinstance(raw, dict) else {}
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return lexical_hit
    allowed = {f.id: f for f in frames}
    triggered: list[BiasFrame] = []
    seen: set[str] = set()
    recognized = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("frame_id") or "").strip()
        if fid not in allowed or fid in seen:
            continue
        recognized += 1
        if not _as_bool(item.get("triggered")):
            seen.add(fid)
            continue
        seen.add(fid)
        triggered.append(allowed[fid])
    if recognized == 0:
        return lexical_hit
    return triggered


def build_h_bind_messages(
    frames: list[BiasFrame],
    graspings: list[GraspingHypothesis],
) -> list[dict[str, str]]:
    h_payload = [{"frame_id": f.id, "text": f.text} for f in frames]
    g_payload = [{"id": h.id, "text": h.text} for h in graspings]
    system = (
        "这些 H 已经确定进入本场。你只决定每条 H 具体作用于哪几条当场执取。\n"
        "grasping_ids 只能从给定执取 id 中选；对不上则给空列表。\n"
        "不要再判断 H 该不该触发。不要改写正文。不要输出心所名或行为。"
    )
    user = (
        f"已触发 H：\n{h_payload}\n\n"
        f"当场执取：\n{g_payload}\n\n"
        '返回 JSON：{"bindings":[{"frame_id":"bf_1","grasping_ids":[]}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _lexical_bind(
    frames: list[BiasFrame],
    graspings: list[GraspingHypothesis],
) -> dict[str, list[str]]:
    by_h: dict[str, list[str]] = {f.id: [] for f in frames}
    for frame in frames:
        for hyp in graspings:
            score = lexical_h_relevance(hyp.text, frame.text)
            if score >= H_SCOPE_LEXICAL_FLOOR:
                by_h[frame.id].append(hyp.id)
    return by_h


def bind_h_to_fresh(
    frames: list[BiasFrame],
    graspings: list[GraspingHypothesis],
    llm: LlmClient | None = None,
) -> dict[str, list[str]]:
    """第二级：已触发的 H 落到 F1/F2/F3。不再问该不该触发。"""
    known_g = {h.id for h in graspings}
    known_h = {f.id for f in frames}
    fallback = _lexical_bind(frames, graspings)
    if llm is None or not frames or not graspings:
        return fallback
    try:
        with llm_call_context(stage="continuity.h_bind"):
            raw = llm.complete(
                build_h_bind_messages(frames, graspings),
                schema=H_BIND_SCHEMA,
            )
    except Exception:  # noqa: BLE001
        return fallback
    data = raw if isinstance(raw, dict) else {}
    bindings = data.get("bindings")
    if not isinstance(bindings, list):
        return fallback
    out: dict[str, list[str]] = {fid: [] for fid in known_h}
    for item in bindings:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("frame_id") or "").strip()
        if fid not in known_h:
            continue
        ids: list[str] = []
        seen: set[str] = set()
        for raw_id in item.get("grasping_ids") or []:
            gid = str(raw_id or "").strip()
            if gid in known_g and gid not in seen:
                seen.add(gid)
                ids.append(gid)
        out[fid] = ids
    return out


def modulate_fresh_graspings(
    graspings: list[GraspingHypothesis],
    frames: list[BiasFrame],
    bindings: dict[str, list[str]],
    *,
    gain: float = H_MODULATION_GAIN,
) -> None:
    """就地加强被绑到的 Fresh 执；不改 text，不改 source_type。"""
    by_id = {h.id: h for h in graspings}
    by_frame = {f.id: f for f in frames}
    for frame_id, grasping_ids in bindings.items():
        frame = by_frame.get(frame_id)
        if frame is None:
            continue
        for gid in grasping_ids:
            hyp = by_id.get(gid)
            if hyp is None:
                continue
            mass = max(0.0, float(frame.mass))
            hyp.confidence = clamp_unit_interval(
                float(hyp.confidence) + gain * mass * (1.0 - float(hyp.confidence))
            )
            if frame_id not in hyp.source_bias_frame_ids:
                hyp.source_bias_frame_ids.append(frame_id)


def _h_prior_trace(
    working: list[BiasFrame],
    triggered: list[BiasFrame],
    bindings: dict[str, list[str]] | None = None,
) -> dict:
    return {
        "working_set": [{"id": f.id, "text": f.text, "mass": f.mass} for f in working],
        "triggered_ids": [f.id for f in triggered],
        "bindings": {fid: list(gids) for fid, gids in (bindings or {}).items()},
    }


def resolve_h_scope(
    yuanwen: str,
    frames: list[BiasFrame] | list[dict] | None,
    llm: LlmClient | None = None,
) -> dict:
    """本事件只算一次：哪些 H 有资格进场。"""
    working = working_set_by_mass(frames_from_injection(frames))
    triggered = trigger_h_scope(yuanwen, working, llm) if working else []
    return {
        "working": working,
        "triggered": triggered,
        "trace": _h_prior_trace(working, triggered),
    }


def bind_h_scope_to_fresh(
    scope: dict,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    llm: LlmClient | None = None,
) -> dict:
    """已触发的 H 落到当前执；不再问该不该触发。"""
    working = list(scope.get("working") or [])
    triggered = list(scope.get("triggered") or [])
    graspings = [h for h in (*self_grasping, *dharma_grasping) if h is not None]
    bindings = bind_h_to_fresh(triggered, graspings, llm) if triggered else {}
    if triggered and bindings:
        modulate_fresh_graspings(graspings, triggered, bindings)
    return _h_prior_trace(working, triggered, bindings)


def apply_h_prior_to_fresh(
    yuanwen: str,
    frames: list[BiasFrame] | list[dict] | None,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    llm: LlmClient | None = None,
) -> dict:
    """库按 mass 取工作集 → 作用域资格 → 绑定并调制 Fresh 执。"""
    scope = resolve_h_scope(yuanwen, frames, llm)
    return bind_h_scope_to_fresh(scope, self_grasping, dharma_grasping, llm)


def h_texts_by_id(h_prior: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for fid, rec in h_records_by_id(h_prior).items():
        text = str(rec.get("text") or "").strip()
        if text:
            out[fid] = text
    return out


def h_records_by_id(h_prior: dict | None) -> dict[str, dict]:
    """working_set → {id: {id, text, mass}}。"""
    out: dict[str, dict] = {}
    if not isinstance(h_prior, dict):
        return out
    for item in h_prior.get("working_set") or []:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id") or "").strip()
        if not fid:
            continue
        try:
            mass = float(item.get("mass") or 0.0)
        except (TypeError, ValueError):
            mass = 0.0
        out[fid] = {
            "id": fid,
            "text": str(item.get("text") or "").strip(),
            "mass": round(mass, 4),
        }
    return out


def h_payload_for_candidate(cand, h_prior: dict | None) -> list[dict]:
    """本支已绑定 H 的只读载荷；未知 id / 无正文的条目跳过。"""
    records = h_records_by_id(h_prior)
    ids: list[str] = []
    raw_ids = list(
        getattr(cand, "reactivated_h_ids", None)
        or getattr(cand, "source_bias_frame_ids", None)
        or []
    )
    for hid in raw_ids:
        token = str(hid or "").strip()
        if token and token not in ids:
            ids.append(token)
    payload: list[dict] = []
    for hid in ids:
        rec = records.get(hid)
        if rec is None or not rec.get("text"):
            continue
        payload.append(rec)
    return payload
