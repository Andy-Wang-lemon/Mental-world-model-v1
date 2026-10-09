"""主观世界建构层校验与概率护栏。"""

from __future__ import annotations

import re

from mental_world_model.engine.extraction.validators import (
    is_grounded_in_yuanwen,
    normalize_match_text,
)
from mental_world_model.engine.perception.models import (
    FieldSource,
    GraspingHypothesis,
    GraspingType,
    ObjectiveEventStructure,
    ObjectiveFact,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import PSYCH_BAN_TERMS
from mental_world_model.engine.perception.config import ALAMBANA_DEFAULT_CONFIDENCE
from mental_world_model.engine.seeds.config import INFERENCE_CONFIDENCE_MAX
from mental_world_model.utils.confidence import as_unit_float, clamp_unit_interval


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return clamp_unit_interval(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_source(raw: object, *, yuanwen: str, text: str) -> tuple[FieldSource, float]:
    source: FieldSource = "原文" if raw == "原文" else "推理"
    confidence = 1.0 if source == "原文" else 0.6
    if source == "原文" and not is_grounded_in_yuanwen(text, yuanwen):
        return "推理", INFERENCE_CONFIDENCE_MAX
    if source == "推理":
        confidence = min(confidence, INFERENCE_CONFIDENCE_MAX)
    return source, confidence


def normalize_objective_facts(raw_list: object, yuanwen: str) -> list[ObjectiveFact]:
    if not isinstance(raw_list, list):
        return []
    facts: list[ObjectiveFact] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        span = str(item.get("source_span", "")).strip()
        if not text or not span:
            continue
        if not is_grounded_in_yuanwen(span, yuanwen):
            continue
        facts.append(ObjectiveFact(text=text, source_span=span, confidence=1.0))
    return facts


def normalize_grasping_hypothesis(
    raw: dict,
    *,
    grasping_type: GraspingType,
    yuanwen: str = "",
    index: int = 0,
) -> GraspingHypothesis | None:
    text = str(raw.get("text", "")).strip()
    if not text:
        return None
    # 硬过滤：心理语言 / 行为预测句式不得进入二执槽
    if contains_psych_language(text) or looks_like_behavior_prediction(text):
        return None
    hyp_id = str(raw.get("id", "")).strip() or f"{grasping_type}_{index + 1}"
    # 主字段 confidence；旧输出 salience 仅静默兼容
    confidence = _as_float(
        raw.get("confidence"),
        _as_float(raw.get("salience"), 0.5),
    )
    source, _ = _parse_source(raw.get("source"), yuanwen=yuanwen, text=text)
    if source == "推理":
        confidence = min(confidence, INFERENCE_CONFIDENCE_MAX)
    raw_ids = raw.get("source_bias_frame_ids") or []
    source_ids = _as_str_list(raw_ids)
    source_type_raw = str(raw.get("source_type") or "").strip()
    source_type = source_type_raw if source_type_raw else "fresh"
    return GraspingHypothesis(
        id=hyp_id,
        type=grasping_type,
        text=text,
        confidence=confidence,
        supporting_evidence=_as_str_list(raw.get("supporting_evidence")),
        counter_evidence=_as_str_list(raw.get("counter_evidence")),
        missing_information=_as_str_list(raw.get("missing_information")),
        source=source,
        source_type=source_type,  # type: ignore[arg-type]
        source_bias_frame_ids=source_ids,
    )


def normalize_probability_distribution(values: list[float]) -> list[float]:
    clipped = [max(0.0, float(v)) for v in values]
    total = sum(clipped)
    if total <= 0:
        if not clipped:
            return []
        uniform = 1.0 / len(clipped)
        return [uniform] * len(clipped)
    return [v / total for v in clipped]


def texts_semantically_near(
    a: str,
    b: str,
    *,
    min_len: int | None = None,
    jaccard_min: float | None = None,
) -> bool:
    """弱召回门闸：字面/字符袋近似（非真语义）。

    规则：
    1) 归一化后全等 → 近
    2) 一方包含另一方：较短边长度 ≥ min_len → 近
    3) 对称字符 Jaccard ≥ jaccard_min 且两边长度均 ≥ min_len → 近
    否则不算近。

    默认阈值偏松（召回门闸）；二执去重 / H 合并 / persist 在过闸后应再 LLM 确认。
    行为合并等可仅用本规则。
    """
    from mental_world_model.engine.perception.config import (
        SEMANTIC_RECALL_JACCARD_MIN,
        SEMANTIC_RECALL_MIN_LEN,
    )

    min_len = SEMANTIC_RECALL_MIN_LEN if min_len is None else int(min_len)
    jaccard_min = (
        SEMANTIC_RECALL_JACCARD_MIN if jaccard_min is None else float(jaccard_min)
    )
    na = normalize_match_text(a)
    nb = normalize_match_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= min_len and shorter in longer:
        return True
    if len(na) >= min_len and len(nb) >= min_len:
        set_a, set_b = set(na), set(nb)
        union = len(set_a | set_b)
        if union > 0 and len(set_a & set_b) / union >= jaccard_min:
            return True
    return False


def text_fingerprint(text: str, *, max_len: int = 96) -> str:
    """跨事件稳定指纹：归一化文本截断（不依赖易变 id）。"""
    blob = normalize_match_text(text)
    if not blob:
        return ""
    return blob[:max_len]


_BEHAVIOR_PREDICTION_PATTERNS = (
    re.compile(
        r"会(?:再次|继续|马上|立刻|主动)?(?:帮助|帮他|帮她|帮你|提供帮助|拒绝|采取|做出|进行)"
    ),
    re.compile(r"(?:将要|将会|准备要|打算|决定)(?:帮助|帮|拒绝|采取|做出|进行|做|提供)"),
    re.compile(
        r"(?:他|她|其|该角色|角色).{0,16}(?:会|将要|将会|打算|决定).{0,12}"
        r"(?:帮助|帮|拒绝|采取|提供|做出)"
    ),
)


def looks_like_behavior_prediction(text: str) -> bool:
    blob = str(text or "").strip()
    if not blob:
        return False
    return any(p.search(blob) for p in _BEHAVIOR_PREDICTION_PATTERNS)


def contains_psych_language(text: str) -> list[str]:
    hits: list[str] = []
    for term in PSYCH_BAN_TERMS:
        if term and term in text:
            hits.append(term)
    return hits


def parse_objective_event(raw: dict) -> ObjectiveEventStructure:
    return ObjectiveEventStructure(
        actors=_as_str_list(raw.get("actors")),
        action=str(raw.get("action") or "").strip(),
        object=str(raw.get("object") or "").strip(),
        scene=str(raw.get("scene") or "").strip(),
        outcome_changes=_as_str_list(raw.get("outcome_changes")),
        role_changes=_as_str_list(raw.get("role_changes")),
        relationship_changes=_as_str_list(raw.get("relationship_changes")),
        constraint_changes=_as_str_list(raw.get("constraint_changes")),
        resource_changes=_as_str_list(raw.get("resource_changes")),
    )


def validate_objective_event(
    event: ObjectiveEventStructure,
) -> tuple[ObjectiveEventStructure | None, list[str]]:
    notes: list[str] = []
    blob = " ".join(
        [
            *event.actors,
            event.action,
            event.object,
            event.scene,
            *event.outcome_changes,
            *event.role_changes,
            *event.relationship_changes,
            *event.constraint_changes,
            *event.resource_changes,
        ]
    )
    hits = contains_psych_language(blob)
    if hits:
        notes.append(f"客观事件含心理语言，已丢弃：{','.join(hits)}")
        return None, notes
    if not event.actors and not event.action.strip():
        notes.append("客观事件缺少 actors/action。")
        return None, notes
    return event, notes


def parse_grasping_list(
    raw_list: object,
    *,
    grasping_type: GraspingType,
    max_items: int = 8,
) -> tuple[list[GraspingHypothesis], list[str]]:
    """解析二执列表；被禁词/行为句式丢弃的条目写入 notes。"""
    notes: list[str] = []
    if not isinstance(raw_list, list):
        return [], notes
    out: list[GraspingHypothesis] = []
    label = "我执" if grasping_type == "self_grasping" else "法执"
    for index, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        hits = contains_psych_language(text)
        if hits:
            rid = str(item.get("id", "")).strip() or f"{grasping_type}_{index + 1}"
            notes.append(f"{label} {rid} 含禁词已丢弃：{','.join(hits)}")
            continue
        if looks_like_behavior_prediction(text):
            rid = str(item.get("id", "")).strip() or f"{grasping_type}_{index + 1}"
            notes.append(f"{label} {rid} 像行为预测已丢弃。")
            continue
        hyp = normalize_grasping_hypothesis(
            item, grasping_type=grasping_type, index=index
        )
        if hyp is not None:
            out.append(hyp)
        if len(out) >= max_items:
            break
    return out, notes


def parse_alambana_candidate_raw(
    raw: dict,
    *,
    index: int,
    valid_self_ids: set[str] | None = None,
    valid_dharma_ids: set[str] | None = None,
) -> tuple[SubjectiveAlambanaCandidate | None, list[str]]:
    notes: list[str] = []
    text = str(raw.get("text") or "").strip()
    if not text:
        text = str(raw.get("experienced_scene") or "").strip()
    if not text:
        return None, ["缺少 text"]

    # 一致性字段已停用（不进 prior）；若旧输出仍带则忽略
    used_self = _as_str_list(raw.get("used_self_ids") or raw.get("self_grasping_ids"))
    used_dharma = _as_str_list(
        raw.get("used_dharma_ids") or raw.get("dharma_grasping_ids")
    )
    if valid_self_ids is not None:
        bad = [i for i in used_self if i not in valid_self_ids]
        if bad:
            notes.append(f"非法我执 id：{bad}")
            used_self = [i for i in used_self if i in valid_self_ids]
    if valid_dharma_ids is not None:
        bad = [i for i in used_dharma if i not in valid_dharma_ids]
        if bad:
            notes.append(f"非法法执 id：{bad}")
            used_dharma = [i for i in used_dharma if i in valid_dharma_ids]

    cand_id = str(raw.get("id") or "").strip() or f"O{index}"
    confidence = as_unit_float(raw.get("confidence"), 0.0)
    if confidence <= 0:
        confidence = ALAMBANA_DEFAULT_CONFIDENCE
    cand = SubjectiveAlambanaCandidate(
        id=cand_id,
        text=text,
        used_self_ids=used_self,
        used_dharma_ids=used_dharma,
        self_grasping_ids=list(used_self),
        dharma_grasping_ids=list(used_dharma),
        combo_reason=str(raw.get("combo_reason") or "").strip(),
        confidence=confidence,
        objective_basis=_as_str_list(raw.get("objective_basis")),
    )
    cand.sync_grasping_aliases()
    return cand, notes


def validate_alambana_candidate(
    candidate: SubjectiveAlambanaCandidate,
) -> tuple[SubjectiveAlambanaCandidate | None, list[str]]:
    notes: list[str] = []
    blob = candidate.text.strip()
    hits = contains_psych_language(blob)
    if hits:
        notes.append(f"所缘 {candidate.id} 含禁词，已丢弃：{','.join(hits)}")
        return None, notes
    if looks_like_behavior_prediction(blob):
        notes.append(f"所缘 {candidate.id} 像行为预测，已丢弃。")
        return None, notes
    if not blob:
        notes.append(f"所缘 {candidate.id} 缺少 text。")
        return None, notes
    return candidate, notes


def normalize_alambana_candidate(
    raw: dict, *, yuanwen: str = "", index: int = 0
) -> SubjectiveAlambanaCandidate | None:
    cand, _ = parse_alambana_candidate_raw(raw, index=index + 1)
    return cand


def validate_combo_transform(
    candidate: SubjectiveAlambanaCandidate,
    *,
    expected_self_ids: list[str],
    expected_dharma_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    if looks_like_behavior_prediction(candidate.text):
        reasons.append("行为预测")
    if set(candidate.self_grasping_ids) != set(expected_self_ids):
        reasons.append("我执 id 不匹配")
    if set(candidate.dharma_grasping_ids) != set(expected_dharma_ids):
        reasons.append("法执 id 不匹配")
    return reasons
