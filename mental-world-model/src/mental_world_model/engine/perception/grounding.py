"""L3 当前事件 grounding 校验：幻觉事实 / 关键变化保留。"""

from __future__ import annotations

from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
    ObjectiveEventStructure,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import (
    ALAMBANA_GROUNDING_SCHEMA,
    SUBJECTIVE_ALAMBANA_SCHEMA,
    build_alambana_grounding_messages,
    build_alambana_grounding_retry_messages,
)
from mental_world_model.engine.perception.semantic_confirm import (
    PerspectiveNearCache,
    perspectives_same,
)
from mental_world_model.engine.perception.validators import (
    parse_alambana_candidate_raw,
    validate_alambana_candidate,
)
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context


def _parse_grounding_items(raw: dict) -> dict[str, dict]:
    items = raw.get("items")
    if not isinstance(items, list):
        return {}
    out: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        aid = str(item.get("alambana_id") or "").strip()
        if not aid:
            continue
        hallucinated = bool(item.get("hallucinated_facts"))
        preserves = item.get("preserves_critical_event")
        if preserves is None:
            preserves = True
        grounded = bool(item.get("grounded")) and (not hallucinated) and bool(preserves)
        out[aid] = {
            "grounded": grounded,
            "reason": str(item.get("reason") or "").strip(),
        }
    return out


def fetch_alambana_grounding(
    text: str,
    objective_event: ObjectiveEventStructure,
    candidates: list[SubjectiveAlambanaCandidate],
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[dict[str, dict], list[str]]:
    notes: list[str] = []
    if not candidates:
        return {}, notes
    try:
        with llm_call_context(stage="perception.alambana_grounding"):
            response = llm.complete(
                build_alambana_grounding_messages(
                    text,
                    objective_event,
                    candidates,
                    focus_person=focus_person,
                ),
                schema=ALAMBANA_GROUNDING_SCHEMA,
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"grounding 校验失败，默认视为通过：{exc}")
        return {}, notes
    raw = response if isinstance(response, dict) else {}
    parsed = _parse_grounding_items(raw)
    if not parsed:
        notes.append("grounding 未返回 items，默认视为通过。")
    return parsed, notes


def _rewrite_alambana(
    text: str,
    objective_event: ObjectiveEventStructure,
    combo: GraspingCombination,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    previous: SubjectiveAlambanaCandidate,
    fail_reason: str,
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> SubjectiveAlambanaCandidate | None:
    try:
        with llm_call_context(stage="perception.alambana_grounding_retry"):
            response = llm.complete(
                build_alambana_grounding_retry_messages(
                    text,
                    objective_event,
                    combo,
                    self_grasping,
                    dharma_grasping,
                    previous.text,
                    fail_reason,
                    focus_person=focus_person,
                ),
                schema=SUBJECTIVE_ALAMBANA_SCHEMA,
            )
    except Exception:  # noqa: BLE001
        return None
    raw = response if isinstance(response, dict) else {}
    items = raw.get("candidates")
    if not isinstance(items, list) or not items:
        return None
    item = items[0] if isinstance(items[0], dict) else None
    if item is None:
        return None
    forced = dict(item)
    forced["id"] = previous.id
    forced["used_self_ids"] = list(combo.used_self_ids)
    forced["used_dharma_ids"] = list(combo.used_dharma_ids)
    forced["combo_reason"] = combo.combo_id
    cand, _ = parse_alambana_candidate_raw(forced, index=1)
    if cand is None:
        return None
    valid, _ = validate_alambana_candidate(cand)
    if valid is None:
        return None
    valid.id = previous.id
    valid.used_self_ids = list(combo.used_self_ids)
    valid.used_dharma_ids = list(combo.used_dharma_ids)
    valid.combo_reason = combo.combo_id
    valid.prior_probability = previous.prior_probability
    valid.posterior_probability = previous.posterior_probability
    valid.source_type = previous.source_type
    valid.source_bias_frame_ids = list(previous.source_bias_frame_ids)
    valid.reactivated_h_ids = list(
        getattr(previous, "reactivated_h_ids", None) or previous.source_bias_frame_ids
    )
    valid.raw_combo_score = previous.raw_combo_score
    valid.activation_alignment = previous.activation_alignment
    valid.perspective_text = previous.perspective_text
    valid.perspective_fingerprint = previous.perspective_fingerprint
    valid.confidence = previous.confidence
    valid.sync_grasping_aliases()
    return valid


def apply_grounding_with_retry(
    text: str,
    objective_event: ObjectiveEventStructure,
    candidates: list[SubjectiveAlambanaCandidate],
    combinations: list[GraspingCombination],
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[list[SubjectiveAlambanaCandidate], list[str]]:
    """失败先按原 grasping+L1 重写一次；再失败才淘汰该支。"""
    notes: list[str] = []
    if not candidates:
        return [], notes
    combo_by_id = {c.combo_id: c for c in combinations}
    verdicts, gnotes = fetch_alambana_grounding(
        text, objective_event, candidates, llm, focus_person=focus_person
    )
    notes.extend(gnotes)

    kept: list[SubjectiveAlambanaCandidate] = []
    retry_targets: list[SubjectiveAlambanaCandidate] = []
    for cand in candidates:
        info = verdicts.get(cand.id)
        if not info:
            cand.grounding_valid = True
            cand.grounding_reason = "未单独返回，默认通过"
            kept.append(cand)
            continue
        if info["grounded"]:
            cand.grounding_valid = True
            cand.grounding_reason = info["reason"] or "grounded"
            kept.append(cand)
            continue
        cand.grounding_valid = False
        cand.grounding_reason = info["reason"] or "未通过 grounding"
        retry_targets.append(cand)

    if retry_targets:
        rewritten: list[SubjectiveAlambanaCandidate] = []
        for cand in retry_targets:
            combo = combo_by_id.get((cand.combo_reason or "").strip())
            if combo is None:
                notes.append(f"{cand.id} grounding 失败且无对应组合，已淘汰。")
                continue
            notes.append(f"{cand.id} grounding 失败，尝试按当前事件重写所缘。")
            retry = _rewrite_alambana(
                text,
                objective_event,
                combo,
                self_grasping,
                dharma_grasping,
                cand,
                cand.grounding_reason,
                llm,
                focus_person=focus_person,
            )
            if retry is None:
                notes.append(f"{cand.id} 重写失败，已淘汰。")
                continue
            rewritten.append(retry)
        if rewritten:
            retry_verdicts, rnotes = fetch_alambana_grounding(
                text,
                objective_event,
                rewritten,
                llm,
                focus_person=focus_person,
            )
            notes.extend(rnotes)
            for cand in rewritten:
                info = retry_verdicts.get(cand.id)
                if not info or info["grounded"]:
                    cand.grounding_valid = True
                    cand.grounding_reason = (
                        (info or {}).get("reason") or "重写后通过"
                    )
                    kept.append(cand)
                else:
                    cand.grounding_valid = False
                    cand.grounding_reason = info["reason"] or "重写后仍未 grounding"
                    notes.append(f"{cand.id} 重写后仍未 grounding，已淘汰。")

    return kept, notes


def dedup_subjective_alambanas(
    candidates: list[SubjectiveAlambanaCandidate],
    *,
    llm: LlmClient | None = None,
    focus_person: str = "",
    max_keep: int = 3,
) -> list[SubjectiveAlambanaCandidate]:
    """跨支所缘语义去重：同一行为预测视角只留 prior 更高的一条。"""
    ranked = sorted(
        candidates,
        key=lambda c: (c.prior_probability, c.confidence),
        reverse=True,
    )
    kept: list[SubjectiveAlambanaCandidate] = []
    cache = PerspectiveNearCache()
    for cand in ranked:
        if any(
            perspectives_same(
                cand.text,
                other.text,
                llm=llm,
                cache=cache,
                focus_person=focus_person,
            )
            for other in kept
        ):
            continue
        kept.append(cand)
        if len(kept) >= max_keep:
            break
    total = sum(max(c.prior_probability, 0.0) for c in kept)
    if total > 0:
        for cand in kept:
            cand.prior_probability /= total
            cand.posterior_probability = cand.prior_probability
    elif kept:
        n = len(kept)
        for cand in kept:
            cand.prior_probability = 1.0 / n
            cand.posterior_probability = cand.prior_probability
    return kept
