"""Fresh 执取、语义去重、accommodation、neutral fallback。"""

from __future__ import annotations

from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    GraspingSourceType,
    ObjectiveEventStructure,
)
from mental_world_model.engine.perception.prompts import (
    GRASPING_PAIR_SCHEMA,
    build_accommodation_grasping_messages,
    build_fresh_grasping_messages,
)
from mental_world_model.engine.perception.semantic_confirm import (
    PerspectiveNearCache,
    perspectives_same,
)
from mental_world_model.engine.perception.validators import parse_grasping_list
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context


def reprefix_hypotheses(
    items: list[GraspingHypothesis],
    *,
    prefix: str,
    source_type: GraspingSourceType,
    default_frame_ids: list[str] | None = None,
) -> list[GraspingHypothesis]:
    """统一 id 前缀并写入来源，避免 Fresh/Historical 冲突。"""
    out: list[GraspingHypothesis] = []
    defaults = list(default_frame_ids or [])
    for index, hyp in enumerate(items, start=1):
        frame_ids = list(hyp.source_bias_frame_ids)
        if source_type == "historical" and not frame_ids:
            frame_ids = list(defaults)
        out.append(
            GraspingHypothesis(
                id=f"{prefix}{index}",
                type=hyp.type,
                text=hyp.text,
                confidence=hyp.confidence,
                supporting_evidence=list(hyp.supporting_evidence),
                counter_evidence=list(hyp.counter_evidence),
                missing_information=list(hyp.missing_information),
                source=hyp.source,
                source_type=source_type,
                source_bias_frame_ids=frame_ids,
            )
        )
    return out


_SOURCE_DEDUP_PRIORITY = {
    "fresh": 0,
    "accommodation": 1,
    "mixed": 2,
    "historical": 3,
    "neutral": 4,
}


def _grasping_dedup_rank(hyp: GraspingHypothesis) -> tuple[int, float, str]:
    """同视角时优先保留 Fresh，其次 accommodation，再比 confidence。"""
    source = str(hyp.source_type or "fresh")
    priority = _SOURCE_DEDUP_PRIORITY.get(source, 2)
    return (priority, -float(hyp.confidence or 0.0), hyp.id)


def dedup_grasping_list(
    items: list[GraspingHypothesis],
    *,
    llm: LlmClient | None = None,
    near_cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> list[GraspingHypothesis]:
    """语义近义且执取方向无实质差异时只留一条。

    同视角不按 Historical 的高 confidence 挤掉 Fresh。
    """
    ranked = sorted(items, key=_grasping_dedup_rank)
    kept: list[GraspingHypothesis] = []
    cache = near_cache if near_cache is not None else PerspectiveNearCache()
    for hyp in ranked:
        if any(
            perspectives_same(
                hyp.text,
                other.text,
                llm=llm,
                cache=cache,
                focus_person=focus_person,
            )
            for other in kept
        ):
            continue
        kept.append(hyp)
    return kept


def merge_and_dedup_graspings(
    self_groups: list[list[GraspingHypothesis]],
    dharma_groups: list[list[GraspingHypothesis]],
    *,
    llm: LlmClient | None = None,
    near_cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> tuple[list[GraspingHypothesis], list[GraspingHypothesis]]:
    cache = near_cache if near_cache is not None else PerspectiveNearCache()
    self_all = [h for group in self_groups for h in group]
    dharma_all = [h for group in dharma_groups for h in group]
    return (
        dedup_grasping_list(
            self_all, llm=llm, near_cache=cache, focus_person=focus_person
        ),
        dedup_grasping_list(
            dharma_all, llm=llm, near_cache=cache, focus_person=focus_person
        ),
    )


def make_neutral_graspings(*, focus_person: str = "") -> tuple[
    list[GraspingHypothesis], list[GraspingHypothesis]
]:
    who = str(focus_person or "").strip() or "主体"
    self_g = [
        GraspingHypothesis(
            id="sg_neutral",
            type="self_grasping",
            text=f"事件与{who}相关，执取未明确",
            confidence=0.4,
            source_type="neutral",
        )
    ]
    dharma_g = [
        GraspingHypothesis(
            id="dg_neutral",
            type="dharma_grasping",
            text="相关事物按字面理解",
            confidence=0.4,
            source_type="neutral",
        )
    ]
    return self_g, dharma_g


def _parse_pair(
    raw: dict,
    *,
    prefix_self: str,
    prefix_dharma: str,
    source_type: GraspingSourceType,
    default_frame_ids: list[str] | None = None,
) -> tuple[list[GraspingHypothesis], list[GraspingHypothesis], list[str]]:
    notes: list[str] = []
    self_g, n_self = parse_grasping_list(
        raw.get("self_grasping"), grasping_type="self_grasping"
    )
    dharma_g, n_dharma = parse_grasping_list(
        raw.get("dharma_grasping"), grasping_type="dharma_grasping"
    )
    notes.extend(n_self)
    notes.extend(n_dharma)
    self_g = reprefix_hypotheses(
        self_g,
        prefix=prefix_self,
        source_type=source_type,
        default_frame_ids=default_frame_ids,
    )
    dharma_g = reprefix_hypotheses(
        dharma_g,
        prefix=prefix_dharma,
        source_type=source_type,
        default_frame_ids=default_frame_ids,
    )
    return self_g, dharma_g, notes


def extract_fresh_grasping(
    text: str,
    objective_event: ObjectiveEventStructure,
    llm: LlmClient,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> tuple[list[GraspingHypothesis], list[GraspingHypothesis], list[str]]:
    notes: list[str] = []
    with llm_call_context(stage="perception.grasping_fresh"):
        response = llm.complete(
            build_fresh_grasping_messages(
                text,
                objective_event,
                role_context=role_context,
                focus_person=focus_person,
            ),
            schema=GRASPING_PAIR_SCHEMA,
        )
    raw = response if isinstance(response, dict) else {}
    self_g, dharma_g, pnotes = _parse_pair(
        raw,
        prefix_self="sg_f",
        prefix_dharma="dg_f",
        source_type="fresh",
    )
    notes.extend(pnotes)
    if not self_g and not dharma_g:
        notes.append("Fresh 路径未抽出有效执取。")
    return self_g, dharma_g, notes


def extract_accommodation_grasping(
    text: str,
    objective_event: ObjectiveEventStructure,
    llm: LlmClient,
    *,
    failed_reason: str = "",
    focus_person: str = "",
) -> tuple[list[GraspingHypothesis], list[GraspingHypothesis], list[str]]:
    notes: list[str] = []
    with llm_call_context(stage="perception.grasping_accommodation"):
        response = llm.complete(
            build_accommodation_grasping_messages(
                text,
                objective_event,
                failed_reason=failed_reason,
                focus_person=focus_person,
            ),
            schema=GRASPING_PAIR_SCHEMA,
        )
    raw = response if isinstance(response, dict) else {}
    self_g, dharma_g, pnotes = _parse_pair(
        raw,
        prefix_self="sg_a",
        prefix_dharma="dg_a",
        source_type="accommodation",
    )
    notes.extend(pnotes)
    if not self_g and not dharma_g:
        notes.append("Accommodation 路径未抽出有效执取。")
    return self_g, dharma_g, notes
