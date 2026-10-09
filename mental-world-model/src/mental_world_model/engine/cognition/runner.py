"""多认知分支评估编排（v4）。

流程：
  感知先验所缘（prior=组合激活 P(C)；最多 Top-K 支，不指定主所缘）
  → L0 行动骨架即思（每事件一次，各支共享；不另开思 LLM）
  → 有行动时：compat →（必要时）行动扩支 L2'→combo'→L3' → 旧支后验 → 搭支 → 各支想
  →【屏障】想与后验就绪后，再各支 40 心所（业列=共享思或空）
  → 跨支激活加权期望心所 / 行为贡献融合
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.core.relations.relation import Relation
from mental_world_model.engine.perception.action_expand import (
    finalize_mixed_action_posteriors,
    maybe_expand_action_informed_branch,
)
from mental_world_model.engine.cognition.action_evidence import (
    ActionEvidence,
    action_skeleton_from_perception_split,
    bind_action_as_si,
    generate_branch_xiangs,
    score_action_likelihoods_on_skeleton,
    update_alambana_posteriors_with_action,
)
from mental_world_model.engine.cognition.adapters import (
    attach_causal_context,
    branch_to_scoring_context,
)
from mental_world_model.engine.cognition.behavior import predict_behaviors_for_branch
from mental_world_model.engine.cognition.predicted_action import (
    is_usable_predicted_action,
)
from mental_world_model.engine.cognition.models import (
    AggregatedBehaviorPrediction,
    CognitionBranch,
    ExtractedMentalState,
)
from mental_world_model.engine.cognition.probability import (
    aggregate_behavior_predictions,
    calculate_expected_mental_factor_scores,
    normalize_branch_probabilities,
)
from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.perception.builder import build_perception_state
from mental_world_model.engine.perception.config import (
    AGGREGATED_ACTION_TOP_K,
    BEHAVIOR_NORMALIZATION_ENABLED,
)
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    PerceptionState,
    StimulusActionSplit,
)
from mental_world_model.engine.reconciliation.reconciler import DuizhiReconciler
from mental_world_model.engine.scoring.parallel_exec import run_parallel
from mental_world_model.engine.scoring.pipeline import XinsuoScoringPipeline
from mental_world_model.engine.scoring.results import EvaluationResult, XinsuoScoreResult
from mental_world_model.engine.scoring.weights import ScoringWeights
from mental_world_model.engine.validator.definition_validator import DefinitionValidator
from mental_world_model.llm.client import LlmClient
from mental_world_model.engine.continuity.config import (
    is_cognitive_continuity_enabled,
    is_mental_continuity_enabled,
)
from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.continuity.service import (
    apply_mental_continuity,
    bias_frames_for_injection,
    blend_branches_with_mental_history,
    resolve_mental_history,
    update_continuity_after_event,
)
from mental_world_model.engine.cognition.xiang_si import ExplicitActionDetection


@dataclass
class CognitionEvaluationBundle:
    """多分支评估完整产物。"""

    perception_state: PerceptionState
    branches: list[CognitionBranch]
    aggregated_behaviors: list[AggregatedBehaviorPrediction]
    expected_mental_factor_scores: dict[str, float]
    evaluation_result: EvaluationResult
    continuity_state: ContinuityState | None = None
    # 兼容旧字段：不再计算，恒为 None
    mental_factor_presence_probability: dict[str, float] | None = None
    overall_uncertainty: str | None = None


def _resolve_grasping(
    ids: list[str],
    hypotheses: list[GraspingHypothesis],
    *,
    alambana_id: str,
    kind: str,
) -> tuple[list[GraspingHypothesis], list[str]]:
    """仅精确 id 或精确全文匹配；禁止模糊包含，禁止静默 top-k 回退。"""
    notes: list[str] = []
    cleaned = [str(raw).strip() for raw in ids if str(raw).strip()]
    if not cleaned:
        return [], notes

    by_id = {h.id: h for h in hypotheses}
    by_text = {h.text.strip(): h for h in hypotheses if h.text.strip()}
    resolved: list[GraspingHypothesis] = []
    missing: list[str] = []

    for key in cleaned:
        if key in by_id:
            resolved.append(by_id[key])
            continue
        if key in by_text:
            hit = by_text[key]
            resolved.append(hit)
            notes.append(f"{kind}引用按精确文本匹配：{key!r}→{hit.id}")
            continue
        missing.append(key)

    if missing:
        notes.append(f"{alambana_id} {kind}引用无效（无模糊回退）：{missing}")

    if not resolved and cleaned:
        notes.append(
            f"{alambana_id} {kind}全部引用无效，本支该侧执取留空（不静默回退强执）。"
        )

    seen: set[str] = set()
    unique: list[GraspingHypothesis] = []
    for item in resolved:
        if item.id in seen:
            continue
        seen.add(item.id)
        unique.append(item)
    return unique, notes


def _build_branches_from_candidates(
    perception: PerceptionState,
) -> tuple[
    list[CognitionBranch],
    dict[str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]],
]:
    """搭分支壳 + 解析二执（本地，无 LLM）。"""
    branch_grasping: dict[str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]] = {}
    branches: list[CognitionBranch] = []
    for alambana in perception.alambana_candidates:
        alambana.sync_grasping_aliases()
        branch = CognitionBranch(
            id=f"branch_{alambana.id}",
            alambana=alambana,
            branch_prior_probability=alambana.prior_probability,
        )
        self_g, self_notes = _resolve_grasping(
            alambana.used_self_ids or alambana.self_grasping_ids,
            perception.self_grasping_hypotheses,
            alambana_id=alambana.id,
            kind="我执",
        )
        dharma_g, dharma_notes = _resolve_grasping(
            alambana.used_dharma_ids or alambana.dharma_grasping_ids,
            perception.dharma_grasping_hypotheses,
            alambana_id=alambana.id,
            kind="法执",
        )
        branch.uncertainty_notes.extend(self_notes)
        branch.uncertainty_notes.extend(dharma_notes)
        perception.uncertainty_notes.extend(self_notes)
        perception.uncertainty_notes.extend(dharma_notes)
        branch_grasping[branch.id] = (self_g, dharma_g)
        branches.append(branch)
    return branches, branch_grasping


def _xiang_fallbacks_from_perception(
    perception: PerceptionState,
    reason: str,
) -> dict[str, ExtractedMentalState]:
    return {
        cand.id: ExtractedMentalState(
            text=(
                f"在此所缘下：{cand.text}"
                if cand.text
                else "对该事态形成认知评价"
            ),
            source="推理",
            confidence=0.4,
            alambana_id=cand.id,
            reason_summary=reason,
        )
        for cand in perception.alambana_candidates
    }


def _compat_expand_and_xiang(
    text: str,
    perception: PerceptionState,
    action_evidence: ActionEvidence,
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[
    ActionEvidence,
    list[CognitionBranch],
    dict[str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]],
    dict[str, ExtractedMentalState],
]:
    """在已有所缘上做 compat/扩支/后验/想。不重抽 L2/L3 prior。"""
    focus = str(focus_person or "").strip()
    need_compat = (
        action_evidence.has_observable_action
        and bool(perception.alambana_candidates)
    )

    def _run_xiang(
        evidence: ActionEvidence,
        grasping_map: dict[
            str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]
        ],
    ) -> tuple[dict[str, ExtractedMentalState], list[str]]:
        notes: list[str] = []
        if not perception.alambana_candidates:
            return {}, notes
        try:
            by_id = generate_branch_xiangs(
                text,
                perception.objective_event,
                perception.alambana_candidates,
                perception.self_grasping_hypotheses,
                perception.dharma_grasping_hypotheses,
                evidence,
                llm,
                grasping_by_alambana_id=grasping_map,
                focus_person=focus,
                h_prior=getattr(perception, "h_prior", None) or {},
            )
            notes.append(f"想已分支出成（{len(by_id)} 支隔离并行）。")
            return by_id, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"想生成失败，已按所缘兜底：{exc}")
            return _xiang_fallbacks_from_perception(perception, "想生成异常兜底"), notes

    if need_compat:
        try:
            evidence, score_notes = score_action_likelihoods_on_skeleton(
                text,
                perception.objective_event,
                perception.objective_facts,
                perception.alambana_candidates,
                action_evidence,
                llm,
                focus_person=focus,
            )
            perception.uncertainty_notes.extend(score_notes)
        except Exception as exc:  # noqa: BLE001
            evidence = action_evidence
            perception.uncertainty_notes.append(
                f"[强提示] compat 失败，后验按中性更新：{exc}"
            )
        expanded = maybe_expand_action_informed_branch(
            text,
            perception,
            action_evidence,
            evidence,
            llm,
            perception.uncertainty_notes,
            focus_person=focus,
        )
        update_alambana_posteriors_with_action(
            perception.alambana_candidates,
            evidence,
            perception.uncertainty_notes,
        )
        if expanded:
            finalize_mixed_action_posteriors(
                perception.alambana_candidates,
                perception.uncertainty_notes,
            )
            perception.uncertainty_notes.append(
                "行动扩支完成：旧支已 compat 矫正，新支 action_informed 保持 prior。"
            )
        action_evidence = evidence
        branches, branch_grasping = _build_branches_from_candidates(perception)
        grasping_by_aid = {
            alambana.id: branch_grasping[f"branch_{alambana.id}"]
            for alambana in perception.alambana_candidates
            if f"branch_{alambana.id}" in branch_grasping
        }
        xiang_by_id, xiang_notes = _run_xiang(action_evidence, grasping_by_aid)
        perception.uncertainty_notes.extend(xiang_notes)
        perception.uncertainty_notes.append(
            f"compat/扩支/后验已完成（{len(perception.alambana_candidates)} 支）；"
            f"想 {len(xiang_by_id)} 支；心所等待想就绪。"
        )
    else:
        update_alambana_posteriors_with_action(
            perception.alambana_candidates,
            action_evidence,
            perception.uncertainty_notes,
        )
        branches, branch_grasping = _build_branches_from_candidates(perception)
        grasping_by_aid = {
            alambana.id: branch_grasping[f"branch_{alambana.id}"]
            for alambana in perception.alambana_candidates
            if f"branch_{alambana.id}" in branch_grasping
        }
        xiang_by_id, xiang_notes = _run_xiang(action_evidence, grasping_by_aid)
        perception.uncertainty_notes.extend(xiang_notes)

    if action_evidence.has_observable_action:
        perception.uncertainty_notes.append(
            "行动证据已更新后验："
            f"{action_evidence.action_span or action_evidence.action_description}"
        )
    else:
        perception.uncertainty_notes.append("无行动证据：后验=先验；思将置空。")
    return action_evidence, branches, branch_grasping, xiang_by_id


def _multi_branch_extracted(
    yuanwen: str,
    branches: list[CognitionBranch],
    shared_si: ExtractedMentalState | None,
) -> ExtractedContext:
    """多支顶层 extracted：并列各支所缘/想；思用共享行动证据。"""
    if not branches:
        empty = ExtractedField(text="", source="推理", confidence=0.0)
        return ExtractedContext(
            yuanwen=yuanwen,
            suoyuan=empty,
            xiang=empty,
            si=empty,
            scoring_path="explicit",
        )
    suoyuan_parts: list[str] = []
    xiang_parts: list[str] = []
    for branch in branches:
        a = branch.activation_probability or branch.branch_effective_probability
        label = f"[{branch.id} a={a:.3f}]"
        suoyuan_parts.append(f"{label} {branch.alambana.text}".strip())
        xiang_text = branch.xiang.text if branch.xiang else ""
        if xiang_text:
            xiang_parts.append(f"{label} {xiang_text}".strip())
    si_text = shared_si.text if shared_si and shared_si.text else ""
    if not si_text and branches[0].si:
        si_text = branches[0].si.text
    return ExtractedContext(
        yuanwen=yuanwen,
        suoyuan=ExtractedField(
            text="\n".join(suoyuan_parts),
            source="推理",
            confidence=1.0,
        ),
        xiang=ExtractedField(
            text="\n".join(xiang_parts),
            source="推理",
            confidence=1.0,
        ),
        si=ExtractedField(
            text=si_text,
            source=(shared_si.source if shared_si and si_text else "推理"),
            confidence=(shared_si.confidence if shared_si and si_text else 0.0),
        ),
        scoring_path="explicit",
    )


def _merge_branch_resolutions(branches: list[CognitionBranch]) -> list:
    merged: list = []
    for branch in branches:
        for item in branch.resolutions or []:
            merged.append(item)
    return merged


def _expected_raw_scores(
    branches: list[CognitionBranch],
    factor_names: list[str],
) -> dict[str, float]:
    """跨支期望的消解前心所（权重 = activation）。"""
    expected: dict[str, float] = {name: 0.0 for name in factor_names}
    for branch in branches:
        p = branch.activation_probability or branch.branch_effective_probability
        if p <= 0:
            continue
        for name in factor_names:
            expected[name] += p * branch.mental_factor_scores_raw.get(name, 0.0)
    return expected


def _score_pack_bundle(
    *,
    text: str,
    perception: PerceptionState,
    branches: list[CognitionBranch],
    xiang_by_id: dict[str, ExtractedMentalState],
    action_evidence: ActionEvidence,
    llm: LlmClient,
    xinsuo_list,
    relations,
    weights,
    validator,
    continuity: ContinuityState,
    bias_frames: list,
    focus: str,
    role_context: str,
    update_continuity: bool,
    aggregate_behaviors: bool,
    predict_branch_behaviors: bool,
) -> CognitionEvaluationBundle:
    """屏障后：绑想/思、评分、可选行为预测、可选写 H/M。"""
    for cand in perception.alambana_candidates:
        x = xiang_by_id.get(cand.id)
        if x is None or not x.text:
            perception.uncertainty_notes.append(f"{cand.id} 想缺失已兜底。")

    shared_si_text = bind_action_as_si(action_evidence, "shared")
    for branch in branches:
        xiang = xiang_by_id.get(branch.alambana.id) or ExtractedMentalState(
            text=branch.alambana.text or "对该事态形成认知评价",
            source="推理",
            confidence=0.4,
            alambana_id=branch.alambana.id,
            reason_summary="想兜底",
        )
        branch.xiang = ExtractedMentalState(
            text=xiang.text,
            source=xiang.source,
            confidence=xiang.confidence,
            alambana_id=branch.alambana.id,
            reason_summary=xiang.reason_summary,
        )
        branch.si = ExtractedMentalState(
            text=shared_si_text.text,
            source=shared_si_text.source,
            confidence=shared_si_text.confidence,
            alambana_id=branch.alambana.id,
            reason_summary=shared_si_text.reason_summary,
            xiang_text=branch.xiang.text,
        )

    xinsuo_by_name = {x.name: x for x in xinsuo_list}
    factor_names = [x.name for x in xinsuo_list]
    reconciler = DuizhiReconciler(llm, validator=validator)

    def _score_branch(branch: CognitionBranch) -> CognitionBranch:
        pipeline = XinsuoScoringPipeline(llm, weights=weights)
        context = branch_to_scoring_context(text, branch)
        try:
            stage_results = pipeline.run(xinsuo_list, context)
            scores_raw = pipeline.to_scores_dict(stage_results)
            try:
                scores_final, resolutions = reconciler.reconcile(
                    scores_raw,
                    relations,
                    xinsuo_by_name,
                    context,
                )
            except Exception as rec_exc:  # noqa: BLE001
                scores_final = dict(scores_raw)
                resolutions = []
                branch.uncertainty_notes.append(f"对治消解失败，已回退原始分：{rec_exc}")
            breakdowns = {
                item.name: item.score_breakdown
                for item in stage_results
                if item.score_breakdown is not None
            }
            branch.mental_factor_scores_raw = scores_raw
            branch.mental_factor_scores_final = scores_final
            branch.score_breakdowns = attach_causal_context(breakdowns, branch)
            branch.resolutions = resolutions
            branch.stage_results = stage_results
            branch.behavior_predictions = []
        except Exception as exc:  # noqa: BLE001
            branch.status = "scoring_failed"
            branch.uncertainty_notes.append(f"评分/消解失败：{exc}")
            branch.mental_factor_scores_raw = {n: 0.0 for n in factor_names}
            branch.mental_factor_scores_final = dict(branch.mental_factor_scores_raw)
        return branch

    scored = run_parallel(
        {branch.id: (lambda b=branch: _score_branch(b)) for branch in branches},
        acquire_slot=False,
    )
    branches = [scored[branch.id] for branch in branches]
    normalize_branch_probabilities(branches)

    expected = calculate_expected_mental_factor_scores(branches, factor_names)

    event_scores = dict(expected)
    mental_trace: dict = {"enabled": False}
    scores_final = dict(event_scores)
    report_active_xinsuo = True
    if predict_branch_behaviors:
        mental_trace = {
            "enabled": is_mental_continuity_enabled(),
            "apply_mode": "per_branch_prediction",
            "mental_factors_event": dict(event_scores),
        }
        if is_mental_continuity_enabled():
            hist_scores, lam, mental_trace = resolve_mental_history(
                yuanwen=text,
                state=continuity,
                llm=llm,
                factor_names=factor_names,
                event_scores=event_scores,
            )
            mental_trace["apply_mode"] = "per_branch_prediction"
            blend_branches_with_mental_history(
                branches,
                hist_scores=hist_scores,
                lambda_eff=lam,
            )
        scores_final = dict(event_scores)
        report_active_xinsuo = False
    elif is_mental_continuity_enabled():
        scores_final, mental_trace = apply_mental_continuity(
            yuanwen=text,
            event_scores=event_scores,
            state=continuity,
            llm=llm,
            factor_names=factor_names,
        )

    if predict_branch_behaviors:
        def _predict_one(branch: CognitionBranch):
            return predict_behaviors_for_branch(
                text,
                branch,
                branch.mental_factor_scores_final,
                xinsuo_by_name,
                llm,
                role_context=role_context,
                focus_person=focus,
            )

        predicted = run_parallel(
            {branch.id: (lambda b=branch: _predict_one(b)) for branch in branches},
            acquire_slot=True,
        )
        for branch in branches:
            branch.behavior_predictions = predicted.get(branch.id) or []

    behavior_diagnostics = {
        "raw_candidate_count": 0,
        "distinct_candidate_count": 0,
        "diversity_regenerated": False,
        "candidate_collapsed": False,
    }
    if predict_branch_behaviors:
        diags = [
            dict(getattr(branch, "behavior_diagnostics", None) or {})
            for branch in branches
        ]
        behavior_diagnostics = {
            "raw_candidate_count": sum(
                int(d.get("raw_candidate_count") or 0) for d in diags
            ),
            "distinct_candidate_count": sum(
                int(d.get("distinct_candidate_count") or 0) for d in diags
            ),
            "diversity_regenerated": any(
                bool(d.get("diversity_regenerated")) for d in diags
            ),
            "candidate_collapsed": any(
                bool(d.get("candidate_collapsed")) for d in diags
            ),
            "branches": {
                branch.id: dict(getattr(branch, "behavior_diagnostics", None) or {})
                for branch in branches
            },
        }

    aggregated = (
        aggregate_behavior_predictions(
            branches,
            normalization_enabled=BEHAVIOR_NORMALIZATION_ENABLED,
            top_k=AGGREGATED_ACTION_TOP_K,
        )
        if aggregate_behaviors
        else []
    )

    scores_initial = _expected_raw_scores(branches, factor_names)
    resolutions = _merge_branch_resolutions(branches)
    conflicts_initial: list = []
    conflicts_final: list = []
    shared_si_for_result = (
        shared_si_text if action_evidence.has_observable_action else None
    )
    extracted = _multi_branch_extracted(text, branches, shared_si_for_result)
    stage_results: list[XinsuoScoreResult] = []

    action_detection = ExplicitActionDetection(
        has_explicit_action=action_evidence.has_observable_action,
        action_span=action_evidence.action_span,
        si_text=action_evidence.summary_text(),
        confidence=action_evidence.confidence,
        reason_summary="",
    )

    continuity_trace: dict = {
        "cognitive_enabled": is_cognitive_continuity_enabled(),
        "mental": mental_trace,
        "injected_bias_frames": bias_frames,
        "h_prior": dict(getattr(perception, "h_prior", None) or {}),
    }
    if update_continuity and (
        is_cognitive_continuity_enabled() or is_mental_continuity_enabled()
    ):
        continuity, cont_notes = update_continuity_after_event(
            yuanwen=text,
            state=continuity,
            llm=llm,
            candidates=perception.alambana_candidates,
            branches=branches,
            event_scores=event_scores,
            mental_trace=mental_trace,
            self_grasping=perception.self_grasping_hypotheses,
            dharma_grasping=perception.dharma_grasping_hypotheses,
        )
        if cont_notes:
            continuity_trace["notes"] = list(cont_notes)
            perception.uncertainty_notes.extend(cont_notes)
        continuity_trace["cognitive_frames_after"] = [
            f.to_dict() for f in continuity.cognitive_frames
        ]
        continuity_trace["mental_episode_count"] = len(continuity.mental_episodes)

    evaluation = EvaluationResult(
        extracted=extracted,
        stage_results=stage_results,
        scores_initial=scores_initial,
        conflicts_initial=conflicts_initial,
        conflicts_final=conflicts_final,
        resolutions=resolutions,
        scores_final=scores_final,
        predicted_behavior=None,
        active_xinsuo=(
            (
                [
                    {"name": n, "score": s}
                    for n, s in sorted(
                        event_scores.items(), key=lambda kv: kv[1], reverse=True
                    )
                    if s > 0
                ][:12]
                or None
            )
            if report_active_xinsuo
            else None
        ),
        prediction_reason=None,
        perception_state=perception,
        cognition_branches=branches,
        aggregated_behaviors=aggregated,
        expected_mental_factor_scores=event_scores,
        mental_factor_presence_probability=None,
        overall_uncertainty=None,
        action_detection=action_detection,
        shared_si=shared_si_for_result,
        action_evidence=action_evidence.to_dict(),
        continuity=continuity_trace,
        mental_factors_event=event_scores,
        focus_person=focus,
        behavior_diagnostics=behavior_diagnostics,
    )

    return CognitionEvaluationBundle(
        perception_state=perception,
        branches=branches,
        aggregated_behaviors=aggregated,
        expected_mental_factor_scores=event_scores,
        evaluation_result=evaluation,
        continuity_state=continuity,
    )


def evaluate_text_with_cognition_branches(
    text: str,
    llm: LlmClient,
    *,
    xinsuo_list: list[Xinsuo],
    relations: list[Relation],
    weights: ScoringWeights,
    validator: DefinitionValidator | None = None,
    role_context: str = "",
    continuity_state: ContinuityState | None = None,
    update_continuity: bool = True,
    use_action_feedback: bool = False,
    aggregate_behaviors: bool = False,
    predict_branch_behaviors: bool = True,
    focus_person: str = "",
) -> CognitionEvaluationBundle:
    """单事件：先验所缘 →（可选）compat→扩支→后验→搭支→想 → 屏障后心所 → 连续状态更新。

    思仅来自 L0 行动骨架（每事件一次、各支共享）；想不依赖后验概率。
    """
    validator = validator or DefinitionValidator()
    continuity = continuity_state or ContinuityState()
    focus = str(focus_person or "").strip()
    if focus and continuity.person_id in ("", "default"):
        continuity.person_id = focus
    bias_frames = bias_frames_for_injection(continuity)

    perception = build_perception_state(
        text,
        llm,
        role_context=role_context,
        cognitive_bias_frames=bias_frames or None,
        expect_action=use_action_feedback,
        focus_person=focus,
    )

    # 多支：不指定主所缘；按 prior 稳定排序（后验并行更新不改排序键）
    perception.alambana_candidates.sort(
        key=lambda c: (c.prior_probability, c.confidence),
        reverse=True,
    )
    perception.uncertainty_notes.append(
        "多支模式：不指定主所缘；结果看各支分布与跨支期望。"
    )

    branches: list[CognitionBranch] = []
    branch_grasping: dict[str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]] = {}
    grasping_by_aid: dict[str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]] = {}

    # 思 = L0 行动骨架（每事件一次）；前序：compat→（可选）扩支→后验→搭支→想
    xiang_by_id: dict[str, ExtractedMentalState] = {}
    if use_action_feedback:
        try:
            action_evidence, act_notes = action_skeleton_from_perception_split(
                text,
                perception.stimulus_action_split,
            )
            perception.uncertainty_notes.extend(act_notes)
        except Exception as exc:  # noqa: BLE001
            action_evidence = ActionEvidence(
                has_observable_action=False,
                reason_summary=f"行动证据提取失败，保持先验：{exc}",
            )
            perception.uncertainty_notes.append(action_evidence.reason_summary)
        (
            action_evidence,
            branches,
            branch_grasping,
            xiang_by_id,
        ) = _compat_expand_and_xiang(
            text,
            perception,
            action_evidence,
            llm,
            focus_person=focus,
        )
    else:
        # 待预测事件：后验=先验；思置空；只生成想
        action_evidence = ActionEvidence(
            has_observable_action=False,
            reason_summary="待预测事件：禁用思反推，后验=先验，思置空。",
        )
        for cand in perception.alambana_candidates:
            cand.posterior_probability = cand.prior_probability
            cand.action_compat = 1.0
            cand.action_likelihood = 1.0
        perception.uncertainty_notes.append(
            "待预测事件：禁用思反推；后验=先验；思置空。"
        )
        branches, branch_grasping = _build_branches_from_candidates(perception)
        grasping_by_aid = {
            alambana.id: branch_grasping[f"branch_{alambana.id}"]
            for alambana in perception.alambana_candidates
            if f"branch_{alambana.id}" in branch_grasping
        }
        if perception.alambana_candidates:
            try:
                xiang_by_id = generate_branch_xiangs(
                    text,
                    perception.objective_event,
                    perception.alambana_candidates,
                    perception.self_grasping_hypotheses,
                    perception.dharma_grasping_hypotheses,
                    action_evidence,
                    llm,
                    grasping_by_alambana_id=grasping_by_aid,
                    focus_person=focus,
                    h_prior=getattr(perception, "h_prior", None) or {},
                )
                perception.uncertainty_notes.append(
                    f"想已分支出成（{len(xiang_by_id)} 支隔离并行）。"
                )
            except Exception as exc:  # noqa: BLE001
                perception.uncertainty_notes.append(
                    f"想生成失败，已按所缘兜底：{exc}"
                )
                xiang_by_id = {
                    cand.id: ExtractedMentalState(
                        text=(
                            f"在此所缘下：{cand.text}"
                            if cand.text
                            else "对该事态形成认知评价"
                        ),
                        source="推理",
                        confidence=0.4,
                        alambana_id=cand.id,
                        reason_summary="想生成异常兜底",
                    )
                    for cand in perception.alambana_candidates
                }

    return _score_pack_bundle(
        text=text,
        perception=perception,
        branches=branches,
        xiang_by_id=xiang_by_id,
        action_evidence=action_evidence,
        llm=llm,
        xinsuo_list=xinsuo_list,
        relations=relations,
        weights=weights,
        validator=validator,
        continuity=continuity,
        bias_frames=bias_frames,
        focus=focus,
        role_context=role_context,
        update_continuity=update_continuity,
        aggregate_behaviors=aggregate_behaviors,
        predict_branch_behaviors=predict_branch_behaviors,
    )


def apply_predicted_action(
    text: str,
    llm: LlmClient,
    predicted: CognitionEvaluationBundle,
    action: str,
    *,
    xinsuo_list: list[Xinsuo],
    relations: list[Relation],
    weights: ScoringWeights,
    validator: DefinitionValidator | None = None,
    role_context: str = "",
    continuity_state: ContinuityState | None = None,
    focus_person: str = "",
    action_probability: float = 0.0,
) -> CognitionEvaluationBundle:
    """保留预测所缘，只接入已发生行动：compat/扩支/后验/想，再评分写 H/M。

    不重抽 L0/L2/L3。明确无反应也算合法行动。
    """
    stripped = (action or "").strip()
    if not is_usable_predicted_action(stripped):
        raise ValueError(f"预测行动不可用，拒绝写入 H/M：{stripped!r}")

    validator = validator or DefinitionValidator()
    continuity = continuity_state or ContinuityState()
    focus = str(focus_person or "").strip()
    if focus and continuity.person_id in ("", "default"):
        continuity.person_id = focus
    bias_frames = bias_frames_for_injection(continuity)

    perception = copy.deepcopy(predicted.perception_state)
    conf = float(action_probability or 0.0)
    if conf <= 0.0:
        conf = 0.7
    else:
        conf = min(1.0, conf)
    old_split = perception.stimulus_action_split
    stimulus_text = (
        old_split.stimulus_text
        if old_split and old_split.stimulus_text
        else (perception.stimulus_text or text)
    )
    perception.stimulus_action_split = StimulusActionSplit(
        stimulus_text=stimulus_text,
        has_observable_action=True,
        action_span=stripped,
        action_description=stripped,
        confidence=conf,
        reason_summary="预测 Top-1 已接入；沿用预测所缘，不重抽 L2/L3。",
    )
    action_evidence = ActionEvidence(
        has_observable_action=True,
        action_span=stripped,
        action_description=stripped,
        confidence=conf,
        reason_summary="预测 Top-1 已接入",
    )
    perception.uncertainty_notes.append(
        f"落地：保留预测所缘，接入行动「{stripped}」做 compat/扩支。"
    )
    action_evidence, branches, _, xiang_by_id = _compat_expand_and_xiang(
        text,
        perception,
        action_evidence,
        llm,
        focus_person=focus,
    )
    return _score_pack_bundle(
        text=text,
        perception=perception,
        branches=branches,
        xiang_by_id=xiang_by_id,
        action_evidence=action_evidence,
        llm=llm,
        xinsuo_list=xinsuo_list,
        relations=relations,
        weights=weights,
        validator=validator,
        continuity=continuity,
        bias_frames=bias_frames,
        focus=focus,
        role_context=role_context,
        update_continuity=True,
        aggregate_behaviors=False,
        predict_branch_behaviors=False,
    )
