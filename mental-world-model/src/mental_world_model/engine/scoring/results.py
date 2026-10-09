"""评分阶段与结果。"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from mental_world_model.engine.scoring.breakdown import ScoreBreakdown

if TYPE_CHECKING:
    from mental_world_model.engine.cognition.models import (
        AggregatedBehaviorPrediction,
        CognitionBranch,
        ExtractedMentalState,
    )
    from mental_world_model.engine.cognition.xiang_si import ExplicitActionDetection
    from mental_world_model.engine.extraction.models import ExtractedContext
    from mental_world_model.engine.perception.models import PerceptionState
    from mental_world_model.llm.usage import LlmUsageSummary

StageName = Literal["suoyuan", "shengqi_nian", "ye", "xiandai_xinli"]
ExitStage = Literal["suoyuan", "shengqi_nian"] | None


# 忿 → 恨 → 恼 依序依赖（首次遍历跳过恨/恼两行）：
#   1. 评忿
#   2. 忿 = 0 → 恨、恼均不判断（不调 LLM，直接 0）
#   3. 忿 > 0 → 评恨
#   4. 恨 = 0 → 恼不判断（不调 LLM，直接 0）
#   5. 恨 > 0 → 评恼
FEN_NAME = "忿"
HEN_NAME = "恨"
NAO_NAME = "恼"
DEPENDENT_ON_FEN = frozenset({HEN_NAME, NAO_NAME})
FEN_HEN_NAO_CHAIN = (FEN_NAME, HEN_NAME, NAO_NAME)


@dataclass
class StageResult:
    """单阶段 LLM 评分结果。"""

    stage: StageName
    score: float


@dataclass
class XinsuoScoreResult:
    """单个心所的完整评分结果。"""

    name: str
    exit_at: ExitStage
    stage_results: list[StageResult] = field(default_factory=list)
    column_scores: dict[str, float] = field(default_factory=dict)
    score_breakdown: ScoreBreakdown | None = None
    final_score: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def explain_score(self) -> str:
        """返回最终分的可解释性说明；门控退出或依序跳过时说明原因。"""
        if self.skipped:
            return f"【{self.name}】{self.skip_reason}，未调用 LLM，最终分 = 0"
        if self.score_breakdown is not None:
            return self.score_breakdown.explain()
        if self.exit_at == "suoyuan":
            return f"【{self.name}】在所缘境阶段低于门控阈值，未继续评分，最终分 = 0"
        if self.exit_at == "shengqi_nian":
            suoyuan = float(self.column_scores.get("suoyuan", 0.0))
            shengqi = float(self.column_scores.get("shengqi_nian", 0.0))
            if suoyuan < 0.3 and shengqi < 0.3:
                return (
                    f"【{self.name}】所缘境（{suoyuan:.3f}）与生起念头/想（{shengqi:.3f}）"
                    f"均低于门控阈值，已排除，最终分 = 0"
                )
            return f"【{self.name}】在生起念头阶段低于门控阈值，未继续评分，最终分 = 0"
        return f"【{self.name}】最终分 = {self.final_score:.4f}"


@dataclass
class EvaluationResult:
    """一次完整 evaluate 的结果。

    多分支模式下：
    - mental_factors_event / expected_mental_factor_scores：跨支激活加权期望（落库用）
    - 前序事件：scores_final = 期望再与历史融合；active_xinsuo 可取自期望分
    - 预测事件：不做跨支心所 final；心所看 branches[*].mental_factor_scores_final
      （可已掺历史）；顶层 scores_final 仅等于 event（兼容）；active_xinsuo=None
    - extracted：并列各支所缘/想 + 共享思（非单支摘要）
    - stage_results 顶层为空；过程明细在 cognition_branches[*].stage_results
    - perception_state / cognition_branches / aggregated_* 为完整分支报告
    """

    extracted: "ExtractedContext | None"
    stage_results: list[XinsuoScoreResult]
    scores_initial: dict[str, float]
    conflicts_initial: list
    conflicts_final: list
    resolutions: list
    scores_final: dict[str, float]
    predicted_behavior: str | None = None
    active_xinsuo: list[dict] | None = None
    prediction_reason: str | None = None
    seed_weights_before: dict[str, float] | None = None
    seed_weights_after: dict[str, float] | None = None
    llm_usage: "LlmUsageSummary | None" = None
    perception_state: "PerceptionState | None" = None
    cognition_branches: list["CognitionBranch"] | None = None
    aggregated_behaviors: list["AggregatedBehaviorPrediction"] | None = None
    expected_mental_factor_scores: dict[str, float] | None = None
    mental_factor_presence_probability: dict[str, float] | None = None
    overall_uncertainty: str | None = None
    action_detection: "ExplicitActionDetection | None" = None
    shared_si: "ExtractedMentalState | None" = None
    action_evidence: dict | None = None
    continuity: dict | None = None
    mental_factors_event: dict[str, float] | None = None
    event_reports: list[dict] | None = None
    focus_person: str = ""
    behavior_diagnostics: dict | None = None

    def to_event_snapshot(self, *, event_label: str | None = None) -> dict:
        """单事件分析快照：必要中间结果（客观所缘、二执、所缘、想/思、心所、行为）。"""
        perception = self.perception_state
        branches = self.cognition_branches or []
        payload: dict = {}
        if event_label:
            payload["event"] = event_label
        if self.focus_person:
            payload["focus_person"] = self.focus_person
        if self.extracted is not None:
            payload["yuanwen"] = self.extracted.yuanwen
        elif perception and perception.source_text:
            payload["yuanwen"] = perception.source_text

        # —— L0 刺激/行动拆分 ——
        if perception:
            if perception.stimulus_text:
                payload["stimulus_text"] = perception.stimulus_text
            if perception.stimulus_action_split is not None:
                split = perception.stimulus_action_split
                payload["stimulus_action_split"] = {
                    "stimulus_text": split.stimulus_text,
                    "has_observable_action": split.has_observable_action,
                    "action_span": split.action_span,
                    "action_description": split.action_description,
                }

        # —— L1 客观刺激 ——
        payload["objective_event"] = (
            perception.objective_event.to_dict()
            if perception and perception.objective_event
            else None
        )

        # —— L2 我执 / 法执 ——
        payload["self_grasping"] = (
            [self._slim_grasping(item) for item in perception.self_grasping_hypotheses]
            if perception
            else []
        )
        payload["dharma_grasping"] = (
            [self._slim_grasping(item) for item in perception.dharma_grasping_hypotheses]
            if perception
            else []
        )

        # —— L2.5 二执组合（激活） ——
        if perception and perception.grasping_combos:
            payload["grasping_combos"] = [
                self._slim_combo(raw) for raw in perception.grasping_combos
            ]

        # —— L3 主观所缘（prior / posterior / compat / L） ——
        payload["subjective_alambanas"] = (
            [
                {
                    **self._slim_alambana(item),
                    "combo_id": item.combo_reason,
                }
                for item in perception.alambana_candidates
            ]
            if perception
            else []
        )

        # —— L5 行动证据 + 各所缘 compat ——
        if self.action_evidence is not None:
            ae = dict(self.action_evidence)
            likelihoods = []
            for item in ae.get("likelihoods") or []:
                if isinstance(item, dict):
                    likelihoods.append(
                        {
                            "alambana_id": item.get("alambana_id", ""),
                            "likelihood": item.get("likelihood", 0.0),
                        }
                    )
            payload["action_evidence"] = {
                "has_observable_action": ae.get("has_observable_action"),
                "action_span": ae.get("action_span", ""),
                "action_description": ae.get("action_description", ""),
                "likelihoods": likelihoods,
            }

        # —— 各认知分支：a、所缘、想、思、心所、支内行为 ——
        branch_items: list[dict] = []
        for branch in branches:
            a = branch.activation_probability or branch.branch_effective_probability
            item: dict = {
                "branch_id": branch.id,
                "activation_probability": a,
                "alambana": self._slim_alambana(branch.alambana),
                "xiang": self._slim_mental(branch.xiang),
                "si": self._slim_mental(branch.si),
                "mental_factors_nonzero": self._nonzero_scores(
                    branch.mental_factor_scores_final
                ),
            }
            if branch.status and branch.status != "ok":
                item["status"] = branch.status
            if branch.behavior_predictions:
                item["behaviors"] = [
                    {
                        "text": pred.text,
                        "probability": pred.conditional_probability,
                        "supporting_mental_factors": list(
                            pred.supporting_mental_factors
                        ),
                    }
                    for pred in branch.behavior_predictions
                ]
            branch_items.append(item)
        payload["branches"] = branch_items

        # —— 跨支行为分布（预测事件） ——
        if self.aggregated_behaviors:
            payload["aggregated_behaviors"] = [
                {
                    "text": item.canonical_text,
                    "probability": item.total_probability,
                    "dominant_mental_factors": list(item.dominant_mental_factors),
                    "branch_contributions": [
                        {
                            "branch_id": c.branch_id,
                            "activation_probability": c.activation_probability,
                            "conditional_probability": (
                                c.conditional_behavior_probability
                            ),
                            "contribution": c.contribution,
                        }
                        for c in item.branch_contributions
                    ],
                }
                for item in self.aggregated_behaviors
            ]

        # —— 心所：事件期望 vs 连续性融合后 ——
        event_scores = self.mental_factors_event or self.expected_mental_factor_scores
        mental = (
            (self.continuity or {}).get("mental")
            if isinstance(self.continuity, dict)
            else None
        )
        apply_mode = (
            mental.get("apply_mode") if isinstance(mental, dict) else None
        )
        if event_scores is not None:
            payload["mental_factors_event_nonzero"] = self._nonzero_scores(event_scores)
        if apply_mode == "per_branch_prediction":
            # 预测事件：无跨支心所 final；看 branches[*].mental_factors_nonzero
            payload["mental_report_mode"] = "per_branch_only"
        elif self.scores_final is not None:
            payload["mental_factors_final_nonzero"] = self._nonzero_scores(
                self.scores_final
            )

        if perception and perception.pipeline_trace:
            payload["cognition_candidates"] = dict(
                perception.pipeline_trace.get("cognition_candidates") or {}
            )
            payload["accommodation"] = dict(
                perception.pipeline_trace.get("accommodation") or {}
            )
            payload["neutral_fallback"] = dict(
                perception.pipeline_trace.get("neutral_fallback") or {}
            )
            payload["used_neutral_fallback"] = bool(
                perception.pipeline_trace.get("used_neutral_fallback")
            )
        if self.behavior_diagnostics:
            payload["behavior_diagnostics"] = dict(self.behavior_diagnostics)

        # —— 连续性（摘要） ——
        if self.continuity:
            payload["continuity"] = self._slim_continuity(self.continuity)

        return payload

    def to_event_report_dict(self) -> dict:
        """默认报告：focus_person + 各事件中间结果（分析用）。"""
        if self.event_reports:
            report: dict = {"events": list(self.event_reports)}
        else:
            report = {"events": [self.to_event_snapshot()]}
        if self.focus_person:
            report["focus_person"] = self.focus_person
        if self.llm_usage is not None:
            u = self.llm_usage
            report["llm_usage"] = {
                "total_calls": getattr(u, "total_calls", None),
                "total_duration_ms": getattr(u, "total_duration_ms", None),
                "wall_clock_ms": getattr(u, "wall_clock_ms", None),
            }
        return report

    @staticmethod
    def _slim_combo(raw: dict) -> dict:
        """组合层只保留分析必需字段。"""
        return {
            "combo_id": raw.get("combo_id", ""),
            "used_self_ids": list(raw.get("used_self_ids") or []),
            "used_dharma_ids": list(raw.get("used_dharma_ids") or []),
            "combo_activation_probability": raw.get(
                "combo_activation_probability",
                raw.get("activation_probability", raw.get("probability", 0.0)),
            ),
            "alambana_prior_probability": raw.get("alambana_prior_probability", 0.0),
            "activation_alignment": raw.get("activation_alignment", 1.0),
            "source_type": raw.get("source_type", ""),
            "source_bias_frame_ids": list(raw.get("source_bias_frame_ids") or []),
            "reactivated_h_ids": list(
                raw.get("reactivated_h_ids") or raw.get("source_bias_frame_ids") or []
            ),
            "raw_combo_score": raw.get("raw_combo_score", raw.get("score", 0.0)),
        }

    @staticmethod
    def _slim_continuity(raw: dict) -> dict:
        """连续性只留摘要。"""
        out: dict = {}
        for key in (
            "cognitive_enabled",
            "mental_episode_count",
            "injected_bias_frames",
            "h_prior",
        ):
            if key in raw:
                out[key] = raw[key]
        mental = raw.get("mental")
        if isinstance(mental, dict):
            slim_m = {
                k: mental[k]
                for k in ("enabled", "blend_alpha", "apply_mode", "lambda_eff")
                if k in mental
            }
            if slim_m:
                out["mental"] = slim_m
        frames = raw.get("cognitive_frames_after")
        if isinstance(frames, list) and frames:
            out["cognitive_frames"] = [
                {
                    "text": f.get("text", f.get("frame_text", "")),
                    "mass": f.get("mass", f.get("weight", 0.0)),
                }
                for f in frames[:8]
                if isinstance(f, dict)
            ]
        return out

    def _explanation_score_lookup(self) -> dict[str, float]:
        """解释过滤用分：多支模式用跨支期望 scores_final。"""
        return self.scores_final

    @property
    def conflicts(self) -> list:
        """消解前的冲突列表（兼容旧字段名）。"""
        return self.conflicts_initial

    def score_explanations(self) -> dict[str, str]:
        """逐心所评分解释。

        多支时合并各支过程；顶层无 stage_results 时回退空。
        """
        if self.cognition_branches:
            merged: dict[str, list[str]] = {}
            for branch in self.cognition_branches:
                a = (
                    branch.activation_probability
                    or branch.branch_effective_probability
                )
                for item in branch.stage_results or []:
                    merged.setdefault(item.name, []).append(
                        f"[{branch.id} a={a:.3f}] {item.explain_score()}"
                    )
            return {name: "\n".join(parts) for name, parts in merged.items()}
        return {item.name: item.explain_score() for item in self.stage_results}

    def explain_all_scores(self, *, only_nonzero: bool = False) -> str:
        """全部心所评分解释合并文本。

        多支：按支输出过程明细；only_nonzero 按跨支期望 scores_final 过滤。
        """
        lookup = self._explanation_score_lookup()
        if self.cognition_branches:
            sections: list[str] = []
            for branch in self.cognition_branches:
                a = (
                    branch.activation_probability
                    or branch.branch_effective_probability
                )
                branch_sections: list[str] = []
                for item in branch.stage_results or []:
                    if only_nonzero and lookup.get(item.name, 0.0) <= 0:
                        # 该心所期望为 0 时跳过；支内明细仍可在 verbose 分支报告看
                        continue
                    branch_score = branch.mental_factor_scores_final.get(
                        item.name, 0.0
                    )
                    if only_nonzero and branch_score <= 0:
                        continue
                    branch_sections.append(item.explain_score())
                if branch_sections:
                    sections.append(
                        f"=== {branch.id} (activation={a:.3f}) ===\n"
                        + "\n\n".join(branch_sections)
                    )
            return "\n\n".join(sections)

        sections = []
        for item in self.stage_results:
            if only_nonzero and lookup.get(item.name, 0.0) <= 0:
                continue
            sections.append(item.explain_score())
        return "\n\n".join(sections)

    def score_breakdowns(self, *, include_explanation: bool = False) -> dict[str, dict | None]:
        """逐心所 score_breakdown。

        多支时返回 {name: {branch_id: breakdown}}；单路仍为扁平字典。
        """
        if self.cognition_branches:
            output: dict[str, dict] = {}
            for branch in self.cognition_branches:
                for item in branch.stage_results or []:
                    bucket = output.setdefault(item.name, {})
                    if item.score_breakdown is None:
                        bucket[branch.id] = None
                    else:
                        bucket[branch.id] = item.score_breakdown.to_dict(
                            include_explanation=include_explanation
                        )
            return output  # type: ignore[return-value]
        output_flat: dict[str, dict | None] = {}
        for item in self.stage_results:
            if item.score_breakdown is None:
                output_flat[item.name] = None
            else:
                output_flat[item.name] = item.score_breakdown.to_dict(
                    include_explanation=include_explanation
                )
        return output_flat

    @staticmethod
    def _nonzero_scores(scores: dict[str, float] | None) -> dict[str, float]:
        if not scores:
            return {}
        return {name: float(score) for name, score in scores.items() if float(score) > 0}

    @staticmethod
    def _slim_grasping(item) -> dict:
        return {
            "id": item.id,
            "text": item.text,
            "confidence": item.confidence,
            "source": item.source,
            "source_type": getattr(item, "source_type", ""),
            "source_bias_frame_ids": list(
                getattr(item, "source_bias_frame_ids", []) or []
            ),
        }

    @staticmethod
    def _slim_alambana(item) -> dict:
        item.sync_grasping_aliases()
        used_self = list(item.used_self_ids or item.self_grasping_ids)
        used_dharma = list(item.used_dharma_ids or item.dharma_grasping_ids)
        return {
            "id": item.id,
            "text": item.text,
            "prior_probability": item.prior_probability,
            "posterior_probability": item.posterior_probability,
            # compat=LLM raw；action_likelihood=有效 L=g(a,compat)
            "action_compat": getattr(item, "action_compat", 1.0),
            "action_likelihood": item.action_likelihood,
            "action_informed": bool(getattr(item, "action_informed", False)),
            "confidence": item.confidence,
            "used_self_ids": used_self,
            "used_dharma_ids": used_dharma,
            "source": item.source,
            "source_type": getattr(item, "source_type", ""),
            "source_bias_frame_ids": list(
                getattr(item, "source_bias_frame_ids", []) or []
            ),
            "reactivated_h_ids": list(
                getattr(item, "reactivated_h_ids", None)
                or getattr(item, "source_bias_frame_ids", [])
                or []
            ),
            "raw_combo_score": float(getattr(item, "raw_combo_score", 0.0) or 0.0),
            "activation_alignment": float(
                getattr(item, "activation_alignment", 1.0) or 0.0
            ),
            "grounding_valid": bool(getattr(item, "grounding_valid", True)),
            "grounding_reason": str(getattr(item, "grounding_reason", "") or ""),
        }

    @staticmethod
    def _slim_mental(state) -> dict | None:
        if state is None:
            return None
        return {
            "text": state.text,
            "source": state.source,
            "confidence": state.confidence,
        }

    def build_pipeline_trace(self, *, slim: bool = True) -> dict | None:
        """按评估流水线整理中间结果（多分支模式）；旧路径返回 None。"""
        if self.perception_state is None and self.cognition_branches is None:
            return None

        perception = self.perception_state
        branches = self.cognition_branches or []
        branch_summaries: list[dict] = []
        for branch in branches:
            if slim:
                behaviors = [
                    {
                        "text": item.text,
                        "conditional_probability": item.conditional_probability,
                    }
                    for item in branch.behavior_predictions
                ]
                branch_summaries.append(
                    {
                        "branch_id": branch.id,
                        "status": branch.status,
                        "branch_prior_probability": branch.branch_prior_probability,
                        "branch_effective_probability": (
                            branch.activation_probability
                            or branch.branch_effective_probability
                        ),
                        "activation_probability": (
                            branch.activation_probability
                            or branch.branch_effective_probability
                        ),
                        "alambana": self._slim_alambana(branch.alambana),
                        "xiang": self._slim_mental(branch.xiang),
                        "si": self._slim_mental(branch.si),
                        "mental_factor_scores_final": self._nonzero_scores(
                            branch.mental_factor_scores_final
                        ),
                        "behavior_predictions": behaviors,
                    }
                )
            else:
                branch_summaries.append(
                    {
                        "branch_id": branch.id,
                        "status": branch.status,
                        "branch_prior_probability": branch.branch_prior_probability,
                        "branch_confidence": branch.branch_confidence,
                        "branch_effective_probability": (
                            branch.activation_probability
                            or branch.branch_effective_probability
                        ),
                        "activation_probability": (
                            branch.activation_probability
                            or branch.branch_effective_probability
                        ),
                        "behavior_contribution_probability": (
                            branch.behavior_contribution_probability
                        ),
                        "alambana": branch.alambana.to_dict(),
                        "xiang": branch.xiang.to_dict() if branch.xiang else None,
                        "si": branch.si.to_dict() if branch.si else None,
                        "mental_factor_scores_final": dict(
                            branch.mental_factor_scores_final
                        ),
                        "behavior_predictions": [
                            item.to_dict() for item in branch.behavior_predictions
                        ],
                        "uncertainty_notes": list(branch.uncertainty_notes),
                    }
                )

        if slim:
            agg_behaviors = [
                {
                    "text": item.canonical_text,
                    "total_probability": item.total_probability,
                    "confidence": item.confidence,
                    "top_contributions": [
                        {
                            "branch_id": c.branch_id,
                            "contribution": c.contribution,
                        }
                        for c in item.branch_contributions[:3]
                    ],
                }
                for item in (self.aggregated_behaviors or [])
            ]
            eval_block = {
                "scores_final": self._nonzero_scores(self.scores_final),
                "predicted_behavior": self.predicted_behavior,
            }
            if self.active_xinsuo:
                eval_block["active_xinsuo"] = list(self.active_xinsuo)

            shared = None
            if self.shared_si is not None:
                shared = self._slim_mental(self.shared_si)

            return {
                "objective_event": (
                    perception.objective_event.to_dict()
                    if perception and perception.objective_event
                    else None
                ),
                "self_grasping": (
                    [self._slim_grasping(item) for item in perception.self_grasping_hypotheses]
                    if perception
                    else []
                ),
                "dharma_grasping": (
                    [
                        self._slim_grasping(item)
                        for item in perception.dharma_grasping_hypotheses
                    ]
                    if perception
                    else []
                ),
                "subjective_alambanas": (
                    [
                        self._slim_alambana(item)
                        for item in perception.alambana_candidates
                    ]
                    if perception
                    else []
                ),
                "action_evidence": self.action_evidence,
                "shared_si": shared,
                "continuity": self._slim_continuity(self.continuity)
                if self.continuity
                else None,
                "mental_factors_event": (
                    self._nonzero_scores(self.mental_factors_event)
                    if self.mental_factors_event
                    else None
                ),
                "branches": branch_summaries,
                "aggregated": {
                    "expected_mental_factor_scores": self._nonzero_scores(
                        self.expected_mental_factor_scores
                    ),
                    "scores_final": self._nonzero_scores(self.scores_final),
                    "behavior_predictions": agg_behaviors,
                },
                "evaluation_result": eval_block,
                "cognition_candidates": dict(
                    (perception.pipeline_trace or {}).get("cognition_candidates") or {}
                )
                if perception
                else {},
                "accommodation": dict(
                    (perception.pipeline_trace or {}).get("accommodation") or {}
                )
                if perception
                else {},
                "neutral_fallback": dict(
                    (perception.pipeline_trace or {}).get("neutral_fallback") or {}
                )
                if perception
                else {},
                "used_neutral_fallback": bool(
                    perception
                    and (perception.pipeline_trace or {}).get("used_neutral_fallback")
                ),
                "behavior_diagnostics": dict(self.behavior_diagnostics or {}),
            }

        return {
            "objective_event": (
                perception.objective_event.to_dict()
                if perception and perception.objective_event
                else None
            ),
            "objective_facts": (
                [item.to_dict() for item in perception.objective_facts]
                if perception
                else []
            ),
            "self_grasping": (
                [item.to_dict() for item in perception.self_grasping_hypotheses]
                if perception
                else []
            ),
            "dharma_grasping": (
                [item.to_dict() for item in perception.dharma_grasping_hypotheses]
                if perception
                else []
            ),
            "base_alambana_text": (
                perception.base_alambana_text if perception else ""
            ),
            "subjective_alambanas": (
                [item.to_dict() for item in perception.alambana_candidates]
                if perception
                else []
            ),
            "action_evidence": self.action_evidence,
            "action_detection": (
                self.action_detection.to_dict() if self.action_detection else None
            ),
            "shared_si": self.shared_si.to_dict() if self.shared_si else None,
            "continuity": self.continuity,
            "mental_factors_event": (
                dict(self.mental_factors_event) if self.mental_factors_event else None
            ),
            "branches": branch_summaries,
            "aggregated": {
                "expected_mental_factor_scores": dict(
                    self.expected_mental_factor_scores or {}
                ),
                "scores_final": dict(self.scores_final),
                "behavior_predictions": [
                    item.to_dict() for item in (self.aggregated_behaviors or [])
                ],
            },
            "evaluation_result": {
                "scores_final": dict(self.scores_final),
                "predicted_behavior": self.predicted_behavior,
                "active_xinsuo": list(self.active_xinsuo or []),
            },
            "cognition_candidates": dict(
                (perception.pipeline_trace or {}).get("cognition_candidates") or {}
            )
            if perception
            else {},
            "accommodation": dict(
                (perception.pipeline_trace or {}).get("accommodation") or {}
            )
            if perception
            else {},
            "neutral_fallback": dict(
                (perception.pipeline_trace or {}).get("neutral_fallback") or {}
            )
            if perception
            else {},
            "used_neutral_fallback": bool(
                perception
                and (perception.pipeline_trace or {}).get("used_neutral_fallback")
            ),
            "behavior_diagnostics": dict(self.behavior_diagnostics or {}),
        }

    def _legacy_slim_report(self) -> dict:
        """旧单路径精简报告。"""
        report: dict = {
            "scores_final": self._nonzero_scores(self.scores_final),
            "conflicts_initial": [str(c) for c in self.conflicts_initial],
            "conflicts_final": [str(c) for c in self.conflicts_final],
        }
        if self.predicted_behavior is not None:
            report["predicted_behavior"] = self.predicted_behavior
        if self.prediction_reason is not None:
            report["prediction_reason"] = self.prediction_reason
        if self.active_xinsuo:
            report["active_xinsuo"] = list(self.active_xinsuo)
        if self.extracted is not None:
            report["extracted"] = self.extracted.to_dict()
        return report

    def _verbose_report(self, *, include_explanations: bool = False) -> dict:
        """完整调试报告（含重复明细字段）。"""
        report = {
            "extracted": self.extracted.to_dict() if self.extracted else None,
            "scores_initial": dict(self.scores_initial),
            "scores_final": dict(self.scores_final),
            "conflicts_initial": [str(c) for c in self.conflicts_initial],
            "conflicts_final": [str(c) for c in self.conflicts_final],
            "conflicts": [str(c) for c in self.conflicts_initial],
            "resolutions": [
                {
                    "name_a": resolution.name_a,
                    "name_b": resolution.name_b,
                    "winner": resolution.winner,
                    "loser": resolution.loser,
                    **({"note": resolution.note} if resolution.note else {}),
                }
                for resolution in self.resolutions
            ],
            "score_breakdowns": self.score_breakdowns(
                include_explanation=include_explanations
            ),
        }
        if self.predicted_behavior is not None:
            report["predicted_behavior"] = self.predicted_behavior
        if self.active_xinsuo is not None:
            report["active_xinsuo"] = list(self.active_xinsuo)
        if self.prediction_reason is not None:
            report["prediction_reason"] = self.prediction_reason
        if include_explanations:
            report["score_explanations"] = self.score_explanations()
            report["explanation_all"] = self.explain_all_scores()
            report["explanation_nonzero"] = self.explain_all_scores(only_nonzero=True)
        if self.seed_weights_before is not None:
            report["seed_weights_before"] = dict(self.seed_weights_before)
        if self.seed_weights_after is not None:
            report["seed_weights_after"] = dict(self.seed_weights_after)
        if self.perception_state is not None:
            report["perception_inference"] = self.perception_state.to_dict()
        if self.cognition_branches is not None:
            report["cognition_branches"] = [
                branch.to_dict(include_breakdown_explanations=include_explanations)
                for branch in self.cognition_branches
            ]
        if self.aggregated_behaviors is not None or self.expected_mental_factor_scores:
            report["aggregated_results"] = {
                "behavior_predictions": [
                    b.to_dict() for b in (self.aggregated_behaviors or [])
                ],
                "aggregated_action_distribution": [
                    {
                        "behavior_id": item.behavior_id,
                        "text": item.canonical_text,
                        "probability": item.total_probability,
                        "confidence": item.confidence,
                        "dominant_mental_factors": list(item.dominant_mental_factors),
                        "branch_contributions": [
                            {
                                "branch_id": c.branch_id,
                                "activation_probability": c.activation_probability,
                                "behavior_contribution_probability": (
                                    c.behavior_contribution_probability
                                ),
                                "conditional_behavior_probability": (
                                    c.conditional_behavior_probability
                                ),
                                "contribution": c.contribution,
                            }
                            for c in item.branch_contributions
                        ],
                    }
                    for item in (self.aggregated_behaviors or [])
                ],
                "expected_mental_factor_scores": dict(
                    self.expected_mental_factor_scores or {}
                ),
            }
        if self.action_detection is not None:
            report["action_detection"] = self.action_detection.to_dict()
        if self.shared_si is not None:
            report["shared_si"] = self.shared_si.to_dict()
        pipeline_trace = self.build_pipeline_trace(slim=False)
        if pipeline_trace is not None:
            report["pipeline_trace"] = pipeline_trace
        return report

    def to_report_dict(
        self,
        *,
        include_explanations: bool = False,
        verbose: bool = False,
    ) -> dict:
        """生成报告字典。

        多事件时始终把 ``events``（含历史事件完整中间结果）放在最前，便于排查。
        默认（verbose=False）另附最后一事件的精简 ``pipeline_trace``；
        ``verbose=True`` 另附最后一事件的完整调试字段。
        """
        if verbose:
            report = self._verbose_report(include_explanations=include_explanations)
        elif self.cognition_branches is not None:
            report = {
                "pipeline_trace": self.build_pipeline_trace(slim=True),
            }
            if include_explanations:
                report["score_explanations"] = self.score_explanations()
                report["score_breakdowns"] = self.score_breakdowns(
                    include_explanation=True
                )
                report["explanation_all"] = self.explain_all_scores()
                report["explanation_nonzero"] = self.explain_all_scores(
                    only_nonzero=True
                )
        else:
            report = self._legacy_slim_report()
            if include_explanations:
                report["score_explanations"] = self.score_explanations()
                report["score_breakdowns"] = self.score_breakdowns(
                    include_explanation=True
                )
                report["explanation_all"] = self.explain_all_scores()
                report["explanation_nonzero"] = self.explain_all_scores(
                    only_nonzero=True
                )

        if self.event_reports:
            report = {"events": list(self.event_reports), **report}
        elif self.cognition_branches is not None:
            # 单事件也保证有 events，与默认精简报告结构一致
            report = {"events": [self.to_event_snapshot(event_label="事件1")], **report}

        if self.llm_usage is not None:
            report["llm_usage"] = self.llm_usage.to_dict()
        return report
