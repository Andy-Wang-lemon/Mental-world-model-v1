"""认知分支数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate

if TYPE_CHECKING:
    from mental_world_model.engine.scoring.breakdown import ScoreBreakdown

FieldSource = Literal["原文", "推理"]
BranchStatus = Literal[
    "active",
    "low_probability",
    "scoring_failed",
]


@dataclass
class ExtractedMentalState:
    """分支内想或思。"""

    text: str
    source: FieldSource
    confidence: float
    alambana_id: str
    supporting_grasping_ids: list[str] = field(default_factory=list)
    reason_summary: str = ""
    xiang_text: str = ""  # 思可附带引用的想

    def to_dict(self) -> dict:
        payload = {
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "alambana_id": self.alambana_id,
            "supporting_grasping_ids": list(self.supporting_grasping_ids),
            "reason_summary": self.reason_summary,
        }
        if self.xiang_text:
            payload["xiang_text"] = self.xiang_text
        return payload


@dataclass
class BranchBehaviorPrediction:
    """单分支内的条件行为预测。"""

    behavior_id: str
    text: str
    conditional_probability: float
    confidence: float
    supporting_mental_factors: list[str] = field(default_factory=list)
    opposing_mental_factors: list[str] = field(default_factory=list)
    supporting_scores: dict[str, float] = field(default_factory=dict)
    alambana_id: str = ""
    xiang_text: str = ""
    si_text: str = ""
    reason_summary: str = ""
    canonical_text: str = ""

    def to_dict(self) -> dict:
        return {
            "behavior_id": self.behavior_id,
            "text": self.text,
            "canonical_text": self.canonical_text or self.text,
            "conditional_probability": self.conditional_probability,
            "confidence": self.confidence,
            "supporting_mental_factors": list(self.supporting_mental_factors),
            "opposing_mental_factors": list(self.opposing_mental_factors),
            "supporting_scores": dict(self.supporting_scores),
            "alambana_id": self.alambana_id,
            "xiang_text": self.xiang_text,
            "si_text": self.si_text,
            "reason_summary": self.reason_summary,
        }


@dataclass
class BehaviorBranchContribution:
    """某行为在某一分支上的贡献。"""

    branch_id: str
    conditional_behavior_probability: float
    contribution: float
    alambana_text: str
    xiang_text: str
    si_text: str
    supporting_mental_factors: list[str] = field(default_factory=list)
    supporting_scores: dict[str, float] = field(default_factory=dict)
    activation_probability: float = 0.0
    behavior_contribution_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "activation_probability": self.activation_probability,
            "behavior_contribution_probability": (
                self.behavior_contribution_probability
            ),
            "conditional_behavior_probability": self.conditional_behavior_probability,
            "contribution": self.contribution,
            "alambana_text": self.alambana_text,
            "xiang_text": self.xiang_text,
            "si_text": self.si_text,
            "supporting_mental_factors": list(self.supporting_mental_factors),
            "supporting_scores": dict(self.supporting_scores),
        }


@dataclass
class AggregatedBehaviorPrediction:
    """跨分支汇总后的行为。"""

    behavior_id: str
    canonical_text: str
    total_probability: float
    confidence: float
    branch_contributions: list[BehaviorBranchContribution] = field(default_factory=list)
    dominant_mental_factors: list[str] = field(default_factory=list)
    alternative_mental_factors: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "behavior_id": self.behavior_id,
            "canonical_text": self.canonical_text,
            "total_probability": self.total_probability,
            "confidence": self.confidence,
            "branch_contributions": [
                item.to_dict() for item in self.branch_contributions
            ],
            "dominant_mental_factors": list(self.dominant_mental_factors),
            "alternative_mental_factors": list(self.alternative_mental_factors),
            "uncertainty_notes": list(self.uncertainty_notes),
        }


@dataclass
class CognitionBranch:
    """独立认知分支。

    - activation_probability：内心激活 = 归一 prior（心所期望 / 跨事件 / 行为融合）
    - branch_confidence：诊断用，不乘进 a
    - behavior_contribution_probability：兼容别名，恒等于 activation
    - branch_effective_probability：兼容别名，等同 activation_probability
    """

    id: str
    alambana: SubjectiveAlambanaCandidate
    xiang: ExtractedMentalState | None = None
    si: ExtractedMentalState | None = None
    branch_prior_probability: float = 0.0
    branch_confidence: float = 0.0
    branch_effective_probability_raw: float = 0.0
    branch_effective_probability: float = 0.0
    activation_probability_raw: float = 0.0
    activation_probability: float = 0.0
    behavior_contribution_probability_raw: float = 0.0
    behavior_contribution_probability: float = 0.0
    mental_factor_scores_raw: dict[str, float] = field(default_factory=dict)
    mental_factor_scores_final: dict[str, float] = field(default_factory=dict)
    score_breakdowns: dict[str, "ScoreBreakdown"] = field(default_factory=dict)
    behavior_predictions: list[BranchBehaviorPrediction] = field(default_factory=list)
    status: BranchStatus = "active"
    uncertainty_notes: list[str] = field(default_factory=list)
    resolutions: list = field(default_factory=list)
    stage_results: list = field(default_factory=list)
    behavior_diagnostics: dict = field(default_factory=dict)

    def to_dict(self, *, include_breakdown_explanations: bool = False) -> dict:
        breakdowns: dict[str, dict | None] = {}
        for name, breakdown in self.score_breakdowns.items():
            breakdowns[name] = breakdown.to_dict(
                include_explanation=include_breakdown_explanations
            )
        activation = self.activation_probability or self.branch_effective_probability
        behavior = self.behavior_contribution_probability
        return {
            "branch_id": self.id,
            "branch_prior_probability": self.branch_prior_probability,
            "branch_confidence": self.branch_confidence,
            "branch_effective_probability_raw": (
                self.activation_probability_raw
                or self.branch_effective_probability_raw
            ),
            "branch_effective_probability": activation,
            "activation_probability_raw": (
                self.activation_probability_raw
                or self.branch_effective_probability_raw
            ),
            "activation_probability": activation,
            "behavior_contribution_probability_raw": (
                self.behavior_contribution_probability_raw
            ),
            "behavior_contribution_probability": behavior,
            "alambana": self.alambana.to_dict(),
            "xiang": self.xiang.to_dict() if self.xiang else None,
            "si": self.si.to_dict() if self.si else None,
            "mental_factor_scores_raw": dict(self.mental_factor_scores_raw),
            "mental_factor_scores_final": dict(self.mental_factor_scores_final),
            "score_breakdowns": breakdowns,
            "behavior_predictions": [
                item.to_dict() for item in self.behavior_predictions
            ],
            "action_distribution": [
                {
                    "behavior_id": item.behavior_id,
                    "text": item.text,
                    "probability": item.conditional_probability,
                    "conditional_probability": item.conditional_probability,
                    "confidence": item.confidence,
                    "supporting_mental_factors": list(item.supporting_mental_factors),
                    "reason_summary": item.reason_summary,
                }
                for item in self.behavior_predictions
            ],
            "status": self.status,
            "uncertainty_notes": list(self.uncertainty_notes),
            "behavior_diagnostics": dict(self.behavior_diagnostics),
            "resolutions": [
                {
                    "name_a": r.name_a,
                    "name_b": r.name_b,
                    "winner": r.winner,
                    "loser": r.loser,
                    "branch_id": self.id,
                    **({"note": r.note} if r.note else {}),
                }
                for r in self.resolutions
            ],
        }
