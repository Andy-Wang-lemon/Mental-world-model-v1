"""主观世界建构层数据模型。

流程：刺激/行动拆分 → 客观刺激 → 我执+法执 → 组合激活 → 主观所缘（prior=P(C)）。
行动证据进入 cognition（思 / L / 想）；本层不看主体行动、不改所缘正文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GraspingType = Literal["self_grasping", "dharma_grasping"]
FieldSource = Literal["原文", "推理"]
GraspingSourceType = Literal["fresh", "historical", "accommodation", "neutral", "mixed"]


@dataclass
class StimulusActionSplit:
    """方案 A：一次拆分刺激文本与行动证据骨架（likelihood 在认知层补）。

    has_observable_action / confidence 由本地路由写入，不由 L0 LLM 判定。
    """

    stimulus_text: str = ""
    has_observable_action: bool = False
    action_span: str = ""
    action_description: str = ""
    confidence: float = 0.0
    reason_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "stimulus_text": self.stimulus_text,
            "has_observable_action": self.has_observable_action,
            "action_span": self.action_span,
            "action_description": self.action_description,
            "confidence": self.confidence,
            "reason_summary": self.reason_summary,
        }


@dataclass
class ObjectiveFact:
    """客观事件原子事实（兼容下游）。"""

    text: str
    source_span: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_span": self.source_span,
            "confidence": self.confidence,
        }


@dataclass
class ObjectiveEventStructure:
    """客观刺激结构：面对主体的可观察刺激（不含主体随后造作/成败）。

    ``action``：他人/环境施加的刺激动作；禁止写主体反应。
    ``outcome_changes``：刺激侧局面变化；禁止写主体行动结果。
    ``role_changes`` / ``relationship_changes`` / ``constraint_changes`` /
    ``resource_changes``：客观结构变化，禁止心理解释。
    """

    actors: list[str] = field(default_factory=list)
    action: str = ""
    object: str = ""
    scene: str = ""
    outcome_changes: list[str] = field(default_factory=list)
    role_changes: list[str] = field(default_factory=list)
    relationship_changes: list[str] = field(default_factory=list)
    constraint_changes: list[str] = field(default_factory=list)
    resource_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "actors": list(self.actors),
            "action": self.action,
            "object": self.object,
            "scene": self.scene,
            "outcome_changes": list(self.outcome_changes),
            "role_changes": list(self.role_changes),
            "relationship_changes": list(self.relationship_changes),
            "constraint_changes": list(self.constraint_changes),
            "resource_changes": list(self.resource_changes),
        }

    def as_facts(self) -> list[ObjectiveFact]:
        facts: list[ObjectiveFact] = []
        for actor in self.actors:
            if actor.strip():
                facts.append(
                    ObjectiveFact(text=f"参与者：{actor.strip()}", source_span=actor.strip())
                )
        if self.action.strip():
            facts.append(
                ObjectiveFact(
                    text=f"刺激动作：{self.action.strip()}",
                    source_span=self.action.strip(),
                )
            )
        if self.object.strip():
            facts.append(
                ObjectiveFact(text=f"对象：{self.object.strip()}", source_span=self.object.strip())
            )
        if self.scene.strip():
            facts.append(
                ObjectiveFact(text=f"场景：{self.scene.strip()}", source_span=self.scene.strip())
            )
        for change in self.outcome_changes:
            if change.strip():
                facts.append(
                    ObjectiveFact(
                        text=f"刺激侧变化：{change.strip()}",
                        source_span=change.strip(),
                    )
                )
        for label, items in (
            ("角色变化", self.role_changes),
            ("关系变化", self.relationship_changes),
            ("约束变化", self.constraint_changes),
            ("资源变化", self.resource_changes),
        ):
            for change in items:
                if change.strip():
                    facts.append(
                        ObjectiveFact(
                            text=f"{label}：{change.strip()}",
                            source_span=change.strip(),
                        )
                    )
        return facts

    def summary_text(self) -> str:
        parts: list[str] = []
        if self.actors:
            parts.append("参与者：" + "、".join(a.strip() for a in self.actors if a.strip()))
        if self.action.strip():
            parts.append(f"刺激动作：{self.action.strip()}")
        if self.object.strip():
            parts.append(f"对象：{self.object.strip()}")
        if self.scene.strip():
            parts.append(f"场景：{self.scene.strip()}")
        if self.outcome_changes:
            parts.append(
                "刺激侧变化："
                + "；".join(c.strip() for c in self.outcome_changes if c.strip())
            )
        for label, items in (
            ("角色变化", self.role_changes),
            ("关系变化", self.relationship_changes),
            ("约束变化", self.constraint_changes),
            ("资源变化", self.resource_changes),
        ):
            if items:
                parts.append(
                    f"{label}：" + "；".join(c.strip() for c in items if c.strip())
                )
        return "；".join(parts)


@dataclass
class GraspingHypothesis:
    """我执或法执条目（开放列表）。"""

    id: str
    type: GraspingType
    text: str
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    source: FieldSource = "推理"
    source_type: GraspingSourceType = "fresh"
    source_bias_frame_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "confidence": self.confidence,
            "supporting_evidence": list(self.supporting_evidence),
            "counter_evidence": list(self.counter_evidence),
            "missing_information": list(self.missing_information),
            "source": self.source,
            "source_type": self.source_type,
            "source_bias_frame_ids": list(self.source_bias_frame_ids),
        }


@dataclass
class GraspingCombination:
    """一个潜在心理状态种子：选定的我执×法执侧重。

    感知只评 activation（刺激是否点燃该视角）；行为融合在认知层用 a。
    """

    combo_id: str
    used_self_ids: list[str] = field(default_factory=list)
    used_dharma_ids: list[str] = field(default_factory=list)
    focus_axis: str = ""  # 可选调试字段，不参与剪枝/决策
    probability: float = 0.0  # 兼容别名：= activation_probability（组合层原始激活）
    reason: str = ""
    score: float = 0.0  # 激活分数（用于排序/保留）
    activation_alignment: float = 1.0
    activation_probability: float = 0.0  # 组合层原始激活概率
    # 融合所缘一致性后的最终 prior；与 activation 分字段，禁止覆盖
    alambana_prior_probability: float = 0.0
    self_confidence: float = 0.0
    dharma_confidence: float = 0.0
    # 视角指纹：我执｜法执｜reason
    perspective_fingerprint: str = ""
    source_type: GraspingSourceType = "fresh"
    source_bias_frame_ids: list[str] = field(default_factory=list)
    raw_combo_score: float = 0.0
    reactivated_h_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.reactivated_h_ids:
            self.reactivated_h_ids = list(self.source_bias_frame_ids)

    def id_set(self) -> frozenset[str]:
        return frozenset([*self.used_self_ids, *self.used_dharma_ids])

    def perspective_text(
        self,
        self_by_id: dict[str, GraspingHypothesis],
        dharma_by_id: dict[str, GraspingHypothesis],
    ) -> str:
        parts: list[str] = []
        for sid in self.used_self_ids:
            hyp = self_by_id.get(sid)
            if hyp and hyp.text.strip():
                parts.append(hyp.text.strip())
        for did in self.used_dharma_ids:
            hyp = dharma_by_id.get(did)
            if hyp and hyp.text.strip():
                parts.append(hyp.text.strip())
        if self.reason.strip():
            parts.append(self.reason.strip())
        return "｜".join(parts)

    def stamp_perspective_fingerprint(
        self,
        self_by_id: dict[str, GraspingHypothesis],
        dharma_by_id: dict[str, GraspingHypothesis],
    ) -> str:
        from mental_world_model.engine.perception.validators import text_fingerprint

        text = self.perspective_text(self_by_id, dharma_by_id)
        self.perspective_fingerprint = text_fingerprint(text) if text else ""
        return text

    def to_dict(self) -> dict:
        combo_act = self.activation_probability or self.probability
        return {
            "combo_id": self.combo_id,
            "used_self_ids": list(self.used_self_ids),
            "used_dharma_ids": list(self.used_dharma_ids),
            "probability": combo_act,
            "activation_probability": combo_act,
            "combo_activation_probability": combo_act,
            "alambana_prior_probability": self.alambana_prior_probability,
            "score": self.score,
            "activation_alignment": self.activation_alignment,
            "self_confidence": self.self_confidence,
            "dharma_confidence": self.dharma_confidence,
            "perspective_fingerprint": self.perspective_fingerprint,
            "source_type": self.source_type,
            "source_bias_frame_ids": list(self.source_bias_frame_ids),
            "reactivated_h_ids": list(self.reactivated_h_ids),
            "raw_combo_score": self.raw_combo_score or self.score,
        }


@dataclass
class SubjectiveAlambanaCandidate:
    """主观所缘候选：文本 + 组合引用 + prior（=组合激活）。

    一致性等诊断字段已停用，不再进 LLM / prior。
    """

    id: str
    text: str = ""
    used_self_ids: list[str] = field(default_factory=list)
    used_dharma_ids: list[str] = field(default_factory=list)
    combo_reason: str = ""
    event_consistency: float = 0.0
    self_consistency: float = 0.0
    world_consistency: float = 0.0
    raw_score: float = 0.0
    # prior：主路径=组合激活；行动更新后写入 posterior
    prior_probability: float = 0.0
    posterior_probability: float = 0.0
    # compat=LLM 行动贴合 raw；action_likelihood=合成后有效 L（用于 p'）
    action_compat: float = 1.0
    action_likelihood: float = 1.0
    confidence: float = 0.0
    # 与 combo 视角同一空间
    perspective_text: str = ""
    perspective_fingerprint: str = ""
    # 下游兼容别名
    self_grasping_ids: list[str] = field(default_factory=list)
    dharma_grasping_ids: list[str] = field(default_factory=list)
    objective_basis: list[str] = field(default_factory=list)
    added_meaning: list[str] = field(default_factory=list)
    distortion_risk: float = 0.0
    missing_conditions: list[str] = field(default_factory=list)
    source: FieldSource = "推理"
    # True：由行动扩支感知子路径生成；跳过后验 compat 矫正
    action_informed: bool = False
    source_type: GraspingSourceType = "fresh"
    source_bias_frame_ids: list[str] = field(default_factory=list)
    reactivated_h_ids: list[str] = field(default_factory=list)
    raw_combo_score: float = 0.0
    activation_alignment: float = 1.0
    grounding_valid: bool = True
    grounding_reason: str = ""
    # 旧结构字段兼容（可选空）
    experienced_scene: str = ""
    central_focus: str = ""
    self_world_relation: str = ""
    world_structure: str = ""
    uncertainty: str = ""

    def compose_text(self) -> str:
        return self.text.strip() or self.experienced_scene.strip()

    def sync_grasping_aliases(self) -> None:
        if self.used_self_ids and not self.self_grasping_ids:
            self.self_grasping_ids = list(self.used_self_ids)
        if self.used_dharma_ids and not self.dharma_grasping_ids:
            self.dharma_grasping_ids = list(self.used_dharma_ids)
        if self.self_grasping_ids and not self.used_self_ids:
            self.used_self_ids = list(self.self_grasping_ids)
        if self.dharma_grasping_ids and not self.used_dharma_ids:
            self.used_dharma_ids = list(self.dharma_grasping_ids)

    def to_dict(self) -> dict:
        self.sync_grasping_aliases()
        return {
            "id": self.id,
            "text": self.text or self.compose_text(),
            "used_self_ids": list(self.used_self_ids),
            "used_dharma_ids": list(self.used_dharma_ids),
            "combo_reason": self.combo_reason,
            "prior_probability": self.prior_probability,
            "posterior_probability": self.posterior_probability,
            "action_compat": self.action_compat,
            "action_likelihood": self.action_likelihood,
            "perspective_text": self.perspective_text,
            "perspective_fingerprint": self.perspective_fingerprint,
            "confidence": self.confidence,
            "self_grasping_ids": list(self.self_grasping_ids),
            "dharma_grasping_ids": list(self.dharma_grasping_ids),
            "source": self.source,
            "action_informed": self.action_informed,
            "source_type": self.source_type,
            "source_bias_frame_ids": list(self.source_bias_frame_ids),
            "reactivated_h_ids": list(self.reactivated_h_ids),
            "raw_combo_score": self.raw_combo_score,
            "activation_alignment": self.activation_alignment,
            "grounding_valid": self.grounding_valid,
            "grounding_reason": self.grounding_reason,
        }


@dataclass
class PerceptionState:
    """主观世界建构结果（先验所缘集合）。"""

    source_text: str = ""
    stimulus_text: str = ""
    stimulus_action_split: StimulusActionSplit | None = None
    objective_event: ObjectiveEventStructure | None = None
    objective_facts: list[ObjectiveFact] = field(default_factory=list)
    self_grasping_hypotheses: list[GraspingHypothesis] = field(default_factory=list)
    dharma_grasping_hypotheses: list[GraspingHypothesis] = field(default_factory=list)
    base_alambana_text: str = ""
    grasping_combos: list[dict] = field(default_factory=list)
    alambana_candidates: list[SubjectiveAlambanaCandidate] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    pipeline_trace: dict = field(default_factory=dict)
    h_prior: dict = field(default_factory=dict)
    # 兼容旧字段
    self_representation: dict | None = None
    world_representation: dict | None = None

    def to_dict(self) -> dict:
        return {
            "source_text": self.source_text,
            "stimulus_text": self.stimulus_text,
            "stimulus_action_split": (
                self.stimulus_action_split.to_dict()
                if self.stimulus_action_split
                else None
            ),
            "objective_event": (
                self.objective_event.to_dict() if self.objective_event else None
            ),
            "objective_facts": [item.to_dict() for item in self.objective_facts],
            "self_grasping_hypotheses": [
                item.to_dict() for item in self.self_grasping_hypotheses
            ],
            "dharma_grasping_hypotheses": [
                item.to_dict() for item in self.dharma_grasping_hypotheses
            ],
            "base_alambana_text": self.base_alambana_text,
            "grasping_combos": list(self.grasping_combos),
            "alambana_candidates": [
                item.to_dict() for item in self.alambana_candidates
            ],
            "uncertainty_notes": list(self.uncertainty_notes),
            "pipeline_trace": dict(self.pipeline_trace),
            "h_prior": dict(self.h_prior),
        }
