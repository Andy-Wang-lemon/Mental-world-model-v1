"""行动证据（思）与各支想：后验更新，不改所缘正文。

思 = L0 刺激/行动拆分骨架（每事件一次、各支共享；不另开思 LLM）。
compat 仅补各所缘 L；想吃所缘 text/二执/本支已绑定 H/行动骨架，不吃概率。

L 合成（不对称）：compat=行动贴合（LLM raw）；a=归一化 prior（内心激活代理）；
compat≥中性：h=h_up(a)=0.5+0.5a（支持侧充分生效）；
compat<中性：h=h_down(a)=0.25+0.25a（矛盾侧少压，避免高激活被打穿）；
L=1+h·(compat−1)；再 p'∝p·L。

扩支见 perception.action_expand：行动子路径生成的新支 action_informed=True，
跳过后验 compat 矫正。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mental_world_model.engine.cognition.models import ExtractedMentalState
from mental_world_model.engine.cognition.prompts import (
    ACTION_LIKELIHOOD_SCHEMA,
    BRANCH_XIANGS_SCHEMA,
    build_action_likelihood_messages,
    build_branch_xiang_messages,
)
from mental_world_model.engine.continuity.prior import h_payload_for_candidate
from mental_world_model.engine.extraction.validators import is_grounded_in_yuanwen
from mental_world_model.engine.perception.config import (
    ACTION_COMPAT_NEUTRAL,
    ACTION_LIKELIHOOD_H_DOWN_BASE,
    ACTION_LIKELIHOOD_H_DOWN_SLOPE,
    ACTION_LIKELIHOOD_H_UP_BASE,
    ACTION_LIKELIHOOD_H_UP_SLOPE,
)
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    ObjectiveEventStructure,
    ObjectiveFact,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.validators import normalize_probability_distribution
from mental_world_model.engine.scoring.parallel_exec import run_parallel
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.confidence import as_unit_float, clamp_unit_interval


@dataclass
class ActionLikelihood:
    alambana_id: str
    likelihood: float
    support_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "alambana_id": self.alambana_id,
            "likelihood": self.likelihood,
        }

@dataclass
class ActionEvidence:
    """思 = 行动证据：内心造作 + 外在表现 + 明确无表现（原文有的全收）。"""

    has_observable_action: bool = False
    action_span: str = ""
    action_description: str = ""
    likelihoods: list[ActionLikelihood] = field(default_factory=list)
    confidence: float = 0.0
    reason_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "has_observable_action": self.has_observable_action,
            "action_span": self.action_span,
            "action_description": self.action_description,
            "likelihoods": [item.to_dict() for item in self.likelihoods],
            "confidence": self.confidence,
        }

    def summary_text(self) -> str:
        if not self.has_observable_action:
            return ""
        return (
            self.action_description.strip()
            or self.action_span.strip()
            or "已记录行动证据"
        )

    def likelihood_map(self) -> dict[str, float]:
        return {item.alambana_id: item.likelihood for item in self.likelihoods}


def activation_proxy_from_priors(
    candidates: list[SubjectiveAlambanaCandidate],
) -> list[float]:
    """后验更新前用 prior 归一化作为内心激活 a 的代理（尚无支置信度）。"""
    raw = [max(0.0, float(c.prior_probability or 0.0)) for c in candidates]
    if sum(raw) <= 0:
        n = len(candidates)
        return [1.0 / n] * n if n else []
    return normalize_probability_distribution(raw)


def action_likelihood_gate(
    activation: float,
    compat: float,
    *,
    neutral: float = ACTION_COMPAT_NEUTRAL,
) -> float:
    """按 compat 相对中性选择 h_up 或 h_down。"""
    a = clamp_unit_interval(activation)
    c = clamp_unit_interval(compat)
    if c >= neutral:
        return ACTION_LIKELIHOOD_H_UP_BASE + ACTION_LIKELIHOOD_H_UP_SLOPE * a
    return ACTION_LIKELIHOOD_H_DOWN_BASE + ACTION_LIKELIHOOD_H_DOWN_SLOPE * a


def synthesize_action_likelihood(
    activation: float,
    compat: float,
) -> float:
    """L = 1 + h·(compat−1)；h 不对称（支持用 h_up，矛盾用 h_down）。

    a 高且 compat 低时不再用满额 h「打穿」强激活视角（表里可能不一）。
    """
    a = clamp_unit_interval(activation)
    c = clamp_unit_interval(compat)
    h = action_likelihood_gate(a, c)
    return clamp_unit_interval(1.0 + h * (c - 1.0))


def update_alambana_posteriors_with_action(
    candidates: list[SubjectiveAlambanaCandidate],
    evidence: ActionEvidence,
    notes: list[str] | None = None,
) -> list[SubjectiveAlambanaCandidate]:
    """p' ∝ p · L；L=g(a,compat) 不对称门控；不改 text。无行动则 p'=p。

    ``action_informed=True`` 的所缘（行动扩支子路径生成）跳过后验矫正。
    """
    notes = notes if notes is not None else []
    actionable = [c for c in candidates if not getattr(c, "action_informed", False)]
    if not actionable:
        if candidates:
            notes.append("无可行动后验矫正的所缘（均为 action_informed）。")
        return candidates

    if not evidence.has_observable_action:
        for cand in actionable:
            cand.posterior_probability = cand.prior_probability
            cand.action_likelihood = 1.0
            cand.action_compat = 1.0
        notes.append("无行动证据：后验保持先验。")
        return candidates

    a_proxy = activation_proxy_from_priors(actionable)
    lik_map = evidence.likelihood_map()
    raw: list[float] = []
    effective_ls: list[float] = []
    for cand, a in zip(actionable, a_proxy):
        compat = lik_map.get(cand.id)
        if compat is None:
            compat = 1.0
            notes.append(
                f"[强提示] {cand.id} 缺 compat，按中性 1.0（非静默 0.5）。"
            )
        compat = as_unit_float(compat, 1.0)
        L = synthesize_action_likelihood(a, compat)
        cand.action_compat = compat
        cand.action_likelihood = L  # 存有效 L（已用于改 prior）
        effective_ls.append(L)
        raw.append(max(0.0, cand.prior_probability) * L)

    if sum(raw) <= 0:
        notes.append("行动似然全无效，退回先验。")
        for cand in actionable:
            cand.posterior_probability = cand.prior_probability
        return candidates

    norms = normalize_probability_distribution(raw)
    for cand, p in zip(actionable, norms):
        cand.posterior_probability = p

    notes.append(
        "后验已用 L=g(a,compat) 更新（a=prior 归一；"
        "compat≥中性用 h_up，矛盾用 h_down；action_informed 支已跳过）。"
    )
    if max(effective_ls) - min(effective_ls) < 0.08:
        notes.append("有效 L 区分力弱，后验接近先验。")
    return candidates


def _parse_likelihoods(raw_likes: object) -> list[ActionLikelihood]:
    likelihoods: list[ActionLikelihood] = []
    if not isinstance(raw_likes, list):
        return likelihoods
    for item in raw_likes:
        if not isinstance(item, dict):
            continue
        aid = str(item.get("alambana_id") or "").strip()
        if not aid:
            continue
        notes_list = item.get("support_notes")
        support = (
            [str(x).strip() for x in notes_list if str(x).strip()]
            if isinstance(notes_list, list)
            else []
        )
        likelihoods.append(
            ActionLikelihood(
                alambana_id=aid,
                likelihood=as_unit_float(item.get("likelihood"), 1.0),
                support_notes=support,
            )
        )
    return likelihoods


def _align_likelihoods_to_candidates(
    likelihoods: list[ActionLikelihood],
    candidate_ids: list[str],
    notes: list[str],
) -> tuple[list[ActionLikelihood], list[str]]:
    """只保留已知候选 ID；返回已对齐项与仍缺失的 ID。"""
    known = set(candidate_ids)
    by_id: dict[str, ActionLikelihood] = {}
    unknown: list[str] = []
    for item in likelihoods:
        if item.alambana_id not in known:
            unknown.append(item.alambana_id)
            continue
        by_id[item.alambana_id] = item
    if unknown:
        notes.append(
            f"[强提示] 似然含未知所缘 ID，已丢弃：{unknown}；"
            f"须严格返回候选 ID={candidate_ids}。"
        )
    missing = [cid for cid in candidate_ids if cid not in by_id]
    aligned = [by_id[cid] for cid in candidate_ids if cid in by_id]
    return aligned, missing


def build_action_skeleton_from_split(
    yuanwen: str,
    *,
    has_observable_action: bool,
    action_span: str,
    action_description: str,
    confidence: float,
    reason_summary: str,
) -> tuple[ActionEvidence, list[str]]:
    """从 L0 拆分结果本地组装行动骨架（思的唯一来源；无 LLM）。"""
    notes: list[str] = []
    if not has_observable_action:
        return (
            ActionEvidence(
                has_observable_action=False,
                reason_summary=reason_summary or "无行动证据。",
            ),
            notes,
        )

    span = action_span.strip()
    desc = action_description.strip()
    if span:
        span_parts = [p.strip() for p in span.replace("；", ";").split(";") if p.strip()]
        grounded_parts = [p for p in span_parts if is_grounded_in_yuanwen(p, yuanwen)]
        if span_parts and not grounded_parts:
            notes.append(
                "拆分行动片段未能严格匹配原文，按契约保留行动（不作废）。"
            )
        elif grounded_parts and len(grounded_parts) < len(span_parts):
            notes.append("部分拆分行动片段未匹配原文，已剔除未落地段。")
            span = ";".join(grounded_parts)
        elif grounded_parts:
            span = ";".join(grounded_parts)

    if not span and not desc:
        notes.append("拆分行动缺少片段与描述，视为无行动。")
        return (
            ActionEvidence(
                has_observable_action=False,
                reason_summary="行动证据不完整。",
            ),
            notes,
        )

    return (
        ActionEvidence(
            has_observable_action=True,
            action_span=span,
            action_description=desc,
            confidence=as_unit_float(confidence, 0.6),
            reason_summary=reason_summary,
        ),
        notes,
    )


def action_skeleton_from_perception_split(
    yuanwen: str,
    split,
) -> tuple[ActionEvidence, list[str]]:
    """从感知 L0 拆分取行动骨架（思）；缺 split 直接报错。"""
    if split is None:
        raise ValueError(
            "缺少 stimulus_action_split：主路径必须先经 L0 刺激/行动拆分，"
            "已删除全文行动提取兜底。"
        )
    return build_action_skeleton_from_split(
        yuanwen,
        has_observable_action=bool(getattr(split, "has_observable_action", False)),
        action_span=str(getattr(split, "action_span", "") or ""),
        action_description=str(getattr(split, "action_description", "") or ""),
        confidence=float(getattr(split, "confidence", 0.0) or 0.0),
        reason_summary=str(getattr(split, "reason_summary", "") or ""),
    )


def score_action_likelihoods_on_skeleton(
    yuanwen: str,
    objective_event: ObjectiveEventStructure | None,
    facts: list[ObjectiveFact],
    candidates: list[SubjectiveAlambanaCandidate],
    skeleton: ActionEvidence,
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[ActionEvidence, list[str]]:
    """对已有行动骨架补各所缘 compat/L（LLM）；不改骨架 span/description。"""
    notes: list[str] = []
    if not skeleton.has_observable_action:
        return skeleton, notes
    if not candidates:
        return skeleton, notes

    candidate_ids = [c.id for c in candidates]

    def _call_likelihood(*, retry_missing: list[str] | None = None) -> dict:
        with llm_call_context(stage="cognition.action_likelihood"):
            response = llm.complete(
                build_action_likelihood_messages(
                    yuanwen,
                    objective_event,
                    facts,
                    candidates,
                    skeleton,
                    required_ids=candidate_ids,
                    retry_missing_ids=retry_missing,
                    focus_person=focus_person,
                ),
                schema=ACTION_LIKELIHOOD_SCHEMA,
            )
        return response if isinstance(response, dict) else {}

    try:
        raw = _call_likelihood()
        aligned, missing = _align_likelihoods_to_candidates(
            _parse_likelihoods(raw.get("likelihoods")), candidate_ids, notes
        )
        if missing:
            notes.append(
                f"[强提示] 似然缺少候选 ID={missing}，强制重试并要求严格返回这些 ID。"
            )
            raw = _call_likelihood(retry_missing=missing)
            aligned, missing = _align_likelihoods_to_candidates(
                _parse_likelihoods(raw.get("likelihoods")), candidate_ids, notes
            )
        if missing:
            notes.append(
                f"[强提示] 重试后仍缺似然 ID={missing}，按中性 L=1.0（非静默 0.5）。"
            )
            by_id = {x.alambana_id: x for x in aligned}
            for cid in missing:
                by_id[cid] = ActionLikelihood(
                    alambana_id=cid,
                    likelihood=1.0,
                    support_notes=["缺失似然，中性 L=1.0"],
                )
            aligned = [by_id[cid] for cid in candidate_ids]
        skeleton.likelihoods = aligned
        if raw.get("reason_summary"):
            skeleton.reason_summary = str(raw.get("reason_summary")).strip()
    except Exception as exc:  # noqa: BLE001
        notes.append(
            f"[强提示] 行动似然评分失败，各所缘按中性 L=1.0（非静默 0.5）：{exc}"
        )
        skeleton.likelihoods = [
            ActionLikelihood(
                alambana_id=cid,
                likelihood=1.0,
                support_notes=["评分失败，中性 L=1.0"],
            )
            for cid in candidate_ids
        ]

    return skeleton, notes


def score_action_likelihoods_from_split(
    yuanwen: str,
    objective_event: ObjectiveEventStructure | None,
    facts: list[ObjectiveFact],
    candidates: list[SubjectiveAlambanaCandidate],
    *,
    has_observable_action: bool,
    action_span: str,
    action_description: str,
    confidence: float,
    reason_summary: str,
    llm: LlmClient,
    focus_person: str = "",
) -> tuple[ActionEvidence, list[str]]:
    """用 L0 拆分出的行动骨架，只补各所缘 L(Act|O)。"""
    skeleton, notes = build_action_skeleton_from_split(
        yuanwen,
        has_observable_action=has_observable_action,
        action_span=action_span,
        action_description=action_description,
        confidence=confidence,
        reason_summary=reason_summary,
    )
    if not skeleton.has_observable_action or not candidates:
        return skeleton, notes
    evidence, score_notes = score_action_likelihoods_on_skeleton(
        yuanwen,
        objective_event,
        facts,
        candidates,
        skeleton,
        llm,
        focus_person=focus_person,
    )
    notes.extend(score_notes)
    return evidence, notes


def action_evidence_from_perception_split(
    yuanwen: str,
    objective_event: ObjectiveEventStructure | None,
    facts: list[ObjectiveFact],
    candidates: list[SubjectiveAlambanaCandidate],
    split,
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[ActionEvidence, list[str]]:
    """用感知 L0 拆分骨架补各所缘 L(Act|O)。缺 split 直接报错，不再回退全文抽取。"""
    skeleton, notes = action_skeleton_from_perception_split(yuanwen, split)
    if not skeleton.has_observable_action or not candidates:
        return skeleton, notes
    evidence, score_notes = score_action_likelihoods_on_skeleton(
        yuanwen,
        objective_event,
        facts,
        candidates,
        skeleton,
        llm,
        focus_person=focus_person,
    )
    notes.extend(score_notes)
    return evidence, notes


def generate_branch_xiangs(
    yuanwen: str,
    objective_event: ObjectiveEventStructure | None,
    candidates: list[SubjectiveAlambanaCandidate],
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    evidence: ActionEvidence,
    llm: LlmClient,
    *,
    grasping_by_alambana_id: dict[
        str, tuple[list[GraspingHypothesis], list[GraspingHypothesis]]
    ]
    | None = None,
    focus_person: str = "",
    h_prior: dict | None = None,
) -> dict[str, ExtractedMentalState]:
    """各候选所缘隔离并行生成想；有所缘必须有想。

    若提供 ``grasping_by_alambana_id``，每支只用该支已解析二执（真正支隔离）。
    ``h_prior`` 提供本支已绑定 H 的 id/text/mass；想只读，不再判相关度。
    """
    if not candidates:
        return {}

    def _one(cand: SubjectiveAlambanaCandidate) -> ExtractedMentalState:
        if grasping_by_alambana_id and cand.id in grasping_by_alambana_id:
            self_g, dharma_g = grasping_by_alambana_id[cand.id]
            prefer_passed = True
        else:
            self_g, dharma_g = self_grasping, dharma_grasping
            prefer_passed = False
        with llm_call_context(stage="cognition.branch_xiang"):
            response = llm.complete(
                build_branch_xiang_messages(
                    yuanwen,
                    objective_event,
                    cand,
                    self_g,
                    dharma_g,
                    evidence,
                    prefer_passed_grasping=prefer_passed,
                    focus_person=focus_person,
                    reactivated_h=h_payload_for_candidate(cand, h_prior),
                ),
                schema=BRANCH_XIANGS_SCHEMA,
            )
        raw = response if isinstance(response, dict) else {}
        items = raw.get("xiangs")
        text = ""
        source = "推理"
        confidence = 0.6
        reason = ""
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                aid = str(item.get("alambana_id") or "").strip()
                item_text = str(item.get("text") or "").strip()
                if not item_text:
                    continue
                # 严格：必须对本支 alambana_id；错 id / 无 id 一律不采纳
                if aid != cand.id:
                    continue
                text = item_text
                source = "推理" if item.get("source") != "原文" else "原文"
                confidence = as_unit_float(item.get("confidence"), 0.6)
                reason = str(item.get("reason_summary") or "").strip()
                break
        if not text:
            fallback = cand.text.strip() or "对该事态形成相应认知评价"
            text = (
                f"在此所缘下：{fallback}"
                if not fallback.startswith("在")
                else fallback
            )
            confidence = 0.35
            reason = "xiang_degraded：想缺失或 id 未对齐，已用所缘正文兜底"
        return ExtractedMentalState(
            text=text,
            source=source,
            confidence=confidence,
            alambana_id=cand.id,
            reason_summary=reason,
        )

    by_id = run_parallel(
        {cand.id: (lambda c=cand: _one(c)) for cand in candidates},
        acquire_slot=True,
    )
    return by_id


def bind_action_as_si(
    evidence: ActionEvidence,
    alambana_id: str,
) -> ExtractedMentalState:
    """业列：有行动证据（内心/外在/明确无表现）则写入思；全无则为空。"""
    text = evidence.summary_text()
    return ExtractedMentalState(
        text=text,
        source="原文" if evidence.has_observable_action and text else "推理",
        confidence=(
            as_unit_float(evidence.confidence, 0.8)
            if evidence.has_observable_action and text
            else 0.0
        ),
        alambana_id=alambana_id,
        reason_summary="行动证据作为思" if text else "无行动证据，思为空",
    )
