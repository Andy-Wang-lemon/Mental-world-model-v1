"""分支概率与跨分支汇总纯函数。

单概率契约（已去掉独立 b）：
- activation_probability (a)：内心激活；心所期望 / 跨事件 / 行为融合
- 主路径 a = normalize(prior)；prior=组合激活 P(C)，不再乘支置信
- branch_effective_probability：兼容别名，始终同步为 a
- behavior_contribution_probability：兼容字段，恒等于 a（不再 a×L）
"""

from __future__ import annotations

from mental_world_model.engine.cognition.models import (
    AggregatedBehaviorPrediction,
    BehaviorBranchContribution,
    BranchBehaviorPrediction,
    CognitionBranch,
)
from mental_world_model.engine.perception.config import MENTAL_FACTOR_PRESENT_THRESHOLD
from mental_world_model.engine.perception.validators import (
    normalize_probability_distribution,
    texts_semantically_near,
)
from mental_world_model.utils.confidence import clamp_unit_interval


def geometric_mean(values: list[float]) -> float:
    valid = [clamp_unit_interval(v) for v in values]
    if not valid:
        return 0.0
    if any(v == 0 for v in valid):
        return 0.0
    product = 1.0
    for value in valid:
        product *= value
    return product ** (1.0 / len(valid))


def compute_branch_confidence(
    alambana_confidence: float,
    xiang_confidence: float,
    si_confidence: float,
) -> float:
    """支置信（诊断/不确定性估计用；不再乘进激活 a）。"""
    return geometric_mean(
        [alambana_confidence, xiang_confidence, si_confidence]
    )


def _activation_prior(branch: CognitionBranch) -> float:
    """激活权重基底：所缘 prior（不含行动似然）。"""
    alambana = branch.alambana
    prior = float(getattr(alambana, "prior_probability", 0.0) or 0.0)
    if prior > 0:
        return prior
    return float(branch.branch_prior_probability or 0.0)


def normalize_branch_probabilities(branches: list[CognitionBranch]) -> None:
    """只写入 activation a（原地）。

    a_raw = prior(O)（=组合激活 P(C)）；跨支再归一。
    失败分支 raw=0。全部失败时 a=0。
    behavior_contribution_probability 兼容同步为 a。
    """
    failed = frozenset({"scoring_failed"})
    for branch in branches:
        if branch.status in failed:
            branch.activation_probability_raw = 0.0
            branch.behavior_contribution_probability_raw = 0.0
            branch.branch_effective_probability_raw = 0.0
            continue
        a_raw = _activation_prior(branch)
        branch.activation_probability_raw = a_raw
        branch.behavior_contribution_probability_raw = a_raw
        branch.branch_effective_probability_raw = a_raw

    a_raws = [b.activation_probability_raw for b in branches]

    if not a_raws or sum(a_raws) <= 0:
        for branch in branches:
            branch.activation_probability = 0.0
            branch.branch_effective_probability = 0.0
            branch.behavior_contribution_probability = 0.0
    else:
        a_norms = normalize_probability_distribution(a_raws)
        for branch, norm in zip(branches, a_norms):
            branch.activation_probability = norm
            branch.branch_effective_probability = norm
            branch.behavior_contribution_probability = norm


def normalize_behavior_id(text: str) -> str:
    """轻量规范化：压缩空白并小写化标识。"""
    compact = " ".join(text.strip().split())
    if not compact:
        return "empty_behavior"
    # 用作 id：保留可读中文，去掉过长尾部
    return compact[:80]


def merge_behavior_ids(
    predictions: list[tuple[CognitionBranch, BranchBehaviorPrediction]],
    *,
    enabled: bool = True,
) -> dict[str, list[tuple[CognitionBranch, BranchBehaviorPrediction]]]:
    """按规范化 id 或近义文本聚类。"""
    groups: dict[str, list[tuple[CognitionBranch, BranchBehaviorPrediction]]] = {}
    canonical_texts: dict[str, str] = {}

    for branch, pred in predictions:
        key = pred.behavior_id or normalize_behavior_id(pred.text)
        if enabled:
            matched: str | None = None
            for existing_id, sample_text in canonical_texts.items():
                if texts_semantically_near(pred.text, sample_text):
                    matched = existing_id
                    break
            if matched is not None:
                key = matched
            else:
                canonical_texts[key] = pred.text
        else:
            canonical_texts.setdefault(key, pred.text)
        groups.setdefault(key, []).append((branch, pred))
    return groups


def aggregate_behavior_predictions(
    branches: list[CognitionBranch],
    *,
    normalization_enabled: bool = True,
    top_k: int | None = None,
) -> list[AggregatedBehaviorPrediction]:
    """跨支融合：P(行为)=Σ a(支)×P(行为|支)。

    ``top_k`` 非空时保留总概率最高的 K 条并重新归一化。
    """
    flat: list[tuple[CognitionBranch, BranchBehaviorPrediction]] = [
        (branch, pred)
        for branch in branches
        for pred in branch.behavior_predictions
    ]
    groups = merge_behavior_ids(flat, enabled=normalization_enabled)
    aggregated: list[AggregatedBehaviorPrediction] = []

    for behavior_id, items in groups.items():
        contributions: list[BehaviorBranchContribution] = []
        total = 0.0
        conf_num = 0.0
        conf_den = 0.0
        factor_votes: dict[str, float] = {}

        for branch, pred in items:
            a_prob = (
                branch.activation_probability
                or branch.branch_effective_probability
            )
            contribution = a_prob * pred.conditional_probability
            total += contribution
            conf_num += contribution * pred.confidence
            conf_den += contribution
            for name in pred.supporting_mental_factors:
                factor_votes[name] = factor_votes.get(name, 0.0) + contribution
            contributions.append(
                BehaviorBranchContribution(
                    branch_id=branch.id,
                    conditional_behavior_probability=pred.conditional_probability,
                    contribution=contribution,
                    alambana_text=branch.alambana.text,
                    xiang_text=branch.xiang.text if branch.xiang else "",
                    si_text=branch.si.text if branch.si else "",
                    supporting_mental_factors=list(pred.supporting_mental_factors),
                    supporting_scores=dict(pred.supporting_scores),
                    activation_probability=a_prob,
                    behavior_contribution_probability=a_prob,
                )
            )

        contributions.sort(key=lambda c: c.contribution, reverse=True)
        ranked_factors = sorted(
            factor_votes.items(), key=lambda kv: kv[1], reverse=True
        )
        dominant = [name for name, _ in ranked_factors[:5]]
        alternative = [
            name
            for name, _ in ranked_factors[5:10]
            if name not in dominant
        ]
        confidence = (conf_num / conf_den) if conf_den > 0 else 0.0
        canonical = items[0][1].text
        aggregated.append(
            AggregatedBehaviorPrediction(
                behavior_id=behavior_id,
                canonical_text=canonical,
                total_probability=clamp_unit_interval(total),
                confidence=clamp_unit_interval(confidence),
                branch_contributions=contributions,
                dominant_mental_factors=dominant,
                alternative_mental_factors=alternative,
            )
        )

    aggregated.sort(key=lambda a: a.total_probability, reverse=True)
    if top_k is not None and top_k > 0 and len(aggregated) > top_k:
        aggregated = aggregated[:top_k]
        norms = normalize_probability_distribution(
            [a.total_probability for a in aggregated]
        )
        for item, norm in zip(aggregated, norms):
            item.total_probability = norm
    return aggregated


def calculate_expected_mental_factor_scores(
    branches: list[CognitionBranch],
    factor_names: list[str],
) -> dict[str, float]:
    """心所期望用 activation_probability。"""
    expected: dict[str, float] = {name: 0.0 for name in factor_names}
    for branch in branches:
        p = branch.activation_probability or branch.branch_effective_probability
        if p <= 0:
            continue
        for name in factor_names:
            expected[name] += p * branch.mental_factor_scores_final.get(name, 0.0)
    return expected


def calculate_mental_factor_presence_probabilities(
    branches: list[CognitionBranch],
    factor_names: list[str],
    *,
    threshold: float = MENTAL_FACTOR_PRESENT_THRESHOLD,
) -> dict[str, float]:
    presence: dict[str, float] = {name: 0.0 for name in factor_names}
    for branch in branches:
        p = branch.activation_probability or branch.branch_effective_probability
        if p <= 0:
            continue
        for name in factor_names:
            if branch.mental_factor_scores_final.get(name, 0.0) >= threshold:
                presence[name] += p
    return presence


def calculate_expected_and_presence(
    branches: list[CognitionBranch],
    factor_names: list[str],
    *,
    threshold: float = MENTAL_FACTOR_PRESENT_THRESHOLD,
) -> tuple[dict[str, float], dict[str, float]]:
    """单次遍历同时计算期望分与心所存在概率（权重 = activation）。"""
    expected: dict[str, float] = {name: 0.0 for name in factor_names}
    presence: dict[str, float] = {name: 0.0 for name in factor_names}
    for branch in branches:
        p = branch.activation_probability or branch.branch_effective_probability
        if p <= 0:
            continue
        for name in factor_names:
            score = branch.mental_factor_scores_final.get(name, 0.0)
            expected[name] += p * score
            if score >= threshold:
                presence[name] += p
    return expected, presence
