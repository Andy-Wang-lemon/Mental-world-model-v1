"""心理连续性：多事件按 θ 加权平均，再按相关度混入本事件。"""

from __future__ import annotations

from mental_world_model.engine.continuity.config import (
    MENTAL_LAMBDA0,
    MENTAL_LAMBDA_MAX,
)
from mental_world_model.engine.continuity.models import MentalEpisode
from mental_world_model.utils.confidence import clamp_unit_interval


def weighted_average_mental_history(
    episodes: list[MentalEpisode],
    *,
    factor_names: list[str] | None = None,
) -> tuple[dict[str, float], list[float], float]:
    """M_hist：按 θ 加权平均。"""
    if not episodes:
        return {}, [], 0.0

    weights = [clamp_unit_interval(ep.theta) for ep in episodes]
    sum_w = sum(weights)
    names = factor_names or sorted(
        {name for ep in episodes for name in ep.scores}
    )
    if sum_w <= 1e-12:
        return {n: 0.0 for n in names}, weights, 0.0

    hist: dict[str, float] = {}
    for name in names:
        total = 0.0
        for ep, w in zip(episodes, weights):
            total += w * float(ep.scores.get(name, 0.0))
        hist[name] = total / sum_w
    return hist, weights, sum_w


def compute_lambda_eff(
    relevance: float,
    *,
    lambda0: float = MENTAL_LAMBDA0,
    lambda_max: float = MENTAL_LAMBDA_MAX,
) -> float:
    """λ = λ0 × R，上限 λ_max。"""
    r = clamp_unit_interval(relevance)
    return min(lambda_max, max(0.0, lambda0 * r))


def blend_mental_scores(
    event_scores: dict[str, float],
    hist_scores: dict[str, float],
    lambda_eff: float,
) -> dict[str, float]:
    """M_final = (1-λ) M_event + λ M_hist。"""
    lam = clamp_unit_interval(lambda_eff)
    names = set(event_scores) | set(hist_scores)
    return {
        name: (1.0 - lam) * float(event_scores.get(name, 0.0))
        + lam * float(hist_scores.get(name, 0.0))
        for name in names
    }
