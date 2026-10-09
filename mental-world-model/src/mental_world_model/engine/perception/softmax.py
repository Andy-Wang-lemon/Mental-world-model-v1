"""主观所缘本地加权 Softmax。"""

from __future__ import annotations

import math

from mental_world_model.engine.perception.config import (
    ALAMBANA_SCORE_W_EVENT,
    ALAMBANA_SCORE_W_SELF,
    ALAMBANA_SCORE_W_WORLD,
)
from mental_world_model.utils.confidence import as_unit_float


def weighted_alambana_score(
    event_consistency: float,
    self_consistency: float,
    world_consistency: float,
    *,
    w_event: float = ALAMBANA_SCORE_W_EVENT,
    w_self: float = ALAMBANA_SCORE_W_SELF,
    w_world: float = ALAMBANA_SCORE_W_WORLD,
) -> float:
    """Score(O) = w1·Event + w2·Self + w3·World。"""
    return (
        w_event * as_unit_float(event_consistency)
        + w_self * as_unit_float(self_consistency)
        + w_world * as_unit_float(world_consistency)
    )


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """数值稳定 Softmax；空列表返回空。"""
    if not scores:
        return []
    t = max(float(temperature), 1e-6)
    scaled = [s / t for s in scores]
    peak = max(scaled)
    exps = [math.exp(s - peak) for s in scaled]
    total = sum(exps)
    if total <= 0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def apply_local_weighted_softmax(
    consistencies: list[tuple[float, float, float]],
    *,
    w_event: float = ALAMBANA_SCORE_W_EVENT,
    w_self: float = ALAMBANA_SCORE_W_SELF,
    w_world: float = ALAMBANA_SCORE_W_WORLD,
    temperature: float = 1.0,
) -> tuple[list[float], list[float]]:
    """对多条候选的一致性三元组：先加权得分，再 Softmax。

    返回 (raw_scores, probabilities)，probabilities 和为 1。
    """
    raw = [
        weighted_alambana_score(
            e, s, w, w_event=w_event, w_self=w_self, w_world=w_world
        )
        for e, s, w in consistencies
    ]
    probs = softmax(raw, temperature=temperature)
    return raw, probs
