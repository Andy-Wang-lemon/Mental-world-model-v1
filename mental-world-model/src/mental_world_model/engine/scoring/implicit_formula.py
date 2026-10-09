"""隐式路径最终分公式：S = a · (β·s_sy + (1-β)·W)。"""

from mental_world_model.engine.scoring.weights import GATE_THRESHOLD
from mental_world_model.utils.confidence import clamp_unit_interval

# 所缘相似度权重；种子为大众敏感度
IMPLICIT_SUOYUAN_BETA = 0.75


def compute_implicit_score(
    suoyuan_score: float,
    seed_weight: float,
    activation: float,
    *,
    beta: float = IMPLICIT_SUOYUAN_BETA,
    gate_threshold: float = GATE_THRESHOLD,
) -> float:
    """隐式路径单心所最终分（消解前）。

    仅当所缘分严格大于门控阈值时参与；否则为 0。
    """
    s = clamp_unit_interval(suoyuan_score)
    if s <= gate_threshold:
        return 0.0
    w = clamp_unit_interval(seed_weight)
    a = clamp_unit_interval(activation)
    beta = clamp_unit_interval(beta)
    return clamp_unit_interval(a * (beta * s + (1.0 - beta) * w))
