"""交互后更新运行时种子库。"""

from mental_world_model.engine.seeds.config import (
    SEED_EMA_ALPHA,
    SEED_ZERO_DECAY,
    is_seed_library_enabled,
)


def update_seed_weights(
    current: dict[str, float],
    scores_final: dict[str, float],
    *,
    alpha: float = SEED_EMA_ALPHA,
    zero_decay: float = SEED_ZERO_DECAY,
) -> dict[str, float]:
    """EMA 更新：有分时向分数靠拢，零分时缓慢衰减。"""
    if not is_seed_library_enabled():
        return dict(current)
    updated: dict[str, float] = {}
    for name, old_weight in current.items():
        score = float(scores_final.get(name, 0.0))
        if score > 0:
            updated[name] = (1.0 - alpha) * old_weight + alpha * score
        else:
            updated[name] = (1.0 - zero_decay) * old_weight
    return updated
