"""评分权重配置与最终分合成。"""

from dataclasses import dataclass, field
from pathlib import Path

from mental_world_model.engine.extraction.models import ExtractedContext
from mental_world_model.engine.seeds.config import is_seed_library_enabled

# 门控阈值：低于此值跳过（双门控等）
GATE_THRESHOLD = 0.3

# 所缘硬地板：低于此值直接出局，不打生起念头及后续列
SUOYUAN_HARD_FLOOR = 0.1

# 保留常量供旧测试/文档引用；分支评分不再使用置信度下限调节
INFERENCE_CONFIDENCE_FLOOR = 0.5

# 四列基础权重（w₁～w₄，之和为 1）
DEFAULT_COLUMN_WEIGHTS: dict[str, float] = {
    "suoyuan": 0.35,
    "shengqi_nian": 0.30,
    "ye": 0.10,
    "xiandai_xinli": 0.25,
}

COLUMN_ORDER: tuple[str, ...] = tuple(DEFAULT_COLUMN_WEIGHTS)

# 种子库单心所权重占位（后续从种子库加载）
DEFAULT_SEED_WEIGHT = 1.0

@dataclass
class ScoringWeights:
    """评分权重配置。"""

    column_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COLUMN_WEIGHTS)
    )
    seed_weights: dict[str, float] = field(default_factory=dict)

    def get_seed_weight(self, xinsuo_name: str) -> float:
        """获取心所配置的种子库权重（未启用时不参与合成）。"""
        return self.seed_weights.get(xinsuo_name, DEFAULT_SEED_WEIGHT)


def build_default_seed_weights(
    xinsuo_names: list[str],
    definitions_dir: str | Path | None = None,
) -> dict[str, float]:
    """从 definitions/seeds/xinsuo_seeds.json 加载种子库权重。"""
    if not is_seed_library_enabled():
        return {name: DEFAULT_SEED_WEIGHT for name in xinsuo_names}
    if definitions_dir is None:
        return {name: DEFAULT_SEED_WEIGHT for name in xinsuo_names}

    from mental_world_model.engine.loader.seed_loader import load_xinsuo_seed_weights

    return load_xinsuo_seed_weights(Path(definitions_dir), xinsuo_names)


def compute_source_modifiers(extracted: ExtractedContext) -> dict[str, float]:
    """四列来源调节系数 m：恒为 1.0。

    分支体制下，所缘/想/思的不确定性由 activation_probability
    承担；分支内心所 final 按固定 w 加权。
    """
    _ = extracted
    return {
        "suoyuan": 1.0,
        "shengqi_nian": 1.0,
        "ye": 1.0,
        "xiandai_xinli": 1.0,
    }


def compute_normalized_column_weights(
    column_weights: dict[str, float],
    modifiers: dict[str, float],
) -> dict[str, float]:
    """按来源调节系数归一化列权重，保证有效权重之和为 1。"""
    weighted = {
        column: column_weights.get(column, 0.0) * modifiers.get(column, 1.0)
        for column in COLUMN_ORDER
    }
    total = sum(weighted.values())
    if total <= 0:
        return {column: 0.0 for column in COLUMN_ORDER}
    return {column: weighted[column] / total for column in COLUMN_ORDER}


def compute_column_composite(
    column_scores: dict[str, float],
    normalized_weights: dict[str, float],
) -> float:
    """四列归一化加权求和。"""
    return sum(
        normalized_weights.get(column, 0.0) * column_scores.get(column, 0.0)
        for column in COLUMN_ORDER
    )


def compute_column_contributions(
    column_scores: dict[str, float],
    normalized_weights: dict[str, float],
) -> dict[str, float]:
    """各列贡献 ŵᵢ·sᵢ。"""
    return {
        column: normalized_weights.get(column, 0.0) * column_scores.get(column, 0.0)
        for column in COLUMN_ORDER
    }


def compute_score_breakdown(
    xinsuo_name: str,
    column_scores: dict[str, float],
    weights: ScoringWeights,
    extracted: ExtractedContext,
    *,
    source_modifiers: dict[str, float] | None = None,
    normalized_weights: dict[str, float] | None = None,
):
    """计算完整合成明细（含 m、ŵ、各列贡献）。"""
    from mental_world_model.engine.scoring.breakdown import ScoreBreakdown

    modifiers = source_modifiers or compute_source_modifiers(extracted)
    base_weights = {
        column: weights.column_weights.get(column, 0.0)
        for column in COLUMN_ORDER
    }
    normalized = normalized_weights or compute_normalized_column_weights(
        weights.column_weights, modifiers
    )
    contributions = compute_column_contributions(column_scores, normalized)
    column_composite = compute_column_composite(column_scores, normalized)
    seed_enabled = is_seed_library_enabled()
    configured_seed = weights.get_seed_weight(xinsuo_name)
    final_score = column_composite

    return ScoreBreakdown(
        xinsuo_name=xinsuo_name,
        column_scores={column: column_scores.get(column, 0.0) for column in COLUMN_ORDER},
        column_weights=base_weights,
        source_modifiers=modifiers,
        normalized_weights=normalized,
        column_contributions=contributions,
        column_composite=column_composite,
        seed_weight=configured_seed if seed_enabled else DEFAULT_SEED_WEIGHT,
        final_score=final_score,
        seed_weights_enabled=seed_enabled,
    )


def compute_final_score(
    xinsuo_name: str,
    column_scores: dict[str, float],
    weights: ScoringWeights,
    extracted: ExtractedContext,
) -> float:
    """根据四列分数、来源调节与种子库权重计算最终分。

    S_col = Σ ŵᵢ·sᵢ，其中 ŵᵢ = (wᵢ·mᵢ) / Σ(wⱼ·mⱼ)
    final = S_col（W_seed 仅用于路由/提示/画像，不参与乘分）
    """
    return compute_score_breakdown(
        xinsuo_name,
        column_scores,
        weights,
        extracted,
    ).final_score
