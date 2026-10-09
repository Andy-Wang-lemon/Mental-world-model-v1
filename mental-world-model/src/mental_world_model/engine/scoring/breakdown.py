"""最终分合成明细与可解释性输出。"""

from dataclasses import dataclass

from mental_world_model.engine.scoring.weights import COLUMN_ORDER

COLUMN_LABELS: dict[str, str] = {
    "suoyuan": "所缘境",
    "shengqi_nian": "生起念头",
    "ye": "业",
    "xiandai_xinli": "现代心理学",
}


@dataclass
class ScoreBreakdown:
    """单个心所最终分的完整合成明细。"""

    column_scores: dict[str, float]
    column_weights: dict[str, float]
    source_modifiers: dict[str, float]
    normalized_weights: dict[str, float]
    column_contributions: dict[str, float]
    column_composite: float
    seed_weight: float
    final_score: float
    xinsuo_name: str = ""
    seed_weights_enabled: bool = False

    def to_dict(self, *, include_explanation: bool = False) -> dict:
        """转为可序列化字典。"""
        payload = {
            "xinsuo_name": self.xinsuo_name,
            "column_scores": dict(self.column_scores),
            "column_weights": dict(self.column_weights),
            "source_modifiers": dict(self.source_modifiers),
            "normalized_weights": dict(self.normalized_weights),
            "column_contributions": dict(self.column_contributions),
            "column_composite": self.column_composite,
            "final_score": self.final_score,
        }
        if include_explanation:
            payload["explanation"] = self.explain()
        if self.seed_weights_enabled:
            payload["seed_weight"] = self.seed_weight
        causal = getattr(self, "causal_context", None)
        if isinstance(causal, dict):
            payload["causal_context"] = dict(causal)
        return payload

    def explain(self) -> str:
        """生成中文可解释性说明。"""
        lines = [f"【{self.xinsuo_name}】最终分 = {self.final_score:.4f}", ""]

        lines.append("1. 四列匹配分 s：")
        for column in COLUMN_ORDER:
            label = COLUMN_LABELS[column]
            score = self.column_scores.get(column, 0.0)
            lines.append(f"   - {label} ({column}): {score:.4f}")

        lines.append("")
        lines.append("2. 基础列权重 w：")
        for column in COLUMN_ORDER:
            label = COLUMN_LABELS[column]
            weight = self.column_weights.get(column, 0.0)
            lines.append(f"   - {label}: {weight:.4f}")

        lines.append("")
        lines.append("3. 来源调节系数 m（分支内恒为 1.0；不确定性由分支概率承担）：")
        for column in COLUMN_ORDER:
            label = COLUMN_LABELS[column]
            modifier = self.source_modifiers.get(column, 1.0)
            lines.append(f"   - {label}: {modifier:.4f}")

        lines.append("")
        lines.append("4. 归一化有效权重 ŵ = (w·m) / Σ(w·m)：")
        for column in COLUMN_ORDER:
            label = COLUMN_LABELS[column]
            normalized = self.normalized_weights.get(column, 0.0)
            lines.append(f"   - {label}: {normalized:.4f}")

        lines.append("")
        lines.append("5. 各列贡献 ŵ·s：")
        for column in COLUMN_ORDER:
            label = COLUMN_LABELS[column]
            contribution = self.column_contributions.get(column, 0.0)
            normalized = self.normalized_weights.get(column, 0.0)
            score = self.column_scores.get(column, 0.0)
            lines.append(
                f"   - {label}: {normalized:.4f} × {score:.4f} = {contribution:.4f}"
            )

        lines.append("")
        lines.append(
            f"6. 四列合成 S_col = Σ(ŵ·s) = {self.column_composite:.4f}"
        )
        if self.seed_weights_enabled:
            lines.append(
                f"7. 个性化种子 W_seed = {self.seed_weight:.4f}（仅路由/提示，不参与乘分）"
            )
            lines.append(
                f"8. 最终分 = S_col = {self.final_score:.4f}"
            )
        else:
            lines.append(
                f"7. 最终分 = S_col = {self.final_score:.4f}"
            )
        return "\n".join(lines)
