"""预测行为选取与合法性（空预测 vs 明确无反应）。"""

from __future__ import annotations

INVALID_PREDICTED_ACTIONS = frozenset(
    {
        "（无行为预测）",
        "（空行为）",
        "（无行为）",
    }
)


def is_usable_predicted_action(text: str) -> bool:
    """明确无反应/没表现算合法行动；系统没给出预测才否。"""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return stripped not in INVALID_PREDICTED_ACTIONS


def top_aggregated_action(result) -> tuple[str, float, list[dict]]:
    """跨支融合 Top-1；空则兜底分支第一条。"""
    items = list(getattr(result, "aggregated_behaviors", None) or [])
    if not items:
        for branch in getattr(result, "cognition_branches", None) or []:
            preds = list(getattr(branch, "behavior_predictions", None) or [])
            if preds:
                p0 = preds[0]
                text = str(getattr(p0, "text", "") or "").strip()
                prob = float(getattr(p0, "conditional_probability", 0.0) or 0.0)
                return text or "（空行为）", prob, [{"text": text, "probability": prob}]
        return "（无行为预测）", 0.0, []

    ranked = sorted(
        items,
        key=lambda x: float(getattr(x, "total_probability", 0.0) or 0.0),
        reverse=True,
    )
    top_k = [
        {
            "text": str(getattr(item, "canonical_text", "") or "").strip(),
            "probability": float(getattr(item, "total_probability", 0.0) or 0.0),
        }
        for item in ranked
    ]
    best = top_k[0]
    return best["text"] or "（空行为）", best["probability"], top_k
