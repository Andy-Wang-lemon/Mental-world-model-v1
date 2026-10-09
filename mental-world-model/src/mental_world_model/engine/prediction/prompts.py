"""行为预测层 prompt 组装。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.scoring.context import ScoringContext

STAGE_BEHAVIOR_MARKER = "【行为预测】"

_BEHAVIOR_SYSTEM = (
    "你是心理行为预测助手。根据原文语境与当前心所（心智）状态，"
    "综合判断此人最可能做出的行为，并说明如何根据这些心所得出该判断。"
    "只依据给定信息，不要编造与原文无关的情节。"
)

BEHAVIOR_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_behavior": {"type": "string"},
        "prediction_reason": {"type": "string"},
    },
    "required": ["predicted_behavior", "prediction_reason"],
}


def is_behavior_stage_content(content: str) -> bool:
    """判断 LLM 消息是否为行为预测阶段。"""
    return STAGE_BEHAVIOR_MARKER in content


def _prediction_context_block(context: ScoringContext) -> str:
    """原文 + 所缘 + 想（不含思）。"""
    extracted = context.extracted
    return (
        f"原文：\n{context.yuanwen}\n\n"
        f"{context.field_block(extracted.suoyuan, '提取的所缘境')}\n\n"
        f"{context.field_block(extracted.xiang, '提取的想')}\n"
    )


def _format_active_xinsuo_block(
    active_items: list[tuple[str, float, Xinsuo]],
) -> str:
    lines: list[str] = []
    for idx, (name, score, xinsuo) in enumerate(active_items, start=1):
        lines.append(
            f"{idx}. {name}（分数：{score:.4f}）\n"
            f"   业：{xinsuo.ye}\n"
            f"   现代心理学解释：{xinsuo.xiandai_xinli}"
        )
    return "\n".join(lines)


def build_behavior_messages(
    context: ScoringContext,
    active_items: list[tuple[str, float, Xinsuo]],
) -> list[dict[str, str]]:
    """组装行为预测 LLM 消息。

    ``active_items`` 为 ``(name, score, xinsuo)``，建议已按分数降序。
    """
    user_content = (
        f"{STAGE_BEHAVIOR_MARKER}\n\n"
        "当前有值心所/心智（按分数从高到低；分数表示该心所强度）：\n"
        f"{_format_active_xinsuo_block(active_items)}\n\n"
        "任务：\n"
        "1. predicted_behavior：综合以上所有有值心所（不要只看最高分），"
        "结合各自强度、业与现代心理学解释，并落实到原文具体语境，"
        "预测此人在这种心理状态下最可能做出的行为（一段连贯中文）。\n"
        "2. prediction_reason：说明你是如何根据这些有值心所及其分数强弱，"
        "共同判断出上述行为的；必须点名所依据的心所名称，并结合其业/心理含义说明推理过程，"
        "不要空泛套话。有值心所清单与分数字段由系统另行输出，此处专注写判断逻辑。\n"
        '只返回 JSON：{"predicted_behavior": "...", "prediction_reason": "..."}'
    )
    return [
        {
            "role": "system",
            "content": f"{_BEHAVIOR_SYSTEM}\n\n{_prediction_context_block(context)}",
        },
        {"role": "user", "content": user_content},
    ]
