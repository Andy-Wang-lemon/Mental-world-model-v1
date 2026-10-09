"""原文所缘境/想/思提取 prompt。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.extraction.models import SuoyuanXiangSiDefinitions
from mental_world_model.engine.seeds.config import HINT_FIELD_MAX_CHARS, is_seed_library_enabled


def _truncate(text: str, limit: int = HINT_FIELD_MAX_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}…"


def build_suoyuan_messages(
    yuanwen: str,
    definitions: SuoyuanXiangSiDefinitions,
) -> list[dict[str, str]]:
    """步骤 1：从原文语义提取所缘境（含完整定义）。"""
    content = (
        "请根据下列定义，从原文中判断并提取「所缘境」。\n\n"
        f"【原文】\n{yuanwen}\n\n"
        f"【所缘境定义】\n{definitions.suoyuan_jing}\n\n"
        f"【抽取说明】\n{definitions.summary}\n\n"
        "所缘境指：本次心理状态面对、指向、处理的对象或情境。"
        "请用一句话概括原文中的所缘境。\n\n"
        '返回 JSON：{"suoyuan": {"text": "..."}}'
    )
    return [{"role": "user", "content": content}]


def build_router_single_xinsuo_messages(
    yuanwen: str,
    suoyuan_text: str,
    xinsuo: Xinsuo,
) -> list[dict[str, str]]:
    """步骤 2（种子库启用）：单心所所缘境与本次所缘的语义相近度。"""
    user_content = (
        f"心所名称：{xinsuo.name}\n\n"
        f"该心所的所缘境定义：\n{xinsuo.suoyuan}\n\n"
        "请用语义理解，评估该心所「所缘境定义」与本次所缘境的相近程度。\n"
        "返回 JSON：{\"relevance\": 0.0}（0.0~1.0，1.0 表示高度一致，0.0 表示完全无关）"
    )
    return [
        {
            "role": "system",
            "content": (
                f"【原文】\n{yuanwen}\n\n"
                f"【本次所缘境】\n{suoyuan_text}"
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _format_hint_xinsuo_block(
    hint_xinsuo: list[Xinsuo],
    seed_weights: dict[str, float],
) -> str:
    lines: list[str] = []
    for xinsuo in hint_xinsuo:
        weight = seed_weights.get(xinsuo.name, 0.0)
        lines.append(f"- {xinsuo.name}（种子权重 {weight:.2f}）")
        lines.append(f"  生起念头：{_truncate(xinsuo.shengqi_nian)}")
        lines.append(f"  业：{_truncate(xinsuo.ye)}")
    return "\n".join(lines)


def build_xiang_evidence_messages(
    yuanwen: str,
    suoyuan_text: str,
    definitions: SuoyuanXiangSiDefinitions,
) -> list[dict[str, str]]:
    """判定原文是否有明确的想；有则只摘原文，无则走隐式百法（不推理想/思）。"""
    content = (
        "请根据下列定义，判断原文中分析对象是否有「明确写出的想」（内心念头/态度）。\n\n"
        f"【原文】\n{yuanwen}\n\n"
        f"【已提取的所缘境】\n{suoyuan_text}\n\n"
        f"【想定义】\n{definitions.xiang}\n\n"
        f"【思定义】\n{definitions.si}\n\n"
        f"【抽取说明】\n{definitions.summary}\n\n"
        "规则：\n"
        "1. has_explicit_xiang=true：仅当原文有可直接依据的内心语言、态度或心理描写"
        "（非仅靠场面/行为猜测）。此时 xiang.text 必须能在原文找到依据，source 固定为「原文」。\n"
        "2. has_explicit_xiang=false：原文没有明确的想。不要推理编造想/思；"
        "xiang/si 可留空字符串。\n"
        "3. has_explicit_si 同理：仅当原文明确写出行为意向/意志方向时为 true。\n\n"
        "返回 JSON：\n"
        '{"has_explicit_xiang": true|false, '
        '"xiang": {"text": "...", "source": "原文", "confidence": 1.0}, '
        '"has_explicit_si": true|false, '
        '"si": {"text": "...", "source": "原文", "confidence": 1.0}}\n'
        "当 has_explicit_*=false 时，对应 text 请返回空字符串。"
    )
    return [{"role": "user", "content": content}]


def build_xiang_si_messages(
    yuanwen: str,
    suoyuan_text: str,
    definitions: SuoyuanXiangSiDefinitions,
    hint_xinsuo: list[Xinsuo],
    seed_weights: dict[str, float],
) -> list[dict[str, str]]:
    """步骤 3：提取想与思；种子库启用时附带 H 心所参考。"""
    lines = [
        "请根据下列定义，从原文中提取「想」与「思」。",
        "",
        f"【原文】\n{yuanwen}",
        "",
        f"【已提取的所缘境】\n{suoyuan_text}",
        "",
        f"【想定义】\n{definitions.xiang}",
        "",
        f"【思定义】\n{definitions.si}",
        "",
        f"【抽取说明】\n{definitions.summary}",
    ]
    if is_seed_library_enabled() and hint_xinsuo:
        hints = _format_hint_xinsuo_block(hint_xinsuo, seed_weights)
        lines.extend(["", f"【与本次所缘境相关的心所参考】\n{hints}"])
    lines.extend(
        [
            "",
            "请分别写出想（内心念头/态度）与思（行为倾向/心理功能）各一句，并给出来源与置信度：",
            "- source=原文：文本能在原文中找到直接依据，confidence=1.0",
            "- source=推理：由所缘/上下文推断，confidence 按依据强弱给 0~1",
            "- 与所缘完全重复、照抄定义而无原文支撑时，应标为推理且 confidence 偏低",
            "",
            '返回 JSON：{"xiang": {"text": "...", "source": "原文|推理", "confidence": 0.0}, '
            '"si": {"text": "...", "source": "原文|推理", "confidence": 0.0}}',
        ]
    )
    return [{"role": "user", "content": "\n".join(lines)}]
