"""心所名称别名映射，统一 raw 文档与 xinsuo.json 中的名称差异。"""

import re

# 别名 → 心所定义中的规范名称
XINSUO_NAME_ALIASES: dict[str, str] = {
    "痴": "无明",
    "无嗔": "无瞋",
}

_PAREN_ALIAS_PATTERN = re.compile(r"^(.+?)[（(](.+?)[）)]$")


def normalize_xinsuo_name(name: str) -> str:
    """将心所名称规范化为 xinsuo.json 中的标准名称。"""
    cleaned = name.strip()
    if not cleaned:
        return cleaned

    if cleaned in XINSUO_NAME_ALIASES:
        return XINSUO_NAME_ALIASES[cleaned]

    match = _PAREN_ALIAS_PATTERN.match(cleaned)
    if match:
        base = match.group(1).strip()
        inner = match.group(2).strip()
        canonical_inner = XINSUO_NAME_ALIASES.get(inner, inner)
        if canonical_inner == base or base == XINSUO_NAME_ALIASES.get(inner):
            return base
        if inner in XINSUO_NAME_ALIASES:
            return XINSUO_NAME_ALIASES[inner]

    return cleaned


def normalize_relation_type(relation_type: str) -> str:
    """障碍与对治语义相同（完全互斥），统一规范为「对治」。"""
    if relation_type == "障碍":
        return "对治"
    return relation_type


def format_relation_description(
    relation_type: str,
    target: str,
    is_direct: bool,
) -> str:
    """根据规范化字段重建关系描述文本。"""
    canonical_type = normalize_relation_type(relation_type)
    prefix = "直接" if is_direct else ""
    return f"{prefix}{canonical_type}{target}。"
