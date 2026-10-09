"""分析对象（focus_person）契约：一次评估只建模一个人的内心。"""

from __future__ import annotations

import re

# 原文前言常见写法；CLI/API 显式传入优先于解析结果
_FOCUS_PATTERNS = (
    re.compile(
        r"(?m)^\s*(?:分析对象|建模对象|心智对象|评估对象|主角|主人公)\s*[：:]\s*(.+?)\s*$"
    ),
    re.compile(
        r"(?m)^\s*(?:本次|这次)?(?:分析|建模|评估)(?:的)?(?:对象|人物)\s*(?:是|为)\s*[：:]?\s*(.+?)\s*$"
    ),
)


def parse_focus_person_from_text(yuanwen: str) -> str:
    """从原文解析「分析对象：某某」；未标明则返回空串。"""
    if not yuanwen or not str(yuanwen).strip():
        return ""
    text = str(yuanwen)
    for pattern in _FOCUS_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = str(match.group(1) or "").strip()
        # 去掉句末标点与多余说明
        name = re.split(r"[，,。；;（(]", name, maxsplit=1)[0].strip()
        if name:
            return name
    return ""


def resolve_focus_person(
    explicit: str | None,
    yuanwen: str = "",
) -> str:
    """CLI/API 显式值优先；否则从原文标记解析。"""
    name = str(explicit or "").strip()
    if name:
        return name
    return parse_focus_person_from_text(yuanwen)


def focus_person_instruction(focus_person: str) -> str:
    """写入 system 的硬约束（空则无约束块）。"""
    name = str(focus_person or "").strip()
    if not name:
        return ""
    return (
        f"【分析对象锁定】本次唯一建模对象是「{name}」。\n"
        f"我执 / 法执 / 主观所缘 / 想 / 思 / 行为预测，主语必须是「{name}」的内心。\n"
        f"他人（对话中的其他人）只作为刺激侧事实，禁止改写为他人的第一人称内心世界。\n"
        f"文中「主体」一律指「{name}」。\n"
    )


def focus_person_user_line(focus_person: str) -> str:
    """写入 user 的显式字段行。"""
    name = str(focus_person or "").strip()
    if not name:
        return ""
    return f"分析对象（必须遵守）：{name}\n\n"
