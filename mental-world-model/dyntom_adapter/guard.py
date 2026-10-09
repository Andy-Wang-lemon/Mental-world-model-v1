"""建模输入与冻结状态的泄漏检查。"""

from __future__ import annotations

BANNED_SNIPPETS = (
    "mental states analysis in every scenario",
    "analysis of mental states across scenarios",
    "option reasons",
    "true answer",
    "true_answer",
)

# 这些键属于故事原文或标注，不得进入冻结快照。
STORY_KEYS = frozenset(
    {
        "yuanwen",
        "stimulus_text",
        "stimulus_action_split",
        "objective_event",
        "sketch",
        "action_span",
        "background",
        "dialogue",
        "options",
        "option reasons",
        "true answer",
        "true_answer",
        "explain",
        "short",
        "reason",
    }
)


class LeakError(ValueError):
    """建模可见文本里出现了标注或标准答案。"""


def assert_model_visible(text: str, *, where: str) -> None:
    lowered = text.lower()
    for snippet in BANNED_SNIPPETS:
        if snippet in lowered:
            raise LeakError(f"{where} 含有禁止字段「{snippet}」")


def assert_no_story_keys(payload: dict, *, where: str) -> None:
    found = sorted(STORY_KEYS.intersection(payload.keys()))
    if found:
        raise LeakError(f"{where} 含有故事或标注字段：{found}")
