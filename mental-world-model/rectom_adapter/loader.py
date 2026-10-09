"""读取 RecTOM 的一道题。一次调用只取出一个下标。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_FOCUS_RE = re.compile(r"\b(RECOMMENDER|SEEKER)\b")


@dataclass(frozen=True)
class RectomItem:
    index: int
    dialogue_id: str
    utterance_pos: int
    utterance_context: str
    question: str
    choices: tuple[str, ...]
    gold: tuple[str, ...]
    focus_person: str

    @property
    def item_id(self) -> str:
        return f"d{self.dialogue_id}_u{self.utterance_pos}"


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 应为题目列表")
    return [item for item in data if isinstance(item, dict)]


def load_one(path: Path, index: int) -> RectomItem:
    items = load_items(path)
    if index < 0 or index >= len(items):
        raise IndexError(f"下标 {index} 超出范围，这份数据共 {len(items)} 题")
    raw = items[index]
    choices = tuple(str(choice).strip() for choice in raw.get("choices") or [] if str(choice).strip())
    gold = tuple(
        letter
        for letter in (_letter(answer) for answer in raw.get("answer") or [])
        if letter
    )
    question = str(raw.get("question") or "").strip()
    focus = _focus(question)
    if not choices or not gold or not question or not focus:
        raise ValueError(f"第 {index} 题缺少问题、选项、标准答案或焦点人物")
    context = str(raw.get("utterance_context") or "").strip()
    if not context:
        raise ValueError(f"第 {index} 题没有对话")
    return RectomItem(
        index=index,
        dialogue_id=str(raw.get("dialogue_id") or ""),
        utterance_pos=int(raw.get("utterance_pos") or 0),
        utterance_context=context,
        question=question,
        choices=choices,
        gold=gold,
        focus_person=focus,
    )


def _focus(question: str) -> str:
    match = _FOCUS_RE.search(question)
    return match.group(1) if match else ""


def _letter(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[0].lower()
