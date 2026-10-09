"""读取 SimpleToM 行为预测的一道题。一次调用只取出一个下标。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_FOCUS_NEXT_ACTION = re.compile(
    r"\bwill (.+?)(?:'s)? next action likely be\b",
    re.IGNORECASE,
)
_FOCUS_LIKELY = re.compile(r"\bwill (.+?) likely\b", re.IGNORECASE)
_FOCUS_DO = re.compile(r"\bwill (.+?) do\b", re.IGNORECASE)


@dataclass(frozen=True)
class SimpleTomItem:
    index: int
    item_id: str
    story: str
    question: str
    choices: tuple[str, ...]
    gold: tuple[str, ...]
    focus_person: str
    scenario_name: str


def load_one(path: Path, index: int) -> SimpleTomItem:
    items = _load_rows(path)
    if index < 0 or index >= len(items):
        raise IndexError(f"下标 {index} 超出范围，这份数据共 {len(items)} 题")
    raw = items[index]
    question = str(raw.get("question") or "").strip()
    focus = focus_person(question)
    choices = _choices(raw.get("choices"))
    gold = _gold(raw.get("answerKey"))
    story = str(raw.get("story") or "").strip()
    item_id = str(raw.get("id") or "").strip()
    if not question or not focus or not choices or not gold or not story or not item_id:
        raise ValueError(f"第 {index} 题缺少故事、问题、焦点人物、选项或标准答案")
    if gold[0] not in {letter for letter, _body in (_split(choice) for choice in choices)}:
        raise ValueError(f"第 {index} 题的标准答案不在选项里")
    return SimpleTomItem(
        index=index,
        item_id=item_id,
        story=story,
        question=question,
        choices=choices,
        gold=gold,
        focus_person=focus,
        scenario_name=str(raw.get("scenario_name") or "").strip(),
    )


def focus_person(question: str) -> str:
    for pattern in (_FOCUS_NEXT_ACTION, _FOCUS_LIKELY, _FOCUS_DO):
        match = pattern.search(question)
        if match:
            name = " ".join(match.group(1).split()).strip(" .")
            if name:
                return name
    return ""


def _load_rows(path: Path) -> list[dict]:
    text = path.expanduser().resolve().read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError(f"{path} 应为题目列表")
        return [item for item in data if isinstance(item, dict)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _choices(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, dict):
        return ()
    labels = raw.get("label") or []
    texts = raw.get("text") or []
    choices = []
    for label, body in zip(labels, texts):
        letter = str(label or "").strip()[:1].upper()
        content = " ".join(str(body or "").split())
        if letter and content:
            choices.append(f"{letter}: {content}")
    return tuple(choices)


def _gold(value: object) -> tuple[str, ...]:
    letter = str(value or "").strip()[:1].lower()
    return (letter,) if letter else ()


def _split(option: str) -> tuple[str, str]:
    if ":" not in option:
        return "", option.strip()
    letter, body = option.split(":", 1)
    return letter.strip().lower()[:1], body.strip()
