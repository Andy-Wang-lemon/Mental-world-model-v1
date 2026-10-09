"""只认最后一行 ANSWER: <letter>。不按选项原文猜测。"""

from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"(?im)^\s*ANSWER:\s*([A-Za-z])\b")
_OPTION_RE = re.compile(r"^\s*([A-Za-z])\s*[:.]")


def choice_letters(choices: list[str] | tuple[str, ...]) -> set[str]:
    letters: set[str] = set()
    for choice in choices:
        match = _OPTION_RE.match(str(choice))
        if match:
            letters.add(match.group(1).lower())
    return letters


def parse_answer(text: str, choices: list[str] | tuple[str, ...]) -> str:
    matches = _ANSWER_RE.findall(str(text or ""))
    if not matches:
        return ""
    letter = matches[-1].lower()
    allowed = choice_letters(choices)
    if allowed and letter not in allowed:
        return ""
    return letter


def is_correct(prediction: str, gold: tuple[str, ...]) -> bool:
    return bool(prediction) and prediction in set(gold)
