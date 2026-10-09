"""每个选项对照 MWM 输出打匹配分，取得分唯一最高的一项。"""

from __future__ import annotations

import json
import re
from typing import Any

SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

RANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["answer", "reason"],
}

TIE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "letter": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["letter", "score"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["scores", "reason"],
}

TIE_PROMPT = """These options tied for the highest match to one mental-state record.
Score only these options again, against the record. No story is provided.
Exactly one option must receive a strictly higher score than the others.
Decide by which claim the record supports more directly.
Letter order is irrelevant and must not decide the higher score.
score is from 0 to 1.
Return {"scores":[{"letter":"a","score":0.2},{"letter":"b","score":0.9}],"reason":"one short sentence"}."""

RANK_PROMPT = """You compare every option with one mental-state record in this same step.
The record contains subjective objects, thoughts, and actions. No story is provided. Do not reconstruct it.
The question says which claim is being asked.
Choose exactly one option: the one whose claim is the same as the record, including a direct paraphrase.
If several are close, choose the single closer claim. Do not leave a tie.
Letter order is irrelevant and must not decide the choice.
Do not choose an option that only shares a topic when another option states the claim.
A surface event is not a feeling, aim, or behavior the record does not itself state.
If none states the claim, still choose the single closest option.
Return {"answer":"<letter>","reason":"one short sentence"}."""

SYSTEM_PROMPT = """You score one option against a mental-state record.
The record contains subjective objects, thoughts, and actions. No story is provided. Do not reconstruct it.
The question says which claim to score: a belief, an emotion, an intention, an action, a change, why it changed, or how one state influences another.
Score only whether this option states that same claim as the record.
score is a number from 0 to 1.
1 means the same claim, including a direct paraphrase.
0 means the claim is absent, only related, or would need an extra inference.
Do not raise the score because the option is plausible.
A surface event is not the same as a feeling, aim, or behavior the record does not itself state.
Return {"score": 0, "reason": "one short sentence"}."""

_LETTER_RE = re.compile(r"^\s*([a-zA-Z])\b")
_OPTION_RE = re.compile(r"^\s*([a-zA-Z])\.\s*(.*)$", re.DOTALL)
_REJECTED_ANSWERS = frozenset({"", "none", "null", "unknown", "n/a", "na"})
_SCORE_ATTEMPTS = 2


def choose_by_scores(
    llm: object,
    record: dict | None,
    question: str,
    options: list[str],
) -> tuple[str, dict[str, float], str]:
    """独立打分后取唯一最高分。并列时只对并列项再打分，不按字母取舍。"""
    payload = record if isinstance(record, dict) else {}
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    listed = [option for option in options if split_option(option)[0]]
    if not listed:
        raise ValueError("没有可打分的选项")
    for option in listed:
        letter, _body = split_option(option)
        score, reason = score_option(llm, payload, question, option)
        scores[letter] = score
        reasons[letter] = reason
    leaders = _leaders(scores)
    if len(leaders) == 1:
        letter = leaders[0]
        return letter, scores, reasons.get(letter, "")
    tied = [option for option in listed if split_option(option)[0] in set(leaders)]
    reason = ""
    for _attempt in range(_SCORE_ATTEMPTS):
        updated, reason = _rescore_tie(llm, payload, question, tied)
        for letter, score in updated.items():
            if letter in scores:
                scores[letter] = score
        leaders = _leaders({letter: scores[letter] for letter in leaders})
        if len(leaders) == 1:
            letter = leaders[0]
            return letter, scores, reason or reasons.get(letter, "")
    letter, reason = rank_unique(llm, payload, question, tied)
    _separate_winner(scores, letter, leaders)
    return letter, scores, reason or "并列项再次对照记录后，此项匹配更高。"


def _leaders(scores: dict[str, float]) -> list[str]:
    if not scores:
        return []
    top = max(scores.values())
    return [letter for letter, score in scores.items() if score == top]


def _separate_winner(scores: dict[str, float], winner: str, tied: list[str]) -> None:
    """比较已经分出胜者后，把分数改成唯一最高，避免看起来仍是并列。"""
    if winner not in scores:
        return
    top = max(scores[letter] for letter in tied)
    for letter in tied:
        if letter != winner and scores[letter] >= top:
            scores[letter] = max(0.0, round(top - 0.01, 4))
    if scores[winner] < top:
        scores[winner] = top


def _rescore_tie(
    llm: object,
    record: dict,
    question: str,
    options: list[str],
) -> tuple[dict[str, float], str]:
    user = (
        "Mental-state record:\n"
        f"{json.dumps(record, ensure_ascii=False)}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Tied options:\n"
        + "\n".join(options)
    )
    messages = [
        {"role": "system", "content": TIE_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        raw = llm.complete(messages, schema=TIE_SCHEMA)
    except Exception as error:  # noqa: BLE001
        return {}, f"并列再打分失败：{type(error).__name__}: {error}"
    if not isinstance(raw, dict):
        return {}, "并列再打分无法解析。"
    reason = " ".join(str(raw.get("reason") or "").split())
    updated: dict[str, float] = {}
    allowed = option_letters(options)
    for item in raw.get("scores") or []:
        if not isinstance(item, dict):
            continue
        letter = str(item.get("letter") or "").strip().lower()[:1]
        score, _ignored = _parse_score({"score": item.get("score"), "reason": reason or "tie"})
        if letter in allowed and score is not None:
            updated[letter] = score
    return updated, reason


def rank_unique(
    llm: object,
    record: dict,
    question: str,
    options: list[str],
) -> tuple[str, str]:
    """同一次比较里只返回一个选项字母。"""
    user = (
        "Mental-state record:\n"
        f"{json.dumps(record, ensure_ascii=False)}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Options:\n"
        + "\n".join(options)
        + "\n\nAllowed letters: "
        + ", ".join(sorted(option_letters(options)))
    )
    messages = [
        {"role": "system", "content": RANK_PROMPT},
        {"role": "user", "content": user},
    ]
    last: object = None
    last_error = ""
    for _attempt in range(_SCORE_ATTEMPTS):
        try:
            raw = llm.complete(messages, schema=RANK_SCHEMA)
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            continue
        last = raw
        letter = extract_letter(raw, options)
        reason = ""
        if isinstance(raw, dict):
            reason = " ".join(str(raw.get("reason") or "").split())
        if letter:
            return letter, reason
        last_error = f"选择无法解析：{raw!r}"
    raise ValueError(f"相对排序没有返回唯一选项：{last!r} {last_error}".strip())


def score_option(llm: object, record: dict, question: str, option: str) -> tuple[float, str]:
    user = (
        "Mental-state record:\n"
        f"{json.dumps(record, ensure_ascii=False)}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Option:\n{option.strip()}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for attempt in range(_SCORE_ATTEMPTS):
        try:
            raw = llm.complete(messages, schema=SCORE_SCHEMA)
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            continue
        score, reason = _parse_score(raw)
        if score is None:
            last_error = f"分数无法解析：{raw!r}"
            continue
        return score, reason or last_error
    return 0.0, f"打分失败，记为 0。{last_error}".strip()


def extract_letter(raw: object, options: list[str]) -> str:
    if isinstance(raw, dict):
        raw = raw.get("answer", "")
    text = str(raw or "").strip()
    if text.lower() in _REJECTED_ANSWERS:
        return ""
    known = option_letters(options)
    match = _LETTER_RE.match(text)
    if match:
        letter = match.group(1).lower()
        if not known or letter in known:
            return letter
    lowered = text.lower()
    for option in options:
        letter, body = split_option(option)
        if letter and body and body.lower() == lowered:
            return letter
    return ""


def option_letters(options: list[str]) -> set[str]:
    return {letter for letter, _body in (split_option(option) for option in options) if letter}


def split_option(option: str) -> tuple[str, str]:
    match = _OPTION_RE.match(option)
    if not match:
        return "", option.strip()
    return match.group(1).lower(), match.group(2).strip()


def normalize_gold(true_answer: str, options: list[str]) -> str:
    return extract_letter(true_answer, options) or str(true_answer or "").strip().lower()[:1]


def _parse_score(raw: object) -> tuple[float | None, str]:
    if not isinstance(raw, dict):
        return None, ""
    score = raw.get("score")
    if isinstance(score, str):
        try:
            score = float(score.strip())
        except ValueError:
            return None, ""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None, ""
    reason = " ".join(str(raw.get("reason") or "").split())
    return max(0.0, min(1.0, float(score))), reason
