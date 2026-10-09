"""把下一步行动归进各个策略选项，概率相加后取得分唯一最高的一项。"""

from __future__ import annotations

import json
import re
from typing import Any

ASSIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "letter": {"type": "string"},
                    "indexes": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["letter", "indexes"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["groups", "reason"],
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

ASSIGN_PROMPT = """You group predicted next actions under next-strategy options.
Look at each option separately.
An action belongs to an option only when that action states the option's strategy, including a direct paraphrase.
An action that only shares a movie title, genre, or topic does not belong.
An action that would need a further guess does not belong.
Each action index belongs to at most one option. Some actions belong to no option.
Use only the predicted next actions listed below.
Return {"groups":[{"letter":"a","indexes":[0]}],"reason":"one short sentence"}.
Include one group for every option. Use an empty indexes list when none belong."""

TIE_PROMPT = """These next-strategy options tied for the highest match to one record.
Score only these options again. No dialogue is provided.
Exactly one option must receive a strictly higher score than the others.
Decide by which next strategy the record supports more directly.
Letter order is irrelevant and must not decide the higher score.
score is from 0 to 1.
Return {"scores":[{"letter":"a","score":0.2},{"letter":"c","score":0.9}],"reason":"one short sentence"}."""

RANK_PROMPT = """Choose exactly one next strategy from the tied options.
The record contains subjective objects, thoughts, and actions. No dialogue is provided.
Letter order is irrelevant.
If none is stated, still choose the single closest option.
Return {"answer":"<letter>","reason":"one short sentence"}."""

_LETTER_RE = re.compile(r"^\s*([a-zA-Z])\b")
_OPTION_RE = re.compile(r"^\s*([a-zA-Z])[:.]\s*(.*)$", re.DOTALL)
_SCORE_ATTEMPTS = 2


def choose_by_scores(
    llm: object,
    record: dict | None,
    question: str,
    options: list[str],
) -> tuple[str, dict[str, float], str]:
    payload = record if isinstance(record, dict) else {}
    listed = [option for option in options if split_option(option)[0]]
    if not listed:
        raise ValueError("没有可打分的选项")
    letters = [split_option(option)[0] for option in listed]
    actions = _predicted_actions(payload)
    groups, note = _assign_actions(llm, question, actions, listed)
    scores = _sum_groups(actions, groups, letters)
    assigned_reason = _format_reason(actions, groups, scores, note)
    leaders = _leaders(scores)
    if len(leaders) == 1:
        letter = leaders[0]
        return letter, scores, assigned_reason
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
            return letter, scores, reason or assigned_reason
    letter, reason = rank_unique(llm, payload, question, tied)
    _separate_winner(scores, letter, leaders)
    return letter, scores, reason or "并列项再次对照记录后，此项匹配更高。"


def extract_letter(raw: object, options: list[str]) -> str:
    if isinstance(raw, dict):
        raw = raw.get("answer", "")
    text = str(raw or "").strip()
    known = {letter for letter, _body in (split_option(option) for option in options) if letter}
    match = _LETTER_RE.match(text)
    if match:
        letter = match.group(1).lower()
        if not known or letter in known:
            return letter
    return ""


def split_option(option: str) -> tuple[str, str]:
    match = _OPTION_RE.match(option)
    if not match:
        return "", option.strip()
    return match.group(1).lower(), match.group(2).strip()


def option_text(letter: str, options: list[str]) -> str:
    wanted = letter.lower()
    for option in options:
        found, _body = split_option(option)
        if found == wanted:
            return option.strip()
    return wanted


def _predicted_actions(record: dict) -> list[tuple[str, float]]:
    actions: list[tuple[str, float]] = []
    for item in record.get("predicted_actions") or []:
        if isinstance(item, str):
            text, probability = item, 0.0
        elif isinstance(item, dict):
            text = str(item.get("content") or item.get("text") or "")
            probability = item.get("probability", 0.0)
        else:
            continue
        text = " ".join(text.split())
        if not text:
            continue
        try:
            probability = float(probability or 0.0)
        except (TypeError, ValueError):
            probability = 0.0
        actions.append((text, max(0.0, probability)))
    return actions


def _assign_actions(
    llm: object,
    question: str,
    actions: list[tuple[str, float]],
    options: list[str],
) -> tuple[dict[str, list[int]], str]:
    letters = [split_option(option)[0] for option in options]
    empty = {letter: [] for letter in letters}
    if not actions:
        return empty, "没有下一步行动。"
    numbered = "\n".join(
        f"{index}. {text} | probability={probability:.4f}"
        for index, (text, probability) in enumerate(actions)
    )
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Predicted next actions:\n{numbered}\n\n"
        "Options:\n"
        + "\n".join(options)
    )
    last_error = ""
    for _attempt in range(_SCORE_ATTEMPTS):
        try:
            raw = llm.complete(
                [
                    {"role": "system", "content": ASSIGN_PROMPT},
                    {"role": "user", "content": user},
                ],
                schema=ASSIGN_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            continue
        groups, reason = _parse_groups(raw, letters, len(actions))
        if groups is not None:
            return groups, reason
        last_error = f"归类无法解析：{raw!r}"
    return empty, f"归类失败，各选项记为 0。{last_error}".strip()


def _parse_groups(
    raw: object,
    letters: list[str],
    action_count: int,
) -> tuple[dict[str, list[int]] | None, str]:
    if not isinstance(raw, dict):
        return None, ""
    allowed = set(letters)
    claimed: dict[str, list[int]] = {letter: [] for letter in letters}
    seen: dict[int, str] = {}
    conflicts: set[int] = set()
    found = False
    for item in raw.get("groups") or []:
        if not isinstance(item, dict):
            continue
        letter = str(item.get("letter") or "").strip().lower()[:1]
        if letter not in allowed:
            continue
        found = True
        for index in item.get("indexes") or []:
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            if index < 0 or index >= action_count:
                continue
            previous = seen.get(index)
            if previous is None:
                seen[index] = letter
                claimed[letter].append(index)
            elif previous != letter:
                conflicts.add(index)
    if not found:
        return None, ""
    for index in conflicts:
        owner = seen.get(index)
        if owner is not None:
            claimed[owner] = [item for item in claimed[owner] if item != index]
    reason = " ".join(str(raw.get("reason") or "").split())
    return claimed, reason


def _sum_groups(
    actions: list[tuple[str, float]],
    groups: dict[str, list[int]],
    letters: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for letter in letters:
        total = sum(actions[index][1] for index in groups.get(letter) or [])
        scores[letter] = round(total, 4)
    return scores


def _format_reason(
    actions: list[tuple[str, float]],
    groups: dict[str, list[int]],
    scores: dict[str, float],
    note: str,
) -> str:
    parts: list[str] = []
    for letter, score in scores.items():
        indexes = groups.get(letter) or []
        if not indexes:
            parts.append(f"{letter.upper()} {score:.4f}")
            continue
        bits = [f"{actions[index][0]} {actions[index][1]:.4f}" for index in indexes]
        parts.append(f"{letter.upper()} {score:.4f} = " + " + ".join(bits))
    text = "；".join(parts)
    if note:
        text = f"{text}。{note}"
    return text


def _leaders(scores: dict[str, float]) -> list[str]:
    if not scores:
        return []
    top = max(scores.values())
    return [letter for letter, score in scores.items() if score == top]


def _separate_winner(scores: dict[str, float], winner: str, tied: list[str]) -> None:
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
    try:
        raw = llm.complete(
            [
                {"role": "system", "content": TIE_PROMPT},
                {"role": "user", "content": user},
            ],
            schema=TIE_SCHEMA,
        )
    except Exception as error:  # noqa: BLE001
        return {}, f"并列再打分失败：{type(error).__name__}: {error}"
    if not isinstance(raw, dict):
        return {}, "并列再打分无法解析。"
    reason = " ".join(str(raw.get("reason") or "").split())
    allowed = {letter for letter, _body in (split_option(option) for option in options) if letter}
    updated: dict[str, float] = {}
    for item in raw.get("scores") or []:
        if not isinstance(item, dict):
            continue
        letter = str(item.get("letter") or "").strip().lower()[:1]
        score, _ignored = _parse_score({"score": item.get("score"), "reason": reason or "tie"})
        if letter in allowed and score is not None:
            updated[letter] = score
    return updated, reason


def rank_unique(llm: object, record: dict, question: str, options: list[str]) -> tuple[str, str]:
    allowed = sorted({letter for letter, _body in (split_option(option) for option in options) if letter})
    user = (
        "Mental-state record:\n"
        f"{json.dumps(record, ensure_ascii=False)}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Options:\n"
        + "\n".join(options)
        + "\n\nAllowed letters: "
        + ", ".join(allowed)
    )
    last_error = ""
    for _attempt in range(_SCORE_ATTEMPTS):
        try:
            raw = llm.complete(
                [
                    {"role": "system", "content": RANK_PROMPT},
                    {"role": "user", "content": user},
                ],
                schema=RANK_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            continue
        letter = extract_letter(raw, options)
        reason = " ".join(str(raw.get("reason") or "").split()) if isinstance(raw, dict) else ""
        if letter:
            return letter, reason
        last_error = f"选择无法解析：{raw!r}"
    raise ValueError(f"并列项没有分出唯一选项：{last_error}".strip())


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
