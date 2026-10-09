"""选题判错之后，只核对 MWM 输出和标准答案是不是同一件事。"""

from __future__ import annotations

import json
from typing import Any

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "level": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["level", "reason"],
}

_CRITERION = """Judge whether the record semantically supports the claim. Identical wording is not required.
level=explicit: the record directly expresses that same mental state or claim.
level=direct: the record does not use the same word, but a behavior, thought, or state in it is a direct and highly specific manifestation of that claim. No new mental state is added.
level=inferred: the claim needs a further psychological guess, and other clearly different mental states would also fit the same record.
Use only the record. Do not add information.
An event outcome is not automatically the person's reaction.
A bare action does not establish one feeling when guilt, politeness, responsibility, or repairing a relationship would also explain it.
High vigilance while checking a monitor directly manifests caution or vigilance.
Apologizing after realizing a mistake, together with discomfort, directly manifests embarrassment.
Apologizing alone does not.
Learning that an event was a misunderstanding does not by itself manifest relief.
If the same record can naturally fit several different mental states, level=inferred.
Return {"level":"inferred","reason":"one short sentence"}."""

SYSTEM_PROMPT = """You review one missed multiple-choice item.
The record contains subjective objects, thoughts, and actions. No story and no other options are provided.
The question names the dimension: belief, emotion, intention, or action.
Decide whether the record already supports the gold claim itself, including the cause and the mental state it names.
level=explicit: the record states that claim.
level=direct: the record states that same claim in other words, or shows one direct and highly specific manifestation of it.
level=inferred: the gold claim adds a feeling, motive, or cause the record does not state, and another mental state would also fit.
A nearby event or a different dimension in the record does not erase a claim the record does state.
It also does not create a claim the record does not state.
An apology supports apologizing, and supports reconciliation when the record says the person apologizes after realizing a misunderstanding.
An apology does not by itself support embarrassment, relief, or a desire to mend the relationship.
Identifying a specific person through the investigation, and confirming that suspicion, is the claim that evidence accumulated against that person.
Dismissing a danger is that person's reaction. It is not the claim that someone escalated because support was lacking, unless the record says support was lacking.
An apology after realizing a misunderstanding supports reconciliation. It does not by itself support embarrassment, relief, or a desire to mend the relationship.
Discovering a misunderstanding does not by itself support relief.
Frustration is supported only when the record states frustration.
If it is not explicit or direct, level=inferred.
Return {"level":"inferred","reason":"one short sentence"}."""

SUPPORT_PROMPT = (
    "You compare a mental-state record with one option.\n"
    "No story and no other options are provided.\n"
    + _CRITERION
)

_SUPPORTED_LEVELS = frozenset({"explicit", "direct"})

_ATTEMPTS = 2


def audit_match(
    llm: object,
    *,
    question: str,
    mwm_text: str,
    gold_text: str,
) -> tuple[bool, str, str]:
    """返回 (是否语义支持, 等级, 理由)。失败时保持不支持。"""
    if not mwm_text.strip() or not gold_text.strip():
        return False, "缺少 MWM 输出或标准答案，不能判为同一件事。"
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Mental-state record:\n{mwm_text.strip()}\n\n"
        f"Gold answer:\n{gold_text.strip()}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for attempt in range(_ATTEMPTS):
        try:
            raw = llm.complete(messages, schema=AUDIT_SCHEMA)
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            if attempt + 1 == _ATTEMPTS:
                return False, "inferred", f"核对失败，保持原判。{last_error}"
            continue
        supported, level, reason = _parse_support(raw)
        if reason:
            return supported, level, reason
        last_error = f"核对结果无法解析：{raw!r}"
    return False, "inferred", f"核对失败，保持原判。{last_error}"


def support_match(
    llm: object,
    *,
    question: str,
    mwm_text: str,
    option_text: str,
) -> tuple[bool, str, str]:
    """选中项是否被记录语义支持。不看标准答案，也不看其他选项。"""
    if not mwm_text.strip() or not option_text.strip():
        return False, "inferred", "缺少 MWM 输出或选中选项，不能算语义支持。"
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Mental-state record:\n{mwm_text.strip()}\n\n"
        f"Option:\n{option_text.strip()}"
    )
    messages = [
        {"role": "system", "content": SUPPORT_PROMPT},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for attempt in range(_ATTEMPTS):
        try:
            raw = llm.complete(messages, schema=AUDIT_SCHEMA)
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            if attempt + 1 == _ATTEMPTS:
                return False, "inferred", f"支持检查失败，不把字母碰中算作对。{last_error}"
            continue
        supported, level, reason = _parse_support(raw)
        if reason:
            return supported, level, reason
        last_error = f"支持检查无法解析：{raw!r}"
    return False, "inferred", f"支持检查失败，不把字母碰中算作对。{last_error}"


def finalize_judgment(row: dict, llm: object, *, placeholder: bool = False) -> dict:
    """最高分就是标准答案时直接算对。只有选错的题才核对记录是否已支持标准答案。"""
    predicted = str(row.get("predicted_answer") or "")
    gold = str(row.get("ground_truth") or "")
    letter_correct = bool(predicted) and predicted == gold
    row["letter_correct"] = letter_correct
    row["selector_correct"] = letter_correct
    if letter_correct or placeholder:
        row["adapter_correct"] = letter_correct
        row["mwm_matches_gold"] = None
        row["audit_reason"] = ""
        row["correct"] = letter_correct
        return row
    row["adapter_correct"] = False
    row["correct"] = False
    return apply_audit(row, llm, placeholder=False)


def apply_audit(row: dict, llm: object, *, placeholder: bool = False) -> dict:
    """选题已对的不核对。判错的只在 MWM 与标准是同一件事时改为对。"""
    if "adapter_correct" in row:
        adapter_correct = row.get("adapter_correct") is True
    else:
        adapter_correct = row.get("correct") is True
    row["adapter_correct"] = adapter_correct
    if adapter_correct:
        row["mwm_matches_gold"] = None
        row["audit_reason"] = ""
        row["correct"] = True
        return row
    if placeholder:
        row["mwm_matches_gold"] = False
        row["audit_reason"] = "占位选题，不把错题改成对。"
        row["correct"] = False
        return row
    same, level, reason = audit_match(
        llm,
        question=str(row.get("question") or ""),
        mwm_text=_mwm_text(row),
        gold_text=str(row.get("gold_output") or row.get("ground_truth") or ""),
    )
    row["mwm_matches_gold"] = same
    row["audit_level"] = level
    row["audit_reason"] = reason
    row["correct"] = same
    return row


def _mwm_text(row: dict) -> str:
    representation = row.get("representation")
    if isinstance(representation, dict) and representation:
        return json.dumps(representation, ensure_ascii=False)
    used = row.get("mwm_state_used")
    if isinstance(used, list) and used:
        return "\n".join(str(item) for item in used)
    return ""


def _parse_support(raw: object) -> tuple[bool, str, str]:
    """返回 (是否支持, explicit|direct|inferred, 理由)。"""
    if not isinstance(raw, dict):
        return False, "", ""
    reason = " ".join(str(raw.get("reason") or "").split())
    level = str(raw.get("level") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if level in {"directly_manifested", "manifested"}:
        level = "direct"
    if level in _SUPPORTED_LEVELS or level == "inferred":
        return level in _SUPPORTED_LEVELS, level, reason
    same = raw.get("same")
    if isinstance(same, str):
        same = same.strip().lower() == "true"
    if isinstance(same, bool):
        return same, "explicit" if same else "inferred", reason
    return False, "", ""
