"""选题不在标准答案里时，核对记录是否已经支持其中一条标准策略。"""

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

SYSTEM_PROMPT = """You review one missed next-strategy item.
The record contains subjective objects, thoughts, the observed action, and predicted next actions.
No dialogue and no other options are provided.
The question asks what strategy this person will use next.
Support a gold strategy only from the predicted next actions, or from a thought about what to do next.
The observed action is what already happened. It is not the next strategy.
level=explicit: a predicted next action states that strategy.
level=direct: a predicted next action states it in other words.
level=inferred: only the observed past action, or a further guess, would connect the record to the gold strategy.
A mentioned movie, genre, or past remark does not by itself become a strategy the record does not predict next.
If it is not explicit or direct, level=inferred.
Return {"level":"inferred","reason":"one short sentence"}."""

_SUPPORTED = frozenset({"explicit", "direct"})
_ATTEMPTS = 2


def review_miss(llm: object, *, question: str, record: dict, gold_text: str) -> tuple[bool, str, str]:
    if not gold_text.strip():
        return False, "inferred", "没有标准策略，保持错。"
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Mental-state record:\n{json.dumps(record, ensure_ascii=False)}\n\n"
        f"Accepted strategies:\n{gold_text.strip()}"
    )
    last_error = ""
    for _attempt in range(_ATTEMPTS):
        try:
            raw = llm.complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                schema=AUDIT_SCHEMA,
            )
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            continue
        if not isinstance(raw, dict):
            last_error = f"核对无法解析：{raw!r}"
            continue
        level = str(raw.get("level") or "").strip().lower()
        reason = " ".join(str(raw.get("reason") or "").split())
        if level in {"explicit", "direct", "inferred"} and reason:
            return level in _SUPPORTED, level, reason
        last_error = f"核对无法解析：{raw!r}"
    return False, "inferred", f"核对失败，保持错。{last_error}".strip()
