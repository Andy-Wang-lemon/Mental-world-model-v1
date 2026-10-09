"""LLM 直接看对话和选项，选一个策略字母。不经过 MWM。"""

from __future__ import annotations

from rectom_adapter.loader import RectomItem
from rectom_adapter.selector import ANSWER_SCHEMA, extract_letter

SYSTEM_PROMPT = """You answer one multiple-choice question about the next conversational strategy.
Use only the dialogue, the question, and the options.
Return only {"answer":"<letter>"}."""


def answer_baseline(llm: object, item: RectomItem) -> str:
    user = (
        "Dialogue:\n"
        f"{item.utterance_context}\n\n"
        f"Question:\n{item.question}\n\n"
        "Options:\n"
        + "\n".join(item.choices)
    )
    raw = llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        schema=ANSWER_SCHEMA,
    )
    letter = extract_letter(raw, list(item.choices))
    if not letter:
        raise ValueError(f"baseline 没有返回有效选项字母：{raw!r}")
    return letter
