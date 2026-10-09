"""LLM baseline：故事 + 题目 + 选项，直接选字母。不经过 MWM。"""

from __future__ import annotations

from dyntom_adapter.loader import QuestionItem, TrialStory, render_story
from dyntom_adapter.selector import ANSWER_SCHEMA, extract_letter

SYSTEM_PROMPT = """You answer a multiple-choice theory-of-mind question.
Use only the story, the question, and the options below.
Do not assume hidden annotations.
Return only {"answer":"<letter>"}."""


def answer_baseline(llm: object, story: TrialStory, question: QuestionItem) -> str:
    user = (
        "Story:\n"
        f"{render_story(story)}\n\n"
        f"Question:\n{question.question}\n\n"
        "Options:\n"
        + "\n".join(question.options)
    )
    raw = llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        schema=ANSWER_SCHEMA,
    )
    letter = extract_letter(raw, question.options)
    if not letter:
        raise ValueError(f"baseline 没有返回有效选项字母：{raw!r}")
    return letter
