"""PICTURE：完整故事始终可见，显式区分人物知道和不知道的事件。"""

from __future__ import annotations

from inference_baselines.items import BenchmarkItem
from inference_baselines.parse_answer import is_correct, parse_answer
from inference_baselines.prompts import format_choices

_DYNTOM = """The following is a sequence of events involving several characters.

{context}

The main question to be solved is:
{question}

Choices:
{choices}

Before answering, explicitly reason about the target character's knowledge state.

What events does and does not {target_character} know about?

Keep events unknown to the character visible in the original story, but explicitly distinguish them from events the character knows. Do not attribute an event, fact, or update to the character merely because it appears in the story. Do not delete or hide events the character does not know.

If the question asks what one character thinks another character believes, identify that nested perspective. Explicitly distinguish objective events, what the outer character knows and does not know, and what the outer character thinks the inner character knows and does not know. Do not follow only the last name in the question. Reason in natural language. Do not use a symbolic belief solver.

Then solve the main question step by step from the relevant character perspective.

At the end output exactly:
ANSWER: <letter>"""

_RECTOM = """The following is a dialogue involving multiple participants.

{context}

The main question to be solved is:
{question}

Choices:
{choices}

Before answering, explicitly reason about the relevant character's knowledge state.

What utterances, facts, preferences, intentions, and interaction events does and does not {target_character} know about?

Do not hide unknown information from the model. Instead, explicitly mark that it is unknown to the target character and do not use it as if the character knew it.

Then solve the main question step by step from the required perspective. Choose one of the listed choices.

At the end output exactly:
ANSWER: <letter>"""


def picture_prompt(item: BenchmarkItem) -> str:
    template = _DYNTOM if item.dataset == "dyntom" else _RECTOM
    return _fill(
        template,
        context=item.context,
        question=item.question,
        choices=format_choices(item.choices),
        target_character=item.target_character,
    )


def run_picture(client: object, item: BenchmarkItem) -> dict:
    prompt = picture_prompt(item)
    raw = client.complete_text([{"role": "user", "content": prompt}])
    prediction = parse_answer(raw, item.choices)
    return {
        "id": item.id,
        "method": "picture",
        "target_character": item.target_character,
        "raw_response": raw,
        "prediction": prediction,
        "gold": item.gold[0] if len(item.gold) == 1 else list(item.gold),
        "correct": is_correct(prediction, item.gold),
    }


def _fill(template: str, **values: str) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text
