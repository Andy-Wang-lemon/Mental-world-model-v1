"""Agentic-ToM：先规划，再由模型逐次调用认知工具，最后作答。"""

from __future__ import annotations

import json
import re

from inference_baselines.items import BenchmarkItem
from inference_baselines.parse_answer import is_correct, parse_answer
from inference_baselines.prompts import format_choices, format_events

MAX_TOOL_CALLS = 8
_FORMAT_ATTEMPTS = 3
_TOOLS = (
    "shift_perspective",
    "mental_state_analysis",
    "choice_reflection",
    "answer",
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_TOOL_TEXT = """shift_perspective
arguments: {"characters": ["outer character", "inner character"]}
Simulates one perspective, or a nested perspective from outermost to innermost.
A nested call such as ["Alice", "Bob"] is allowed only after ["Alice"] has already been called.
Use it for what that perspective can know or perceive.

mental_state_analysis
arguments: {"character": "name", "mental_states": "belief and knowledge"}
Tracks only the requested states across events: belief, desire, intention, knowledge, emotion, preference, expectation.
For a recommendation dialogue, the plan may also request preferences, recommendation goals, conversational intention, beliefs about the other participant, knowledge of previous recommendations, or the likely next conversational strategy, when the question needs them.

choice_reflection
arguments: {}
Reviews the question, earlier tool outputs, and every choice. It does not see a gold answer.

answer
arguments: {}
Chooses one listed letter and ends with ANSWER: <letter>."""

_PLAN = """You are the planner for Agentic-ToM.
Write a plan only. Do not answer the question. Do not write ANSWER.

Full story or dialogue:
{context}

The main question to be solved is:
{question}

Choices:
{choices}

Target character named by the question:
{target_character}

Available cognitive tools:
{tools}

Decide from this question:
1. What kind of theory-of-mind question this is.
2. Which character perspectives are needed.
3. Whether a nested perspective is needed.
4. Which mental states need tracking.
5. Whether reflection is needed.
6. A recommended tool sequence.

The sequence is your choice for this question. It is not fixed.
If a nested perspective is needed, plan the calls from the outside inward.
An illustration of that nesting, not a default plan, is:
1. shift_perspective with characters ["Alice"]
2. shift_perspective with characters ["Alice", "Bob"]

Write:
PLAN:
1. ..."""

_DECIDE = """You are choosing the next Agentic-ToM cognitive tool.
Do not answer the question in this step.

Original problem:
{problem}

Plan:
{plan}

Previous tool outputs:
{history}

Available cognitive tools:
{tools}

Return only a JSON object with keys "tool" and "arguments".
The tool must be one of: shift_perspective, mental_state_analysis, choice_reflection, answer."""

_SHIFT = """You are the shift_perspective cognitive tool.
Perspective chain, from outermost to innermost: {characters}

Full story or dialogue, unchanged:
{context}

The main question to be solved is:
{question}

Choices:
{choices}

Previous tool outputs:
{history}

Simulate this perspective. Consider only information this perspective can know or perceive.
If the chain has more than one name, the outer character is representing what the next character can perceive.
Return this perspective as a new reasoning state.
Do not write ANSWER."""

_MENTAL = """You are the mental_state_analysis cognitive tool.
Character: {character}
Mental states to track: {mental_states}

Full story or dialogue:
{context}

Events in chronological order:
{events}

The main question to be solved is:
{question}

Previous tool outputs:
{history}

For each important event, track the specified character's mental state and how it changes over time.
There is no gold answer in this prompt. Do not invent one.

Output:
Initial State:
...

Event-by-Event Analysis:
Event 1 -> state or change
Event 2 -> state or change

Current Mental State:
...

Do not write ANSWER."""

_REFLECT = """You are the choice_reflection cognitive tool.
Integrate the cognitive-tool outputs below. There is no gold answer here.

The main question to be solved is:
{question}

Choices:
{choices}

Previous tool outputs:
{history}

1. Restate the question.
2. Review the previous tool outputs.
3. Re-evaluate evidence that may have been overlooked.
4. Examine every answer choice.
5. Eliminate choices that are impossible or inconsistent with the perspective.
6. Compare the remaining choices.
7. Decide whether another cognitive tool is still required.
8. If the evidence is sufficient, identify the most likely choice.

Do not write a line that starts with ANSWER. The answer tool does that."""

_ANSWER = """You are the answer cognitive tool.
The task is multiple choice. Map the reasoning onto one listed choice.

The main question to be solved is:
{question}

Choices:
{choices}

Previous tool outputs:
{history}

At the end output exactly:
ANSWER: <letter>"""


class FormatFailure(RuntimeError):
    def __init__(self, message: str, saved: dict) -> None:
        super().__init__(message)
        self.saved = saved


def run_agentic(client: object, item: BenchmarkItem) -> dict:
    problem = _problem(item)
    plan_prompt = _fill(
        _PLAN,
        context=item.context,
        question=item.question,
        choices=format_choices(item.choices),
        target_character=item.target_character,
        tools=_TOOL_TEXT,
    )
    plan = client.complete_text([{"role": "user", "content": plan_prompt}])
    steps: list[dict] = []
    answered = ""
    for _step in range(MAX_TOOL_CALLS):
        decision_prompt = _fill(
            _DECIDE,
            problem=problem,
            plan=plan,
            history=_history(steps),
            tools=_TOOL_TEXT,
        )
        try:
            call, decision_raw = _read_call(client, decision_prompt)
            record = _execute(client, item, call, decision_raw, steps)
        except RuntimeError as error:
            raise FormatFailure(str(error), _saved(item, plan, plan_prompt, steps, "")) from error
        steps.append(record)
        if record["tool"] == "answer":
            answered = str(record["raw_response"])
            break
    if not answered:
        reflection = _execute(
            client,
            item,
            {"tool": "choice_reflection", "arguments": {}},
            "",
            steps,
        )
        steps.append(reflection)
        try:
            final = _execute(client, item, {"tool": "answer", "arguments": {}}, "", steps)
        except RuntimeError as error:
            raise FormatFailure(str(error), _saved(item, plan, plan_prompt, steps, "")) from error
        steps.append(final)
        answered = str(final["raw_response"])
    prediction = parse_answer(answered, item.choices)
    row = _saved(item, plan, plan_prompt, steps, answered)
    row["prediction"] = prediction
    row["correct"] = is_correct(prediction, item.gold)
    return row


def _execute(client: object, item: BenchmarkItem, call: dict, decision_raw: str, steps: list[dict]) -> dict:
    name = str(call.get("tool") or "")
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if name == "shift_perspective":
        characters = _characters(arguments.get("characters"))
        arguments = {"characters": characters}
        missing = _missing_prefix(characters, steps)
        if missing:
            text = (
                "Nested perspective must be built from the outside inward. "
                f"Call shift_perspective with {missing} before {characters}."
            )
            return _record(name, arguments, text, text, decision_raw)
        prompt = _fill(
            _SHIFT,
            characters=json.dumps(characters, ensure_ascii=False),
            context=item.context,
            question=item.question,
            choices=format_choices(item.choices),
            history=_history(steps),
        )
        raw = client.complete_text([{"role": "user", "content": prompt}])
        return _record(name, arguments, prompt, raw, decision_raw)
    if name == "mental_state_analysis":
        character = str(arguments.get("character") or item.target_character).strip()
        states = _states(arguments.get("mental_states"))
        arguments = {"character": character, "mental_states": states}
        prompt = _fill(
            _MENTAL,
            character=character,
            mental_states=states,
            context=item.context,
            events=format_events(item.events),
            question=item.question,
            history=_history(steps),
        )
        raw = client.complete_text([{"role": "user", "content": prompt}])
        return _record(name, arguments, prompt, raw, decision_raw)
    if name == "choice_reflection":
        prompt = _fill(
            _REFLECT,
            question=item.question,
            choices=format_choices(item.choices),
            history=_history(steps),
        )
        raw = client.complete_text([{"role": "user", "content": prompt}])
        return _record(name, {}, prompt, raw, decision_raw)
    if name == "answer":
        prompt, raw = _answer(client, item, steps)
        return _record(name, {}, prompt, raw, decision_raw)
    text = f"Unknown tool {name}. Use one of: {', '.join(_TOOLS)}."
    return _record(name or "unknown", arguments, text, text, decision_raw)


def _answer(client: object, item: BenchmarkItem, steps: list[dict]) -> tuple[str, str]:
    prompt = _fill(
        _ANSWER,
        question=item.question,
        choices=format_choices(item.choices),
        history=_history(steps),
    )
    last = ""
    for _attempt in range(_FORMAT_ATTEMPTS):
        last = client.complete_text([{"role": "user", "content": prompt}])
        if parse_answer(last, item.choices):
            return prompt, last
    raise RuntimeError("answer 工具没有返回合法选项字母")


def _read_call(client: object, prompt: str) -> tuple[dict, str]:
    last = ""
    for attempt in range(_FORMAT_ATTEMPTS):
        text = prompt
        if attempt:
            text += "\n\nThe previous reply was not valid JSON. Return only the JSON object."
        last = client.complete_text([{"role": "user", "content": text}])
        parsed = _parse_call(last)
        if parsed is not None and parsed["tool"] in _TOOLS:
            return parsed, last
    raise RuntimeError(f"工具调用格式无效：{last[:200]}")


def _parse_call(raw: str) -> dict | None:
    stripped = _FENCE_RE.sub("", str(raw or "").strip()).strip()
    data = _object(stripped)
    if data is None:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            data = _object(stripped[start : end + 1])
    if not isinstance(data, dict):
        return None
    name = str(data.get("tool") or data.get("tool_name") or "").strip()
    arguments = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
    if not name:
        return None
    return {"tool": name, "arguments": arguments}


def _object(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _characters(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        name = str(item or "").strip()
        if name:
            names.append(name)
    return names


def _states(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "belief and knowledge").strip()


def _missing_prefix(characters: list[str], steps: list[dict]) -> list[str]:
    if len(characters) <= 1:
        return []
    done = {
        tuple(step["arguments"].get("characters") or [])
        for step in steps
        if step.get("tool") == "shift_perspective" and not str(step.get("raw_response") or "").startswith("Nested perspective")
    }
    for length in range(1, len(characters)):
        prefix = characters[:length]
        if tuple(prefix) not in done:
            return prefix
    return []


def _problem(item: BenchmarkItem) -> str:
    return (
        f"Full story or dialogue:\n{item.context}\n\n"
        f"The main question to be solved is:\n{item.question}\n\n"
        f"Choices:\n{format_choices(item.choices)}"
    )


def _history(steps: list[dict]) -> str:
    if not steps:
        return "(none)"
    chunks = []
    for index, step in enumerate(steps, start=1):
        chunks.append(
            f"{index}. {step.get('tool')}\n"
            f"arguments: {json.dumps(step.get('arguments') or {}, ensure_ascii=False)}\n"
            f"output:\n{step.get('raw_response') or ''}"
        )
    return "\n\n".join(chunks)


def _record(name: str, arguments: dict, prompt: str, raw: str, decision_raw: str) -> dict:
    return {
        "tool": name,
        "arguments": arguments,
        "tool_prompt": prompt,
        "raw_response": raw,
        "decision_response": decision_raw,
    }


def _saved(item: BenchmarkItem, plan: str, plan_prompt: str, steps: list[dict], answered: str) -> dict:
    return {
        "id": item.id,
        "method": "agentictom",
        "implementation": "Agentic-ToM",
        "target_character": item.target_character,
        "plan_prompt": plan_prompt,
        "plan": plan,
        "steps": steps,
        "raw_response": answered,
        "prediction": "",
        "gold": item.gold[0] if len(item.gold) == 1 else list(item.gold),
        "correct": False,
    }


def _fill(template: str, **values: str) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text
