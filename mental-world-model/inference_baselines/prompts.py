"""三套方法的提示。第一阶段不放选项，也不放标准答案。"""

from __future__ import annotations

from inference_baselines.items import BenchmarkItem

COT_PROMPT = """You are answering a Theory of Mind reasoning task.

Read the story or dialogue carefully.

Before selecting an answer, reason step by step about:
- what the target character knows
- what the target character does not know
- the character's beliefs
- mental state
- intention
- relevant previous events

Base the prediction on the target character's own perspective rather than on information unavailable to them.

Question:
{question}

Choices:
{choices}

Reason step by step.

At the end output exactly:

ANSWER: <letter>"""

SIMTOM_STAGE1_PROMPT = """You are reconstructing the subjective perspective of a target character.

Full story or dialogue:
{context}

Target character:
{target_character}

Identify ONLY the information that the target character has access to.

Include:
- events directly observed by the character
- statements the character heard
- information explicitly told to the character
- the character's own actions
- relevant previous interactions they participated in

Exclude:
- hidden events
- private information of other characters
- narrator-only information unavailable to the character
- events the character did not observe
- information inferred only because the reader knows the full story

Return a concise reconstruction of the target character's available information."""

SIMTOM_STAGE2_PROMPT = """You must answer the following question strictly from the target character's perspective.

Information available to the target character:

{perspective}

Question:
{question}

Choices:
{choices}

Reason only from the information available to the target character.

Do not use hidden information from the original story that is absent from the reconstructed perspective.

At the end output exactly:

ANSWER: <letter>"""

TIMETOM_DYNTOM_PROMPT = """You are tracking a target character's mental state over time.

This is a TimeToM-style temporal reconstruction, not a final answer.

Target character:
{target_character}

Events in chronological order:
{events}

For each event, determine:

1. What the target character observes or learns
2. Their belief before the event
3. Whether the event changes their belief
4. Their belief after the event
5. Their likely emotion
6. Their current intention

Do not use future information when analyzing an earlier event.

Maintain temporal continuity.

Return structured JSON:

[
  {{
    "time": 1,
    "observed": "...",
    "belief_before": "...",
    "belief_update": "...",
    "belief_after": "...",
    "emotion": "...",
    "intention": "..."
  }}
]"""

TIMETOM_RECTOM_PROMPT = """You are tracking a target speaker's mental state across a recommendation dialogue.

This is a TimeToM-style temporal reconstruction, not a final answer.

Target character:
{target_character}

Dialogue turns in chronological order:
{events}

For each turn, determine:
- information available to the target speaker
- preferences
- beliefs
- intentions
- reaction to the other speaker
- likely conversational goal
- whether the new turn changes the previous interpretation

Do not use a later turn to explain an earlier turn.

Maintain temporal continuity.

Return structured JSON:

[
  {{
    "time": 1,
    "observed": "...",
    "belief_before": "...",
    "belief_update": "...",
    "belief_after": "...",
    "emotion": "...",
    "intention": "...",
    "conversational_goal": "..."
  }}
]"""

TIMETOM_STAGE2_PROMPT = """You are given the temporal mental-state history of a target character.

This answer uses a TimeToM-style state chain.

Temporal state chain:

{temporal_state_chain}

Question:
{question}

Choices:
{choices}

Use the latest mental state together with its historical evolution to answer the question.

Do not introduce information unavailable to the target character.

At the end output exactly:

ANSWER: <letter>"""


def format_choices(choices: tuple[str, ...]) -> str:
    return "\n".join(choices)


def format_events(events: tuple[str, ...]) -> str:
    return "\n\n".join(f"{index}. {event}" for index, event in enumerate(events, start=1))


def cot_messages(item: BenchmarkItem) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Story or dialogue:\n"
                f"{item.context}\n\n"
                + COT_PROMPT.format(question=item.question, choices=format_choices(item.choices))
            ),
        }
    ]


def simtom_stage1_messages(item: BenchmarkItem) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": SIMTOM_STAGE1_PROMPT.format(
                context=item.context,
                target_character=item.target_character,
            ),
        }
    ]


def simtom_stage2_messages(perspective: str, item: BenchmarkItem) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": SIMTOM_STAGE2_PROMPT.format(
                perspective=perspective.strip(),
                question=item.question,
                choices=format_choices(item.choices),
            ),
        }
    ]


def timetom_stage1_messages(item: BenchmarkItem) -> list[dict[str, str]]:
    template = TIMETOM_DYNTOM_PROMPT if item.dataset == "dyntom" else TIMETOM_RECTOM_PROMPT
    return [
        {
            "role": "user",
            "content": template.format(
                target_character=item.target_character,
                events=format_events(item.events),
            ),
        }
    ]


def timetom_stage2_messages(chain_text: str, item: BenchmarkItem) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": TIMETOM_STAGE2_PROMPT.format(
                temporal_state_chain=chain_text.strip(),
                question=item.question,
                choices=format_choices(item.choices),
            ),
        }
    ]
