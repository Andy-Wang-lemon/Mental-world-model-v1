"""第一层：冻结快照 → belief / emotion / intention / action / changes。"""

from __future__ import annotations

import json
from typing import Any

from dyntom_adapter.guard import assert_no_story_keys
from dyntom_adapter.router import DIMENSIONS

PROJECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "belief": {"type": "string"},
        "emotion": {"type": "string"},
        "intention": {"type": "string"},
        "action": {"type": "string"},
        "influences": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "belief_to_emotion": {"type": "string"},
                "belief_emotion_to_intention": {"type": "string"},
                "intention_to_action": {"type": "string"},
            },
            "required": [
                "belief_to_emotion",
                "belief_emotion_to_intention",
                "intention_to_action",
            ],
        },
        "changes": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "belief": {"type": "string"},
                "emotion": {"type": "string"},
                "intention": {"type": "string"},
                "action": {"type": "string"},
            },
            "required": ["belief", "emotion", "intention", "action"],
        },
    },
    "required": ["belief", "emotion", "intention", "action", "influences", "changes"],
}

SYSTEM_PROMPT = """You are a state-projection module for a theory-of-mind benchmark.
Map the supplied internal snapshot onto four short English labels: belief, emotion, intention, and action.
Also state why each label changed since the previous scenario, and how the four states influence each other inside this scenario.

Rules:
- Use ONLY the JSON snapshot and the previous projection in the user message.
- The original story is not included. Do not invent plot events or causes that the snapshot does not support.
- belief: one short noun phrase for what this person now thinks is happening, based on objects, thoughts, and history. Not a sentence.
- emotion: one short English everyday emotion phrase summarized from mental_state intensities and thoughts. Do not output raw factor names or the source language. Not a sentence.
- intention: a short phrase with a verb of aiming plus its target, based on thoughts, mental_state, and predicted_actions. Name the immediate aim toward a person or an outcome. Do not name only the situation. Intention is not an action already taken.
- action: one short phrase for what this person does in this scenario. If gathering information is the act itself, name what information is sought, even if they also speak to another person. Prefer observed_action when it is specific; otherwise use the highest-probability predicted action. Action is not the goal.
- changes: one short cause for that dimension, based on deltas. Do not restate the before-and-after labels. Belief: the new information or reinterpretation. Emotion: how the person now takes the situation, not only who spoke. Intention: which new aim the updated belief or emotion makes necessary. Action: which constraint or goal changed the behavior. Use an empty string when there is no previous projection.
- influences: one short English sentence each for belief→emotion, belief and emotion→intention, and intention→action inside this scenario.
- belief, emotion, intention, and action must each stay within about eight words.
Return JSON only."""


def project_state(
    llm: object,
    snapshot: dict,
    previous: dict | None,
    *,
    deltas: dict | None = None,
) -> dict:
    clean = {
        key: value
        for key, value in snapshot.items()
        if key != "_deltas_from_prior_snapshot"
    }
    assert_no_story_keys(clean, where="projection input")
    if deltas is None:
        deltas = snapshot.get("_deltas_from_prior_snapshot") or {}
    user = {
        "snapshot": clean,
        "previous_projection": _public_projection(previous),
        "deltas": deltas,
    }
    raw = llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        schema=PROJECTION_SCHEMA,
    )
    return normalize_projection(raw, scenario=int(snapshot["scenario"]))


def normalize_projection(raw: object, *, scenario: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("投影结果不是 JSON 对象")
    influences = raw.get("influences") if isinstance(raw.get("influences"), dict) else {}
    changes = raw.get("changes") if isinstance(raw.get("changes"), dict) else {}
    projection = {
        "scenario": scenario,
        "belief": _phrase(raw.get("belief")),
        "emotion": _phrase(raw.get("emotion")),
        "intention": _phrase(raw.get("intention")),
        "action": _phrase(raw.get("action")),
        "influences": {
            "belief_to_emotion": _phrase(influences.get("belief_to_emotion")),
            "belief_emotion_to_intention": _phrase(
                influences.get("belief_emotion_to_intention")
            ),
            "intention_to_action": _phrase(influences.get("intention_to_action")),
        },
        "changes": {name: _phrase(changes.get(name)) for name in DIMENSIONS},
    }
    if scenario == 1:
        projection["changes"] = {name: "" for name in DIMENSIONS}
    missing = [name for name in DIMENSIONS if not projection[name]]
    if missing:
        raise ValueError(f"scenario {scenario} 投影缺少 {missing}")
    return projection


def attach_deltas(previous_snapshot: dict | None, snapshot: dict) -> dict:
    """把相邻快照的数值差交给投影层。不加入故事原文。"""
    payload = dict(snapshot)
    payload["_deltas_from_prior_snapshot"] = _deltas(previous_snapshot, snapshot)
    return payload


def _public_projection(previous: dict | None) -> dict | None:
    if not previous:
        return None
    return {
        "scenario": previous.get("scenario"),
        "belief": previous.get("belief", ""),
        "emotion": previous.get("emotion", ""),
        "intention": previous.get("intention", ""),
        "action": previous.get("action", ""),
    }


def _phrase(value: object) -> str:
    return " ".join(str(value or "").split())


def _deltas(previous_snapshot: dict | None, current: dict) -> dict:
    if not previous_snapshot:
        return {}
    return {
        "objects": _ranked_shift(
            previous_snapshot.get("objects") or [],
            current.get("objects") or [],
            text_key="content",
            weight_key="probability",
        ),
        "mental_event": _score_shift(
            (previous_snapshot.get("mental_state") or {}).get("event") or {},
            (current.get("mental_state") or {}).get("event") or {},
        ),
        "history": _ranked_shift(
            previous_snapshot.get("history") or [],
            current.get("history") or [],
            text_key="content",
            weight_key="mass",
        ),
    }


def _ranked_shift(
    previous: list,
    current: list,
    *,
    text_key: str,
    weight_key: str,
) -> list[dict]:
    before = {
        str(item.get(text_key) or ""): float(item.get(weight_key) or 0.0)
        for item in previous
        if isinstance(item, dict) and str(item.get(text_key) or "").strip()
    }
    after = {
        str(item.get(text_key) or ""): float(item.get(weight_key) or 0.0)
        for item in current
        if isinstance(item, dict) and str(item.get(text_key) or "").strip()
    }
    rows = []
    for text in set(before) | set(after):
        delta = round(after.get(text, 0.0) - before.get(text, 0.0), 4)
        if abs(delta) < 0.02 and text not in after:
            continue
        if abs(delta) < 0.02:
            continue
        rows.append({"content": text, "delta": delta})
    rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rows[:8]


def _score_shift(previous: dict, current: dict) -> list[dict]:
    keys = set(previous) | set(current)
    rows = []
    for name in keys:
        delta = round(float(current.get(name) or 0.0) - float(previous.get(name) or 0.0), 4)
        if abs(delta) < 0.02:
            continue
        rows.append({"name": str(name), "delta": delta})
    rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rows[:8]
