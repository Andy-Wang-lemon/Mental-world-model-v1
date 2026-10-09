"""把一次 MWM 评估裁成 Adapter 可见的状态快照。"""

from __future__ import annotations

from dyntom_adapter.guard import assert_no_story_keys


def _round(value: object) -> float:
    try:
        return round(float(value or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _text(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "").strip()
    return str(value or "").strip()


def freeze_snapshot(result: object, scenario: int) -> dict:
    """从 EvaluationResult 抽出 O / X / M / S* / H。不保留原文。"""
    raw = result.to_event_snapshot(event_label=f"Scenario {scenario}")
    if not isinstance(raw, dict):
        raise TypeError("to_event_snapshot 必须返回 dict")
    frozen = freeze_raw_snapshot(raw, scenario)
    assert_no_story_keys(frozen, where=f"state {scenario}")
    return frozen


def freeze_raw_snapshot(raw: dict, scenario: int) -> dict:
    objects = []
    for item in raw.get("subjective_alambanas") or []:
        if not isinstance(item, dict):
            continue
        content = _text(item)
        if not content:
            continue
        probability = item.get("posterior_probability", item.get("prior_probability"))
        objects.append({"content": content, "probability": _round(probability)})
    objects.sort(key=lambda item: item["probability"], reverse=True)

    thoughts = []
    branch_mental = []
    for branch in raw.get("branches") or []:
        if not isinstance(branch, dict):
            continue
        activation = _round(
            branch.get("activation_probability", branch.get("branch_effective_probability"))
        )
        thought = _text(branch.get("xiang"))
        if thought:
            thoughts.append({"content": thought, "activation": activation})
        factors = branch.get("mental_factors_nonzero") or {}
        if isinstance(factors, dict) and factors:
            branch_mental.append(
                {
                    "branch_id": str(branch.get("branch_id") or ""),
                    "activation": activation,
                    "factors": {str(name): _round(score) for name, score in factors.items()},
                }
            )
    thoughts.sort(key=lambda item: item["activation"], reverse=True)

    predicted = []
    for item in raw.get("aggregated_behaviors") or []:
        if not isinstance(item, dict):
            continue
        content = _text(item)
        if not content:
            continue
        predicted.append(
            {"content": content, "probability": _round(item.get("probability"))}
        )
    predicted.sort(key=lambda item: item["probability"], reverse=True)

    evidence = raw.get("action_evidence") if isinstance(raw.get("action_evidence"), dict) else {}
    observed = str(evidence.get("action_description") or "").strip()

    event_factors = raw.get("mental_factors_event_nonzero") or {}
    if not isinstance(event_factors, dict):
        event_factors = {}

    history = []
    continuity = raw.get("continuity") if isinstance(raw.get("continuity"), dict) else {}
    for frame in continuity.get("cognitive_frames") or []:
        if not isinstance(frame, dict):
            continue
        content = str(frame.get("text") or frame.get("frame_text") or "").strip()
        if not content:
            continue
        history.append({"content": content, "mass": _round(frame.get("mass", frame.get("weight")))})

    return {
        "scenario": int(scenario),
        "focus_person": str(raw.get("focus_person") or ""),
        "objects": objects,
        "thoughts": thoughts,
        "mental_state": {
            "event": {str(name): _round(score) for name, score in event_factors.items()},
            "branches": branch_mental,
        },
        "observed_action": observed,
        "predicted_actions": predicted,
        "history": history,
    }
