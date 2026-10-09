"""把一次 MWM 评估裁成选题可见的记录。不保留故事原文。"""

from __future__ import annotations


def _round(value: object) -> float:
    try:
        return round(float(value or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _text(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "").strip()
    return str(value or "").strip()


def freeze_snapshot(result: object) -> dict:
    raw = result.to_event_snapshot(event_label="SimpleToM")
    if not isinstance(raw, dict):
        raise TypeError("to_event_snapshot 必须返回 dict")
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
    for branch in raw.get("branches") or []:
        if not isinstance(branch, dict):
            continue
        thought = _text(branch.get("xiang"))
        if not thought:
            continue
        thoughts.append(
            {
                "content": thought,
                "activation": _round(
                    branch.get("activation_probability", branch.get("branch_effective_probability"))
                ),
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
        predicted.append({"content": content, "probability": _round(item.get("probability"))})
    predicted.sort(key=lambda item: item["probability"], reverse=True)

    evidence = raw.get("action_evidence") if isinstance(raw.get("action_evidence"), dict) else {}
    return {
        "focus_person": str(raw.get("focus_person") or ""),
        "objects": objects,
        "thoughts": thoughts,
        "observed_action": str(evidence.get("action_description") or "").strip(),
        "predicted_actions": predicted,
    }
