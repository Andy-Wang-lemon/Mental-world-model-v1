"""按题干确定要用哪几幕，并从冻结状态取出所缘、想、行动。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DIMENSIONS = ("belief", "emotion", "intention", "action")

_WHAT = re.compile(
    r"What is the (belief|emotion|intention|action) of .+? in scenario (\d+)\?",
    re.IGNORECASE,
)
_HOW = re.compile(
    r"How does the (belief|emotion|intention|action) of .+? change across the (\d+) scenarios\?",
    re.IGNORECASE,
)
_WHY = re.compile(
    r"Why does the (belief|emotion|intention|action) of .+? changes? from scenario (\d+) to scenario (\d+)\?",
    re.IGNORECASE,
)
_WHETHER = re.compile(
    r"Whether the (belief|emotion|intention|action) of .+? changes? from scenario (\d+) to scenario (\d+)\?",
    re.IGNORECASE,
)
_INF_BELIEF_EMOTION = re.compile(
    r"In scenario (\d+), how does the belief of .+? influence the emotion of",
    re.IGNORECASE,
)
_INF_BELIEF_EMOTION_INTENTION = re.compile(
    r"In scenario (\d+), how does the belief&emotion of .+? influence the intention of",
    re.IGNORECASE,
)
_INF_INTENTION_ACTION = re.compile(
    r"In scenario (\d+), how does the intention of .+? influence the action of",
    re.IGNORECASE,
)


class RouterError(ValueError):
    """题干不在已知模板里。"""


@dataclass
class Route:
    question_id: str
    question_type: str
    target_dimension: str
    operation: str
    required_scenarios: list[int]
    source_dimensions: list[str] = field(default_factory=list)
    scope: str = "single_scenario"


def parse_route(question_id: str, question: str, *, question_type: str = "") -> Route:
    text = " ".join(question.split())
    qtype = question_type or _type_from_id(question_id)

    match = _INF_BELIEF_EMOTION_INTENTION.search(text)
    if match:
        scenario = int(match.group(1))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_c",
            target_dimension="intention",
            operation="influence",
            required_scenarios=[scenario],
            source_dimensions=["belief", "emotion"],
            scope="single_scenario",
        )
    match = _INF_BELIEF_EMOTION.search(text)
    if match:
        scenario = int(match.group(1))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_c",
            target_dimension="emotion",
            operation="influence",
            required_scenarios=[scenario],
            source_dimensions=["belief"],
            scope="single_scenario",
        )
    match = _INF_INTENTION_ACTION.search(text)
    if match:
        scenario = int(match.group(1))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_c",
            target_dimension="action",
            operation="influence",
            required_scenarios=[scenario],
            source_dimensions=["intention"],
            scope="single_scenario",
        )
    match = _WHAT.search(text)
    if match:
        scenario = int(match.group(2))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_a",
            target_dimension=match.group(1).lower(),
            operation="what",
            required_scenarios=[scenario],
            scope="single_scenario",
        )
    match = _WHY.search(text)
    if match:
        start, end = int(match.group(2)), int(match.group(3))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_d",
            target_dimension=match.group(1).lower(),
            operation="why",
            required_scenarios=[start, end],
            scope="cross_scenario",
        )
    match = _WHETHER.search(text)
    if match:
        start, end = int(match.group(2)), int(match.group(3))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_d",
            target_dimension=match.group(1).lower(),
            operation="whether",
            required_scenarios=[start, end],
            scope="cross_scenario",
        )
    match = _HOW.search(text)
    if match:
        count = int(match.group(2))
        return Route(
            question_id=question_id,
            question_type=qtype or "type_d",
            target_dimension=match.group(1).lower(),
            operation="how",
            required_scenarios=list(range(1, count + 1)),
            scope="cross_scenario",
        )
    raise RouterError(f"无法路由题目 {question_id}：{question}")


def bundle_for(route: Route, snapshots: list[dict]) -> dict:
    """题目所需各幕的主观所缘、想、行动。不含投影，不含故事。"""
    by_scenario = {}
    for item in snapshots:
        if isinstance(item, dict) and item.get("scenario") is not None:
            by_scenario[int(item["scenario"])] = item
    missing = [index for index in route.required_scenarios if index not in by_scenario]
    if missing:
        raise RouterError(f"缺少 scenario {missing} 的 state")
    return {
        "scenarios": [_scenario_view(by_scenario[index]) for index in route.required_scenarios]
    }


def bundle_lines(bundle: dict) -> list[str]:
    lines = []
    for scenario in bundle.get("scenarios") or []:
        if not isinstance(scenario, dict):
            continue
        index = scenario.get("scenario")
        for key in ("objects", "thoughts", "predicted_actions"):
            for item in scenario.get(key) or []:
                if isinstance(item, dict) and str(item.get("content") or "").strip():
                    lines.append(f"s{index} {key}: {item['content']}")
        observed = str(scenario.get("observed_action") or "").strip()
        if observed:
            lines.append(f"s{index} observed_action: {observed}")
    return lines


def _scenario_view(snapshot: dict) -> dict:
    return {
        "scenario": int(snapshot.get("scenario") or 0),
        "objects": _passages(snapshot.get("objects"), "probability"),
        "thoughts": _passages(snapshot.get("thoughts"), "activation"),
        "observed_action": str(snapshot.get("observed_action") or "").strip(),
        "predicted_actions": _passages(snapshot.get("predicted_actions"), "probability"),
    }


def _passages(items: object, weight_key: str) -> list[dict]:
    passages = []
    if not isinstance(items, list):
        return passages
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        passages.append({"content": content, "weight": item.get(weight_key)})
    return passages


def representation_for(route: Route, projections: list[dict]) -> dict:
    by_scenario = {int(item["scenario"]): item for item in projections}
    missing = [index for index in route.required_scenarios if index not in by_scenario]
    if missing:
        raise RouterError(f"缺少 scenario {missing} 的投影")

    def label(index: int, dimension: str) -> str:
        return str(by_scenario[index].get(dimension) or "").strip()

    dimension = route.target_dimension
    if route.operation == "what":
        index = route.required_scenarios[0]
        return {"scenario": index, dimension: label(index, dimension)}
    if route.operation == "how":
        return {
            "dimension": dimension,
            "trajectory": [label(index, dimension) for index in route.required_scenarios],
        }
    if route.operation == "why":
        start, end = route.required_scenarios
        changes = by_scenario[end].get("changes") or {}
        return {
            "dimension": dimension,
            "from_scenario": start,
            "to_scenario": end,
            "change_reason": str(changes.get(dimension) or "").strip(),
        }
    if route.operation == "whether":
        start, end = route.required_scenarios
        return {
            "dimension": dimension,
            "from_scenario": start,
            "to_scenario": end,
            "before": label(start, dimension),
            "after": label(end, dimension),
        }
    index = route.required_scenarios[0]
    state = by_scenario[index]
    influences = state.get("influences") or {}
    payload = {"scenario": index}
    for name in route.source_dimensions + [dimension]:
        payload[name] = label(index, name)
    if route.source_dimensions == ["belief"] and dimension == "emotion":
        payload["influence"] = str(influences.get("belief_to_emotion") or "").strip()
    elif "emotion" in route.source_dimensions and dimension == "intention":
        payload["influence"] = str(
            influences.get("belief_emotion_to_intention") or ""
        ).strip()
    elif route.source_dimensions == ["intention"] and dimension == "action":
        payload["influence"] = str(influences.get("intention_to_action") or "").strip()
    return payload


def state_lines(representation: dict) -> list[str]:
    """答题记录里的 mwm_state_used。"""
    if "trajectory" in representation:
        return [str(item) for item in representation["trajectory"]]
    if "change_reason" in representation and "before" not in representation:
        reason = str(representation.get("change_reason") or "").strip()
        return [reason] if reason else []
    if "before" in representation and "after" in representation:
        lines = [
            str(representation.get("before") or ""),
            str(representation.get("after") or ""),
        ]
        reason = str(representation.get("change_reason") or "").strip()
        if reason:
            lines.append(reason)
        return lines
    lines = []
    for key in ("belief", "emotion", "intention", "action", "influence"):
        if key in representation:
            lines.append(str(representation.get(key) or ""))
    return lines


def _type_from_id(question_id: str) -> str:
    match = re.match(r"(type_[a-z])_", question_id)
    return match.group(1) if match else ""
