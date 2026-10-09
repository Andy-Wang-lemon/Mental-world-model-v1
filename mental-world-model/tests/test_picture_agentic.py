"""PICTURE 保留全文。Agentic-ToM 分次调用工具，答错不重试。"""

from __future__ import annotations

import json

from inference_baselines.agentic_tom import MAX_TOOL_CALLS, run_agentic
from inference_baselines.items import BenchmarkItem, load_split
from inference_baselines.picture import picture_prompt, run_picture


def _toy() -> BenchmarkItem:
    return BenchmarkItem(
        id="toy",
        dataset="rectom",
        context="SEEKER says: hello there",
        events=("SEEKER says: hello there",),
        question="What strategy will SEEKER use next?",
        choices=(
            "A: Ask for preference",
            "B: Say thanks",
            "C: Give a recommendation",
        ),
        target_character="SEEKER",
        gold=("c",),
        source="toy",
    )


def test_picture_keeps_the_full_story_and_choices():
    dyntom = load_split("dyntom")[0]
    rectom = load_split("rectom")[0]
    dyntom_prompt = picture_prompt(dyntom)
    rectom_prompt = picture_prompt(rectom)
    assert dyntom.context in dyntom_prompt
    assert "What events does and does not Henry Lozano know about?" in dyntom_prompt
    assert "Do not delete or hide events" in dyntom_prompt
    assert all(choice in dyntom_prompt for choice in dyntom.choices)
    assert "true answer" not in dyntom_prompt
    assert rectom.context in rectom_prompt
    assert "What utterances, facts, preferences, intentions, and interaction events does and does not RECOMMENDER know about?" in rectom_prompt
    assert "Identify ONLY the information" not in rectom_prompt
    assert all(choice in rectom_prompt for choice in rectom.choices)

    class _Client:
        def complete_text(self, messages):
            assert "Choices:" in messages[-1]["content"]
            return "Henry does not know the later events.\nANSWER: G"

    row = run_picture(_Client(), dyntom)
    assert row["method"] == "picture"
    assert row["prediction"] == "g"
    assert row["correct"] is True


def test_agent_calls_tools_separately_and_does_not_retry_a_wrong_letter():
    item = _toy()
    calls: list[str] = []

    class _Client:
        def complete_text(self, messages):
            text = messages[-1]["content"]
            calls.append(text)
            if text.startswith("You are the planner"):
                assert "A: Ask for preference" in text
                assert "true answer" not in text
                return 'PLAN:\n1. shift_perspective(["SEEKER"])\n2. answer()'
            if text.startswith("You are choosing the next"):
                assert "A: Ask for preference" in text
                if not any(chunk.startswith("You are the shift_perspective") for chunk in calls):
                    return json.dumps({"tool": "shift_perspective", "arguments": {"characters": ["SEEKER"]}})
                return json.dumps({"tool": "answer", "arguments": {}})
            if text.startswith("You are the shift_perspective"):
                assert "SEEKER says: hello there" in text
                assert "A: Ask for preference" in text
                return "SEEKER knows only the greeting."
            if text.startswith("You are the answer"):
                assert "A: Ask for preference" in text
                assert "SEEKER knows only the greeting." in text
                return "ANSWER: A"
            raise AssertionError(text[:80])

    row = run_agentic(_Client(), item)
    assert row["implementation"] == "Agentic-ToM"
    assert [step["tool"] for step in row["steps"]] == ["shift_perspective", "answer"]
    assert row["steps"][0]["tool_prompt"].startswith("You are the shift_perspective")
    assert row["prediction"] == "a"
    assert row["correct"] is False
    assert sum(text.startswith("You are the answer") for text in calls) == 1


def test_nested_perspective_cannot_skip_the_outer_call():
    item = _toy()
    calls: list[str] = []
    decisions = {"n": 0}

    class _Client:
        def complete_text(self, messages):
            text = messages[-1]["content"]
            calls.append(text)
            if text.startswith("You are the planner"):
                return "PLAN:\n1. nested"
            if text.startswith("You are choosing the next"):
                decisions["n"] += 1
                if decisions["n"] == 1:
                    return json.dumps(
                        {"tool": "shift_perspective", "arguments": {"characters": ["SEEKER", "RECOMMENDER"]}}
                    )
                if decisions["n"] == 2:
                    return json.dumps({"tool": "shift_perspective", "arguments": {"characters": ["SEEKER"]}})
                return json.dumps({"tool": "answer", "arguments": {}})
            if text.startswith("You are the shift_perspective"):
                chain = text.split("Perspective chain, from outermost to innermost: ", 1)[1].split("\n", 1)[0]
                assert chain == '["SEEKER"]'
                return "outer perspective"
            if text.startswith("You are the answer"):
                return "ANSWER: C"
            raise AssertionError(text[:80])

    row = run_agentic(_Client(), item)
    assert row["steps"][0]["raw_response"].startswith("Nested perspective")
    assert row["steps"][0]["tool_prompt"].startswith("Nested perspective")
    assert row["steps"][1]["tool"] == "shift_perspective"
    assert row["steps"][1]["arguments"]["characters"] == ["SEEKER"]
    assert row["prediction"] == "c"
    assert sum(text.startswith("You are the shift_perspective") for text in calls) == 1


def test_tool_limit_forces_reflection_then_answer():
    item = _toy()

    class _Client:
        def complete_text(self, messages):
            text = messages[-1]["content"]
            if text.startswith("You are the planner"):
                return "PLAN:\n1. keep analyzing"
            if text.startswith("You are choosing the next"):
                return json.dumps(
                    {
                        "tool": "mental_state_analysis",
                        "arguments": {"character": "SEEKER", "mental_states": ["belief", "intention"]},
                    }
                )
            if text.startswith("You are the mental_state_analysis"):
                assert "belief, intention" in text
                return "Current Mental State:\nuncertain"
            if text.startswith("You are the choice_reflection"):
                assert "A: Ask for preference" in text
                assert "Current Mental State" in text
                return "The likely choice is C. Another tool is not required."
            if text.startswith("You are the answer"):
                return "ANSWER: C"
            raise AssertionError(text[:80])

    row = run_agentic(_Client(), item)
    names = [step["tool"] for step in row["steps"]]
    assert names[:MAX_TOOL_CALLS] == ["mental_state_analysis"] * MAX_TOOL_CALLS
    assert names[-2:] == ["choice_reflection", "answer"]
    assert row["correct"] is True
