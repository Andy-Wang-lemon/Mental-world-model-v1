"""SimpleToM 一次只取一道行为预测题，概率相加后取最高，选错才核对。"""

from __future__ import annotations

import json
from pathlib import Path

from simpletom_adapter.loader import focus_person, load_one
from simpletom_adapter.selector import choose_by_scores

_DATA = Path("/Users/shitou/Desktop/其他文件/心智世界模型/公开数据集/SimpleToM/test.jsonl")


def test_load_one_behavior_item():
    item = load_one(_DATA, 0)
    assert item.index == 0
    assert item.item_id == "potato_chip_food_sev1_action"
    assert item.focus_person == "Mary"
    assert item.gold == ("a",)
    assert item.choices[0].startswith("A:")
    assert item.choices[1].startswith("B:")


def test_every_question_has_a_focus_person():
    rows = [json.loads(line) for line in _DATA.read_text().splitlines() if line.strip()]
    missing = [row["question"] for row in rows if not focus_person(row["question"])]
    assert missing == []


def test_actions_of_one_option_are_summed():
    class _Llm:
        def complete(self, messages, schema=None):
            return {
                "groups": [
                    {"letter": "a", "indexes": [0, 2]},
                    {"letter": "b", "indexes": [1]},
                ],
                "reason": "two actions pay",
            }

    letter, scores, reason = choose_by_scores(
        _Llm(),
        {
            "predicted_actions": [
                {"content": "去收银台付钱", "probability": 0.4},
                {"content": "向店员报告发霉", "probability": 0.35},
                {"content": "买下这袋薯片", "probability": 0.25},
            ]
        },
        "What will Mary likely do next?",
        ["A: pay for the chips", "B: report the moldy chips"],
    )
    assert letter == "a"
    assert scores["a"] == 0.65
    assert scores["b"] == 0.35
    assert "去收银台付钱 0.4000" in reason


def test_tie_is_rescored_without_letter_order():
    class _Llm:
        def complete(self, messages, schema=None):
            if schema and "scores" in json.dumps(schema):
                return {
                    "scores": [
                        {"letter": "a", "score": 0.4},
                        {"letter": "b", "score": 0.8},
                    ],
                    "reason": "reporting is more direct",
                }
            return {
                "groups": [
                    {"letter": "a", "indexes": [0]},
                    {"letter": "b", "indexes": [1]},
                ],
                "reason": "same probability",
            }

    letter, scores, _reason = choose_by_scores(
        _Llm(),
        {
            "predicted_actions": [
                {"content": "付钱", "probability": 0.5},
                {"content": "报告发霉", "probability": 0.5},
            ]
        },
        "What will Mary likely do next?",
        ["A: pay for the chips", "B: report the moldy chips"],
    )
    assert letter == "b"
    assert scores["b"] > scores["a"]


def test_wrong_letter_can_be_checked_against_gold_only(tmp_path):
    from simpletom_adapter.loader import SimpleTomItem
    from simpletom_adapter.runner import _finish

    item = SimpleTomItem(
        index=0,
        item_id="potato_chip_food_sev1_action",
        story="The bag of potato chips has moldy chips in it.",
        question="What will Mary likely do next?",
        choices=("A: pay for the chips", "B: report the moldy chips"),
        gold=("a",),
        focus_person="Mary",
        scenario_name="food_item_in_grocery_store",
    )

    class _Llm:
        def complete(self, messages, schema=None):
            text = messages[-1]["content"]
            if "Accepted behavior:" in text:
                assert "report the moldy chips" not in text
                return {"level": "direct", "reason": "The record says Mary will pay next."}
            return {
                "groups": [
                    {"letter": "a", "indexes": []},
                    {"letter": "b", "indexes": [0]},
                ],
                "reason": "reports mold",
            }

    row = _finish(
        item,
        {"predicted_actions": [{"content": "report the mold", "probability": 0.7}]},
        _Llm(),
        tmp_path,
        "test",
    )
    assert row["selector_correct"] is False
    assert row["correct"] is True
    assert (tmp_path / "answer.json").is_file()
    assert (tmp_path / "report.txt").is_file()
