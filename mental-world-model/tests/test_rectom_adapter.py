"""RecTOM 一次只取一道题，并列要重评，选错才核对。"""

from __future__ import annotations

import json
from pathlib import Path

from rectom_adapter.loader import load_one
from rectom_adapter.selector import choose_by_scores


def test_load_one_reads_only_the_requested_index():
    path = Path("/Users/shitou/Desktop/其他文件/心智世界模型/公开数据集/RecTOM/3_pred_rec.json")
    first = load_one(path, 0)
    second = load_one(path, 1)
    assert first.index == 0
    assert first.focus_person == "RECOMMENDER"
    assert first.gold == ("c",)
    assert first.item_id != second.item_id
    seeker = load_one(
        Path("/Users/shitou/Desktop/其他文件/心智世界模型/公开数据集/RecTOM/4_pred_seeker.json"),
        1,
    )
    assert seeker.focus_person == "SEEKER"
    assert seeker.gold == ("a", "b")


def test_unique_top_score_is_the_choice():
    class _Llm:
        def complete(self, messages, schema=None):
            return {
                "groups": [
                    {"letter": "a", "indexes": []},
                    {"letter": "c", "indexes": [0]},
                ],
                "reason": "recommends next",
            }

    letter, scores, _reason = choose_by_scores(
        _Llm(),
        {"predicted_actions": [{"content": "recommend a thriller", "probability": 0.55}]},
        "What strategy will RECOMMENDER use next?",
        [
            "A: Ask for preference",
            "C: Provides a recommendation directly",
        ],
    )
    assert letter == "c"
    assert scores["c"] == 0.55
    assert scores["a"] == 0.0


def test_actions_of_one_option_are_summed():
    class _Llm:
        def complete(self, messages, schema=None):
            return {
                "groups": [
                    {"letter": "a", "indexes": [1, 2]},
                    {"letter": "c", "indexes": [0]},
                ],
                "reason": "two actions give feedback",
            }

    letter, scores, reason = choose_by_scores(
        _Llm(),
        {
            "predicted_actions": [
                {"content": "请求更多推荐", "probability": 0.2502},
                {"content": "选定 The Lego Batman Movie", "probability": 0.2146},
                {"content": "接受推荐", "probability": 0.1738},
            ]
        },
        "What strategy will SEEKER use next?",
        [
            "A: Give feedback on the previously recommended movie or suggestion",
            "C: Asks for a recommendation or expresses their preference",
        ],
    )
    assert letter == "a"
    assert scores["a"] == 0.3884
    assert scores["c"] == 0.2502
    assert "选定 The Lego Batman Movie 0.2146" in reason
    assert "接受推荐 0.1738" in reason


def test_one_action_is_not_counted_under_two_options():
    class _Llm:
        def complete(self, messages, schema=None):
            return {
                "groups": [
                    {"letter": "a", "indexes": [0]},
                    {"letter": "c", "indexes": [0, 1]},
                ],
                "reason": "overlap",
            }

    _letter, scores, _reason = choose_by_scores(
        _Llm(),
        {
            "predicted_actions": [
                {"content": "请求确认", "probability": 0.8},
                {"content": "请求推荐", "probability": 0.2},
            ]
        },
        "What strategy will SEEKER use next?",
        [
            "A: Give feedback on the previously recommended movie or suggestion",
            "C: Asks for a recommendation or expresses their preference",
        ],
    )
    assert scores["a"] == 0.0
    assert scores["c"] == 0.2


def test_tie_is_rescored_without_letter_order():
    class _Llm:
        def complete(self, messages, schema=None):
            if schema and "scores" in json.dumps(schema):
                return {
                    "scores": [
                        {"letter": "a", "score": 0.4},
                        {"letter": "c", "score": 0.8},
                    ],
                    "reason": "recommendation is more direct",
                }
            return {
                "groups": [
                    {"letter": "a", "indexes": [0]},
                    {"letter": "c", "indexes": [1]},
                ],
                "reason": "same probability",
            }

    letter, scores, _reason = choose_by_scores(
        _Llm(),
        {
            "predicted_actions": [
                {"content": "询问偏好", "probability": 0.4},
                {"content": "推荐一部惊悚片", "probability": 0.4},
            ]
        },
        "What strategy will RECOMMENDER use next?",
        ["A: Ask for preference", "C: Provides a recommendation directly"],
    )
    assert letter == "c"
    assert scores["c"] > scores["a"]


def test_wrong_letter_can_be_checked_against_gold_only(tmp_path):
    from rectom_adapter.loader import RectomItem
    from rectom_adapter.runner import _finish

    item = RectomItem(
        index=0,
        dialogue_id="1",
        utterance_pos=1,
        utterance_context="SEEKER says: hi",
        question="What strategy will RECOMMENDER use next?",
        choices=(
            "A: Ask for preference",
            "C: Provides a recommendation directly",
        ),
        gold=("c",),
        focus_person="RECOMMENDER",
    )

    class _Llm:
        def complete(self, messages, schema=None):
            text = messages[-1]["content"]
            if "Accepted strategies:" in text:
                assert "Ask for preference" not in text
                return {"level": "direct", "reason": "The record says the next step is a recommendation."}
            return {
                "groups": [
                    {"letter": "a", "indexes": [0]},
                    {"letter": "c", "indexes": []},
                ],
                "reason": "asks",
            }

    row = _finish(
        item,
        {"predicted_actions": [{"content": "recommend a movie next", "probability": 0.7}]},
        _Llm(),
        tmp_path,
        "3_pred_rec",
        method="mwm",
    )
    assert row["selector_correct"] is False
    assert row["correct"] is True
    assert (tmp_path / "answer.json").is_file()
    assert (tmp_path / "report.txt").is_file()
