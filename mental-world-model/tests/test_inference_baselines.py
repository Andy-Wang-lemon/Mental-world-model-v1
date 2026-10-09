"""已评条目、ANSWER 解析、两阶段提示，以及结果目录。"""

from __future__ import annotations

import json

from inference_baselines.items import load_split
from inference_baselines.methods import parse_state_chain, run_cot, run_simtom, run_timetom
from inference_baselines.parse_answer import parse_answer
from inference_baselines.prompts import simtom_stage1_messages, timetom_stage1_messages
from inference_baselines.runner import result_dir, run_items


def test_split_matches_previous_items():
    dyntom = load_split("dyntom")
    rectom = load_split("rectom")
    assert len(dyntom) == 142
    assert dyntom[0].id == "trial497__type_d_how_1"
    assert dyntom[0].target_character == "Henry Lozano"
    assert dyntom[0].gold == ("g",)
    assert len(dyntom[0].events) == 5
    assert dyntom[71].id == "trial235__type_d_how_1"
    assert dyntom[71].target_character == "Sara Manuelito"
    assert len(rectom) == 40
    assert rectom[0].id == "3_pred_rec__0000_d474_u1"
    assert rectom[0].target_character == "RECOMMENDER"
    assert rectom[0].gold == ("c",)
    assert rectom[20].source == "4_pred_seeker"
    assert rectom[20].target_character == "SEEKER"
    assert "true answer" not in dyntom[0].context


def test_answer_parser_uses_only_answer_line():
    choices = ["A: pay for the chips", "B: report the moldy chips"]
    text = "The closest wording is pay for the chips.\nANSWER: B\nANSWER: A"
    assert parse_answer(text, choices) == "a"
    assert parse_answer("I think she will pay for the chips.", choices) == ""
    assert parse_answer("ANSWER: Z", choices) == ""


def test_first_stage_does_not_see_choices(tmp_path):
    item = load_split("rectom")[0]
    stage1 = simtom_stage1_messages(item)[0]["content"]
    tracked = timetom_stage1_messages(item)[0]["content"]
    assert "A:" not in stage1
    assert item.question not in stage1
    assert "A:" not in tracked
    assert "TimeToM-style" in tracked
    assert item.context.splitlines()[0] in stage1

    class _Client:
        def complete_text(self, messages):
            content = messages[-1]["content"]
            if "subjective perspective" in content:
                assert "Choices:" not in content
                return "Mary only knows she is holding the bag."
            if "Temporal state chain" in content or "tracking" in content:
                return '[{"time": 1, "observed": "hello", "belief_before": "", "belief_update": "", "belief_after": "", "emotion": "", "intention": ""}]'
            assert "Story or dialogue:" in content or "Information available" in content or "Temporal state chain" in content
            return "step by step\nANSWER: C"

    row = run_simtom(_Client(), item)
    assert row["prediction"] == "c"
    assert row["correct"] is True
    assert "Mary only knows" in row["perspective"]
    chain = run_timetom(_Client(), item)
    assert chain["implementation"] == "TimeToM-style"
    assert chain["temporal_state_chain"][0]["time"] == 1
    saved = tmp_path / "results"
    failed = run_items([item], method="cot", client=_Client(), out_dir=result_dir(saved, "rectom", "cot"), resume=False)
    assert failed == 0
    path = saved / "rectom" / "cot" / f"{item.id}.json"
    assert path.is_file()
    written = json.loads(path.read_text())
    assert written["method"] == "cot"
    assert written["raw_response"].endswith("ANSWER: C")
    again = run_items([item], method="cot", client=_Client(), out_dir=path.parent, resume=False)
    assert again == 0
    assert json.loads(path.read_text())["raw_response"] == written["raw_response"]


def test_malformed_chain_is_kept_raw():
    assert parse_state_chain("not json") is None
    assert parse_state_chain('```json\n[{"time": 2}]\n```') == [{"time": 2}]


def test_result_dir_uses_reports_method_folder(tmp_path):
    out = result_dir(tmp_path / "reports", "dyntom", "cot")
    assert out == (tmp_path / "reports" / "dyntom" / "cot").resolve()
    blocked = tmp_path / "reports" / "dyntom" / "dyntom mwm"
    try:
        result_dir(blocked, "dyntom", "cot")
    except SystemExit as error:
        assert "不能写进" in str(error)
    else:
        raise AssertionError("已有模型目录不应接受新结果")
