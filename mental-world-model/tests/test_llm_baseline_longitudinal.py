"""LLM 原始 baseline：纯文本历史纵向 rollout。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "experiments"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import chenmo_longitudinal_7 as mwm  # noqa: E402
import llm_baseline_longitudinal_rollout as mod  # noqa: E402


def test_parse_matches_mwm_event_file():
    path = _ROOT / "data" / "自建实验" / "纵向" / "陈默纵向7事件.txt"
    assert mod.parse_event_file(path) == mwm.parse_event_file(path)
    bodies = mod.parse_event_file(path)
    assert len(bodies) == 7
    assert mod.parse_focus_person(path) == mwm.parse_focus_person(path) == "陈默"


def test_parse_chengye_events_and_focus():
    path = _ROOT / "data" / "自建实验" / "纵向" / "程野纵向7事件.txt"
    assert mod.parse_event_file(path) == mwm.parse_event_file(path)
    assert len(mod.parse_event_file(path)) == 7
    assert mod.parse_focus_person(path) == "程野"


def test_e1_prompt_has_no_history_block():
    event = "林可拿出数据。"
    prompt = mod.build_prompt("", event, agent="陈默")
    assert "此前已经发生" not in prompt
    assert "当前事件：" in prompt
    assert event in prompt
    assert "分析对象：陈默" in prompt
    assert "不解释原因" in prompt


def test_history_append_and_e2_prompt_only_keeps_top1():
    event1 = "公司招聘林可。"
    event2 = "林可地位提升。"
    top1 = "要求林可补充数据"
    other = "拒绝讨论"
    history = mod.append_history("", 1, event1, top1, agent="陈默")
    assert "事件1：" in history
    assert top1 in history
    assert "陈默行动：" in history
    prompt = mod.build_prompt(history, event2, agent="陈默")
    assert "此前已经发生" in prompt
    assert event1 in prompt
    assert top1 in prompt
    assert other not in prompt
    assert event2 in prompt
    assert prompt.index("当前事件：") > prompt.index(top1)


def test_normalize_renormalizes_and_caps_at_three():
    rows = mod.normalize_behaviors(
        {
            "behaviors": [
                {"text": "A", "probability": 2},
                {"text": "B", "probability": 1},
                {"text": "C", "probability": 1},
                {"text": "D", "probability": 8},
                {"text": "", "probability": 9},
                {"reason": "应被忽略"},
            ]
        }
    )
    assert [row["text"] for row in rows] == ["D", "A", "B"]
    assert abs(sum(row["probability"] for row in rows) - 1.0) < 1e-9
    assert "reason" not in rows[0]


def test_normalize_rejects_empty_or_zero():
    with pytest.raises(mod.BaselineAbort, match="无可用行为"):
        mod.normalize_behaviors({"behaviors": []})
    with pytest.raises(mod.BaselineAbort, match="概率全为 0"):
        mod.normalize_behaviors({"behaviors": [{"text": "A", "probability": 0}]})


def test_script_does_not_import_mwm_engine():
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "from mental_world_model.model" not in src
    assert "import MentalWorldModel" not in src
    assert "mental_world_model.engine.cognition" not in src
    assert "mental_world_model.engine.continuity" not in src
    assert "mental_world_model.engine.perception" not in src


def _runner(tmp_path, *, resume=False, client=None):
    return mod.LlmBaselineRollout(
        event_file=_ROOT / "data" / "自建实验" / "纵向" / "陈默纵向7事件.txt",
        out_dir=tmp_path,
        use_mock=True,
        resume=resume,
        client=client,
    )


class TestBaselineRollout:
    def test_full_mock_writes_report(self, tmp_path):
        runner = _runner(tmp_path)
        summary = runner.run()
        assert summary["protocol"] == "llm_raw_history_open_rollout"
        assert len(summary["completed_steps"]) == 7
        for i in range(1, 8):
            step_path = tmp_path / "steps" / f"S{i:02d}.json"
            assert step_path.is_file()
        traj = (tmp_path / "behavior_trajectory.txt").read_text(encoding="utf-8")
        assert traj.startswith("E1 → mock行动1甲")
        assert "E7 → mock行动7甲" in traj
        history = (tmp_path / "history_final.txt").read_text(encoding="utf-8")
        assert history.count("陈默行动：") == 7
        assert "mock行动1甲" in history
        assert "mock行动1乙" not in history
        s02 = json.loads((tmp_path / "steps" / "S02.json").read_text(encoding="utf-8"))
        assert list(s02.keys()) == [
            "event_id",
            "history_before",
            "current_event",
            "behaviors",
            "top1",
            "history_after",
        ]
        assert s02["event_id"] == "S02"
        assert s02["top1"] == "mock行动2甲"
        assert "mock行动1甲" in s02["history_before"]
        assert "mock行动1乙" not in s02["history_before"]
        assert s02["current_event"]
        assert not (tmp_path / "prompts").exists()
        assert not (tmp_path / "steps" / "S01_prediction.json").exists()
        assert not (tmp_path / "steps" / "S01_landed.json").exists()

    def test_refuse_overwrite_without_resume(self, tmp_path):
        _runner(tmp_path)
        with pytest.raises(FileExistsError, match="progress.json"):
            _runner(tmp_path, resume=False)

    def test_resume_continues_history(self, tmp_path):
        r1 = _runner(tmp_path)
        r1.run_step(1)
        hist1 = r1.history_text
        r2 = _runner(tmp_path, resume=True)
        assert r2.completed == ["S01"]
        assert r2.history_text == hist1
        r2.run_step(1)
        r2.run_step(2)
        assert r2.completed == ["S01", "S02"]
        assert "mock行动1甲" in r2.history_text
        assert "mock行动2甲" in r2.history_text

    def test_empty_top1_aborts_without_appending_history(self, tmp_path):
        class EmptyClient:
            def complete(self, messages, *, schema=None):
                return {"behaviors": []}

        runner = _runner(tmp_path, client=EmptyClient())
        with pytest.raises(mod.BaselineAbort, match="无可用行为"):
            runner.run_step(1)
        assert runner.history_text == ""
        assert runner.completed == []
        abort = json.loads((tmp_path / "abort.json").read_text(encoding="utf-8"))
        assert abort["step_id"] == "S01"
        assert not (tmp_path / "steps" / "S01.json").exists()


def test_event_count_and_agent_follow_file(tmp_path):
    path = tmp_path / "two.txt"
    path.write_text(
        "分析对象：程野\n\n事件1：散会后苏妍追上程野。\n\n事件2：领导让程野配合推进。\n",
        encoding="utf-8",
    )
    runner = mod.LlmBaselineRollout(
        event_file=path,
        out_dir=tmp_path / "out",
        use_mock=True,
    )
    assert runner.agent == "程野"
    assert runner.total_events == 2
    summary = runner.run()
    assert summary["agent"] == "程野"
    assert summary["total_events"] == 2
    assert summary["completed_steps"] == ["S01", "S02"]
    history = (tmp_path / "out" / "history_final.txt").read_text(encoding="utf-8")
    assert history.count("程野行动：") == 2
    assert "陈默" not in history
    s01 = json.loads((tmp_path / "out" / "steps" / "S01.json").read_text(encoding="utf-8"))
    assert "分析对象：程野" in mod.build_prompt(
        s01["history_before"], s01["current_event"], agent="程野"
    )
    assert not (tmp_path / "out" / "steps" / "S03.json").exists()
