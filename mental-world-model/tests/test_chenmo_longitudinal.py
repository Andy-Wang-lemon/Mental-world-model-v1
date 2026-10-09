"""陈默 7 事件纵向：单实例增量预测+落地。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "experiments"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import chenmo_longitudinal_7 as mod  # noqa: E402


def test_remaining_tasks():
    assert mod.task_no(1, "predict") == 1
    assert mod.task_no(1, "land") == 2
    assert mod.task_no(7, "land") == 14
    assert mod.remaining_tasks(1, done=False, total_tasks=14) == 14
    assert mod.remaining_tasks(1, done=True, total_tasks=14) == 13
    assert mod.remaining_tasks(14, done=True, total_tasks=14) == 0
    bodies = mod.parse_event_file(_ROOT / "data" / "自建实验" / "纵向" / "陈默纵向7事件.txt")
    assert len(bodies) == 7
    assert mod.parse_focus_person(_ROOT / "data" / "自建实验" / "纵向" / "陈默纵向7事件.txt") == "陈默"
    assert "数据证明林可的判断部分正确" in bodies[0]
    assert "你怎么看" in bodies[6]


def test_parse_chengye_events_and_focus():
    path = _ROOT / "data" / "自建实验" / "纵向" / "程野纵向7事件.txt"
    bodies = mod.parse_event_file(path)
    assert len(bodies) == 7
    assert mod.parse_focus_person(path) == "程野"
    assert "今晚我们还庆祝吗" in bodies[0]
    assert "一起去上海" in bodies[6]


def test_event_count_follows_file(tmp_path):
    path = tmp_path / "two.txt"
    path.write_text(
        "分析对象：程野\n\n事件1：散会后苏妍追上程野。\n\n事件2：领导让程野配合推进。\n",
        encoding="utf-8",
    )
    runner = mod.ChenmoLongitudinalRunner(
        definitions_dir=_ROOT / "definitions",
        event_file=path,
        out_dir=tmp_path / "out",
        use_mock=True,
    )
    assert runner.agent == "程野"
    assert runner.total_events == 2
    assert runner.total_tasks == 4
    summary = runner.run()
    assert summary["agent"] == "程野"
    assert summary["total_events"] == 2
    assert summary["completed_steps"] == ["S01", "S02"]
    assert (tmp_path / "out" / "steps" / "S02_landed.json").is_file()
    assert not (tmp_path / "out" / "steps" / "S03_prediction.json").exists()


def test_focus_person_required(tmp_path):
    path = tmp_path / "no_agent.txt"
    path.write_text("事件1：有人开会。\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未找到分析对象"):
        mod.parse_focus_person(path)


def _runner(tmp_path, *, resume=False):
    return mod.ChenmoLongitudinalRunner(
        definitions_dir=_ROOT / "definitions",
        event_file=_ROOT / "data" / "自建实验" / "纵向" / "陈默纵向7事件.txt",
        out_dir=tmp_path,
        use_mock=True,
        resume=resume,
    )


class TestLongitudinalIncremental:
    def test_one_instance_lands_without_rewinding(self, tmp_path):
        runner = _runner(tmp_path)
        model_id = id(runner.model)
        runner.run_step(1)
        after1 = mod.continuity_brief(runner.model.continuity_state)
        assert after1["event_count"] == 1
        assert after1["mental_episode_count"] == 1
        runner.run_step(2)
        after2 = mod.continuity_brief(runner.model.continuity_state)
        assert id(runner.model) == model_id
        assert after2["event_count"] == 2
        assert after2["mental_episode_count"] == 2
        pred1 = json.loads(
            (tmp_path / "steps" / "S01_prediction.json").read_text(encoding="utf-8")
        )
        land1 = json.loads(
            (tmp_path / "steps" / "S01_landed.json").read_text(encoding="utf-8")
        )
        assert pred1["_longitudinal"]["phase"] == "prediction"
        assert land1["_longitudinal"]["phase"] == "landed"
        assert land1["_longitudinal"]["landed_action"]
        assert (tmp_path / "behavior_trajectory.txt").is_file()

    def test_refuse_overwrite_without_resume(self, tmp_path):
        _runner(tmp_path)
        with pytest.raises(FileExistsError, match="progress.json"):
            _runner(tmp_path, resume=False)

    def test_resume_continues_same_continuity(self, tmp_path):
        r1 = _runner(tmp_path)
        r1.run_step(1)
        brief1 = mod.continuity_brief(r1.model.continuity_state)
        r2 = _runner(tmp_path, resume=True)
        assert r2.completed == ["S01"]
        assert r2.model.continuity_state.event_count == brief1["event_count"]
        r2.run_step(1)
        r2.run_step(2)
        assert r2.model.continuity_state.event_count == 2
        assert "S01" in r2.completed and "S02" in r2.completed

    def test_empty_top1_aborts_without_writing_m(self, tmp_path, monkeypatch):
        runner = _runner(tmp_path)
        monkeypatch.setattr(
            mod,
            "top_aggregated_action",
            lambda result: ("（无行为预测）", 0.0, []),
        )
        with pytest.raises(mod.LongitudinalAbort, match="闭环中止"):
            runner.run_step(1)
        brief = mod.continuity_brief(runner.model.continuity_state)
        assert brief["event_count"] == 0
        assert brief["mental_episode_count"] == 0
        abort = json.loads((tmp_path / "abort.json").read_text(encoding="utf-8"))
        assert abort["step_id"] == "S01"
