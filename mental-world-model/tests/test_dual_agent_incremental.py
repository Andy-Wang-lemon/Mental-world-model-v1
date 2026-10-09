"""双人闭环：增量 H/M（前序只算一次）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "experiments"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import dual_agent_closed_loop as mod  # noqa: E402


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    out = tmp_path_factory.mktemp("dual_inc")
    return mod.DualLoopRunner(
        definitions_dir=_ROOT / "definitions",
        chenmo_file=_ROOT / "data" / "陈默事件（负面版）.txt",
        linke_file=_ROOT / "data" / "林可负面事件.txt",
        out_dir=out,
        use_mock=True,
        rebuild_prologue=False,
    )


class TestIncrementalContinuity:
    def test_warmup_once_then_predict_adds_one_episode(self, runner):
        a = runner.ingest_prologue("陈默")
        b = runner.ingest_prologue("林可")
        assert a["mental_episodes"] == 7
        assert b["mental_episodes"] == 8
        # 再预热不得重复加
        a2 = runner.ingest_prologue("陈默")
        assert a2["mental_episodes"] == 7

        before = a["mental_episodes"]
        rec = runner.predict(
            "陈默",
            step_id="S01_A1",
            give_to_other=True,
            note="增量冒烟",
        )
        after = mod.continuity_brief(runner.model_a.continuity_state)
        # 预测不落库，接入行动 +1
        assert after["mental_episodes"] == before + 1
        assert rec.observable_event
        assert runner.agent_a.pending_final is None
        assert "随后的行动" in runner.agent_a.history[-1]


class TestAbortOnEmptyAction:
    def test_placeholder_top1_aborts_without_writing_m(self, tmp_path, monkeypatch):
        runner = mod.DualLoopRunner(
            definitions_dir=_ROOT / "definitions",
            chenmo_file=_ROOT / "data" / "陈默事件（负面版）.txt",
            linke_file=_ROOT / "data" / "林可负面事件.txt",
            out_dir=tmp_path,
            use_mock=True,
            rebuild_prologue=False,
        )
        before = runner.ingest_prologue("陈默")
        monkeypatch.setattr(
            mod,
            "top_behavior",
            lambda result: ("（无行为预测）", 0.0, []),
        )
        with pytest.raises(mod.DualLoopAbort, match="闭环中止"):
            runner.predict("陈默", step_id="S01_A1", give_to_other=True)
        after = mod.continuity_brief(runner.model_a.continuity_state)
        assert after["mental_episodes"] == before["mental_episodes"]
        assert runner.agent_a.pending_final is not None
        abort_path = tmp_path / "abort.json"
        assert abort_path.is_file()
        progress_path = tmp_path / "progress.json"
        assert progress_path.is_file()
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        assert "S01_A1" not in progress.get("completed_steps", [])
        assert progress["agent_a"]["prologue_warmed"] is True


class TestResume:
    def _make(self, tmp_path, *, resume=False):
        return mod.DualLoopRunner(
            definitions_dir=_ROOT / "definitions",
            chenmo_file=_ROOT / "data" / "陈默事件（负面版）.txt",
            linke_file=_ROOT / "data" / "林可负面事件.txt",
            out_dir=tmp_path,
            use_mock=True,
            rebuild_prologue=False,
            resume=resume,
        )

    def test_refuse_overwrite_without_resume(self, tmp_path):
        self._make(tmp_path)
        with pytest.raises(FileExistsError, match="progress.json"):
            self._make(tmp_path, resume=False)

    def test_rebuild_and_resume_conflict(self, tmp_path):
        with pytest.raises(ValueError, match="不能同时"):
            mod.DualLoopRunner(
                definitions_dir=_ROOT / "definitions",
                chenmo_file=_ROOT / "data" / "陈默事件（负面版）.txt",
                linke_file=_ROOT / "data" / "林可负面事件.txt",
                out_dir=tmp_path,
                use_mock=True,
                rebuild_prologue=True,
                resume=True,
            )

    def test_resume_rejects_empty_continuity(self, tmp_path):
        r1 = self._make(tmp_path)
        r1.ingest_prologue("陈默")
        r1.ingest_prologue("林可")
        runtime = tmp_path / "runtime" / "continuity_chenmo.json"
        runtime.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="continuity"):
            self._make(tmp_path, resume=True)

    def test_ingest_skips_when_event_count_warm(self, tmp_path, monkeypatch):
        runner = self._make(tmp_path)
        runner.model_a.continuity_state.event_count = 7

        def boom(*_a, **_k):
            raise AssertionError("已达预热条数，不应再跑前序")

        monkeypatch.setattr(
            runner.model_a, "evaluate_text_with_cognition_branches", boom
        )
        runner.ingest_prologue("陈默")
        assert runner.agent_a.prologue_warmed is True

    def test_resume_keeps_h_and_skips_completed(self, tmp_path):
        r1 = self._make(tmp_path)
        r1.ingest_prologue("陈默")
        r1.ingest_prologue("林可")
        rec = r1.predict("陈默", step_id="S01_A1", give_to_other=True)
        ep = int(r1.model_a.continuity_state.event_count or 0)
        frames = len(r1.model_a.continuity_state.cognitive_frames or [])
        assert "S01_A1" in r1._completed_steps
        assert r1.agent_b.pending_final == rec.observable_event

        r2 = self._make(tmp_path, resume=True)
        assert int(r2.model_a.continuity_state.event_count or 0) == ep
        assert len(r2.model_a.continuity_state.cognitive_frames or []) == frames
        assert r2.agent_a.prologue_warmed is True
        assert r2.agent_b.pending_final
        assert "S01_A1" in r2._completed_steps

        called: list[str] = []

        def wrapped(who, *, step_id, give_to_other, note=""):
            called.append(step_id)
            if step_id == "S01_A1":
                raise AssertionError("不应重跑已完成步骤")
            raise mod.DualLoopAbort("stop after skip")

        r2.predict = wrapped  # type: ignore[method-assign]
        with pytest.raises(mod.DualLoopAbort, match="stop after skip"):
            r2.run()
        assert "S01_A1" not in called
        assert called[0] == "S02_B1"

    def test_landed_step_skips_reevaluate(self, tmp_path, monkeypatch):
        runner = self._make(tmp_path)
        runner.ingest_prologue("陈默")
        runner.agent_a.pending_final = "林可来求助。"
        runner._landed = {
            "step_id": "S01_A1",
            "who": "陈默",
            "action": "陈默当场点头答应。",
            "prob": 0.4,
            "stimulus": "林可来求助。",
            "yuanwen": "分析对象：陈默",
            "give_to_other": True,
            "note": "",
            "top_k": [{"text": "陈默当场点头答应。", "probability": 0.4}],
            "extra_meta": {},
        }
        runner.agent_a.pending_final = None

        def boom(*_a, **_k):
            raise AssertionError("已落地步骤不应再评估")

        monkeypatch.setattr(
            runner.model_a, "evaluate_text_with_cognition_branches", boom
        )
        rec = runner.predict("陈默", step_id="S01_A1", give_to_other=True)
        assert rec.action == "陈默当场点头答应。"
        assert "S01_A1" in runner._completed_steps
        assert runner._landed is None
