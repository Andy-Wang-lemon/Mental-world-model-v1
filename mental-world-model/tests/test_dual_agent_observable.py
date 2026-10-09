"""双人闭环脚本层：观察者可见性过滤（不进引擎）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "experiments"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import dual_agent_closed_loop as mod  # noqa: E402


class TestHeuristicObservable:
    def test_only_prefixes_actor(self):
        proj = mod.heuristic_observable_projection(
            "陈默",
            "林可",
            "表面没有反应，但是内心非常生气。",
        )
        assert proj.observable_event.startswith("陈默")
        assert "内心" in proj.observable_event
        assert proj.source == "heuristic"

    def test_keeps_public_private_mention(self):
        proj = mod.heuristic_observable_projection(
            "林可",
            "陈默",
            "林可当众追问陈默为何私下补数据。",
        )
        assert "私下补数据" in proj.observable_event
        assert proj.observable_event.startswith("林可")

    def test_keeps_detail_not_support_template(self):
        proj = mod.heuristic_observable_projection(
            "陈默",
            "林可",
            "陈默主动提出解决方案，详细讲了分三步把实验设计补齐。",
        )
        assert "或提供支持" not in proj.observable_event
        assert "分三步" in proj.observable_event


class TestProjectObservableEvent:
    def test_no_llm_uses_heuristic(self):
        proj = mod.project_observable_event(
            llm=None,
            use_mock=True,
            actor="陈默",
            observer="林可",
            raw_action="点头答应了。",
        )
        assert proj.source == "heuristic"
        assert "点头答应" in proj.observable_event
        assert proj.observable_event.startswith("陈默")

    def test_llm_response_used(self):
        class _Fake:
            def complete(self, messages, schema=None):
                return {
                    "observable_event": "陈默很开心",
                    "dropped": ["回家改数据"],
                    "visibility": "partial",
                    "reason": "私下不可见",
                }

        proj = mod.project_observable_event(
            llm=_Fake(),
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="表面开心，回家改数据。",
        )
        assert proj.source == "llm"
        assert proj.observable_event == "陈默很开心。"
        assert proj.dropped == ["回家改数据"]
        assert proj.visibility == "partial"

    def test_empty_llm_falls_back(self):
        class _Empty:
            def complete(self, messages, schema=None):
                return {}

        proj = mod.project_observable_event(
            llm=_Empty(),
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="点头答应了。",
        )
        assert proj.source == "llm_empty_fallback"
        assert "点头答应" in proj.observable_event

    def test_inner_raw_does_not_leak_to_observer(self):
        class _Inner:
            def complete(self, messages, schema=None):
                return {
                    "observable_event": "陈默内心非常生气。",
                    "dropped": [],
                    "visibility": "visible",
                    "reason": "",
                }

        proj = mod.project_observable_event(
            llm=_Inner(),
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="表面没有反应，但是内心非常生气。",
        )
        assert "内心" not in proj.observable_event
        assert proj.visibility == "none"
        assert "可被林可察觉" in proj.observable_event
        assert proj.source == "fallback_none"

    def test_no_llm_inner_raw_becomes_none_visible(self):
        proj = mod.project_observable_event(
            llm=None,
            use_mock=True,
            actor="陈默",
            observer="林可",
            raw_action="表面没有反应，但是内心非常生气。",
        )
        assert "内心" not in proj.observable_event
        assert proj.source == "fallback_none"

    def test_llm_error_falls_back(self):
        class _Boom:
            def complete(self, messages, schema=None):
                raise RuntimeError("down")

        proj = mod.project_observable_event(
            llm=_Boom(),
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="点头答应了。",
        )
        assert proj.source == "heuristic_fallback"
        assert "点头答应" in proj.observable_event


class TestObservableHardChecks:
    def test_wrong_subject_falls_back_to_linke(self):
        class _Swap:
            def complete(self, messages, schema=None):
                return {
                    "observable_event": "陈默可能会主动提出解决方案或提供支持，以帮助林可完成任务。",
                    "dropped": [],
                    "visibility": "visible",
                    "reason": "",
                }

        proj = mod.project_observable_event(
            llm=_Swap(),
            use_mock=False,
            actor="林可",
            observer="陈默",
            raw_action="林可继续主导项目，寻求陈默的帮助，以确保项目顺利推进。",
        )
        assert proj.observable_event.startswith("林可")
        assert "陈默可能会" not in proj.observable_event
        assert "继续主导项目" in proj.observable_event
        assert "寻求陈默的帮助" in proj.observable_event
        assert proj.source == "llm_rejected_heuristic"

    def test_invented_data_task_rejected(self):
        class _Invent:
            def complete(self, messages, schema=None):
                return {
                    "observable_event": "陈默需要处理数据补全问题，以确保项目顺利推进。",
                    "dropped": [],
                    "visibility": "visible",
                    "reason": "林可的预测转化为陈默补数据",
                }

        proj = mod.project_observable_event(
            llm=_Invent(),
            use_mock=False,
            actor="林可",
            observer="陈默",
            raw_action="林可继续寻求陈默的帮助，以确保项目顺利推进。",
            scene="林可发现陈默曾私下补数据、未署名。陈默尚未表态。",
        )
        assert proj.observable_event.startswith("林可")
        assert "补数据" not in proj.observable_event
        assert "需要处理" not in proj.observable_event

    def test_jargon_fragment_retried_then_fallback(self):
        class _Jargon:
            def complete(self, messages, schema=None):
                return {
                    "observable_event": "陈默这符合他内心的无瞋业，即促成善行。",
                    "dropped": [],
                    "visibility": "visible",
                    "reason": "",
                }

        proj = mod.project_observable_event(
            llm=_Jargon(),
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="陈默当场点头答应。",
        )
        assert proj.observable_event.startswith("陈默")
        assert proj.source == "llm_rejected_heuristic"
        assert "点头答应" in proj.observable_event

    def test_observer_sees_rejected(self):
        class _Sees:
            def complete(self, messages, schema=None):
                return {
                    "observable_event": "林可看到陈默主动提出解决方案或提供支持，帮助完成任务。",
                    "dropped": [],
                    "visibility": "visible",
                    "reason": "公开场合",
                }

        proj = mod.project_observable_event(
            llm=_Sees(),
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="陈默可能会主动提出解决方案或提供支持，以帮助林可完成任务。",
        )
        assert not proj.observable_event.startswith("林可")
        assert "林可看到" not in proj.observable_event
        assert proj.observable_event.startswith("陈默")


class TestObservableLlmRetry:
    def test_retry_once_when_subject_wrong(self):
        class _Retry:
            def __init__(self):
                self.n = 0

            def complete(self, messages, schema=None):
                self.n += 1
                if self.n == 1:
                    return {
                        "observable_event": "林可看到陈默点头。",
                        "dropped": [],
                        "visibility": "visible",
                        "reason": "",
                    }
                return {
                    "observable_event": "陈默当场点头答应。",
                    "dropped": [],
                    "visibility": "visible",
                    "reason": "retry",
                }

        llm = _Retry()
        proj = mod.project_observable_event(
            llm=llm,
            use_mock=False,
            actor="陈默",
            observer="林可",
            raw_action="陈默当场点头答应。",
        )
        assert llm.n == 2
        assert proj.source == "llm_retry"
        assert proj.observable_event.startswith("陈默")
        assert "点头" in proj.observable_event


class TestSelectStepAction:
    def test_picks_first_usable(self):
        text, prob, meta = mod.select_step_action(
            "陈默",
            [
                {"text": "陈默当场拒绝继续帮忙。", "probability": 0.64},
                {"text": "陈默点头答应。", "probability": 0.36},
            ],
            step_id="S05_A3",
        )
        assert text == "陈默当场拒绝继续帮忙。"
        assert prob == 0.64
        assert meta["selected_rank"] == 0

    def test_skips_wrong_subject(self):
        text, prob, meta = mod.select_step_action(
            "林可",
            [
                {"text": "陈默可能会主动提出解决方案。", "probability": 0.7},
                {"text": "林可向陈默寻求帮助。", "probability": 0.3},
            ],
            step_id="S04_B2",
        )
        assert text == "林可向陈默寻求帮助。"
        assert prob == 0.3
        assert meta["skipped"][0]["reason"] == "wrong_subject"

    def test_all_unusable_aborts(self):
        with pytest.raises(mod.DualLoopAbort, match="闭环中止"):
            mod.select_step_action(
                "陈默",
                [{"text": "（无行为预测）", "probability": 1.0}],
                step_id="S05_A3",
            )


class TestLedgerKeepsRaw:
    def test_seal_keeps_inner_action(self):
        ledger = mod.AgentLedger(
            focus_person="陈默",
            history=["前序"],
            pending_final="林可来求助。",
        )
        ledger.seal_own_action("表面没反应，内心很生气")
        assert "内心很生气" in ledger.history[-1]
        assert ledger.pending_final is None

    def test_stimulus_is_observable_only(self):
        ledger = mod.AgentLedger(
            focus_person="林可",
            history=["前序"],
            pending_final=None,
        )
        ledger.set_stimulus("陈默没有明显反应。")
        assert "内心" not in (ledger.pending_final or "")
        assert "做出了如下行动" not in (ledger.pending_final or "")

    def test_sealed_event_keeps_pending(self):
        ledger = mod.AgentLedger(
            focus_person="陈默",
            history=["前序"],
            pending_final="林可来求助。",
        )
        sealed = ledger.sealed_event("点头答应")
        assert ledger.pending_final == "林可来求助。"
        assert "林可来求助" in sealed
        assert "点头答应" in sealed


class TestRoleContext:
    def test_keeps_last_three_only(self):
        ctx = mod.build_role_context("陈默", [f"事件正文{i}" for i in range(1, 6)])
        assert "分析对象：陈默" in ctx
        assert "[事件3]" in ctx
        assert "[事件5]" in ctx
        assert "[事件1]" not in ctx
        assert "[事件2]" not in ctx

    def test_truncates_long_body(self):
        body = "甲" * 700
        ctx = mod.build_role_context("林可", [body])
        assert "…" in ctx
        line = [ln for ln in ctx.splitlines() if ln.startswith("[事件1]")][0]
        assert len(line) < 700 + 20
