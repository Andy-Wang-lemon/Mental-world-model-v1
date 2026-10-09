"""LLM 调用统计测试。"""

from pathlib import Path

from mental_world_model.engine.scoring.prompts import (
    is_group_suoyuan_stage_content,
    is_xiandai_stage_content,
    is_ye_stage_content,
)
from mental_world_model.engine.scoring.suoyuan_groups import is_group_suoyuan_schema
from mental_world_model.llm.mock_client import MockLlmClient, _group_suoyuan_mock_response
from mental_world_model.llm.tracing_client import TracingLlmClient, ensure_tracing_client
from mental_world_model.llm.usage import llm_call_context, llm_evaluation_scope
from mental_world_model.llm.message_utils import user_message_content
from mental_world_model.model import MentalWorldModel


class TestTracingClient:
    def test_records_stage_and_duration(self):
        inner = MockLlmClient(default_response={"score": 1.0})
        client = TracingLlmClient(inner)
        with llm_evaluation_scope() as tracker:
            with llm_call_context(stage="test.stage"):
                client.complete([{"role": "user", "content": "x"}])
        summary = tracker.summary()
        assert summary.total_calls == 1
        assert summary.by_stage["test.stage"].count == 1
        assert summary.by_stage["test.stage"].duration_ms >= 0
        assert summary.wall_clock_ms is not None

    def test_ensure_tracing_idempotent(self):
        inner = MockLlmClient()
        wrapped = ensure_tracing_client(inner)
        assert isinstance(wrapped, TracingLlmClient)
        assert ensure_tracing_client(wrapped) is wrapped

    def test_ensure_tracing_keeps_prefix_outer(self):
        from mental_world_model.llm.prefixed_client import StagePrefixLlmClient
        from mental_world_model.llm.usage import current_llm_stage, llm_evaluation_scope

        inner = MockLlmClient(default_response={"ok": True})
        traced = ensure_tracing_client(inner)
        prefixed = StagePrefixLlmClient(traced, "chenmo")
        assert ensure_tracing_client(prefixed) is prefixed
        with llm_evaluation_scope() as tracker:
            with llm_call_context(stage="perception.objective_event"):
                prefixed.complete([{"role": "user", "content": "x"}])
        stages = list(tracker.summary().by_stage)
        assert "chenmo.perception.objective_event" in stages
        assert current_llm_stage() != "chenmo.perception.objective_event"

    def test_progress_logs_when_enabled(self, monkeypatch, capsys):
        from mental_world_model.llm import tracing_client as tc

        monkeypatch.setenv("MWM_LLM_PROGRESS", "1")
        tc.reset_progress_for_tests()
        inner = MockLlmClient(default_response={"score": 1.0})
        client = TracingLlmClient(inner)
        with llm_evaluation_scope() as tracker:
            with llm_call_context(stage="perception.objective_event"):
                client.complete([{"role": "user", "content": "x"}])
            with llm_call_context(stage="perception.objective_event"):
                client.complete([{"role": "user", "content": "x"}])
        err = capsys.readouterr().err
        # 默认里程碑：同阶段只打一行，不逐次 LLM #
        assert err.count("感知·客观所缘") == 1
        assert "LLM #" not in err
        assert tracker.total_calls_so_far() == 2

    def test_progress_all_logs_each_call(self, monkeypatch, capsys):
        from mental_world_model.llm import tracing_client as tc

        monkeypatch.setenv("MWM_LLM_PROGRESS", "all")
        tc.reset_progress_for_tests()
        inner = MockLlmClient(default_response={"score": 1.0})
        client = TracingLlmClient(inner)
        with llm_evaluation_scope() as tracker:
            with llm_call_context(stage="test.progress"):
                client.complete([{"role": "user", "content": "x"}])
        err = capsys.readouterr().err
        assert "LLM #" in err
        assert "test.progress" in err
        assert tracker.total_calls_so_far() == 1


class TestEvaluationLlmUsage:
    def test_evaluate_from_yuanwen_populates_report(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        def handler(messages, schema):
            content = user_message_content(messages)
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "所缘"}}
            if "has_explicit_xiang" in props:
                return {
                    "has_explicit_xiang": False,
                    "xiang": {"text": ""},
                    "has_explicit_si": False,
                    "si": {"text": ""},
                }
            if "activations" in props:
                return {"activations": {}}
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                return _group_suoyuan_mock_response(messages, {"score": 0.0})
            if is_ye_stage_content(content) or is_xiandai_stage_content(content):
                return {"score": 0.0}
            if schema and "winner" in props:
                return {"winner": "none", "reason": "mock"}
            if "predicted_behavior" in props:
                return {
                    "predicted_behavior": "mock预测行为",
                    "prediction_reason": "mock参考心所：信",
                }
            return {"score": 0.0}

        llm = ensure_tracing_client(MockLlmClient(handler=handler))
        result = model.evaluate_from_yuanwen("测试原文", llm)
        assert result.llm_usage is not None
        assert result.llm_usage.total_calls > 0
        # 默认多分支路径：感知层；关闭分支时为 extraction.suoyuan
        assert (
            "perception.objective_event" in result.llm_usage.by_stage
            or "perception.objective_facts" in result.llm_usage.by_stage
            or "extraction.suoyuan" in result.llm_usage.by_stage
        )
        assert "scoring.suoyuan_batch" in result.llm_usage.by_stage

        report = result.to_report_dict()
        assert "llm_usage" in report
        assert report["llm_usage"]["total_calls"] == result.llm_usage.total_calls
        assert "wall_clock_ms" in report["llm_usage"]
