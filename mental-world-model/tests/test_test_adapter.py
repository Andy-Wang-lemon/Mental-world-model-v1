"""测试适配层（批量跑分）测试。"""

from pathlib import Path

from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel
from mwm_test_adapter.runner import TestAdapterRunner as Runner
from mwm_test_adapter.runner import TestInputRow as InputRow
from mwm_test_adapter.runner import build_core_text, iter_input_rows


def _mock_handler(messages, schema):
    props = (schema or {}).get("properties", {})
    if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
        return {"suoyuan": {"text": "mock"}}
    if "relevance" in props:
        return {"relevance": 0.1}
    if "has_explicit_xiang" in props:
        return {
            "has_explicit_xiang": False,
            "xiang": {"text": ""},
            "has_explicit_si": False,
            "si": {"text": ""},
        }
    if "activations" in props:
        return {"activations": {}}
    if schema and "winner" in props:
        return {"winner": "none"}
    if "predicted_behavior" in props:
        return {
            "predicted_behavior": "mock预测行为",
            "prediction_reason": "mock参考心所：信",
        }
    return {"score": 0.0}


class TestRunner:
    def test_build_core_text(self):
        row = InputRow(
            story="Story text.",
            question="Does Emma think the duck is on the table?",
            character="Emma",
        )
        core = build_core_text(row)
        assert "Story text." in core
        assert "Question:" in core
        assert "Character: Emma" in core

    def test_run_file_mock(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        sample = root / "data" / "fixtures" / "benchmark_samples" / "opentom_demo.jsonl"
        assert len(list(iter_input_rows(sample))) == 3

        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )
        runner = Runner(model, MockLlmClient(handler=_mock_handler))
        summary = runner.run_file(sample, limit=1)
        assert summary.succeeded == 1
        result = summary.results[0]
        assert result.scores_final
        assert len(result.scores_final) == 40
        assert "Question:" in result.text
        assert result.error is None

    def test_failed_sample_records_error(self, tmp_path, monkeypatch):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        def boom(*_args, **_kwargs):
            raise RuntimeError("extraction boom")

        monkeypatch.setattr(model, "evaluate_from_yuanwen", boom)
        runner = Runner(model, MockLlmClient())
        row = next(iter_input_rows(root / "data/fixtures/benchmark_samples/opentom_demo.jsonl"))
        result = runner.run_sample(row)
        assert result.error is not None
        assert "extraction boom" in result.error
        assert result.scores_final == {}
