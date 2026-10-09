"""ScoreBreakdown 与可解释性测试。"""

from pathlib import Path

from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.scoring.pipeline import XinsuoScoringPipeline
from mental_world_model.engine.scoring.prompts import (
    is_group_suoyuan_stage_content,
    is_xiandai_stage_content,
    is_ye_stage_content,
)
from mental_world_model.engine.scoring.suoyuan_groups import is_group_suoyuan_schema
from mental_world_model.engine.scoring.weights import (
    DEFAULT_COLUMN_WEIGHTS,
    ScoringWeights,
    compute_score_breakdown,
)
from mental_world_model.llm.mock_client import MockLlmClient, _group_suoyuan_mock_response
from mental_world_model.llm.message_utils import user_message_content
from mental_world_model.model import MentalWorldModel
from tests.test_extraction import make_test_scoring_context


def _all_yuanwen_extracted() -> ExtractedContext:
    return ExtractedContext(
        yuanwen="示例原文",
        suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
        xiang=ExtractedField(text="想", source="原文", confidence=1.0),
        si=ExtractedField(text="思", source="原文", confidence=1.0),
    )


class TestScoreBreakdown:
    def test_compute_score_breakdown_fields(self):
        weights = ScoringWeights(column_weights=dict(DEFAULT_COLUMN_WEIGHTS))
        column_scores = {
            "suoyuan": 0.8,
            "shengqi_nian": 0.8,
            "ye": 0.6,
            "xiandai_xinli": 0.6,
        }
        breakdown = compute_score_breakdown(
            "信",
            column_scores,
            weights,
            _all_yuanwen_extracted(),
        )
        assert breakdown.final_score == breakdown.column_composite
        assert breakdown.source_modifiers["suoyuan"] == 1.0
        assert breakdown.source_modifiers["xiandai_xinli"] == 1.0
        assert abs(sum(breakdown.normalized_weights.values()) - 1.0) < 1e-9
        assert "所缘境" in breakdown.explain()
        assert breakdown.to_dict()["final_score"] == breakdown.final_score

    def test_pipeline_attaches_breakdown(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        xinsuo = model.get_xinsuo("信")

        def handler(messages, schema):
            content = user_message_content(messages)
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                return _group_suoyuan_mock_response(messages, {"score": 0.8})
            if is_ye_stage_content(content):
                return {"score": 0.6}
            if is_xiandai_stage_content(content):
                return {"score": 0.6}
            return {"score": 0.8}

        pipeline = XinsuoScoringPipeline(MockLlmClient(handler=handler))
        result = pipeline.run([xinsuo], make_test_scoring_context())[0]
        assert result.score_breakdown is not None
        assert result.score_breakdown.final_score == result.final_score
        assert len(result.stage_results) == 4
        assert result.explain_score() == result.score_breakdown.explain()

    def test_gate_exit_has_no_breakdown(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        xinsuo = model.get_xinsuo("信")
        pipeline = XinsuoScoringPipeline(
            MockLlmClient(default_response={"score": 0.1})
        )
        result = pipeline.run([xinsuo], make_test_scoring_context())[0]
        assert result.score_breakdown is None
        assert "均低于门控阈值" in result.explain_score()


class TestEvaluationReport:
    def test_to_report_dict_contains_explanations(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "所缘"}}
            if "relevance" in props:
                return {"relevance": 0.9}
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
                return {"winner": "none", "reason": "mock"}
            if "predicted_behavior" in props:
                return {
                    "predicted_behavior": "mock预测行为",
                    "prediction_reason": "mock参考心所：信",
                }
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(
                user_message_content(messages)
            ):
                return _group_suoyuan_mock_response(messages, {"score": 0.0})
            return {"score": 0.0}

        result = model.evaluate_from_yuanwen(
            "测试原文",
            MockLlmClient(handler=handler),
        )
        report = result.to_report_dict(include_explanations=True)
        assert "score_explanations" in report
        assert "score_breakdowns" in report
        assert "explanation_all" in report
        assert len(result.explain_all_scores()) > 0
