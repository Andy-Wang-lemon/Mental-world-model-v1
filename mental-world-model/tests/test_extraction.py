"""测试所缘境/想/思提取逻辑。"""

from pathlib import Path

from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.loader.definition_loader import DefinitionLoader
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.weights import compute_final_score, ScoringWeights
from mental_world_model.engine.extraction.extractor import ContextExtractor
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel
from mental_world_model.core.relations.relation import Relation
from mental_world_model.engine.parser.definition_parser import DefinitionParser
from mental_world_model.engine.validator.definition_validator import DefinitionValidator


class TestSuoyuanXiangSiDefinitionsLoad:
    def test_load_normalized_json(self):
        root = Path(__file__).resolve().parents[1]
        loader = DefinitionLoader()
        data = loader.load_suoyuan_xiang_si_definitions(root / "definitions")
        assert "suoyuan_jing" in data
        assert "xiang" in data
        assert "si" in data
        assert "summary" in data


class TestContextExtractor:
    def test_extract_with_mock_llm(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "xiang" not in props and "has_explicit_xiang" not in props:
                return {"suoyuan": {"text": "正确价值"}}
            if "relevance" in props:
                return {"relevance": 0.9}
            if "has_explicit_xiang" in props:
                return {
                    "has_explicit_xiang": False,
                    "xiang": {"text": ""},
                    "has_explicit_si": False,
                    "si": {"text": ""},
                }
            return {"score": 0.0}

        extracted = model.extract("示例原文", MockLlmClient(handler=handler))
        assert extracted.suoyuan.confidence == 1.0
        assert extracted.routed_hint_names == []
        assert extracted.scoring_path == "implicit"
        assert "隐式" in extracted.xiang.text


class TestSourceModifiersInScoring:
    def test_renormalized_weights_preserve_scale_when_all_scores_one(self):
        """全列 1.0 时，归一化后合成仍为 1.0，不因来源调节而整体压低。"""
        weights = ScoringWeights()
        column_scores = {
            "suoyuan": 1.0,
            "shengqi_nian": 1.0,
            "ye": 1.0,
            "xiandai_xinli": 1.0,
        }
        all_yuanwen = ExtractedContext(
            yuanwen="示例原文",
            suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
            xiang=ExtractedField(text="想", source="原文", confidence=1.0),
            si=ExtractedField(text="思", source="原文", confidence=1.0),
        )
        inferred_xiang = ExtractedContext(
            yuanwen="示例原文",
            suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
            xiang=ExtractedField(text="想", source="推理", confidence=0.5),
            si=ExtractedField(text="思", source="原文", confidence=1.0),
        )
        full = compute_final_score(
            "信",
            column_scores,
            weights,
            all_yuanwen,
        )
        renormalized = compute_final_score(
            "信",
            column_scores,
            weights,
            inferred_xiang,
        )
        assert full == 1.0
        assert renormalized == 1.0


class TestMentalWorldModelLoadValidation:
    def test_load_validates_definitions(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        assert len(model.xinsuo_list) == 40
        assert len(model.relations) == 40

    def test_load_raises_on_invalid_relation_endpoint(self):
        root = Path(__file__).resolve().parents[1]
        loader = DefinitionLoader()
        parser = DefinitionParser()
        xinsuo_list = parser.parse_xinsuo(
            loader.load_json(root / "definitions/normalized/xinsuo.json")
        )
        relations = parser.parse_relation(
            loader.load_json(root / "definitions/normalized/xinsuo_duizhi.json")
        )
        relations[0] = Relation(
            source=relations[0].source,
            target="不存在的心所",
            description=relations[0].description,
            relation_type=relations[0].relation_type,
            is_direct=relations[0].is_direct,
        )

        validator = DefinitionValidator()
        result = validator.validate_relation_list(
            relations,
            xinsuo_names={x.name for x in xinsuo_list},
        )
        assert result.valid is False


class TestEvaluateFromYuanwen:
    def test_end_to_end(self, tmp_path, monkeypatch):
        import mental_world_model.engine.perception.config as perception_config

        monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        call_count = {"n": 0}

        def handler(messages, schema):
            call_count["n"] += 1
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
            return {"score": 0.0}

        result = model.evaluate_from_yuanwen("测试原文", MockLlmClient(handler=handler))
        assert result.extracted is not None
        assert result.extracted.suoyuan.confidence == 1.0
        assert len(result.stage_results) == 40
        assert call_count["n"] > 0
        assert result.seed_weights_after is not None


def _make_context(extracted: ExtractedContext) -> ScoringContext:
    return ScoringContext.from_extracted(extracted)


def make_test_scoring_context() -> ScoringContext:
    """供其他测试使用的评分上下文构造。"""
    extracted = ExtractedContext(
        yuanwen="示例原文",
        suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
        xiang=ExtractedField(text="想", source="原文", confidence=1.0),
        si=ExtractedField(text="思", source="原文", confidence=1.0),
        scoring_path="explicit",
    )
    return _make_context(extracted)
