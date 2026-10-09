"""隐式路径公式与忿恨恼链测试。"""

from pathlib import Path

import pytest

from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.implicit_formula import compute_implicit_score
from mental_world_model.engine.scoring.implicit_pipeline import ImplicitBaifaPipeline
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel


class TestImplicitFormula:
    def test_below_gate_is_zero(self):
        assert compute_implicit_score(0.3, 0.9, 1.0) == 0.0
        assert compute_implicit_score(0.29, 1.0, 1.0) == 0.0

    def test_mix_suoyuan_and_seed(self):
        # a=1, β=0.75 → 0.75*0.8 + 0.25*0.4 = 0.7
        assert compute_implicit_score(0.8, 0.4, 1.0) == pytest.approx(0.7)

    def test_activation_scales(self):
        assert compute_implicit_score(0.8, 0.4, 0.5) == pytest.approx(0.35)


class TestImplicitPipeline:
    def test_fen_zero_skips_hen_nao(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed.json",
        )
        context = ScoringContext.from_extracted(
            ExtractedContext(
                yuanwen="测试",
                suoyuan=ExtractedField("所缘", "原文", 1.0),
                xiang=ExtractedField("（无原文明确想，走百法隐式路径）", "原文", 1.0),
                si=ExtractedField("（无原文明确思，走百法隐式路径）", "原文", 1.0),
                scoring_path="implicit",
            )
        )

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "activations" in props:
                return {"activations": {}}
            if "scores" in props or (
                schema and schema.get("properties", {}).get("scores")
            ):
                from mental_world_model.llm.mock_client import (
                    _group_suoyuan_mock_response,
                )

                return _group_suoyuan_mock_response(messages, {"score": 0.0})
            return {"score": 0.0}

        pipe = ImplicitBaifaPipeline(MockLlmClient(handler=handler), model.default_seed_catalog)
        results = {r.name: r for r in pipe.run(model.xinsuo_list, context)}
        assert results["忿"].final_score == 0.0
        assert results["恨"].skipped
        assert results["恼"].skipped

    def test_evaluate_implicit_end_to_end(self, tmp_path, monkeypatch):
        import mental_world_model.engine.perception.config as perception_config

        monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed.json",
        )

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "所缘对象"}}
            if "has_explicit_xiang" in props:
                return {
                    "has_explicit_xiang": False,
                    "xiang": {"text": ""},
                    "has_explicit_si": False,
                    "si": {"text": ""},
                }
            if "activations" in props:
                content = messages[-1]["content"]
                names = []
                for line in content.splitlines():
                    s = line.strip()
                    if s.startswith("- ") and "（所缘相似度" in s:
                        names.append(s[2:].split("（", 1)[0].strip())
                return {"activations": {n: 0.8 for n in names}}
            if "predicted_behavior" in props:
                return {
                    "predicted_behavior": "可能的行为",
                    "prediction_reason": "根据有值心所",
                }
            if "winner" in props:
                return {"winner": "none"}
            from mental_world_model.engine.scoring.suoyuan_groups import (
                is_group_suoyuan_schema,
            )
            from mental_world_model.llm.mock_client import _group_suoyuan_mock_response

            if is_group_suoyuan_schema(schema):
                return _group_suoyuan_mock_response(messages, {"score": 0.9})
            return {"score": 0.9}

        result = model.evaluate_from_yuanwen(
            "没有明确内心描写，只有场面。",
            MockLlmClient(handler=handler),
        )
        assert result.extracted is not None
        assert result.extracted.scoring_path == "implicit"
        assert any(v > 0 for v in result.scores_final.values())
        assert result.predicted_behavior == "可能的行为"
