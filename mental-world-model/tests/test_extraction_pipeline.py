"""分步提取、路由与种子更新测试。"""

import json
from pathlib import Path

import pytest

from mental_world_model.engine.extraction.extractor import ContextExtractor
from mental_world_model.engine.extraction.router import (
    parse_router_single_response,
    select_hint_xinsuo_names,
)
from mental_world_model.engine.extraction.validators import (
    is_grounded_in_yuanwen,
    normalize_suoyuan_response,
    normalize_xiang_si_field,
)
from mental_world_model.engine.seeds import config as seeds_config
from mental_world_model.engine.seeds.config import SEED_LIBRARY_ENABLED
from mental_world_model.engine.seeds.state_store import SeedStateStore
from mental_world_model.engine.seeds.updater import update_seed_weights
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel


def _enable_seed_library(monkeypatch) -> None:
    """测试用：仅修改 config 模块上的种子库开关。"""
    monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", True)


def _load_model(tmp_path: Path) -> MentalWorldModel:
    root = Path(__file__).resolve().parents[1]
    return MentalWorldModel.load(
        root / "definitions",
        seed_state_path=tmp_path / "seed_state.json",
    )


class TestExtractionValidators:
    def test_grounded_in_yuanwen(self):
        assert is_grounded_in_yuanwen("我知道这样做是对的", "我知道这样做是对的。")

    def test_normalize_llm_yuanwen_bumps_confidence(self):
        field = normalize_xiang_si_field(
            {"text": "我知道这样做是对的", "source": "原文", "confidence": 0.7},
            yuanwen="我知道这样做是对的。",
        )
        assert field.source == "原文"
        assert field.confidence == 1.0

    def test_normalize_inference_confidence_one_without_grounding(self):
        field = normalize_xiang_si_field(
            {"text": "某种推断念头", "source": "推理", "confidence": 1.0},
            yuanwen="示例原文",
        )
        assert field.source == "推理"
        assert field.confidence == 0.8

    def test_normalize_yuanwen_without_grounding_flips_to_inference(self):
        field = normalize_xiang_si_field(
            {"text": "某种推断念头", "source": "原文", "confidence": 0.9},
            yuanwen="示例原文",
        )
        assert field.source == "推理"
        assert field.confidence == 0.8

    def test_normalize_empty_text_zero_confidence(self):
        field = normalize_xiang_si_field(
            {"text": "  ", "source": "推理", "confidence": 0.6},
            yuanwen="示例原文",
        )
        assert field.text == ""
        assert field.confidence == 0.0

    def test_normalize_suoyuan_response(self):
        field = normalize_suoyuan_response({"text": "所缘文本"})
        assert field.source == "原文"
        assert field.confidence == 1.0


class TestRouterSelection:
    def test_select_hint_xinsuo_names(self):
        relevance = {"信": 0.9, "瞋": 0.1, "善见": 0.8}
        seeds = {"信": 0.6, "瞋": 0.3, "善见": 0.5}
        names = select_hint_xinsuo_names(relevance, seeds, top_k=2)
        assert names == ["信", "善见"]

    def test_select_hint_filters_below_min_relevance(self):
        relevance = {"信": 0.9, "瞋": 0.2}
        seeds = {"信": 0.6, "瞋": 0.9}
        names = select_hint_xinsuo_names(relevance, seeds, top_k=6, min_relevance=0.35)
        assert names == ["信"]

    def test_parse_router_single_response_clamps(self):
        assert parse_router_single_response({"relevance": 0.8}) == 0.8
        assert parse_router_single_response({"relevance": 1.5}) == 1.0
        assert parse_router_single_response({"relevance": -0.2}) == 0.0


class TestContextExtractorPipeline:
    def test_extract_without_router_when_seed_disabled(self, tmp_path):
        assert SEED_LIBRARY_ENABLED is False
        model = _load_model(tmp_path)
        yuanwen = "我知道这样做是对的。"
        router_calls = {"n": 0}

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "这样做是对的"}}
            if "relevance" in props:
                router_calls["n"] += 1
                return {"relevance": 0.9}
            if "has_explicit_xiang" in props:
                return {
                    "has_explicit_xiang": True,
                    "xiang": {
                        "text": "我知道这样做是对的",
                        "source": "原文",
                        "confidence": 1.0,
                    },
                    "has_explicit_si": False,
                    "si": {"text": ""},
                }
            return {"score": 0.0}

        extractor = ContextExtractor(
            MockLlmClient(handler=handler),
            model.suoyuan_xiang_si_definitions,
            model.xinsuo_list,
            seed_weights=model.weights.seed_weights,
        )
        extracted = extractor.extract(yuanwen)
        assert extracted.suoyuan.text == "这样做是对的"
        assert extracted.routed_hint_names == []
        assert router_calls["n"] == 0
        assert extracted.xiang.source == "原文"
        assert extracted.scoring_path == "explicit"

    def test_route_one_llm_per_xinsuo_when_seed_enabled(self, tmp_path, monkeypatch):
        _enable_seed_library(monkeypatch)

        model = _load_model(tmp_path)
        yuanwen = "我知道这样做是对的。"
        router_calls = {"n": 0}

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "这样做是对的"}}
            if "relevance" in props:
                router_calls["n"] += 1
                return {"relevance": 0.9 if router_calls["n"] == 1 else 0.1}
            if "has_explicit_xiang" in props:
                return {
                    "has_explicit_xiang": True,
                    "xiang": {
                        "text": "我知道这样做是对的",
                        "source": "原文",
                        "confidence": 1.0,
                    },
                    "has_explicit_si": False,
                    "si": {"text": ""},
                }
            return {"score": 0.0}

        extractor = ContextExtractor(
            MockLlmClient(handler=handler),
            model.suoyuan_xiang_si_definitions,
            model.xinsuo_list,
            seed_weights=model.weights.seed_weights,
        )
        extracted = extractor.extract(yuanwen)
        assert router_calls["n"] == len(model.xinsuo_list)
        assert extracted.routed_hint_names
        assert extracted.scoring_path == "explicit"


class TestSeedRuntimeLoop:
    @pytest.mark.skipif(not SEED_LIBRARY_ENABLED, reason="种子库已禁用")
    def test_seed_state_persists(self, tmp_path):
        store = SeedStateStore(tmp_path / "seed_state.json")
        default = {"信": 0.6, "精进": 0.55}
        loaded = store.load_or_initialize(default)
        assert loaded == default

        updated = update_seed_weights(default, {"信": 0.8, "精进": 0.0})
        store.save(updated)
        reloaded = store.load_or_initialize(default)
        assert reloaded["信"] > 0.6
        assert reloaded["精进"] < 0.55

    def test_evaluate_updates_seed_weights(self, tmp_path, monkeypatch):
        import mental_world_model.engine.perception.config as perception_config

        monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
        _enable_seed_library(monkeypatch)
        model = _load_model(tmp_path)
        before = model.weights.seed_weights["信"]

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "test"}}
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
                return {"activations": {"信": 0.9}}
            if schema and "winner" in props:
                return {"winner": "none", "reason": "mock"}
            if "predicted_behavior" in props:
                return {
                    "predicted_behavior": "mock预测行为",
                    "prediction_reason": "mock参考心所：信",
                }
            return {"score": 0.9}

        result = model.evaluate_from_yuanwen("test", MockLlmClient(handler=handler))
        assert result.seed_weights_after is not None
        assert result.seed_weights_after["信"] != before
        assert json.loads((tmp_path / "seed_state.json").read_text())["weights"]["信"] == pytest.approx(
            result.seed_weights_after["信"]
        )

    def test_evaluate_full_loop_when_seed_enabled(self, tmp_path, monkeypatch):
        import mental_world_model.engine.perception.config as perception_config

        monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
        _enable_seed_library(monkeypatch)
        model = _load_model(tmp_path)
        router_calls = {"n": 0}
        before = model.weights.seed_weights["信"]

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "suoyuan" in props and "has_explicit_xiang" not in props and "xiang" not in props:
                return {"suoyuan": {"text": "所缘"}}
            if "relevance" in props:
                router_calls["n"] += 1
                return {"relevance": 0.9 if router_calls["n"] == 1 else 0.1}
            if "has_explicit_xiang" in props:
                return {
                    "has_explicit_xiang": False,
                    "xiang": {"text": ""},
                    "has_explicit_si": False,
                    "si": {"text": ""},
                }
            if "activations" in props:
                return {"activations": {"信": 0.9}}
            if schema and "winner" in props:
                return {"winner": "none", "reason": "mock"}
            if "predicted_behavior" in props:
                return {
                    "predicted_behavior": "mock预测行为",
                    "prediction_reason": "mock参考心所：信",
                }
            return {"score": 0.9}

        result = model.evaluate_from_yuanwen("测试原文", MockLlmClient(handler=handler))
        assert router_calls["n"] == len(model.xinsuo_list)
        assert result.extracted is not None
        assert result.extracted.routed_hint_names
        assert result.seed_weights_after is not None
        assert result.seed_weights_after["信"] != before
