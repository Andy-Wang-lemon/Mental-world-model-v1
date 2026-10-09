"""测试评分管线、冲突消解与 MentalWorldModel。"""

from pathlib import Path

import pytest

from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.scoring.pipeline import XinsuoScoringPipeline
from mental_world_model.engine.scoring.prompts import (
    is_group_suoyuan_stage_content,
    is_xiandai_stage_content,
    is_ye_stage_content,
)
from mental_world_model.engine.scoring.suoyuan_groups import is_group_suoyuan_schema
from mental_world_model.engine.seeds import config as seeds_config
from mental_world_model.engine.seeds.config import SEED_LIBRARY_ENABLED
from mental_world_model.engine.scoring.weights import (
    DEFAULT_COLUMN_WEIGHTS,
    GATE_THRESHOLD,
    ScoringWeights,
    compute_final_score,
    compute_score_breakdown,
    compute_normalized_column_weights,
    compute_source_modifiers,
)
from mental_world_model.engine.reconciliation.reconciler import DuizhiReconciler
from mental_world_model.llm.mock_client import MockLlmClient, _group_suoyuan_mock_response
from mental_world_model.llm.message_utils import user_message_content
from mental_world_model.model import MentalWorldModel
from tests.test_extraction import make_test_scoring_context


def _sample_context():
    return make_test_scoring_context()


def _all_yuanwen_extracted() -> ExtractedContext:
    return ExtractedContext(
        yuanwen="示例原文",
        suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
        xiang=ExtractedField(text="想", source="原文", confidence=1.0),
        si=ExtractedField(text="思", source="原文", confidence=1.0),
    )


def _stage_handler(
    *,
    suoyuan: float = 0.8,
    shengqi: float = 0.8,
    ye: float = 0.6,
    xiandai: float = 0.6,
):
    def handler(messages, schema):
        content = user_message_content(messages)
        if is_ye_stage_content(content):
            return {"score": ye}
        if is_xiandai_stage_content(content):
            return {"score": xiandai}
        if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
            response = _group_suoyuan_mock_response(
                messages,
                {"score": suoyuan, "reason": "mock"},
            )
            return response
        if "该心所的所缘境定义" in content:
            return {"score": suoyuan}
        if "生起的念头定义" in content:
            return {"score": shengqi}
        return {"score": suoyuan}

    return handler


class TestScoringWeights:
    def test_compute_final_score_all_yuanwen(self):
        weights = ScoringWeights(
            column_weights=dict(DEFAULT_COLUMN_WEIGHTS),
            seed_weights={"信": 1.0},
        )
        column_scores = {
            "suoyuan": 0.8,
            "shengqi_nian": 0.8,
            "ye": 0.6,
            "xiandai_xinli": 0.6,
        }
        final = compute_final_score(
            "信",
            column_scores,
            weights,
            _all_yuanwen_extracted(),
        )
        assert final == 0.73

    def test_seed_weight_does_not_multiply_score(self, monkeypatch):
        weights = ScoringWeights(seed_weights={"信": 2.0})
        column_scores = {
            "suoyuan": 1.0,
            "shengqi_nian": 1.0,
            "ye": 1.0,
            "xiandai_xinli": 1.0,
        }
        monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", True)
        final = compute_final_score(
            "信",
            column_scores,
            weights,
            _all_yuanwen_extracted(),
        )
        assert final == 1.0

        monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", False)
        final = compute_final_score(
            "信",
            column_scores,
            weights,
            _all_yuanwen_extracted(),
        )
        assert final == 1.0

    def test_seed_weight_recorded_but_not_multiplied(self, monkeypatch):
        weights = ScoringWeights(seed_weights={"信": 0.5})

        monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", True)
        breakdown = compute_score_breakdown(
            "信",
            {
                "suoyuan": 1.0,
                "shengqi_nian": 1.0,
                "ye": 1.0,
                "xiandai_xinli": 1.0,
            },
            weights,
            _all_yuanwen_extracted(),
        )
        assert breakdown.seed_weights_enabled is True
        assert breakdown.seed_weight == 0.5
        assert breakdown.final_score == 1.0

        monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", False)
        breakdown = compute_score_breakdown(
            "信",
            {
                "suoyuan": 1.0,
                "shengqi_nian": 1.0,
                "ye": 1.0,
                "xiandai_xinli": 1.0,
            },
            weights,
            _all_yuanwen_extracted(),
        )
        assert breakdown.seed_weights_enabled is False
        assert breakdown.final_score == 1.0

    def test_xiandai_modifier_always_one(self):
        """现代心理学列 m 恒为 1.0；分支体制下四列 m 均为 1.0。"""
        weights = ScoringWeights()
        column_scores = {
            "suoyuan": 0.9,
            "shengqi_nian": 0.9,
            "ye": 0.8,
            "xiandai_xinli": 0.7,
        }
        modifiers = compute_source_modifiers(_all_yuanwen_extracted())
        assert modifiers["xiandai_xinli"] == 1.0
        assert modifiers["shengqi_nian"] == 1.0
        assert modifiers["ye"] == 1.0
        final = compute_final_score(
            "信",
            column_scores,
            weights,
            _all_yuanwen_extracted(),
        )
        assert final == pytest.approx(0.84)

    def test_source_modifiers_always_one_even_for_inference(self):
        """想/思为推理时，m 仍恒为 1.0（不确定性由分支概率承担）。"""
        extracted = ExtractedContext(
            yuanwen="示例原文",
            suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
            xiang=ExtractedField(text="想", source="推理", confidence=0.8),
            si=ExtractedField(text="思", source="原文", confidence=1.0),
        )
        modifiers = compute_source_modifiers(extracted)
        assert modifiers == {
            "suoyuan": 1.0,
            "shengqi_nian": 1.0,
            "ye": 1.0,
            "xiandai_xinli": 1.0,
        }
        normalized = compute_normalized_column_weights(
            dict(DEFAULT_COLUMN_WEIGHTS),
            modifiers,
        )
        assert abs(sum(normalized.values()) - 1.0) < 1e-9
        assert normalized["shengqi_nian"] == pytest.approx(
            DEFAULT_COLUMN_WEIGHTS["shengqi_nian"]
        )
        assert normalized["suoyuan"] == pytest.approx(
            DEFAULT_COLUMN_WEIGHTS["suoyuan"]
        )

    def test_inference_confidence_does_not_affect_modifiers(self):
        extracted = ExtractedContext(
            yuanwen="示例原文",
            suoyuan=ExtractedField(text="所缘", source="原文", confidence=1.0),
            xiang=ExtractedField(text="想", source="推理", confidence=0.2),
            si=ExtractedField(text="思", source="推理", confidence=0.1),
        )
        modifiers = compute_source_modifiers(extracted)
        assert modifiers["shengqi_nian"] == 1.0
        assert modifiers["ye"] == 1.0
        assert modifiers["xiandai_xinli"] == 1.0

class TestXinsuoScoringPipeline:
    def _load_xinsuo(self, name: str):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        return model.get_xinsuo(name)

    def test_gate_both_low_exits_after_shengqi(self):
        """所缘=0.1 未过硬地板，进生起；所缘与生起都低才排除。"""
        xinsuo = self._load_xinsuo("信")
        llm = MockLlmClient(default_response={"score": 0.1, "reason": "low"})
        pipeline = XinsuoScoringPipeline(llm, gate_threshold=GATE_THRESHOLD)
        result = pipeline.run([xinsuo], _sample_context())[0]
        assert result.exit_at == "shengqi_nian"
        assert result.final_score == 0.0
        assert len(result.stage_results) == 2
        assert result.column_scores["suoyuan"] == pytest.approx(0.1)
        assert result.column_scores["shengqi_nian"] == pytest.approx(0.1)

    def test_suoyuan_below_hard_floor_skips_shengqi(self):
        """所缘 < 0.1：直接出局，不打生起念头。"""
        xinsuo = self._load_xinsuo("信")
        llm = MockLlmClient(default_response={"score": 0.05, "reason": "很低"})
        pipeline = XinsuoScoringPipeline(llm, gate_threshold=GATE_THRESHOLD)
        result = pipeline.run([xinsuo], _sample_context())[0]
        assert result.exit_at == "suoyuan"
        assert result.final_score == 0.0
        assert len(result.stage_results) == 1
        assert "shengqi_nian" not in result.column_scores
        assert result.column_scores["suoyuan"] == pytest.approx(0.05)

    def test_suoyuan_low_shengqi_high_continues(self):
        """所缘=0.1（保留）但低于双门控阈值、生起念头过了 → 继续后两列。"""
        xinsuo = self._load_xinsuo("信")

        def _all_content(messages):
            return "\n".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict)
            )

        def smart_handler(messages, schema):
            content = _all_content(messages)
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                return _group_suoyuan_mock_response(
                    messages, {"score": 0.1, "reason": "mock"}
                )
            if "该心所的所缘境定义" in content:
                return {"score": 0.1}
            if "生起的念头" in content:
                return {"score": 0.9}
            return {"score": 0.6}

        pipeline = XinsuoScoringPipeline(
            MockLlmClient(handler=smart_handler), gate_threshold=GATE_THRESHOLD
        )
        result = pipeline.run([xinsuo], _sample_context())[0]
        assert result.exit_at is None
        assert result.final_score > 0
        assert len(result.stage_results) == 4
        assert result.column_scores["suoyuan"] == pytest.approx(0.1)
        assert result.column_scores["shengqi_nian"] == pytest.approx(0.9)

    def test_gate_exit_at_shengqi_nian_alone_no_longer_exits(self):
        """所缘已过门控、仅生起念头低：双门控下不排除（须两者都低才出局）。"""
        xinsuo = self._load_xinsuo("信")

        def _all_content(messages):
            return "\n".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict)
            )

        def smart_handler(messages, schema):
            content = _all_content(messages)
            if is_ye_stage_content(content) or is_xiandai_stage_content(content):
                return {"score": 1.0}
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                return _group_suoyuan_mock_response(
                    messages, {"score": 0.9, "reason": "mock"}
                )
            if "该心所的所缘境定义" in content:
                return {"score": 0.9}
            if "生起的念头" in content:
                return {"score": 0.1}
            return {"score": 0.9}

        pipeline = XinsuoScoringPipeline(MockLlmClient(handler=smart_handler))
        result = pipeline.run([xinsuo], _sample_context())[0]
        assert result.exit_at is None
        assert result.final_score > 0
        assert len(result.stage_results) == 4
        assert result.column_scores["suoyuan"] == pytest.approx(0.9)
        assert result.column_scores["shengqi_nian"] == pytest.approx(0.1)

    def test_full_pass_computes_final_score(self):
        xinsuo = self._load_xinsuo("信")
        pipeline = XinsuoScoringPipeline(
            MockLlmClient(handler=_stage_handler(ye=0.6, xiandai=0.6))
        )
        result = pipeline.run([xinsuo], _sample_context())[0]
        assert result.exit_at is None
        assert result.final_score == 0.73
        assert len(result.stage_results) == 4
        assert [stage.stage for stage in result.stage_results] == [
            "suoyuan",
            "shengqi_nian",
            "ye",
            "xiandai_xinli",
        ]


class TestFenHenNaoDependencyChain:
    """忿 → 恨 → 恼 依序依赖：前项无值则后项跳过 LLM。"""

    def _load_chain_xinsuo(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        return model, [model.get_xinsuo(name) for name in ("忿", "恨", "恼")]

    def _run_chain(self, score_by_name: dict[str, float]):
        model, chain = self._load_chain_xinsuo()
        calls: list[str] = []

        def handler(messages, schema):
            content = user_message_content(messages)
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                response = _group_suoyuan_mock_response(
                    messages, {"score": 0.0, "reason": "mock"}
                )
                for name in response["scores"]:
                    calls.append(name)
                    response["scores"][name]["score"] = score_by_name.get(name, 0.0)
                return response
            name = content.split("心所名称：", 1)[1].split("\n", 1)[0]
            calls.append(name)
            score = score_by_name.get(name, 0.0)
            if is_ye_stage_content(content) or is_xiandai_stage_content(content):
                return {"score": score}
            return {"score": score, "reason": "mock"}

        pipeline = XinsuoScoringPipeline(MockLlmClient(handler=handler))
        results = pipeline.run(chain, _sample_context())
        return results, calls

    def test_fen_zero_skips_hen_and_nao(self):
        results, calls = self._run_chain({"忿": 0.1})
        by_name = {item.name: item for item in results}

        assert by_name["忿"].final_score == 0.0
        assert by_name["恨"].skipped is True
        assert by_name["恼"].skipped is True
        assert "恨" not in calls
        assert "恼" not in calls
        # 所缘 + 生起念头双波后才双门控出局，故忿至少被评两列
        assert calls.count("忿") >= 2
        assert by_name["忿"].exit_at == "shengqi_nian"
        assert len(by_name["忿"].stage_results) == 2

    def test_fen_positive_hen_zero_skips_nao(self):
        results, calls = self._run_chain({"忿": 0.8, "恨": 0.1})
        by_name = {item.name: item for item in results}

        assert by_name["忿"].final_score > 0
        assert by_name["恨"].final_score == 0.0
        assert by_name["恨"].skipped is False
        assert by_name["恼"].skipped is True
        assert "恼" not in calls
        assert "恨" in calls

    def test_all_three_scored_when_chain_active(self):
        results, calls = self._run_chain({"忿": 0.8, "恨": 0.8, "恼": 0.8})
        by_name = {item.name: item for item in results}

        assert all(by_name[name].final_score > 0 for name in ("忿", "恨", "恼"))
        assert all(not by_name[name].skipped for name in ("忿", "恨", "恼"))
        assert "忿" in calls and "恨" in calls and "恼" in calls

    def test_full_list_preserves_order_and_skips(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        scored_names: set[str] = set()

        def handler(messages, schema):
            content = user_message_content(messages)
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                response = _group_suoyuan_mock_response(
                    messages, {"score": 0.1, "reason": "mock"}
                )
                scored_names.update(response["scores"].keys())
                return response
            name = content.split("心所名称：", 1)[1].split("\n", 1)[0]
            scored_names.add(name)
            if is_ye_stage_content(content) or is_xiandai_stage_content(content):
                return {"score": 0.0}
            return {"score": 0.1}

        pipeline = XinsuoScoringPipeline(MockLlmClient(handler=handler))
        results = pipeline.run(model.xinsuo_list, _sample_context())

        assert len(results) == 40
        assert [item.name for item in results] == [x.name for x in model.xinsuo_list]
        assert "恨" not in scored_names
        assert "恼" not in scored_names
        hen = next(item for item in results if item.name == "恨")
        nao = next(item for item in results if item.name == "恼")
        assert hen.skipped and nao.skipped


class TestDuizhiReconciler:
    def test_resolve_winner_keeps_score(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        xinsuo_by_name = {x.name: x for x in model.xinsuo_list}

        llm = MockLlmClient(default_response={"winner": "信", "reason": "更像信"})

        reconciler = DuizhiReconciler(llm)
        scores = {"信": 0.8, "不信": 0.3}
        final, resolutions = reconciler.reconcile(
            scores,
            model.relations,
            xinsuo_by_name,
            _sample_context(),
        )
        assert final["信"] == 0.8
        assert final["不信"] == 0.0
        assert resolutions[0].winner == "信"

    def test_resolve_none_zeros_both(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        xinsuo_by_name = {x.name: x for x in model.xinsuo_list}

        llm = MockLlmClient(default_response={"winner": "none", "reason": "都不像"})

        reconciler = DuizhiReconciler(llm, score_margin=0.08)
        # 近平局才走 LLM；明显分差会走代码决胜
        scores = {"信": 0.55, "不信": 0.52}
        final, resolutions = reconciler.reconcile(
            scores,
            model.relations,
            xinsuo_by_name,
            _sample_context(),
        )
        assert final["信"] == 0.0
        assert final["不信"] == 0.0
        assert resolutions[0].winner is None

    def test_resolve_by_score_skips_llm(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        xinsuo_by_name = {x.name: x for x in model.xinsuo_list}

        calls = {"n": 0}

        def handler(messages, schema):
            calls["n"] += 1
            return {"winner": "不信"}

        reconciler = DuizhiReconciler(MockLlmClient(handler=handler), score_margin=0.08)
        scores = {"信": 0.8, "不信": 0.3}
        final, resolutions = reconciler.reconcile(
            scores,
            model.relations,
            xinsuo_by_name,
            _sample_context(),
        )
        assert calls["n"] == 0
        assert final["信"] == 0.8
        assert final["不信"] == 0.0
        assert resolutions[0].winner == "信"
        assert "比分决胜" in resolutions[0].note


class TestMentalWorldModelEvaluate:
    def test_evaluate_end_to_end_with_mock(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        def handler(messages, schema):
            if schema and "predicted_behavior" in schema.get("properties", {}):
                return {
                    "predicted_behavior": "mock预测行为",
                    "prediction_reason": "mock参考心所：信",
                }
            if schema and "winner" in schema.get("properties", {}):
                return {"winner": "信", "reason": "信更符合"}
            content = user_message_content(messages)
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
                return {
                    "scores": {
                        "信": {"score": 0.8, "reason": "mock"},
                        "不信": {"score": 0.5, "reason": "mock"},
                    }
                }
            if is_ye_stage_content(content) or is_xiandai_stage_content(content):
                if "心所名称：信" in content and "不信" not in content.split("心所名称：")[1][:5]:
                    return {"score": 0.5}
                if "心所名称：不信" in content:
                    return {"score": 0.4}
                return {"score": 0.0}
            if "心所名称：信" in content:
                return {"score": 0.8}
            if "心所名称：不信" in content:
                return {"score": 0.5}
            return {"score": 0.0}

        llm = MockLlmClient(handler=handler)
        context = _sample_context()

        xin = model.get_xinsuo("信")
        xin_bu = model.get_xinsuo("不信")
        pipeline = XinsuoScoringPipeline(llm, weights=model.weights)
        partial_results = pipeline.run([xin, xin_bu], context)
        scores = pipeline.to_scores_dict(partial_results)
        assert scores["信"] > 0
        assert scores["不信"] > 0

        result = model.evaluate(context, llm)
        assert len(result.stage_results) == 40
        assert result.extracted is not None
        assert isinstance(result.scores_final, dict)
        assert result.scores_final["信"] > 0
        assert result.scores_final["不信"] == 0.0

    def test_document_order_preserved(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        names = [x.name for x in model.xinsuo_list]
        assert names[0] == "信"
        assert names[-1] == "散乱"
        assert len(names) == 40

    @pytest.mark.skipif(not SEED_LIBRARY_ENABLED, reason="种子库已禁用")
    def test_seed_weights_loaded_from_file(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions", use_runtime_seeds=False)
        assert len(model.weights.seed_weights) == 40
        assert model.weights.get_seed_weight("信") == 0.6
        assert model.weights.get_seed_weight("恨") == 0.15
        assert model.weights.get_seed_weight("恶见") == 0.2
