"""行为预测层测试。"""

from pathlib import Path

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.prediction.predictor import BehaviorPredictor
from mental_world_model.engine.prediction.prompts import (
    STAGE_BEHAVIOR_MARKER,
    build_behavior_messages,
    is_behavior_stage_content,
)
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.prompts import is_group_suoyuan_stage_content
from mental_world_model.engine.scoring.results import EvaluationResult
from mental_world_model.engine.scoring.suoyuan_groups import is_group_suoyuan_schema
from mental_world_model.llm.message_utils import user_message_content
from mental_world_model.llm.mock_client import (
    MockLlmClient,
    _group_suoyuan_mock_response,
)
from mental_world_model.model import MentalWorldModel


def _context() -> ScoringContext:
    return ScoringContext.from_extracted(
        ExtractedContext(
            yuanwen="他看到朋友需要帮助，心里很想上前。",
            suoyuan=ExtractedField(text="朋友需要帮助", source="原文", confidence=1.0),
            xiang=ExtractedField(text="应该去帮忙", source="原文", confidence=1.0),
            si=ExtractedField(text="打算行动", source="推理", confidence=0.6),
        )
    )


def _xinsuo(name: str, ye: str = "业定义", xiandai: str = "心理解释") -> Xinsuo:
    return Xinsuo(
        name=name,
        suoyuan="所缘",
        shengqi_nian="念头",
        ye=ye,
        xiandai_xinli=xiandai,
    )


class TestBehaviorPrompts:
    def test_build_messages_contains_score_ye_xiandai_sorted(self):
        xin = _xinsuo("信", ye="信的业", xiandai="信的心理")
        jing = _xinsuo("精进", ye="精进的业", xiandai="精进的心理")
        messages = build_behavior_messages(
            _context(),
            [
                ("信", 0.8, xin),
                ("精进", 0.5, jing),
            ],
        )
        assert len(messages) == 2
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert "他看到朋友需要帮助" in system
        assert "朋友需要帮助" in system
        assert "应该去帮忙" in system
        assert "打算行动" not in system
        assert STAGE_BEHAVIOR_MARKER in user
        assert is_behavior_stage_content(user)
        assert "信（分数：0.8000）" in user
        assert "精进（分数：0.5000）" in user
        assert "信的业" in user
        assert "信的心理" in user
        assert "prediction_reason" in user
        assert "心所" in user
        assert user.index("信（分数") < user.index("精进（分数")

    def test_zero_score_items_not_included_by_caller_contract(self):
        """prompt 只接收调用方传入的有值项；零分不应出现。"""
        messages = build_behavior_messages(
            _context(),
            [("信", 0.7, _xinsuo("信"))],
        )
        assert "不信" not in messages[1]["content"]


class TestBehaviorPredictor:
    def test_all_zero_skips_llm(self):
        calls = {"n": 0}

        def handler(messages, schema):
            calls["n"] += 1
            return {
                "predicted_behavior": "不应被调用",
                "prediction_reason": "不应被调用",
            }

        predictor = BehaviorPredictor(MockLlmClient(handler=handler))
        result = predictor.predict(
            _context(),
            {"信": 0.0, "精进": 0.0},
            {"信": _xinsuo("信"), "精进": _xinsuo("精进")},
        )
        assert result.predicted_behavior == ""
        assert result.prediction_reason == ""
        assert result.active_xinsuo == ()
        assert calls["n"] == 0

    def test_predict_returns_three_parts_and_orders_by_score(self):
        seen = {"user": ""}

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            assert "predicted_behavior" in props
            assert "prediction_reason" in props
            seen["user"] = user_message_content(messages)
            return {
                "predicted_behavior": "  他可能会主动上前帮忙。  ",
                "prediction_reason": "  因「信」「精进」共同推动其行动。  ",
            }

        predictor = BehaviorPredictor(MockLlmClient(handler=handler))
        result = predictor.predict(
            _context(),
            {"精进": 0.4, "信": 0.9},
            {
                "信": _xinsuo("信", ye="信业", xiandai="信心理"),
                "精进": _xinsuo("精进", ye="精进业", xiandai="精进心理"),
            },
        )
        assert result.predicted_behavior == "他可能会主动上前帮忙。"
        assert result.prediction_reason == "因「信」「精进」共同推动其行动。"
        assert [item.name for item in result.active_xinsuo] == ["信", "精进"]
        assert result.active_xinsuo[0].score == 0.9
        assert result.active_xinsuo[1].score == 0.4
        assert seen["user"].index("信（分数") < seen["user"].index("精进（分数")
        assert "信业" in seen["user"]
        assert "精进心理" in seen["user"]


class TestEvaluationResultBehavior:
    def test_to_report_dict_includes_three_prediction_fields(self):
        result = EvaluationResult(
            extracted=None,
            stage_results=[],
            scores_initial={},
            conflicts_initial=[],
            conflicts_final=[],
            resolutions=[],
            scores_final={"信": 0.5},
            predicted_behavior="可能去帮忙",
            active_xinsuo=[{"name": "信", "score": 0.5}],
            prediction_reason="根据心所「信」判断",
        )
        report = result.to_report_dict()
        assert report["predicted_behavior"] == "可能去帮忙"
        assert report["active_xinsuo"] == [{"name": "信", "score": 0.5}]
        assert report["prediction_reason"] == "根据心所「信」判断"

    def test_evaluate_wires_three_prediction_fields(self, tmp_path, monkeypatch):
        import mental_world_model.engine.perception.config as perception_config

        monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            if "predicted_behavior" in props:
                return {
                    "predicted_behavior": "端到端预测行为",
                    "prediction_reason": "根据有值心所综合判断",
                }
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
                content = user_message_content(messages)
                names = []
                for line in content.splitlines():
                    s = line.strip()
                    if s.startswith("- ") and "（所缘相似度" in s:
                        names.append(s[2:].split("（", 1)[0].strip())
                return {"activations": {n: 0.9 for n in names}}
            if schema and "winner" in props:
                return {"winner": "none"}
            content = user_message_content(messages)
            if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(
                content
            ):
                return _group_suoyuan_mock_response(messages, {"score": 0.8})
            return {"score": 0.8}

        result = model.evaluate_from_yuanwen(
            "测试原文",
            MockLlmClient(handler=handler),
        )
        assert any(score > 0 for score in result.scores_final.values())
        assert result.predicted_behavior == "端到端预测行为"
        assert result.prediction_reason == "根据有值心所综合判断"
        assert result.active_xinsuo is not None
        assert len(result.active_xinsuo) > 0
        assert all(
            item["score"] > 0 and item["name"] in result.scores_final
            for item in result.active_xinsuo
        )
        # 按分数降序
        scores = [item["score"] for item in result.active_xinsuo]
        assert scores == sorted(scores, reverse=True)
        report = result.to_report_dict()
        assert report["predicted_behavior"] == "端到端预测行为"
        assert report["active_xinsuo"] == result.active_xinsuo
        assert report["prediction_reason"] == "根据有值心所综合判断"
