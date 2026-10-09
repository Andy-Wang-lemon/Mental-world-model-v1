"""想/思置信度不再被所缘 confidence 封顶。"""

from mental_world_model.engine.cognition.xiang_si import _normalize_mental_state
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate
from mental_world_model.engine.seeds.config import INFERENCE_CONFIDENCE_MAX


class TestXiangSiConfidenceNoAlambanaCap:
    def test_inference_confidence_not_capped_by_alambana(self):
        alambana = SubjectiveAlambanaCandidate(
            id="a1",
            text="挑战资格",
            confidence=0.5,
            source="推理",
        )
        state = _normalize_mental_state(
            {
                "text": "这是对我权威的挑衅",
                "source": "推理",
                "confidence": 0.9,
            },
            yuanwen="领导当场打断并否定员工意见。",
            alambana=alambana,
        )
        # 仅受推理上限约束，不被所缘 0.5 压扁
        assert state.confidence == INFERENCE_CONFIDENCE_MAX

    def test_grounded_action_si_can_exceed_alambana(self):
        yuanwen = "领导当场打断并否定员工意见。"
        alambana = SubjectiveAlambanaCandidate(
            id="a1",
            text="挑战资格",
            confidence=0.55,
            source="推理",
        )
        state = _normalize_mental_state(
            {
                "text": "当场压制异议",
                "source": "原文",
                "confidence": 0.7,
            },
            yuanwen=yuanwen,
            alambana=alambana,
            action_span="领导当场打断并否定",
        )
        assert state.confidence >= 0.8
        assert state.confidence > alambana.confidence
