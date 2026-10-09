"""硬约束：二执禁词、行为白名单、评分 clamp、L6 支隔离。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.cognition.behavior import predict_behaviors_for_branch
from mental_world_model.engine.cognition.models import CognitionBranch, ExtractedMentalState
from mental_world_model.engine.cognition.prompts import (
    BRANCH_BEHAVIOR_SCHEMA,
    build_branch_xiang_messages,
)
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.validators import (
    normalize_grasping_hypothesis,
    parse_grasping_list,
)
from mental_world_model.engine.scoring.suoyuan_groups import parse_group_suoyuan_response
from mental_world_model.utils.confidence import as_unit_float


def _g(hid: str, text: str) -> GraspingHypothesis:
    return GraspingHypothesis(
        id=hid,
        type="self_grasping",
        text=text,
        confidence=0.7,
    )


class TestGraspingHardFilter:
    def test_rejects_psych_language(self):
        hyp = normalize_grasping_hypothesis(
            {"text": "这件事让我愤怒", "confidence": 0.8},
            grasping_type="self_grasping",
        )
        assert hyp is None

    def test_rejects_behavior_prediction(self):
        hyp = normalize_grasping_hypothesis(
            {"text": "当事人会再次帮助对方", "confidence": 0.8},
            grasping_type="dharma_grasping",
        )
        assert hyp is None

    def test_keeps_interpretive_grasping(self):
        hyp = normalize_grasping_hypothesis(
            {"text": "把体面与自我价值绑在一起", "confidence": 0.7},
            grasping_type="self_grasping",
        )
        assert hyp is not None
        assert "体面" in hyp.text

    def test_parse_list_drops_banned(self):
        items, notes = parse_grasping_list(
            [
                {"id": "ok", "text": "把体面执为我"},
                {"id": "bad", "text": "情绪上头"},
            ],
            grasping_type="self_grasping",
        )
        assert [h.id for h in items] == ["ok"]
        assert any("禁词" in n and "bad" in n for n in notes)


class TestBehaviorWhitelist:
    def test_schema_omits_code_owned_fields(self):
        props = BRANCH_BEHAVIOR_SCHEMA["properties"]["behavior_predictions"]["items"][
            "properties"
        ]
        assert "supporting_scores" not in props
        assert "alambana_id" not in props
        assert "xiang_text" not in props
        assert "si_text" not in props

    def test_filters_unknown_supporting_factors(self):
        class Client:
            def complete(self, messages, schema=None):
                return {
                    "behavior_predictions": [
                        {
                            "text": "回避",
                            "conditional_probability": 1.0,
                            "confidence": 0.6,
                            "supporting_mental_factors": ["懈怠", "虚构心所", "嗔"],
                            "opposing_mental_factors": ["精进", "未知"],
                            "reason_summary": "mock",
                        }
                    ]
                }

        branch = CognitionBranch(
            id="b1",
            alambana=SubjectiveAlambanaCandidate(
                id="O1",
                text="丢人现场",
                posterior_probability=0.5,
                confidence=0.7,
            ),
            xiang=ExtractedMentalState(
                text="想回避", source="推理", confidence=0.6, alambana_id="O1"
            ),
            si=ExtractedMentalState(
                text="内心：不想去", source="原文", confidence=0.8, alambana_id="O1"
            ),
        )
        xinsuo = {
            "懈怠": Xinsuo(name="懈怠", suoyuan="s", shengqi_nian="n", ye="y", xiandai_xinli="x"),
            "精进": Xinsuo(name="精进", suoyuan="s", shengqi_nian="n", ye="y", xiandai_xinli="x"),
        }
        preds = predict_behaviors_for_branch(
            "同学叫去上课",
            branch,
            {"懈怠": 0.7, "精进": 0.2},
            xinsuo,
            Client(),  # type: ignore[arg-type]
        )
        assert preds[0].supporting_mental_factors == ["懈怠"]
        assert preds[0].opposing_mental_factors == []
        assert preds[0].reason_summary == ""
        assert preds[0].supporting_scores == {"懈怠": 0.7}


class TestScoreClamp:
    def test_as_unit_float_clamps(self):
        assert as_unit_float(1.5) == 1.0
        assert as_unit_float(-0.2) == 0.0
        assert as_unit_float("bad", 0.3) == 0.3

    def test_group_suoyuan_parse_clamps(self):
        parsed = parse_group_suoyuan_response(
            {"scores": {"信": {"score": 1.7}, "不信": {"score": -1}}},
            ("信", "不信"),
        )
        assert parsed == {"信": 1.0, "不信": 0.0}


class TestXiangBranchIsolation:
    def test_no_fallback_to_all_grasping(self):
        """used_ids 对不上时不得把调用方未传入的执塞进提示。"""

        class _Ev:
            has_observable_action = False
            action_span = ""
            action_description = ""

        cand = SubjectiveAlambanaCandidate(
            id="O1",
            text="课堂如刑场",
            posterior_probability=0.5,
            confidence=0.7,
            used_self_ids=["sg_missing"],
            used_dharma_ids=[],
        )
        # 调用方只传入本支已解析执；ids 无匹配且未 prefer → 提示中无执
        branch_only = [_g("sg_shame", "把体面绑在自我上")]
        _system, user = build_branch_xiang_messages(
            "同学叫去上课",
            None,
            cand,
            branch_only,
            [],
            _Ev(),
            prefer_passed_grasping=False,
        )
        blob = user["content"]
        assert "posterior_probability" not in blob
        assert "prior_probability" not in blob
        assert "把体面绑在自我上" not in blob
        assert "本支主观所缘" in blob

        # prefer 时使用本支列表（即使 used_ids 对不上）
        _, user2 = build_branch_xiang_messages(
            "同学叫去上课",
            None,
            cand,
            branch_only,
            [],
            _Ev(),
            prefer_passed_grasping=True,
        )
        assert "把体面绑在自我上" in user2["content"]

    def test_uses_passed_branch_grasping_when_ids_empty(self):
        class _Ev:
            has_observable_action = False
            action_span = ""
            action_description = ""

        cand = SubjectiveAlambanaCandidate(
            id="O2",
            text="义务压过来",
            posterior_probability=0.4,
            confidence=0.6,
            used_self_ids=[],
            used_dharma_ids=[],
        )
        branch_self = [_g("sg_duty", "把角色义务执为我")]
        _, user = build_branch_xiang_messages(
            "同学叫去上课",
            None,
            cand,
            branch_self,
            [],
            _Ev(),
        )
        assert "把角色义务执为我" in user["content"]
