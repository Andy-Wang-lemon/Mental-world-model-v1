"""所缘校验与主观所缘生成。"""

from mental_world_model.engine.perception.builder import generate_subjective_alambanas
from mental_world_model.engine.perception.config import PERCEPTION_PROMPT_VERSION
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    ObjectiveEventStructure,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import (
    build_grasping_pair_messages,
    build_subjective_alambana_messages,
)
from mental_world_model.engine.perception.validators import (
    looks_like_behavior_prediction,
    validate_alambana_candidate,
    validate_combo_transform,
)


def _hyp(
    hid: str,
    text: str,
    *,
    activation: float,
    grasping_type: str = "self_grasping",
) -> GraspingHypothesis:
    return GraspingHypothesis(
        id=hid,
        type=grasping_type,  # type: ignore[arg-type]
        text=text,
        source="推理",
        confidence=activation,
    )


class TestValidateAlambana:
    def test_rejects_behavior_prediction(self):
        cand = SubjectiveAlambanaCandidate(
            id="a1",
            text="当事人会再次帮助对方",
            used_self_ids=["s1"],
            used_dharma_ids=["d1"],
            source="推理",
        )
        valid, notes = validate_alambana_candidate(cand)
        assert valid is None
        assert any("行为预测" in n for n in notes)

    def test_passes_interpretive_text(self):
        cand = SubjectiveAlambanaCandidate(
            id="a1",
            text="在经验即我之下，当前求助被看成对自我价值的再确认",
            used_self_ids=["s1"],
            used_dharma_ids=["d1"],
            source="推理",
        )
        valid, notes = validate_alambana_candidate(cand)
        assert valid is not None
        assert notes == []

    def test_combo_transform_id_check(self):
        cand = SubjectiveAlambanaCandidate(
            id="a1",
            text="当前求助被看成对自我价值的再确认",
            self_grasping_ids=["s1"],
            dharma_grasping_ids=["d1"],
            source="推理",
        )
        assert (
            validate_combo_transform(
                cand, expected_self_ids=["s1"], expected_dharma_ids=["d1"]
            )
            == []
        )
        assert looks_like_behavior_prediction("当事人会再次帮助对方")


class TestGraspingPrompt:
    def test_inference_rich_dimensions_and_version(self):
        system, user = build_grasping_pair_messages(
            "同学叫我去上课",
            ObjectiveEventStructure(actors=["同学"], action="叫去上课", scene="课前"),
        )
        assert f"prompt_version={PERCEPTION_PROMPT_VERSION}" in system["content"]
        assert "当前事件" in system["content"]
        assert "不要依赖任何历史解释框架" in system["content"]
        assert "与自己分不开" in system["content"]
        assert "不能当成与我无关" in system["content"]
        assert "不能互相替代" in system["content"]
        assert "不要只写最显著的那条" in system["content"]
        assert "点燃了这一条" in system["content"]
        assert "挂住" in system["content"] or "锚点" in system["content"]
        assert "无锚" in system["content"] or "空想" in system["content"]
        assert "self_grasping" in user["content"]
        # 无背景时不注入背景激活说明
        assert "仍可能在当前情境激活" not in system["content"]

    def test_background_may_activate_any_grasping(self):
        system, user = build_grasping_pair_messages(
            "同学叫我去上课",
            ObjectiveEventStructure(actors=["同学"], action="叫去上课", scene="课前"),
            role_context="此前老师当众说他丢人。",
        )
        assert "任何类型" in system["content"]
        assert "若其中仍有可激活的执取可写入" in user["content"]
        assert "此前老师当众说他丢人" in user["content"]
        assert "不替代当前事件" in user["content"]
        assert "无关的空想" in system["content"]


class TestSubjectivePrompt:
    def test_renderer_prompt_for_fixed_combos(self):
        from mental_world_model.engine.perception.models import GraspingCombination

        system = build_subjective_alambana_messages(
            "有人前来求助",
            ObjectiveEventStructure(scene="求助", action="前来"),
            [_hyp("s1", "经验即我", activation=0.5)],
            [_hyp("d1", "求助即认可", activation=0.5, grasping_type="dharma_grasping")],
            combinations=[
                GraspingCombination(
                    combo_id="C1",
                    used_self_ids=["s1"],
                    used_dharma_ids=["d1"],
                    focus_axis="status",
                    reason="被需要",
                )
            ],
        )[0]["content"]
        assert "主观所缘境渲染器" in system
        assert "combo_id" in system
        assert "event_consistency" not in system
        assert f"prompt_version={PERCEPTION_PROMPT_VERSION}" in system


class TestGenerateSubjectiveAlambanas:
    def test_binds_combo_ids_and_applies_combo_activation_prior(self):
        from mental_world_model.engine.perception.models import GraspingCombination

        class FakeLlm:
            def complete(self, messages, schema=None):
                # combo_meta 或 alambana
                content = str(messages)
                props = (schema or {}).get("properties") or {}
                items = props.get("items") or {}
                item_props = ((items.get("items") or {}).get("properties") or {})
                if "same_perspective" in item_props:
                    return {
                        "items": [{"pair_id": "p0", "same_perspective": True}]
                    }
                if "二执组合评估器" in content or "activation_alignment" in item_props:
                    return {
                        "items": [
                            {
                                "combo_id": "C1",
                                "activation_alignment": 0.9,
                            }
                        ]
                    }
                return {
                    "candidates": [
                        {
                            "id": "O1",
                            "combo_id": "C1",
                            "text": "在执取下，事态被看成自我价值受威胁后的再确认机会",
                        }
                    ]
                }

        combos = [
            GraspingCombination(
                combo_id="C1",
                used_self_ids=["s1"],
                used_dharma_ids=["d1"],
                probability=1.0,
                score=0.5,
                self_confidence=0.55,
                dharma_confidence=0.5,
            )
        ]
        result, notes = generate_subjective_alambanas(
            "有人前来求助",
            ObjectiveEventStructure(scene="有人前来求助", action="求助"),
            [_hyp("s1", "把经验当成我", activation=0.55)],
            [
                _hyp(
                    "d1",
                    "把方法当成永恒实体",
                    activation=0.5,
                    grasping_type="dharma_grasping",
                )
            ],
            FakeLlm(),  # type: ignore[arg-type]
            combinations=combos,
        )
        assert len(result) == 1
        assert result[0].used_self_ids == ["s1"]
        assert result[0].used_dharma_ids == ["d1"]
        assert result[0].combo_reason == "C1"
        assert abs(result[0].prior_probability - 1.0) < 1e-6
        assert result[0].posterior_probability == result[0].prior_probability
