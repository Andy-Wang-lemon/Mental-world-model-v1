"""二执组合选择：心理视角互异。"""

from mental_world_model.engine.perception.combo_selection import (
    filter_diverse_combinations,
    is_same_perspective,
    jaccard_ids,
    select_grasping_combinations,
)
from mental_world_model.engine.perception.grasping_combos import build_grasping_combos
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
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


class TestBuildGraspingCombos:
    def test_below_threshold_does_not_force_normalize(self):
        """confidence 均低于门槛时不再降级成 P=1，交由上层 accommodation。"""
        self_g = [_hyp("s1", "弱我执A", activation=0.1), _hyp("s2", "弱我执B", activation=0.15)]
        dharma_g = [
            _hyp("d1", "弱法执", activation=0.12, grasping_type="dharma_grasping")
        ]
        selected = select_grasping_combinations(self_g, dharma_g)
        assert selected == []

    def test_zero_alignment_is_not_normalized_to_one(self):
        self_g = [_hyp("s1", "我是决策核心", activation=0.7)]
        dharma_g = [
            _hyp("d1", "求助等于依赖", activation=0.6, grasping_type="dharma_grasping")
        ]
        selected = select_grasping_combinations(
            self_g,
            dharma_g,
            meta_by_combo_id={"C1": {"activation_alignment": 0.0}},
        )
        assert selected == []

    def test_confidence_drives_entry(self):
        """confidence=0.6 应进组合，不得变空。"""
        self_g = [
            GraspingHypothesis(
                id="s1",
                type="self_grasping",
                text="我是决策核心",
                confidence=0.6,
            )
        ]
        dharma_g = [
            GraspingHypothesis(
                id="d1",
                type="dharma_grasping",
                text="求助等于依赖",
                confidence=0.5,
            )
        ]
        selected = select_grasping_combinations(self_g, dharma_g)
        assert len(selected) >= 1
        assert abs(selected[0].self_confidence - 0.6) < 1e-9

    def test_semantic_near_self_dedup(self):
        self_g = [
            _hyp("s1", "我是团队里最懂业务的人之一", activation=0.7),
            _hyp("s2", "我是团队里最懂业务的人", activation=0.65),  # 被前者包含且够长
        ]
        dharma_g = [
            _hyp(
                "d1",
                "新方法必须完全取代旧方法",
                activation=0.6,
                grasping_type="dharma_grasping",
            )
        ]
        selected = select_grasping_combinations(self_g, dharma_g)
        used_self = {sid for c in selected for sid in c.used_self_ids}
        assert len(used_self) == 1

    def test_same_perspective_filters(self):
        self_by = {
            "s1": _hyp("s1", "把资格当成自我身份的核心", activation=0.7),
            "s2": _hyp("s2", "把资格当成自我身份的核心点", activation=0.65),
        }
        dharma_by = {
            "d1": _hyp(
                "d1",
                "把私下求助理解成认错",
                activation=0.6,
                grasping_type="dharma_grasping",
            ),
            "d2": _hyp(
                "d2",
                "把私下求助理解成认错了",
                activation=0.55,
                grasping_type="dharma_grasping",
            ),
        }
        # 视角文本含包含关系 → 同视角
        a = GraspingCombination(
            combo_id="C1",
            used_self_ids=["s1"],
            used_dharma_ids=["d1"],
            self_confidence=0.7,
            dharma_confidence=0.6,
            score=0.5,
        )
        b = GraspingCombination(
            combo_id="C2",
            used_self_ids=["s1"],
            used_dharma_ids=["d1"],
            self_confidence=0.65,
            dharma_confidence=0.55,
            score=0.4,
        )
        assert is_same_perspective(
            a, b, self_by_id=self_by, dharma_by_id=dharma_by
        )
        kept = filter_diverse_combinations(
            [a, b], self_by_id=self_by, dharma_by_id=dharma_by
        )
        assert len(kept) == 1
        assert kept[0].combo_id == "C1"

    def test_activation_controls_keep(self):
        """高 activation 组合应保留。"""
        self_g = [
            _hyp("s1", "我是团队决策核心", activation=0.7),
            _hyp("s2", "我与对方是互助同事", activation=0.6),
        ]
        dharma_g = [
            _hyp(
                "d1",
                "新方法必须完全取代旧方法",
                activation=0.65,
                grasping_type="dharma_grasping",
            ),
            _hyp(
                "d2",
                "私下求助等于承认依赖",
                activation=0.55,
                grasping_type="dharma_grasping",
            ),
        ]
        selected = select_grasping_combinations(
            self_g,
            dharma_g,
            meta_by_combo_id={
                "C1": {
                    "activation_alignment": 0.95,
                    "reason": "事件强烈触及该视角",
                },
                "C2": {
                    "activation_alignment": 0.55,
                    "reason": "激活较弱",
                },
                "C3": {
                    "activation_alignment": 0.5,
                    "reason": "中性",
                },
                "C4": {
                    "activation_alignment": 0.45,
                    "reason": "激活不足主导",
                },
            },
        )
        assert 1 <= len(selected) <= 3
        assert abs(sum(c.activation_probability for c in selected) - 1.0) < 1e-6
        top = max(selected, key=lambda c: c.activation_probability)
        assert top.activation_alignment >= 0.9
        assert "behavior_alignment" not in top.to_dict()
        assert "behavior_contribution_probability" not in top.to_dict()

    def test_jaccard_near_duplicate_ids(self):
        a = GraspingCombination(
            combo_id="C1", used_self_ids=["s1"], used_dharma_ids=["d1"]
        )
        b = GraspingCombination(
            combo_id="C2", used_self_ids=["s1"], used_dharma_ids=["d1", "d2"]
        )
        assert jaccard_ids(a, b) >= 0.5

    def test_compat_build_grasping_combos(self):
        self_g = [_hyp("s1", "我执甲", activation=0.7)]
        dharma_g = [
            _hyp("d1", "法执甲", activation=0.6, grasping_type="dharma_grasping")
        ]
        combos = build_grasping_combos(self_g, dharma_g)
        assert combos
        assert combos[0].activation_score() > 0

    def test_combo_activation_and_alambana_prior_are_separate_fields(self):
        combo = GraspingCombination(
            combo_id="C1",
            used_self_ids=["s1"],
            used_dharma_ids=["d1"],
            activation_probability=0.42,
            probability=0.42,
            alambana_prior_probability=0.71,
        )
        payload = combo.to_dict()
        assert abs(payload["combo_activation_probability"] - 0.42) < 1e-9
        assert abs(payload["activation_probability"] - 0.42) < 1e-9
        assert abs(payload["alambana_prior_probability"] - 0.71) < 1e-9
        # 禁止用 prior 覆盖 activation
        assert payload["activation_probability"] != payload["alambana_prior_probability"]

    def test_unscored_combo_does_not_compete_with_default_alignment(self):
        """只给 C1 打 meta 时，未打分的 C2 不得凭默认 alignment=1 挤掉 C1。"""
        self_g = [
            _hyp("s1", "我是被当众点名的人", activation=0.8),
            _hyp("s2", "我是团队里可有可无的人", activation=0.8),
        ]
        dharma_g = [
            _hyp(
                "d1",
                "公开批评等于人格否定",
                activation=0.8,
                grasping_type="dharma_grasping",
            ),
            _hyp(
                "d2",
                "批评只是任务调整",
                activation=0.8,
                grasping_type="dharma_grasping",
            ),
        ]
        scored = GraspingCombination(
            combo_id="C1",
            used_self_ids=["s1"],
            used_dharma_ids=["d1"],
            self_confidence=0.8,
            dharma_confidence=0.8,
            activation_alignment=0.2,
            score=0.128,
            raw_combo_score=0.128,
        )
        unscored = GraspingCombination(
            combo_id="C2",
            used_self_ids=["s2"],
            used_dharma_ids=["d2"],
            self_confidence=0.8,
            dharma_confidence=0.8,
            activation_alignment=1.0,
            score=0.64,
            raw_combo_score=0.64,
        )
        selected = select_grasping_combinations(
            self_g,
            dharma_g,
            meta_by_combo_id={"C1": {"activation_alignment": 0.2}},
            candidates=[scored, unscored],
        )
        assert len(selected) == 1
        assert selected[0].used_self_ids == ["s1"]
        assert selected[0].used_dharma_ids == ["d1"]
        assert abs(selected[0].activation_alignment - 0.2) < 1e-9
