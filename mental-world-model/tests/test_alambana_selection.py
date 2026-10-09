"""所缘 Softmax 筛选：过阈值才开、最多 3、可为 1。"""

from mental_world_model.engine.perception.builder import select_valid_alambana_candidates
from mental_world_model.engine.perception.config import MAX_ALAMBANA_BRANCHES
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate


def _cand(
    cid: str,
    text: str,
    *,
    event: float,
    self_c: float,
    world: float,
    conf: float,
) -> SubjectiveAlambanaCandidate:
    return SubjectiveAlambanaCandidate(
        id=cid,
        text=text,
        event_consistency=event,
        self_consistency=self_c,
        world_consistency=world,
        confidence=conf,
        source="推理",
    )


class TestSelectValidAlambana:
    def test_max_branches_is_three(self):
        assert MAX_ALAMBANA_BRANCHES == 3

    def test_does_not_pad_below_threshold(self):
        notes: list[str] = []
        # 尖峰一致性 → Softmax 后弱支被阈值滤掉
        selected = select_valid_alambana_candidates(
            [
                _cand("a", "尖峰解释A", event=0.95, self_c=0.95, world=0.95, conf=0.7),
                _cand("b", "弱解释B", event=0.05, self_c=0.05, world=0.05, conf=0.1),
                _cand("c", "弱解释C", event=0.02, self_c=0.02, world=0.02, conf=0.1),
            ],
            uncertainty_notes=notes,
        )
        assert len(selected) == 1
        assert selected[0].id == "a"
        assert abs(selected[0].prior_probability - 1.0) < 1e-6

    def test_keeps_up_to_three_passing(self):
        notes: list[str] = []
        selected = select_valid_alambana_candidates(
            [
                _cand("a", "解释A", event=0.8, self_c=0.7, world=0.7, conf=0.5),
                _cand("b", "解释B", event=0.7, self_c=0.65, world=0.6, conf=0.5),
                _cand("c", "解释C", event=0.65, self_c=0.6, world=0.55, conf=0.5),
                _cand("d", "解释D", event=0.4, self_c=0.4, world=0.4, conf=0.5),
                _cand("e", "解释E", event=0.3, self_c=0.3, world=0.3, conf=0.5),
            ],
            uncertainty_notes=notes,
        )
        assert len(selected) == 3
        assert {c.id for c in selected} == {"a", "b", "c"}
        assert abs(sum(c.prior_probability for c in selected) - 1.0) < 1e-6

    def test_fallback_single_when_all_below(self):
        notes: list[str] = []
        selected = select_valid_alambana_candidates(
            [
                _cand("a", "勉强A", event=0.15, self_c=0.1, world=0.1, conf=0.1),
                _cand("b", "勉强B", event=0.12, self_c=0.1, world=0.1, conf=0.1),
            ],
            uncertainty_notes=notes,
        )
        assert len(selected) == 1
        assert selected[0].id == "a"
        assert any("保留最高" in n for n in notes)

    def test_prompt_allows_one_to_three(self):
        from mental_world_model.engine.perception.models import (
            GraspingHypothesis,
            ObjectiveEventStructure,
        )
        from mental_world_model.engine.perception.prompts import (
            build_subjective_alambana_messages,
        )

        system = build_subjective_alambana_messages(
            "原文",
            ObjectiveEventStructure(scene="会议", action="发言"),
            [
                GraspingHypothesis(
                    id="sg1", type="self_grasping", text="资格执", confidence=0.6
                )
            ],
            [],
        )[0]["content"]
        assert "主观所缘境渲染器" in system
        user = build_subjective_alambana_messages(
            "原文",
            ObjectiveEventStructure(scene="会议", action="发言"),
            [
                GraspingHypothesis(
                    id="sg1", type="self_grasping", text="资格执", confidence=0.6
                )
            ],
            [],
        )[1]["content"]
        assert "最多" in user
        assert "禁止" not in system or "行为" in system


class TestComboDrivenPriorFromActivation:
    def test_combo_probs_do_not_drop_low_confidence_branches(self):
        """组合已选的候选：低置信仍保留；prior 直接等于组合激活。"""
        from mental_world_model.engine.perception.builder import _select_after_softmax

        notes: list[str] = []
        cands = [
            _cand("O_C1", "组合一所缘文本", event=0.9, self_c=0.8, world=0.8, conf=0.7),
            _cand("O_C2", "组合二所缘文本", event=0.1, self_c=0.1, world=0.1, conf=0.1),
        ]
        cands[0].combo_reason = "C1"
        cands[1].combo_reason = "C2"
        kept = _select_after_softmax(
            cands,
            notes,
            combo_probs={"C1": 0.7, "C2": 0.3},
        )
        assert len(kept) == 2
        assert {c.id for c in kept} == {"O_C1", "O_C2"}
        by_id = {c.id: c for c in kept}
        assert abs(by_id["O_C1"].prior_probability - 0.7) < 1e-6
        assert abs(by_id["O_C2"].prior_probability - 0.3) < 1e-6
        assert abs(sum(c.prior_probability for c in kept) - 1.0) < 1e-6
        assert any("仍保留" in n for n in notes)

    def test_prior_ignores_consistency_softmax(self):
        """一致性再尖也不会改 prior（主路径只用 P(C)）。"""
        from mental_world_model.engine.perception.builder import _select_after_softmax

        notes: list[str] = []
        cands = [
            _cand("O_C1", "弱一致但高激活", event=0.1, self_c=0.1, world=0.1, conf=0.5),
            _cand("O_C2", "强一致但低激活", event=0.99, self_c=0.99, world=0.99, conf=0.9),
        ]
        cands[0].combo_reason = "C1"
        cands[1].combo_reason = "C2"
        kept = _select_after_softmax(
            cands,
            notes,
            combo_probs={"C1": 0.8, "C2": 0.2},
        )
        by_id = {c.id: c for c in kept}
        assert abs(by_id["O_C1"].prior_probability - 0.8) < 1e-6
        assert abs(by_id["O_C2"].prior_probability - 0.2) < 1e-6
