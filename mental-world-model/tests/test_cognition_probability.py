"""认知分支概率与汇总纯函数测试。"""

from mental_world_model.engine.cognition.models import (
    BranchBehaviorPrediction,
    CognitionBranch,
    ExtractedMentalState,
)
from mental_world_model.engine.cognition.probability import (
    aggregate_behavior_predictions,
    calculate_expected_mental_factor_scores,
    calculate_mental_factor_presence_probabilities,
    compute_branch_confidence,
    geometric_mean,
    normalize_branch_probabilities,
)
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate
from mental_world_model.engine.perception.validators import (
    normalize_objective_facts,
    normalize_probability_distribution,
)


def _alambana(aid: str, text: str, prob: float = 0.5) -> SubjectiveAlambanaCandidate:
    return SubjectiveAlambanaCandidate(
        id=aid,
        text=text,
        posterior_probability=prob,
        confidence=0.8,
        source="推理",
    )


def _branch(
    bid: str,
    prior: float,
    confidence: float,
    scores: dict[str, float],
    behaviors: list[BranchBehaviorPrediction] | None = None,
) -> CognitionBranch:
    branch = CognitionBranch(
        id=bid,
        alambana=_alambana(bid.replace("branch_", ""), f"所缘{bid}", prior),
        xiang=ExtractedMentalState(
            text="想",
            source="推理",
            confidence=confidence,
            alambana_id=bid,
        ),
        si=ExtractedMentalState(
            text="思",
            source="推理",
            confidence=confidence,
            alambana_id=bid,
        ),
        branch_prior_probability=prior,
        branch_confidence=confidence,
        mental_factor_scores_final=scores,
        behavior_predictions=behaviors or [],
    )
    return branch


class TestProbabilityHelpers:
    def test_geometric_mean(self):
        assert geometric_mean([1.0, 1.0, 1.0]) == 1.0
        assert geometric_mean([0.0, 0.5, 1.0]) == 0.0
        assert abs(geometric_mean([0.25, 0.25]) - 0.25) < 1e-9

    def test_branch_confidence(self):
        assert compute_branch_confidence(1.0, 1.0, 1.0) == 1.0
        assert compute_branch_confidence(0.8, 0.8, 0.8) == 0.8

    def test_normalize_distribution(self):
        assert normalize_probability_distribution([0.2, 0.6, 0.2]) == [
            0.2,
            0.6,
            0.2,
        ]
        uniform = normalize_probability_distribution([0.0, 0.0])
        assert abs(sum(uniform) - 1.0) < 1e-9
        assert uniform[0] == 0.5

    def test_branch_probability_normalization(self):
        branches = [
            _branch("branch_a", 0.7, 1.0, {}),
            _branch("branch_b", 0.3, 1.0, {}),
        ]
        normalize_branch_probabilities(branches)
        total = sum(b.activation_probability for b in branches)
        assert abs(total - 1.0) < 1e-9
        assert abs(branches[0].activation_probability - 0.7) < 1e-9
        assert abs(branches[0].branch_effective_probability - 0.7) < 1e-9
        assert abs(branches[0].behavior_contribution_probability - 0.7) < 1e-9

    def test_activation_ignores_branch_confidence(self):
        """a 只跟 prior；支置信不同也不改激活权重。"""
        a = _branch("branch_a", 0.6, 0.2, {})
        b = _branch("branch_b", 0.4, 1.0, {})
        a.alambana.prior_probability = 0.6
        b.alambana.prior_probability = 0.4
        normalize_branch_probabilities([a, b])
        assert abs(a.activation_probability - 0.6) < 1e-9
        assert abs(b.activation_probability - 0.4) < 1e-9
        assert abs(a.behavior_contribution_probability - 0.6) < 1e-9

    def test_failed_branch_gets_zero_weight(self):
        alive = _branch("branch_a", 0.6, 1.0, {"忿": 1.0})
        failed = _branch("branch_b", 0.4, 1.0, {"忿": 0.0})
        failed.status = "scoring_failed"
        normalize_branch_probabilities([alive, failed])
        assert abs(alive.branch_effective_probability - 1.0) < 1e-9
        assert failed.branch_effective_probability == 0.0
        expected = calculate_expected_mental_factor_scores(
            [alive, failed], ["忿"]
        )
        assert abs(expected["忿"] - 1.0) < 1e-9

    def test_all_failed_branches_zero_mass(self):
        a = _branch("branch_a", 0.5, 1.0, {"忿": 1.0})
        b = _branch("branch_b", 0.5, 1.0, {"忿": 0.5})
        a.status = "scoring_failed"
        b.status = "scoring_failed"
        normalize_branch_probabilities([a, b])
        assert a.branch_effective_probability == 0.0
        assert b.branch_effective_probability == 0.0
        expected = calculate_expected_mental_factor_scores([a, b], ["忿"])
        assert expected["忿"] == 0.0

    def test_expected_and_presence_case1(self):
        branches = [
            _branch("branch_a", 0.2, 1.0, {"忿": 1.0}),
            _branch("branch_b", 0.8, 1.0, {"忿": 0.0}),
        ]
        normalize_branch_probabilities(branches)
        expected = calculate_expected_mental_factor_scores(branches, ["忿"])
        presence = calculate_mental_factor_presence_probabilities(branches, ["忿"])
        assert abs(expected["忿"] - 0.2) < 1e-9
        assert abs(presence["忿"] - 0.2) < 1e-9

    def test_expected_and_presence_case2(self):
        branches = [
            _branch("branch_a", 0.5, 1.0, {"忿": 0.4}),
            _branch("branch_b", 0.5, 1.0, {"忿": 0.4}),
        ]
        normalize_branch_probabilities(branches)
        expected = calculate_expected_mental_factor_scores(branches, ["忿"])
        presence = calculate_mental_factor_presence_probabilities(branches, ["忿"])
        assert abs(expected["忿"] - 0.4) < 1e-9
        assert abs(presence["忿"] - 1.0) < 1e-9

    def test_behavior_aggregation_prefers_high_conditional(self):
        """高条件概率的低权重分支可超过低条件概率的高权重分支。"""
        branches = [
            _branch(
                "branch_a",
                0.7,
                1.0,
                {"忿": 0.8},
                [
                    BranchBehaviorPrediction(
                        behavior_id="X",
                        text="行为X",
                        conditional_probability=0.30,
                        confidence=0.7,
                    )
                ],
            ),
            _branch(
                "branch_b",
                0.3,
                1.0,
                {"信": 0.8},
                [
                    BranchBehaviorPrediction(
                        behavior_id="Y",
                        text="行为Y",
                        conditional_probability=0.90,
                        confidence=0.7,
                    )
                ],
            ),
        ]
        normalize_branch_probabilities(branches)
        aggregated = aggregate_behavior_predictions(branches)
        by_id = {a.behavior_id: a for a in aggregated}
        assert by_id["Y"].total_probability > by_id["X"].total_probability
        assert abs(by_id["X"].total_probability - 0.21) < 1e-9
        assert abs(by_id["Y"].total_probability - 0.27) < 1e-9

    def test_same_behavior_keeps_multiple_paths(self):
        branches = [
            _branch(
                "branch_a",
                0.5,
                1.0,
                {"恨": 0.8},
                [
                    BranchBehaviorPrediction(
                        behavior_id="silence",
                        text="保持沉默",
                        conditional_probability=1.0,
                        confidence=0.7,
                        supporting_mental_factors=["恨", "覆"],
                        supporting_scores={"恨": 0.8},
                    )
                ],
            ),
            _branch(
                "branch_b",
                0.5,
                1.0,
                {"念": 0.7},
                [
                    BranchBehaviorPrediction(
                        behavior_id="silence",
                        text="保持沉默",
                        conditional_probability=1.0,
                        confidence=0.7,
                        supporting_mental_factors=["念", "慧"],
                        supporting_scores={"念": 0.7},
                    )
                ],
            ),
        ]
        normalize_branch_probabilities(branches)
        aggregated = aggregate_behavior_predictions(branches)
        assert len(aggregated) == 1
        assert len(aggregated[0].branch_contributions) == 2
        factors = {
            tuple(c.supporting_mental_factors)
            for c in aggregated[0].branch_contributions
        }
        assert ("恨", "覆") in factors
        assert ("念", "慧") in factors

    def test_activation_and_behavior_use_same_a(self):
        """L/compat 不进分支概率；a 与兼容字段 b 恒等；行为融合跟 a。"""
        a = _branch(
            "branch_a",
            0.6,
            1.0,
            {"忿": 1.0},
            [
                BranchBehaviorPrediction(
                    behavior_id="X",
                    text="行为X",
                    conditional_probability=1.0,
                    confidence=0.7,
                )
            ],
        )
        b = _branch(
            "branch_b",
            0.4,
            1.0,
            {"信": 1.0},
            [
                BranchBehaviorPrediction(
                    behavior_id="Y",
                    text="行为Y",
                    conditional_probability=1.0,
                    confidence=0.7,
                )
            ],
        )
        a.alambana.prior_probability = 0.6
        a.alambana.action_likelihood = 0.2
        b.alambana.prior_probability = 0.4
        b.alambana.action_likelihood = 0.9
        normalize_branch_probabilities([a, b])
        assert abs(a.activation_probability - 0.6) < 1e-9
        assert abs(b.activation_probability - 0.4) < 1e-9
        assert abs(a.behavior_contribution_probability - 0.6) < 1e-9
        assert abs(b.behavior_contribution_probability - 0.4) < 1e-9
        expected = calculate_expected_mental_factor_scores([a, b], ["忿", "信"])
        assert abs(expected["忿"] - 0.6) < 1e-9
        assert abs(expected["信"] - 0.4) < 1e-9
        aggregated = aggregate_behavior_predictions([a, b])
        by_id = {item.behavior_id: item for item in aggregated}
        assert abs(by_id["X"].total_probability - 0.6) < 1e-9
        assert abs(by_id["Y"].total_probability - 0.4) < 1e-9

    def test_behavior_fusion_follows_activation_not_likelihood(self):
        """高激活低 L：行为融合仍跟 a，不再被 L 压低。"""
        latent = _branch(
            "branch_latent",
            0.7,
            1.0,
            {"覆": 1.0},
            [
                BranchBehaviorPrediction(
                    behavior_id="X",
                    text="行为X",
                    conditional_probability=1.0,
                    confidence=0.7,
                )
            ],
        )
        overt = _branch(
            "branch_overt",
            0.3,
            1.0,
            {"念": 1.0},
            [
                BranchBehaviorPrediction(
                    behavior_id="Y",
                    text="行为Y",
                    conditional_probability=1.0,
                    confidence=0.7,
                )
            ],
        )
        latent.alambana.prior_probability = 0.7
        latent.alambana.action_likelihood = 0.2
        overt.alambana.prior_probability = 0.3
        overt.alambana.action_likelihood = 0.9
        normalize_branch_probabilities([latent, overt])
        assert abs(latent.activation_probability - 0.7) < 1e-9
        assert abs(overt.activation_probability - 0.3) < 1e-9
        assert abs(latent.behavior_contribution_probability - 0.7) < 1e-9
        assert abs(overt.behavior_contribution_probability - 0.3) < 1e-9
        expected = calculate_expected_mental_factor_scores(
            [latent, overt], ["覆", "念"]
        )
        assert abs(expected["覆"] - 0.7) < 1e-9
        aggregated = aggregate_behavior_predictions([latent, overt])
        by_id = {item.behavior_id: item for item in aggregated}
        assert by_id["X"].total_probability > by_id["Y"].total_probability


class TestObjectiveFactGuard:
    def test_ungrounded_facts_dropped(self):
        facts = normalize_objective_facts(
            [
                {"text": "他很愤怒", "source_span": "他很愤怒"},
                {"text": "他看了对方一眼", "source_span": "他看了对方一眼"},
            ],
            "他看了对方一眼。",
        )
        assert len(facts) == 1
        assert facts[0].confidence == 1.0


class TestGraspingSimplify:
    def test_confidence_only_and_salience_compat(self):
        from mental_world_model.engine.perception.validators import (
            normalize_grasping_hypothesis,
        )

        hyp = normalize_grasping_hypothesis(
            {
                "id": "g1",
                "text": "把领导资格执为我",
                "supporting_evidence": [],
                "counter_evidence": ["有反证也不再压"],
                "source": "推理",
                "confidence": 0.7,
            },
            grasping_type="self_grasping",
            yuanwen="员工指出风险。",
            index=0,
        )
        assert hyp is not None
        assert hyp.confidence == 0.7
        assert "effective_activation" not in hyp.to_dict()
        assert "salience" not in hyp.to_dict()

        hyp2 = normalize_grasping_hypothesis(
            {
                "id": "g2",
                "text": "把规则执为不可违",
                "salience": 0.65,
                "source": "推理",
            },
            grasping_type="dharma_grasping",
            yuanwen="员工指出风险。",
            index=1,
        )
        assert hyp2 is not None
        assert abs(hyp2.confidence - 0.65) < 1e-9

    def test_ungrounded_yuanwen_source_becomes_inference(self):
        from mental_world_model.engine.perception.validators import (
            normalize_grasping_hypothesis,
        )

        hyp = normalize_grasping_hypothesis(
            {
                "text": "完全编造的执取表述",
                "source": "原文",
                "confidence": 1.0,
            },
            grasping_type="dharma_grasping",
            yuanwen="员工指出风险。",
            index=0,
        )
        assert hyp is not None
        assert hyp.source == "推理"
        assert hyp.confidence <= 0.8
