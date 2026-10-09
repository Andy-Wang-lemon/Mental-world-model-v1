"""支内 Top3 行为分布与跨支融合。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.cognition.behavior import (
    _keep_top_k_behaviors,
    predict_behaviors_for_branch,
)
from mental_world_model.engine.cognition.models import (
    BranchBehaviorPrediction,
    CognitionBranch,
    ExtractedMentalState,
)
from mental_world_model.engine.cognition.probability import (
    aggregate_behavior_predictions,
    normalize_branch_probabilities,
)
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate


def _pred(text: str, p: float) -> BranchBehaviorPrediction:
    return BranchBehaviorPrediction(
        behavior_id=text,
        text=text,
        conditional_probability=p,
        confidence=0.7,
    )


def _branch(bid: str, prior: float, behaviors: list[BranchBehaviorPrediction]) -> CognitionBranch:
    return CognitionBranch(
        id=bid,
        alambana=SubjectiveAlambanaCandidate(
            id=bid,
            text=f"所缘{bid}",
            posterior_probability=prior,
            confidence=1.0,
            source="推理",
        ),
        branch_prior_probability=prior,
        branch_confidence=1.0,
        behavior_predictions=behaviors,
        xiang=ExtractedMentalState(
            text="想", source="推理", confidence=0.7, alambana_id=bid
        ),
    )


class TestBranchActionTopK:
    def test_keep_top_3_and_renormalize(self):
        preds = [
            _pred("a", 0.4),
            _pred("b", 0.3),
            _pred("c", 0.2),
            _pred("d", 0.1),
        ]
        kept = _keep_top_k_behaviors(preds, top_k=3)
        assert [p.text for p in kept] == ["a", "b", "c"]
        assert abs(sum(p.conditional_probability for p in kept) - 1.0) < 1e-9
        assert abs(kept[0].conditional_probability - 0.4 / 0.9) < 1e-9

    def test_predict_truncates_to_top_3(self):
        class Client:
            def complete(self, messages, schema=None):
                return {
                    "behavior_predictions": [
                        {"text": f"行为{i}", "conditional_probability": p, "confidence": 0.6}
                        for i, p in enumerate([0.35, 0.25, 0.2, 0.15, 0.05], start=1)
                    ]
                }

        branch = CognitionBranch(
            id="b1",
            alambana=SubjectiveAlambanaCandidate(
                id="O1", text="场景", posterior_probability=0.5, confidence=0.7
            ),
            xiang=ExtractedMentalState(
                text="想", source="推理", confidence=0.6, alambana_id="O1"
            ),
        )
        xinsuo = {
            "瞋": Xinsuo(name="瞋", suoyuan="s", shengqi_nian="n", ye="y", xiandai_xinli="x"),
        }
        preds = predict_behaviors_for_branch(
            "原文",
            branch,
            {"瞋": 0.8},
            xinsuo,
            Client(),  # type: ignore[arg-type]
        )
        assert len(preds) == 3
        assert abs(sum(p.conditional_probability for p in preds) - 1.0) < 1e-9
        assert preds[0].text == "行为1"


class TestCrossBranchFusion:
    def test_aggregate_top_k_renormalizes(self):
        branches = [
            _branch(
                "branch_a",
                0.6,
                [_pred("帮助", 0.7), _pred("拒绝", 0.3)],
            ),
            _branch(
                "branch_b",
                0.4,
                [_pred("帮助", 0.2), _pred("观望", 0.8)],
            ),
        ]
        normalize_branch_probabilities(branches)
        aggregated = aggregate_behavior_predictions(branches, top_k=2)
        assert len(aggregated) == 2
        assert abs(sum(a.total_probability for a in aggregated) - 1.0) < 1e-9
        # 帮助 = 0.6*0.7 + 0.4*0.2 = 0.50；观望=0.32；拒绝=0.18 → top2 帮助+观望
        assert aggregated[0].canonical_text == "帮助"
        assert {a.canonical_text for a in aggregated} == {"帮助", "观望"}
