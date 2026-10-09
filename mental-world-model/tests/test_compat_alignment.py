"""多支模式下解释 API 对齐跨支期望，不再依赖摘要支。"""

from mental_world_model.engine.cognition.models import CognitionBranch, ExtractedMentalState
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate
from mental_world_model.engine.scoring.results import EvaluationResult, XinsuoScoreResult


def _branch(bid: str, aid: str, scores: dict[str, float], stages: list) -> CognitionBranch:
    return CognitionBranch(
        id=bid,
        alambana=SubjectiveAlambanaCandidate(id=aid, text=f"所缘{aid}"),
        xiang=ExtractedMentalState(
            text="想", source="推理", confidence=0.8, alambana_id=aid
        ),
        activation_probability=0.5,
        mental_factor_scores_final=scores,
        stage_results=stages,
    )


class TestMultiBranchExplain:
    def test_only_nonzero_uses_expected_scores_final(self):
        stage_a = XinsuoScoreResult(name="无嗔", exit_at="suoyuan", final_score=0.0)
        stage_b = XinsuoScoreResult(name="嗔", exit_at=None, final_score=0.8)
        branch = _branch(
            "branch_a",
            "O1",
            {"无嗔": 0.0, "嗔": 0.8},
            [stage_a, stage_b],
        )
        result = EvaluationResult(
            extracted=None,
            stage_results=[],
            scores_initial={"无嗔": 0.0, "嗔": 0.8},
            conflicts_initial=[],
            conflicts_final=[],
            resolutions=[],
            scores_final={"无嗔": 0.0, "嗔": 0.4},
            cognition_branches=[branch],
        )
        text = result.explain_all_scores(only_nonzero=True)
        assert "无嗔" not in text
        assert "嗔" in text or "branch_a" in text

    def test_explain_includes_all_active_branches(self):
        b1 = _branch(
            "branch_1",
            "O1",
            {"嗔": 0.7},
            [XinsuoScoreResult(name="嗔", exit_at=None, final_score=0.7)],
        )
        b2 = _branch(
            "branch_2",
            "O2",
            {"嗔": 0.3},
            [XinsuoScoreResult(name="嗔", exit_at=None, final_score=0.3)],
        )
        b1.activation_probability = 0.6
        b2.activation_probability = 0.4
        result = EvaluationResult(
            extracted=None,
            stage_results=[],
            scores_initial={"嗔": 0.54},
            conflicts_initial=[],
            conflicts_final=[],
            resolutions=[],
            scores_final={"嗔": 0.54},
            cognition_branches=[b1, b2],
        )
        text = result.explain_all_scores(only_nonzero=True)
        assert "branch_1" in text
        assert "branch_2" in text
