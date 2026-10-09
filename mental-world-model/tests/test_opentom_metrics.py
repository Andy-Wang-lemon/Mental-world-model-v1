"""OpenToM 论文级指标测试。"""

from mwm_test_adapter.opentom_metrics import (
    compute_opentom_paper_metrics,
    macro_f1_single_label,
    normalize_paper_question_type,
)


class TestOpenToMMetrics:
    def test_normalize_question_type(self):
        assert normalize_paper_question_type("location_cg_fo") == "location-fo-coarse"
        assert normalize_paper_question_type("multihop_so") == "multihop-so"
        assert normalize_paper_question_type("attitude") == "attitude"

    def test_macro_f1_perfect(self):
        gold = ["yes", "no", "yes"]
        pred = ["yes", "no", "yes"]
        assert macro_f1_single_label(gold, pred) == 1.0

    def test_paper_metrics_by_type(self):
        metrics = compute_opentom_paper_metrics(
            gold_labels=[["negative"], ["yes"], ["no"]],
            pred_labels=[["negative"], ["yes"], ["yes"]],
            question_types=["attitude", "location_cg_fo", "location_cg_fo"],
        )
        assert metrics.accuracy == 2 / 3
        assert "attitude" in metrics.by_question_type
        assert "location-fo-coarse" in metrics.by_question_type
        assert metrics.by_question_type["attitude"] == 1.0
        assert 0.0 <= metrics.macro_avg_f1 <= 1.0
