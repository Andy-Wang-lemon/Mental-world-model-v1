"""预测行动合法性：空预测 vs 明确无反应。"""

from mental_world_model.engine.cognition.predicted_action import (
    is_usable_predicted_action,
    top_aggregated_action,
)


class TestIsUsablePredictedAction:
    def test_empty_and_placeholders_rejected(self):
        assert is_usable_predicted_action("") is False
        assert is_usable_predicted_action("   ") is False
        assert is_usable_predicted_action("（无行为预测）") is False
        assert is_usable_predicted_action("（空行为）") is False
        assert is_usable_predicted_action("（无行为）") is False

    def test_no_reaction_is_usable(self):
        assert is_usable_predicted_action("没表现") is True
        assert is_usable_predicted_action("没有反应") is True
        assert is_usable_predicted_action("表面没有反应") is True


class TestTopAggregatedAction:
    def test_empty_result_is_placeholder(self):
        class _Empty:
            aggregated_behaviors = []
            cognition_branches = []

        text, prob, top_k = top_aggregated_action(_Empty())
        assert text == "（无行为预测）"
        assert prob == 0.0
        assert top_k == []
