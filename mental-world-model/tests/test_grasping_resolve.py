"""执取 ID 解析：仅精确匹配，无模糊/无强执回退。"""

from mental_world_model.engine.cognition.runner import _resolve_grasping
from mental_world_model.engine.perception.models import GraspingHypothesis


def _hyp(hid: str, text: str, activation: float) -> GraspingHypothesis:
    return GraspingHypothesis(
        id=hid,
        type="self_grasping",
        text=text,
        source="推理",
        confidence=activation,
    )


class TestResolveGrasping:
    def test_exact_id_match(self):
        hyps = [_hyp("self_grasping_1", "把资格当成我", 0.8)]
        resolved, notes = _resolve_grasping(
            ["self_grasping_1"],
            hyps,
            alambana_id="a1",
            kind="我执",
        )
        assert [h.id for h in resolved] == ["self_grasping_1"]
        assert notes == []

    def test_text_match_and_invalid_no_fuzzy(self):
        hyps = [
            _hyp("self_grasping_1", "把资格当成我", 0.9),
            _hyp("self_grasping_2", "较弱等同", 0.3),
        ]
        resolved, notes = _resolve_grasping(
            ["把资格当成我", "sg_missing"],
            hyps,
            alambana_id="a1",
            kind="我执",
        )
        assert [h.id for h in resolved] == ["self_grasping_1"]
        assert any("按精确文本匹配" in n for n in notes)
        assert any("引用无效" in n for n in notes)

    def test_empty_ids_stay_empty(self):
        hyps = [
            _hyp("g_low", "低激活", 0.2),
            _hyp("g_high", "高激活", 0.95),
        ]
        resolved, notes = _resolve_grasping(
            [],
            hyps,
            alambana_id="a1",
            kind="我执",
        )
        assert resolved == []
        assert notes == []

    def test_all_invalid_ids_stay_empty(self):
        hyps = [
            _hyp("g_low", "低激活", 0.2),
            _hyp("g_high", "高激活", 0.95),
        ]
        resolved, notes = _resolve_grasping(
            ["sg_missing"],
            hyps,
            alambana_id="a1",
            kind="我执",
        )
        assert resolved == []
        assert any("全部引用无效" in n or "留空" in n for n in notes)
