"""所缘境合并分组测试。"""

from pathlib import Path

from mental_world_model.engine.scoring.results import DEPENDENT_ON_FEN
from mental_world_model.engine.scoring.suoyuan_groups import (
    MERGED_SUOYUAN_GROUPS,
    SINGLETON_SUOYUAN_GROUPS,
    SUOYUAN_BATCH_SKIP_NAMES,
    parse_group_suoyuan_response,
    plan_suoyuan_scoring_groups,
)
from mental_world_model.model import MentalWorldModel


class TestSuoyuanGroups:
    def _xinsuo_by_name(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        return {item.name: item for item in model.xinsuo_list}

    def test_all_main_loop_xinsuo_mapped(self):
        by_name = self._xinsuo_by_name()
        mapped: set[str] = set()
        for group in MERGED_SUOYUAN_GROUPS:
            mapped.update(group.xinsuo_names)
        for group in SINGLETON_SUOYUAN_GROUPS:
            mapped.update(group.xinsuo_names)
        expected = set(by_name) - SUOYUAN_BATCH_SKIP_NAMES
        assert mapped == expected

    def test_plan_batches_count_for_full_list(self):
        by_name = self._xinsuo_by_name()
        batches = plan_suoyuan_scoring_groups(by_name)
        assert len(batches) == 21

    def test_merged_group_batch_contains_all_members(self):
        by_name = self._xinsuo_by_name()
        batches = plan_suoyuan_scoring_groups(by_name)
        xin_batch = next(
            batch for batch in batches if "信" in batch.xinsuo_names
        )
        assert xin_batch.xinsuo_names == ("信", "不信")

    def test_parse_group_response(self):
        parsed = parse_group_suoyuan_response(
            {
                "scores": {
                    "信": {"score": 0.8},
                    "不信": {"score": 0.1},
                }
            },
            ("信", "不信"),
        )
        assert parsed["信"] == 0.8
        assert parsed["不信"] == 0.1
