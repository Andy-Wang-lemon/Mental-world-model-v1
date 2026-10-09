"""执取组合（兼容壳）：委托 combo_selection。

主路径使用 ``combo_selection.select_grasping_combinations``。
本模块保留 ``build_grasping_combos`` 供旧测试与脚本。
"""

from __future__ import annotations

from dataclasses import dataclass

from mental_world_model.engine.perception.combo_selection import (
    select_grasping_combinations,
    select_object_representatives,
)
from mental_world_model.engine.perception.config import (
    GRASPING_COMBO_ACTIVATION,
    GRASPING_COMBO_SECONDARY_RATIO,
    MAX_ALAMBANA_BRANCHES,
)
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
)


@dataclass(frozen=True)
class GraspingCombo:
    """旧结构：一组用于染色的我执/法执。"""

    id: str
    self_grasping: tuple[GraspingHypothesis, ...]
    dharma_grasping: tuple[GraspingHypothesis, ...]

    @property
    def self_ids(self) -> list[str]:
        return [h.id for h in self.self_grasping]

    @property
    def dharma_ids(self) -> list[str]:
        return [h.id for h in self.dharma_grasping]

    def activation_score(self) -> float:
        acts = [
            h.confidence
            for h in (*self.self_grasping, *self.dharma_grasping)
        ]
        if not acts:
            return 0.0
        return sum(acts) / len(acts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "self_grasping_ids": self.self_ids,
            "dharma_grasping_ids": self.dharma_ids,
            "activation_score": self.activation_score(),
        }


def build_grasping_combos(
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    *,
    activation_threshold: float = GRASPING_COMBO_ACTIVATION,
    secondary_ratio: float = GRASPING_COMBO_SECONDARY_RATIO,
    max_combos: int = MAX_ALAMBANA_BRANCHES,
) -> list[GraspingCombo]:
    """兼容入口：返回旧 GraspingCombo 列表。"""
    selected = select_grasping_combinations(
        self_grasping,
        dharma_grasping,
        activation_threshold=activation_threshold,
        max_keep=max_combos,
        secondary_ratio=secondary_ratio,
    )
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    out: list[GraspingCombo] = []
    for combo in selected:
        selves = tuple(
            self_by_id[i] for i in combo.used_self_ids if i in self_by_id
        )
        dharmas = tuple(
            dharma_by_id[i] for i in combo.used_dharma_ids if i in dharma_by_id
        )
        out.append(
            GraspingCombo(
                id=combo.combo_id,
                self_grasping=selves,
                dharma_grasping=dharmas,
            )
        )
    return out


__all__ = [
    "GraspingCombo",
    "GraspingCombination",
    "build_grasping_combos",
    "select_grasping_combinations",
    "select_object_representatives",
]
