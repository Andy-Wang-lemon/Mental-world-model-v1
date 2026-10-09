"""连续事件更新：认知偏置 H + 心理惯性 M。"""

from mental_world_model.engine.continuity.config import (
    is_cognitive_continuity_enabled,
    is_mental_continuity_enabled,
)
from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.continuity.store import ContinuityStateStore

__all__ = [
    "ContinuityState",
    "ContinuityStateStore",
    "is_cognitive_continuity_enabled",
    "is_mental_continuity_enabled",
]
