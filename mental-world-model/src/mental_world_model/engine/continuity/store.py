"""连续事件状态持久化。"""

from __future__ import annotations

import json
from pathlib import Path

from mental_world_model.engine.continuity.config import DEFAULT_CONTINUITY_STATE_FILENAME
from mental_world_model.engine.continuity.models import ContinuityState, utc_now_iso
from mental_world_model.utils.atomic_io import atomic_write_json


class ContinuityStateStore:
    """读写 ``data/runtime/continuity_state.json``。"""

    def __init__(self, state_path: str | Path) -> None:
        self._path = Path(state_path)

    @classmethod
    def default_path(cls, project_root: str | Path) -> "ContinuityStateStore":
        return cls(Path(project_root) / "data" / "runtime" / DEFAULT_CONTINUITY_STATE_FILENAME)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> ContinuityState:
        if not self._path.is_file():
            return ContinuityState()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"连续状态 JSON 无效：{self._path}") from error
        return ContinuityState.from_dict(data if isinstance(data, dict) else {})

    def save(self, state: ContinuityState) -> None:
        state.updated_at = utc_now_iso()
        atomic_write_json(self._path, state.to_dict())
