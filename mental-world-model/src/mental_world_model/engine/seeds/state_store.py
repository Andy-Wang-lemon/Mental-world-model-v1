"""用户运行时种子库持久化。"""

import json
from pathlib import Path

from mental_world_model.engine.seeds.config import DEFAULT_SEED_STATE_FILENAME


class SeedStateStore:
    """读写 ``data/runtime/seed_state.json``（或自定义路径）。"""

    def __init__(self, state_path: str | Path) -> None:
        self._path = Path(state_path)

    @classmethod
    def default_path(cls, project_root: str | Path) -> "SeedStateStore":
        return cls(Path(project_root) / "data" / "runtime" / DEFAULT_SEED_STATE_FILENAME)

    @property
    def path(self) -> Path:
        return self._path

    def load_or_initialize(self, default_weights: dict[str, float]) -> dict[str, float]:
        """若状态文件存在则加载，否则复制默认种子库。"""
        if not self._path.is_file():
            return dict(default_weights)

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"种子状态文件 JSON 无效：{self._path}") from error

        weights = data.get("weights")
        if not isinstance(weights, dict):
            raise ValueError(f"种子状态文件缺少 weights 对象：{self._path}")

        return {str(name): float(value) for name, value in weights.items()}

    def save(self, weights: dict[str, float]) -> None:
        """持久化运行时种子库。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "description": "用户运行时心所种子库（交互后 EMA 更新）",
            "weights": {name: float(value) for name, value in weights.items()},
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
