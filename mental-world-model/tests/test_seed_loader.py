"""心所种子库加载测试。"""

import json
from pathlib import Path

import pytest

from mental_world_model.engine.loader.seed_loader import (
    SeedLoaderError,
    load_xinsuo_seed_weights,
)
from mental_world_model.engine.seeds import config as seeds_config
from mental_world_model.engine.seeds.config import is_seed_library_enabled
from mental_world_model.engine.scoring.weights import build_default_seed_weights
from mental_world_model.model import MentalWorldModel


class TestSeedLoader:
    def test_load_default_seed_file(self):
        root = Path(__file__).resolve().parents[1]
        names = [x.name for x in MentalWorldModel.load(root / "definitions").xinsuo_list]
        weights = load_xinsuo_seed_weights(root / "definitions", names)
        assert len(weights) == 40
        assert weights["信"] == pytest.approx(0.45)
        assert "无明" in weights

    def test_build_default_seed_weights_requires_definitions_dir(self):
        weights = build_default_seed_weights(["信", "精进"])
        assert weights == {"信": 1.0, "精进": 1.0}

    def test_missing_seed_file_raises(self, tmp_path):
        with pytest.raises(SeedLoaderError, match="未找到种子库"):
            load_xinsuo_seed_weights(tmp_path, ["信"])

    def test_incomplete_seed_file_raises(self, tmp_path):
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        (seeds_dir / "xinsuo_seeds.json").write_text(
            json.dumps({"weights": {"信": 0.6}}),
            encoding="utf-8",
        )
        with pytest.raises(SeedLoaderError, match="缺少心所"):
            load_xinsuo_seed_weights(tmp_path, ["信", "精进"])

    def test_is_seed_library_enabled_reads_config(self, monkeypatch):
        monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", True)
        assert is_seed_library_enabled() is True
        monkeypatch.setattr(seeds_config, "SEED_LIBRARY_ENABLED", False)
        assert is_seed_library_enabled() is False
