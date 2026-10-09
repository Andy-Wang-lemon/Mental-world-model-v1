"""心所种子库权重加载。"""

import json
from pathlib import Path
from typing import Any

from mental_world_model.utils.xinsuo_names import normalize_xinsuo_name

DEFAULT_SEED_FILENAME = "xinsuo_seeds.json"


class SeedLoaderError(ValueError):
    """种子库文件无效或心所名称不匹配。"""


def _parse_seed_entries(data: Any) -> dict[str, float]:
    """从种子库 JSON 解析 name → weight。"""
    if isinstance(data, dict) and "weights" in data:
        raw_weights = data["weights"]
        if not isinstance(raw_weights, dict):
            raise SeedLoaderError("weights 应为对象")
        return {normalize_xinsuo_name(str(name)): float(weight) for name, weight in raw_weights.items()}

    if isinstance(data, dict) and "categories" in data:
        categories = data["categories"]
        if not isinstance(categories, list):
            raise SeedLoaderError("categories 应为数组")
        weights: dict[str, float] = {}
        for index, category in enumerate(categories):
            if not isinstance(category, dict):
                raise SeedLoaderError(f"categories[{index}] 应为对象")
            entries = category.get("entries", [])
            if not isinstance(entries, list):
                raise SeedLoaderError(f"categories[{index}].entries 应为数组")
            for entry_index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise SeedLoaderError(
                        f"categories[{index}].entries[{entry_index}] 应为对象"
                    )
                name = normalize_xinsuo_name(str(entry.get("name", "")).strip())
                if not name:
                    raise SeedLoaderError(
                        f"categories[{index}].entries[{entry_index}].name 不能为空"
                    )
                if "weight" not in entry:
                    raise SeedLoaderError(f"{name} 缺少 weight")
                weights[name] = float(entry["weight"])
        return weights

    raise SeedLoaderError("种子库 JSON 需包含 categories 或 weights 字段")


def _validate_seed_weights(weights: dict[str, float], xinsuo_names: list[str]) -> dict[str, float]:
    expected = set(xinsuo_names)
    loaded = set(weights.keys())

    missing = sorted(expected - loaded)
    if missing:
        raise SeedLoaderError(f"种子库缺少心所：{', '.join(missing)}")

    extra = sorted(loaded - expected)
    if extra:
        raise SeedLoaderError(f"种子库包含未知心所：{', '.join(extra)}")

    for name, weight in weights.items():
        if weight < 0:
            raise SeedLoaderError(f"{name} 的 weight 不能为负数：{weight}")

    return {name: weights[name] for name in xinsuo_names}


def parse_seed_library_data(data: Any, xinsuo_names: list[str]) -> dict[str, float]:
    """解析并校验种子库数据结构。"""
    return _validate_seed_weights(_parse_seed_entries(data), xinsuo_names)


def load_xinsuo_seed_weights(
    definitions_dir: Path,
    xinsuo_names: list[str],
    *,
    filename: str = DEFAULT_SEED_FILENAME,
) -> dict[str, float]:
    """从 definitions 加载心所种子库权重（经 DefinitionLoader 同步 raw）。"""
    from mental_world_model.engine.loader.definition_loader import DefinitionLoader

    loader = DefinitionLoader()
    try:
        data = loader.load_xinsuo_seed_definitions(definitions_dir)
    except FileNotFoundError as error:
        seed_path = definitions_dir / "seeds" / filename
        if seed_path.is_file():
            try:
                data = json.loads(seed_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as decode_error:
                raise SeedLoaderError(f"种子库 JSON 解析失败：{decode_error}") from decode_error
        else:
            raise SeedLoaderError(f"未找到种子库：{error}") from error

    return parse_seed_library_data(data, xinsuo_names)
