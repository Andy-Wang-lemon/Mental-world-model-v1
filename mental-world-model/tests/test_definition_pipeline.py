"""测试 Loader 扩展格式与 Normalizer 集成。"""

import json
from pathlib import Path

import pytest

from mental_world_model.engine.loader.definition_loader import DefinitionLoader
from mental_world_model.engine.normalizer.definition_normalizer import DefinitionNormalizer
from mental_world_model.model import MentalWorldModel


class TestDefinitionLoaderFormats:
    def test_load_raw_file_json(self, tmp_path):
        payload = [{"name": "信", "value": 1}]
        path = tmp_path / "sample.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert DefinitionLoader().load_raw_file(path) == payload

    def test_load_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "sample.yaml"
        path.write_text("name: 信\nvalue: 1\n", encoding="utf-8")
        assert DefinitionLoader().load_yaml(path) == {"name": "信", "value": 1}

    def test_load_markdown_json_block(self, tmp_path):
        path = tmp_path / "sample.md"
        path.write_text("# 定义\n\n```json\n{\"name\": \"信\"}\n```\n", encoding="utf-8")
        assert DefinitionLoader().load_markdown(path) == {"name": "信"}

    def test_load_raw_directory(self, tmp_path):
        (tmp_path / "a.json").write_text('{"a": 1}', encoding="utf-8")
        result = DefinitionLoader().load_raw_directory(tmp_path)
        assert result["a"] == {"a": 1}

        yaml = pytest.importorskip("yaml")
        del yaml
        (tmp_path / "b.yaml").write_text("b: 2\n", encoding="utf-8")
        result = DefinitionLoader().load_raw_directory(tmp_path)
        assert result["b"] == {"b": 2}

    def test_load_raw_file_unsupported_suffix(self, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            DefinitionLoader().load_raw_file(path)


class TestDefinitionNormalizerIntegration:
    def test_normalize_relation_maps_aliases(self):
        normalizer = DefinitionNormalizer()
        item = normalizer.normalize_relation_dict(
            {
                "source": "痴",
                "target": "无嗔",
                "description": "直接障碍无瞋。",
                "relation_type": "障碍",
                "is_direct": True,
            }
        )
        assert item["source"] == "无明"
        assert item["target"] == "无瞋"
        assert item["relation_type"] == "对治"


class TestMentalWorldModelLoadPipeline:
    def test_load_applies_normalizer_and_schema_validation(self):
        root = Path(__file__).resolve().parents[1] / "definitions"
        model = MentalWorldModel.load(root)
        assert len(model.xinsuo_list) == 40
        assert len(model.relations) == 40
        assert model.suoyuan_xiang_si_definitions.suoyuan_jing
