"""测试定义加载相关类的导入、实例化与 JSON 加载。"""

from pathlib import Path

from mental_world_model.engine.loader.definition_loader import DefinitionLoader
from mental_world_model.engine.parser.definition_parser import DefinitionParser
from mental_world_model.engine.normalizer.definition_normalizer import DefinitionNormalizer
from mental_world_model.engine.validator.definition_validator import DefinitionValidator
from mental_world_model.engine.indexer.definition_indexer import DefinitionIndexer


class TestDefinitionLoader:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _DEFINITIONS = _PROJECT_ROOT / "definitions"

    def test_import(self):
        assert DefinitionLoader is not None

    def test_instantiate(self):
        loader = DefinitionLoader()
        assert loader is not None

    def test_methods_exist(self):
        loader = DefinitionLoader()
        for method_name in [
            "load_raw_file",
            "load_raw_directory",
            "load_json",
            "load_markdown",
            "load_yaml",
            "load_docx",
            "load_duizhi_docx",
            "load_xinsuo_definitions",
            "load_duizhi_definitions",
            "load_suoyuan_xiang_si_definitions",
            "load_xinsuo_seed_definitions",
        ]:
            assert hasattr(loader, method_name)
            assert callable(getattr(loader, method_name))

    def test_load_xinsuo_definitions_from_json(self):
        loader = DefinitionLoader()
        data = loader.load_xinsuo_definitions(self._DEFINITIONS)
        assert isinstance(data, list)
        assert len(data) == 40
        assert data[0]["name"] == "信"

    def test_load_duizhi_definitions_from_json(self):
        loader = DefinitionLoader()
        data = loader.load_duizhi_definitions(self._DEFINITIONS)
        assert isinstance(data, list)
        assert len(data) == 40
        assert data[0]["source"] == "信"
        assert data[0]["target"] == "不信"

    def test_load_suoyuan_xiang_si_definitions_from_json(self):
        loader = DefinitionLoader()
        data = loader.load_suoyuan_xiang_si_definitions(self._DEFINITIONS)
        assert "suoyuan_jing" in data
        assert "xiang" in data
        assert "si" in data
        assert "summary" in data

    def test_load_xinsuo_seed_definitions_from_json(self):
        loader = DefinitionLoader()
        data = loader.load_xinsuo_seed_definitions(self._DEFINITIONS)
        assert "categories" in data
        assert data["categories"][0]["entries"][0]["name"] == "信"

    def test_load_json_from_normalized(self):
        loader = DefinitionLoader()
        data = loader.load_json(self._DEFINITIONS / "normalized/xinsuo.json")
        assert isinstance(data, list)
        assert len(data) == 40

    def test_parse_xinsuo_from_json(self):
        loader = DefinitionLoader()
        parser = DefinitionParser()
        raw = loader.load_xinsuo_definitions(self._DEFINITIONS)
        result = parser.parse_xinsuo(raw)
        assert len(result) == 40
        assert result[0].name == "信"
        assert result[-1].name == "散乱"

    def test_parse_relation_from_json(self):
        loader = DefinitionLoader()
        parser = DefinitionParser()
        raw = loader.load_duizhi_definitions(self._DEFINITIONS)
        result = parser.parse_relation(raw)
        assert len(result) == 40
        assert result[0].source == "信"
        assert result[0].target == "不信"

    def test_all_duizhi_names_in_xinsuo_list(self):
        loader = DefinitionLoader()
        xinsuo_names = {d["name"] for d in loader.load_xinsuo_definitions(self._DEFINITIONS)}
        duizhi = loader.load_duizhi_definitions(self._DEFINITIONS)
        for item in duizhi:
            assert item["source"] in xinsuo_names, item["source"]
            assert item["target"] in xinsuo_names, item["target"]

    def test_each_xinsuo_has_exactly_one_duizhi_source(self):
        loader = DefinitionLoader()
        xinsuo_names = [d["name"] for d in loader.load_xinsuo_definitions(self._DEFINITIONS)]
        duizhi = loader.load_duizhi_definitions(self._DEFINITIONS)
        sources = [item["source"] for item in duizhi]
        assert len(duizhi) == len(xinsuo_names) == 40
        assert set(sources) == set(xinsuo_names)
        assert len(sources) == len(set(sources))


class TestDefinitionParser:
    def test_import(self):
        assert DefinitionParser is not None

    def test_instantiate(self):
        parser = DefinitionParser()
        assert parser is not None

    def test_methods_exist(self):
        parser = DefinitionParser()
        for method_name in ["parse_xinsuo", "parse_relation"]:
            assert hasattr(parser, method_name)
            assert callable(getattr(parser, method_name))


class TestDefinitionNormalizer:
    def test_import(self):
        assert DefinitionNormalizer is not None

    def test_instantiate(self):
        normalizer = DefinitionNormalizer()
        assert normalizer is not None

    def test_methods_exist(self):
        normalizer = DefinitionNormalizer()
        for method_name in ["normalize", "normalize_name", "normalize_text"]:
            assert hasattr(normalizer, method_name)
            assert callable(getattr(normalizer, method_name))

    def test_normalize_name_alias(self):
        normalizer = DefinitionNormalizer()
        assert normalizer.normalize_name("痴") == "无明"
        assert normalizer.normalize_name("无嗔") == "无瞋"

    def test_normalize_text_strips_and_unifies_newlines(self):
        normalizer = DefinitionNormalizer()
        assert normalizer.normalize_text("  hello \r\n world  ") == "hello\n world"


class TestDefinitionValidator:
    _SCHEMAS = Path(__file__).resolve().parents[1] / "definitions" / "schemas"

    def test_import(self):
        assert DefinitionValidator is not None

    def test_instantiate(self):
        validator = DefinitionValidator()
        assert validator is not None

    def test_methods_exist(self):
        validator = DefinitionValidator()
        for method_name in [
            "validate",
            "validate_xinsuo",
            "validate_relation",
            "validate_suoyuan_xiang_si",
        ]:
            assert hasattr(validator, method_name)
            assert callable(getattr(validator, method_name))

    def test_schema_rejects_extra_xinsuo_field(self):
        validator = DefinitionValidator(schemas_dir=self._SCHEMAS)
        result = validator.validate_xinsuo(
            {
                "name": "信",
                "suoyuan": "x",
                "shengqi_nian": "x",
                "ye": "x",
                "xiandai_xinli": "x",
                "extra": "bad",
            }
        )
        assert result.valid is False

    def test_validate_suoyuan_xiang_si_rejects_missing_field(self):
        validator = DefinitionValidator(schemas_dir=self._SCHEMAS)
        result = validator.validate_suoyuan_xiang_si(
            {"suoyuan_jing": "a", "xiang": "b", "si": "c"}
        )
        assert result.valid is False


class TestDefinitionIndexer:
    def test_import(self):
        assert DefinitionIndexer is not None

    def test_instantiate(self):
        indexer = DefinitionIndexer()
        assert indexer is not None

    def test_methods_exist(self):
        indexer = DefinitionIndexer()
        for method_name in ["build_index", "find", "add"]:
            assert hasattr(indexer, method_name)
            assert callable(getattr(indexer, method_name))
