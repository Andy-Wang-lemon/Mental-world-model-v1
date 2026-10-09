"""测试 Validator、Indexer、GraphBuilder 集成能力。"""

from pathlib import Path

from mental_world_model.engine.graph_builder.mental_graph_builder import MentalGraphBuilder
from mental_world_model.engine.indexer.definition_indexer import DefinitionIndexer
from mental_world_model.engine.loader.definition_loader import DefinitionLoader
from mental_world_model.engine.parser.definition_parser import DefinitionParser
from mental_world_model.engine.validator.definition_validator import DefinitionValidator
from mental_world_model.core.relations.relation import Relation


def _load_definitions():
    root = Path(__file__).resolve().parents[1] / "definitions"
    loader = DefinitionLoader()
    parser = DefinitionParser()
    xinsuo = parser.parse_xinsuo(loader.load_xinsuo_definitions(root))
    relations = parser.parse_relation(loader.load_duizhi_definitions(root))
    return {"xinsuo": xinsuo, "relations": relations}


class TestDefinitionValidatorIntegration:
    def test_validate_xinsuo_list(self):
        definitions = _load_definitions()
        validator = DefinitionValidator()
        result = validator.validate_xinsuo_list(definitions["xinsuo"])
        assert result.valid is True
        assert result.errors == []

    def test_validate_relation_list(self):
        definitions = _load_definitions()
        xinsuo_names = {item.name for item in definitions["xinsuo"]}
        validator = DefinitionValidator()
        result = validator.validate_relation_list(
            definitions["relations"],
            xinsuo_names=xinsuo_names,
        )
        assert result.valid is True
        assert result.errors == []
        assert len(definitions["relations"]) == 40

    def test_check_score_conflicts_detects_conflict(self):
        definitions = _load_definitions()
        validator = DefinitionValidator()
        scores = {"信": 0.8, "不信": 0.3, "贪": 0.0}
        conflicts = validator.check_score_conflicts(scores, definitions["relations"])
        assert len(conflicts) == 1
        assert {conflicts[0].name_a, conflicts[0].name_b} == {"信", "不信"}

    def test_check_score_conflicts_ignores_zero(self):
        definitions = _load_definitions()
        validator = DefinitionValidator()
        scores = {"信": 0.8, "不信": 0.0}
        conflicts = validator.check_score_conflicts(scores, definitions["relations"])
        assert conflicts == []

    def test_check_score_conflicts_deduplicates_bidirectional(self):
        definitions = _load_definitions()
        validator = DefinitionValidator()
        scores = {"信": 0.5, "不信": 0.6}
        conflicts = validator.check_score_conflicts(scores, definitions["relations"])
        assert len(conflicts) == 1

    def test_check_score_conflicts_detects_duizhi_mutex(self):
        """对治关系：双方有分即触发互斥冲突。"""
        validator = DefinitionValidator()
        relations = [
            Relation(
                source="贪",
                target="无贪",
                description="直接对治无贪。",
                relation_type="对治",
                is_direct=True,
            )
        ]
        scores = {"贪": 0.7, "无贪": 0.4}
        conflicts = validator.check_score_conflicts(scores, relations)
        assert len(conflicts) == 1
        assert {conflicts[0].name_a, conflicts[0].name_b} == {"贪", "无贪"}


class TestDefinitionIndexerIntegration:
    def test_build_and_find_xinsuo(self):
        definitions = _load_definitions()
        indexer = DefinitionIndexer()
        indexer.build_index(definitions)
        xinsuo = indexer.find("信")
        assert xinsuo is not None
        assert xinsuo.name == "信"
        assert len(indexer.xinsuo_names()) == 40

    def test_find_relations(self):
        definitions = _load_definitions()
        indexer = DefinitionIndexer()
        indexer.build_index(definitions)
        relations = indexer.find_relations("信")
        names = {(rel.source, rel.target) for rel in relations}
        assert ("信", "不信") in names
        assert ("不信", "信") in names

    def test_find_relations_includes_new_virtue_sources(self):
        definitions = _load_definitions()
        indexer = DefinitionIndexer()
        indexer.build_index(definitions)
        assert ("不慢", "慢") in {
            (rel.source, rel.target) for rel in indexer.find_relations("不慢")
        }
        assert ("善见", "恶见") in {
            (rel.source, rel.target) for rel in indexer.find_relations("善见")
        }
        assert ("不覆", "覆") in {
            (rel.source, rel.target) for rel in indexer.find_relations("不覆")
        }
        assert ("不嫉", "嫉") in {
            (rel.source, rel.target) for rel in indexer.find_relations("不嫉")
        }


class TestMentalGraphBuilderIntegration:
    def test_build_from_definitions(self):
        definitions = _load_definitions()
        builder = MentalGraphBuilder()
        graph = builder.build_from_definitions(definitions)
        assert len(graph.node_ids()) == 40
        assert len(graph.get_edges()) == 40

    def test_find_neighbors(self):
        definitions = _load_definitions()
        builder = MentalGraphBuilder()
        graph = builder.build_from_definitions(definitions)
        neighbors = graph.find_neighbors("信")
        assert "不信" in neighbors
        assert "疑" in neighbors
        assert "不正知" in neighbors
