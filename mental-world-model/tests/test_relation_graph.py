"""测试关系图相关类的导入、实例化与方法存在性。"""

from pathlib import Path

from mental_world_model.core.relations.node import RelationNode
from mental_world_model.core.relations.edge import RelationEdge
from mental_world_model.core.relations.graph import RelationGraph
from mental_world_model.core.relations.relation import Relation
from mental_world_model.engine.graph_builder.mental_graph_builder import MentalGraphBuilder


class TestRelation:
    def test_import(self):
        assert Relation is not None

    def test_instantiate(self):
        obj = Relation(
            source="信",
            target="不信",
            description="直接对治不信",
            relation_type="对治",
            is_direct=True,
        )
        assert obj.source == "信"

    def test_from_dict(self):
        data = {
            "source": "贪",
            "target": "无贪",
            "description": "直接对治无贪。",
            "relation_type": "对治",
            "is_direct": True,
        }
        obj = Relation.from_dict(data)
        assert obj.to_dict() == data


class TestRelationNode:
    def test_import(self):
        assert RelationNode is not None

    def test_instantiate(self):
        node = RelationNode(node_id="信", kind="xinsuo")
        assert node.node_id == "信"


class TestRelationEdge:
    def test_import(self):
        assert RelationEdge is not None

    def test_instantiate(self):
        edge = RelationEdge(source_id="信", target_id="不信")
        assert edge.source_id == "信"


class TestRelationGraph:
    def test_import(self):
        assert RelationGraph is not None

    def test_instantiate(self):
        graph = RelationGraph()
        assert graph is not None

    def test_methods_exist(self):
        graph = RelationGraph()
        for method_name in [
            "add_node",
            "add_edge",
            "get_node",
            "get_edges",
            "find_neighbors",
        ]:
            assert hasattr(graph, method_name)
            assert callable(getattr(graph, method_name))


class TestMentalGraphBuilder:
    def test_import(self):
        assert MentalGraphBuilder is not None

    def test_instantiate(self):
        builder = MentalGraphBuilder()
        assert builder is not None

    def test_methods_exist(self):
        builder = MentalGraphBuilder()
        for method_name in [
            "build_from_definitions",
            "add_xinsuo",
            "add_relation",
        ]:
            assert hasattr(builder, method_name)
            assert callable(getattr(builder, method_name))
