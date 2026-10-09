"""心智关系图构建器。"""

from typing import Any

from mental_world_model.core.relations.edge import RelationEdge
from mental_world_model.core.relations.graph import RelationGraph
from mental_world_model.core.relations.node import RelationNode
from mental_world_model.core.relations.relation import Relation
from mental_world_model.core.xinsuo.model import Xinsuo


class MentalGraphBuilder:
    """从定义构建关系图。"""

    def __init__(self) -> None:
        self._graph = RelationGraph()

    @property
    def graph(self) -> RelationGraph:
        """当前构建中的关系图。"""
        return self._graph

    def build_from_definitions(self, definitions: dict[str, Any]) -> RelationGraph:
        """根据定义集合构建关系图（心所节点 + 对治关系边）。"""
        self._graph = RelationGraph()

        for xinsuo in definitions.get("xinsuo", []):
            self.add_xinsuo(xinsuo)

        for relation in definitions.get("relations", []):
            self.add_relation(relation)

        return self._graph

    def add_xinsuo(self, xinsuo: Xinsuo) -> None:
        """向图中添加心所（Xinsuo）。"""
        node = RelationNode(node_id=xinsuo.name, kind="xinsuo", data=xinsuo)
        self._graph.add_node(node)

    def add_relation(self, relation: Relation) -> None:
        """向图中添加关系（Relation）。"""
        for name in (relation.source, relation.target):
            if self._graph.get_node(name) is None:
                self._graph.add_node(
                    RelationNode(node_id=name, kind="xinsuo", data=None)
                )

        edge = RelationEdge(
            source_id=relation.source,
            target_id=relation.target,
            data=relation,
        )
        self._graph.add_edge(edge)
