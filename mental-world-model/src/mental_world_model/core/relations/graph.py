"""关系图。"""

from collections import defaultdict

from mental_world_model.core.relations.edge import RelationEdge
from mental_world_model.core.relations.node import RelationNode


class RelationGraph:
    """关系图。"""

    def __init__(self) -> None:
        self._nodes: dict[str, RelationNode] = {}
        self._edges: list[RelationEdge] = []
        self._neighbors: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: RelationNode) -> None:
        """添加节点。"""
        self._nodes[node.node_id] = node

    def add_edge(self, edge: RelationEdge) -> None:
        """添加边。"""
        self._edges.append(edge)
        self._neighbors[edge.source_id].add(edge.target_id)
        self._neighbors[edge.target_id].add(edge.source_id)

    def get_node(self, node_id: str) -> RelationNode | None:
        """根据标识获取节点。"""
        return self._nodes.get(node_id)

    def get_edges(self) -> list[RelationEdge]:
        """获取所有边。"""
        return list(self._edges)

    def find_neighbors(self, node_id: str) -> list[str]:
        """查找指定节点的邻接节点标识。"""
        return sorted(self._neighbors.get(node_id, set()))

    def node_ids(self) -> list[str]:
        """返回所有节点标识。"""
        return sorted(self._nodes.keys())
