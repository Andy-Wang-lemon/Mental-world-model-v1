"""定义索引器。"""

from typing import Any

from mental_world_model.core.relations.relation import Relation
from mental_world_model.core.xinsuo.model import Xinsuo


class DefinitionIndexer:
    """构建可检索索引。"""

    def __init__(self) -> None:
        self._xinsuo_by_name: dict[str, Xinsuo] = {}
        self._relations_by_name: dict[str, list[Relation]] = {}

    def build_index(self, definitions: dict[str, Any] | None = None) -> None:
        """根据定义集合构建索引。

        Args:
            definitions: 字典，可含 ``xinsuo`` 与 ``relations`` 键。
        """
        self._xinsuo_by_name.clear()
        self._relations_by_name.clear()

        if not definitions:
            return

        for xinsuo in definitions.get("xinsuo", []):
            self.add(xinsuo.name, xinsuo)

        for relation in definitions.get("relations", []):
            self._index_relation(relation)

    def find(self, key: str) -> Xinsuo | None:
        """根据心所名称查找。"""
        return self._xinsuo_by_name.get(key)

    def add(self, key: str, value: Xinsuo) -> None:
        """添加心所索引项。"""
        self._xinsuo_by_name[key] = value

    def find_relations(self, name: str) -> list[Relation]:
        """查找与指定心所相关的所有关系。"""
        return list(self._relations_by_name.get(name, []))

    def xinsuo_names(self) -> list[str]:
        """返回已索引的心所名称。"""
        return sorted(self._xinsuo_by_name.keys())

    def _index_relation(self, relation: Relation) -> None:
        for endpoint in (relation.source, relation.target):
            bucket = self._relations_by_name.setdefault(endpoint, [])
            if relation not in bucket:
                bucket.append(relation)
