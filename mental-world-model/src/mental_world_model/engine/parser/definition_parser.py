"""定义解析器。"""

from typing import Any

from mental_world_model.core.relations.relation import Relation
from mental_world_model.core.xinsuo.model import Xinsuo


class DefinitionParser:
    """将原始数据解析为核心对象。"""

    def parse_xinsuo(self, raw) -> list[Xinsuo]:
        """解析心所（Xinsuo）定义。"""
        if isinstance(raw, list):
            return [self._build_xinsuo(item) for item in raw]
        return [self._build_xinsuo(raw)]

    def _build_xinsuo(self, item: dict[str, Any]) -> Xinsuo:
        return Xinsuo.from_dict(item)

    def parse_relation(self, raw) -> list[Relation]:
        """解析关系（Relation）定义。"""
        if isinstance(raw, list):
            return [self._build_relation(item) for item in raw]
        return [self._build_relation(raw)]

    def _build_relation(self, item: dict[str, Any]) -> Relation:
        return Relation.from_dict(item)
