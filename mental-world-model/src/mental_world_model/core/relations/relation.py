"""关系定义。"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Relation:
    """心所之间的关系定义。

    字段结构由用户提供的原始定义文件决定。
    """

    source: str
    target: str
    description: str
    relation_type: str
    is_direct: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        """从字典构造关系对象。"""
        return cls(
            source=data["source"],
            target=data["target"],
            description=data["description"],
            relation_type=data["relation_type"],
            is_direct=data["is_direct"],
        )

    def to_dict(self) -> dict[str, str | bool]:
        """转换为字典。"""
        return {
            "source": self.source,
            "target": self.target,
            "description": self.description,
            "relation_type": self.relation_type,
            "is_direct": self.is_direct,
        }
