"""关系图节点。"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RelationNode:
    """关系图节点。"""

    node_id: str
    kind: str
    data: Any = None
