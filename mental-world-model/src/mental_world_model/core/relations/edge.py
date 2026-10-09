"""关系图边。"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RelationEdge:
    """关系图边。"""

    source_id: str
    target_id: str
    data: Any = None
