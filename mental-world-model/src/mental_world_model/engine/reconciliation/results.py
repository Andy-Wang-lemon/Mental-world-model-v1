"""冲突消解结果。"""

from dataclasses import dataclass, field


@dataclass
class ConflictResolution:
    """单次冲突消解记录。"""

    name_a: str
    name_b: str
    winner: str | None
    loser: str | None
    note: str = ""
    scores_before: dict[str, float] = field(default_factory=dict)
    scores_after: dict[str, float] = field(default_factory=dict)
