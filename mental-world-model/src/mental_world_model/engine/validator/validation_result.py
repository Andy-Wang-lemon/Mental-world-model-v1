"""校验结果与分数冲突。"""

from dataclasses import dataclass, field


class DefinitionValidationError(ValueError):
    """定义文件校验失败。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "定义校验失败：\n" + "\n".join(f"- {error}" for error in errors)
        super().__init__(message)


@dataclass
class ValidationResult:
    """定义校验结果。"""

    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class ScoreConflict:
    """打分后互斥冲突。"""

    name_a: str
    name_b: str
    score_a: float
    score_b: float
    description: str

    def __str__(self) -> str:
        return (
            f"冲突：{self.name_a}({self.score_a}) ↔ {self.name_b}({self.score_b})"
            f" — {self.description}"
        )
