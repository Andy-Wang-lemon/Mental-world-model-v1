"""所缘境/想/思提取数据模型。"""

from dataclasses import dataclass
from typing import Literal

FieldSource = Literal["原文", "推理"]
ScoringPath = Literal["explicit", "implicit"]

IMPLICIT_XIANG_PLACEHOLDER = "（无原文明确想，走百法隐式路径）"
IMPLICIT_SI_PLACEHOLDER = "（无原文明确思，走百法隐式路径）"


@dataclass
class SuoyuanXiangSiDefinitions:
    """所缘境、想、思的定义（来自用户原始定义文件）。"""

    suoyuan_jing: str
    xiang: str
    si: str
    summary: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "SuoyuanXiangSiDefinitions":
        return cls(
            suoyuan_jing=data["suoyuan_jing"],
            xiang=data["xiang"],
            si=data["si"],
            summary=data["summary"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "suoyuan_jing": self.suoyuan_jing,
            "xiang": self.xiang,
            "si": self.si,
            "summary": self.summary,
        }


@dataclass
class ExtractedField:
    """单个提取字段。"""

    text: str
    source: FieldSource
    confidence: float


@dataclass
class ExtractedContext:
    """从原文提取的所缘境、想、思。"""

    yuanwen: str
    suoyuan: ExtractedField
    xiang: ExtractedField
    si: ExtractedField
    scoring_path: ScoringPath = "explicit"
    routed_hint_names: list[str] | None = None
    router_relevance: dict[str, float] | None = None

    @property
    def suoyuan_text(self) -> str:
        return self.suoyuan.text

    @property
    def xiang_text(self) -> str:
        return self.xiang.text

    @property
    def si_text(self) -> str:
        return self.si.text

    @property
    def is_implicit_path(self) -> bool:
        return self.scoring_path == "implicit"

    def to_dict(self) -> dict:
        """转为可序列化字典。"""
        payload = {
            "yuanwen": self.yuanwen,
            "suoyuan": {
                "text": self.suoyuan.text,
                "source": self.suoyuan.source,
                "confidence": self.suoyuan.confidence,
            },
            "xiang": {
                "text": self.xiang.text,
                "source": self.xiang.source,
                "confidence": self.xiang.confidence,
            },
            "si": {
                "text": self.si.text,
                "source": self.si.source,
                "confidence": self.si.confidence,
            },
            "scoring_path": self.scoring_path,
        }
        if self.routed_hint_names is not None:
            payload["routed_hint_names"] = list(self.routed_hint_names)
        if self.router_relevance is not None:
            payload["router_relevance"] = dict(self.router_relevance)
        return payload
