"""评分输入上下文。"""

from dataclasses import dataclass
from functools import cached_property

from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField


@dataclass
class ScoringContext:
    """单次评分任务的输入上下文。"""

    yuanwen: str
    extracted: ExtractedContext

    @classmethod
    def from_extracted(cls, extracted: ExtractedContext) -> "ScoringContext":
        """从提取结果构建评分上下文。"""
        return cls(yuanwen=extracted.yuanwen, extracted=extracted)

    @property
    def extracted_suoyuan(self) -> str:
        return self.extracted.suoyuan.text

    @property
    def extracted_xiang(self) -> str:
        return self.extracted.xiang.text

    @property
    def extracted_si(self) -> str:
        return self.extracted.si.text

    @cached_property
    def base_context_block(self) -> str:
        """评分输入：原文 + 所缘/想/思（分支路径下所缘为该分支主观所缘）。"""
        extracted = self.extracted
        return (
            f"原文：\n{self.yuanwen}\n\n"
            f"{self.field_block(extracted.suoyuan, '提取的所缘境')}\n\n"
            f"{self.field_block(extracted.xiang, '提取的想')}\n\n"
            f"{self.field_block(extracted.si, '提取的思')}\n"
        )

    def field_block(self, field: ExtractedField, label: str) -> str:
        """格式化单个提取字段（含来源与置信度）。"""
        return (
            f"{label}：{field.text}\n"
            f"数据来源：{field.source}\n"
            f"置信度：{field.confidence}"
        )
