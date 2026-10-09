"""想/思辅助：兼容报告用动作检测壳、思绑定与规范化。

主路径的想/思生成在 ``action_evidence``（L0 拆分 + 分支想 + 行动作思）。
"""

from __future__ import annotations

from dataclasses import dataclass

from mental_world_model.engine.cognition.models import ExtractedMentalState
from mental_world_model.engine.extraction.validators import is_grounded_in_yuanwen
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate
from mental_world_model.engine.seeds.config import INFERENCE_CONFIDENCE_MAX
from mental_world_model.utils.confidence import as_unit_float


@dataclass(frozen=True)
class ExplicitActionDetection:
    """兼容旧报告字段：由 ActionEvidence 投影，不再单独 LLM 检测。"""

    has_explicit_action: bool
    action_span: str = ""
    si_text: str = ""
    confidence: float = 0.0
    reason_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "has_explicit_action": self.has_explicit_action,
            "action_span": self.action_span,
            "si_text": self.si_text,
            "confidence": self.confidence,
            "reason_summary": self.reason_summary,
        }


def bind_si_to_alambana(
    si: ExtractedMentalState,
    alambana_id: str,
) -> ExtractedMentalState:
    """将共享思绑定到当前分支所缘 id（不改语义文本）。"""
    return ExtractedMentalState(
        text=si.text,
        source=si.source,
        confidence=si.confidence,
        alambana_id=alambana_id,
        supporting_grasping_ids=list(si.supporting_grasping_ids),
        reason_summary=si.reason_summary,
        xiang_text=si.xiang_text,
    )


def _normalize_mental_state(
    raw: dict,
    *,
    yuanwen: str,
    alambana: SubjectiveAlambanaCandidate,
    xiang_text: str = "",
    action_span: str = "",
) -> ExtractedMentalState:
    """规范化想/思；confidence 仅按自身原文/推理护栏，不再被所缘 confidence 封顶。"""
    text = str(raw.get("text", "")).strip()
    if not text:
        return ExtractedMentalState(
            text="",
            source="推理",
            confidence=0.0,
            alambana_id=alambana.id,
            reason_summary=str(raw.get("reason_summary", "")).strip(),
            xiang_text=xiang_text,
        )

    source = "原文" if raw.get("source") == "原文" else "推理"
    confidence = as_unit_float(raw.get("confidence"), 0.6)
    grounded_text = is_grounded_in_yuanwen(text, yuanwen)
    grounded_action = bool(
        action_span and is_grounded_in_yuanwen(action_span, yuanwen)
    )

    if source == "原文":
        if grounded_text:
            confidence = 1.0
        elif grounded_action:
            # 思为动作意向概括：原文未必含意向词，但动作片段可核验
            confidence = min(max(confidence, 0.8), 1.0)
        else:
            source = "推理"
            confidence = min(confidence, INFERENCE_CONFIDENCE_MAX)
    else:
        confidence = min(confidence, INFERENCE_CONFIDENCE_MAX)

    ids = raw.get("supporting_grasping_ids", [])
    supporting = (
        [str(i).strip() for i in ids if str(i).strip()]
        if isinstance(ids, list)
        else []
    )
    return ExtractedMentalState(
        text=text,
        source=source,
        confidence=confidence,
        alambana_id=alambana.id,
        supporting_grasping_ids=supporting,
        reason_summary=str(raw.get("reason_summary", "")).strip(),
        xiang_text=xiang_text or str(raw.get("xiang_text", "")).strip(),
    )
