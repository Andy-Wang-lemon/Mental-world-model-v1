"""提取结果校验与字段定型（LLM 主判，代码护栏）。"""

import re

from mental_world_model.engine.extraction.models import ExtractedField, FieldSource
from mental_world_model.engine.seeds.config import (
    INFERENCE_CONFIDENCE_DEFAULT,
    INFERENCE_CONFIDENCE_MAX,
)
from mental_world_model.utils.confidence import clamp_unit_interval

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_match_text(text: str) -> str:
    """用于匹配的轻量归一化。"""
    return _WHITESPACE_RE.sub("", text.strip().lower())


def is_grounded_in_yuanwen(text: str, yuanwen: str) -> bool:
    """判断文本是否能在原文中找到依据（子串或高重叠）。"""
    normalized_text = normalize_match_text(text)
    normalized_yuanwen = normalize_match_text(yuanwen)
    if not normalized_text or not normalized_yuanwen:
        return False
    if normalized_text in normalized_yuanwen:
        return True
    if len(normalized_text) >= 4 and normalized_yuanwen in normalized_text:
        return True

    text_chars = set(normalized_text)
    overlap = len(text_chars & set(normalized_yuanwen)) / max(len(text_chars), 1)
    return overlap >= 0.85


def _parse_source(raw_source: object) -> FieldSource:
    if raw_source == "原文":
        return "原文"
    return "推理"


def _parse_confidence(raw_confidence: object) -> float:
    if raw_confidence is None:
        return INFERENCE_CONFIDENCE_DEFAULT
    try:
        return clamp_unit_interval(float(raw_confidence))
    except (TypeError, ValueError):
        return INFERENCE_CONFIDENCE_DEFAULT


def normalize_xiang_si_field(raw: dict, *, yuanwen: str) -> ExtractedField:
    """采纳 LLM 的 source/confidence，代码做一致性护栏。"""
    text = str(raw.get("text", "")).strip()
    if not text:
        return ExtractedField(text="", source="推理", confidence=0.0)

    source = _parse_source(raw.get("source"))
    confidence = _parse_confidence(raw.get("confidence"))
    grounded = is_grounded_in_yuanwen(text, yuanwen)

    if source == "原文":
        if confidence < 1.0:
            confidence = 1.0
        if not grounded:
            source = "推理"
            confidence = INFERENCE_CONFIDENCE_MAX

    if source == "推理" and confidence >= 1.0 and not grounded:
        confidence = INFERENCE_CONFIDENCE_MAX

    return ExtractedField(text=text, source=source, confidence=confidence)


def normalize_suoyuan_response(raw: dict) -> ExtractedField:
    """所缘境：恒为原文来源。"""
    if "text" not in raw:
        raise ValueError("suoyuan.text 缺失")
    text = str(raw["text"]).strip()
    if not text:
        raise ValueError("suoyuan.text 为空")
    return ExtractedField(text=text, source="原文", confidence=1.0)
