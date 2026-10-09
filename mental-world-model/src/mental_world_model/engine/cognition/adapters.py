"""将认知分支适配为现有 ScoringContext / ExtractedContext。

分支内心所评分输入为：原文 + 分支所缘 + 分支想 + 分支思。
（ScoringContext.base_context_block 已包含 yuanwen。）
想/思以 confidence=1.0 进入评分上下文，使四列 m 恒为 1.0；
真实 source/confidence 仍保留在 CognitionBranch 上。
"""

from __future__ import annotations

from mental_world_model.engine.cognition.models import CognitionBranch, ExtractedMentalState
from mental_world_model.engine.extraction.models import ExtractedContext, ExtractedField
from mental_world_model.engine.scoring.breakdown import ScoreBreakdown
from mental_world_model.engine.scoring.context import ScoringContext


def _scoring_field(state: ExtractedMentalState | None, *, fallback: str = "") -> ExtractedField:
    """评分用字段：置信度强制 1.0，不再触发旧 λ(想/思)。"""
    text = state.text if state and state.text else fallback
    return ExtractedField(text=text, source="原文", confidence=1.0)


def branch_to_extracted_context(
    yuanwen: str,
    branch: CognitionBranch,
) -> ExtractedContext:
    """构造分支评分用 ExtractedContext（m 全 1）。"""
    return ExtractedContext(
        yuanwen=yuanwen,
        suoyuan=ExtractedField(
            text=branch.alambana.text,
            source="原文",
            confidence=1.0,
        ),
        xiang=_scoring_field(branch.xiang),
        si=_scoring_field(branch.si),
        scoring_path="explicit",
    )


def branch_to_scoring_context(yuanwen: str, branch: CognitionBranch) -> ScoringContext:
    return ScoringContext.from_extracted(branch_to_extracted_context(yuanwen, branch))


def causal_context_for_branch(branch: CognitionBranch) -> dict:
    """ScoreBreakdown 附加因果上下文。"""
    xiang = branch.xiang
    si = branch.si
    return {
        "branch_id": branch.id,
        "alambana_id": branch.alambana.id,
        "alambana_text": branch.alambana.text,
        "alambana_probability": branch.alambana.posterior_probability,
        "alambana_confidence": branch.alambana.confidence,
        "alambana_source": branch.alambana.source,
        "xiang_text": xiang.text if xiang else "",
        "xiang_source": xiang.source if xiang else "推理",
        "xiang_confidence": xiang.confidence if xiang else 0.0,
        "si_text": si.text if si else "",
        "si_source": si.source if si else "推理",
        "si_confidence": si.confidence if si else 0.0,
        "supporting_self_grasping": list(branch.alambana.self_grasping_ids),
        "supporting_dharma_grasping": list(branch.alambana.dharma_grasping_ids),
    }


def attach_causal_context(
    breakdowns: dict[str, ScoreBreakdown | None],
    branch: CognitionBranch,
) -> dict[str, ScoreBreakdown]:
    """为 breakdown 附加 causal_context（写在 to_dict 扩展前暂存到对象属性）。"""
    context = causal_context_for_branch(branch)
    attached: dict[str, ScoreBreakdown] = {}
    for name, breakdown in breakdowns.items():
        if breakdown is None:
            continue
        # ScoreBreakdown 是 dataclass；动态附加供 to_dict 使用
        setattr(breakdown, "causal_context", context)
        attached[name] = breakdown
    return attached
