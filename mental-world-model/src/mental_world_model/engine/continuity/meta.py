"""连续事件元数据：θ、情境相关度 R。"""

from __future__ import annotations

from mental_world_model.engine.continuity.models import ContinuityState, MentalEpisode
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.confidence import as_unit_float


EVENT_META_SCHEMA = {
    "type": "object",
    "properties": {
        "theta": {"type": "number"},
        "relevance": {"type": "number"},
    },
    "required": ["theta", "relevance"],
}


def build_event_meta_messages(
    yuanwen: str,
    *,
    previous_summary: str,
    hist_brief: list[dict],
) -> list[dict[str, str]]:
    system = (
        "你是事件重要性与情境相关度评估器（服务心理连续性，不是认知解释）。\n\n"
        "估计：\n"
        "- theta：本事件对主体的重要性∈[0,1]（主观重要性，非新闻热度）\n"
        "- relevance：当前事件与历史心理残留的情境相关度∈[0,1]\n"
        "禁止输出心所名、情绪诊断。"
    )
    user = (
        f"当前原文：\n{yuanwen}\n\n"
        f"上一事件摘要：\n{previous_summary or '（无）'}\n\n"
        f"历史心理残留摘要：\n{hist_brief}\n\n"
        "返回 JSON：\n"
        '{"theta":0.0,"relevance":0.0}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_event_meta(
    yuanwen: str,
    state: ContinuityState,
    llm: LlmClient,
) -> tuple[float, float]:
    """返回 (theta, relevance)。无历史时 relevance 强制为 0。"""
    prev: MentalEpisode | None = (
        state.mental_episodes[-1] if state.mental_episodes else None
    )
    hist_brief = [
        {
            "theta": ep.theta,
            "summary": ep.text_summary[:80],
            "top_scores": dict(
                sorted(ep.scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
            ),
        }
        for ep in state.mental_episodes[-5:]
    ]
    with llm_call_context(stage="continuity.event_meta"):
        response = llm.complete(
            build_event_meta_messages(
                yuanwen,
                previous_summary=prev.text_summary if prev else "",
                hist_brief=hist_brief,
            ),
            schema=EVENT_META_SCHEMA,
        )
    raw = response if isinstance(response, dict) else {}
    theta = as_unit_float(raw.get("theta"), 0.5)
    relevance = (
        0.0
        if not state.mental_episodes
        else as_unit_float(raw.get("relevance"), 0.3)
    )
    return theta, relevance
