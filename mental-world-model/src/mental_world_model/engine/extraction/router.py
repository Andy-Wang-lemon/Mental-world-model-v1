"""所缘境与心所路由（种子库启用时：逐心所 LLM + Top-K 选 H）。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.extraction.prompts import build_router_single_xinsuo_messages
from mental_world_model.engine.seeds.config import (
    ROUTER_MIN_RELEVANCE,
    ROUTER_RELEVANCE_ALPHA,
    ROUTER_TOP_K,
    is_seed_library_enabled,
)
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context

_ROUTER_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number"},
    },
    "required": ["relevance"],
}


def parse_router_single_response(response: dict) -> float:
    """解析单心所路由 LLM 返回的 relevance。"""
    try:
        score = float(response.get("relevance", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(score, 1.0))


def select_hint_xinsuo_names(
    relevance: dict[str, float],
    seed_weights: dict[str, float],
    *,
    top_k: int = ROUTER_TOP_K,
    min_relevance: float = ROUTER_MIN_RELEVANCE,
    alpha: float = ROUTER_RELEVANCE_ALPHA,
) -> list[str]:
    """融合所缘相关度与种子库权重，选出可参考心所 H（Top-K）。"""
    ranked: list[tuple[str, float]] = []
    for name, rel in relevance.items():
        if rel < min_relevance:
            continue
        seed = float(seed_weights.get(name, 0.0))
        combined = alpha * rel + (1.0 - alpha) * seed
        ranked.append((name, combined))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


class SuoyuanXinsuoRouter:
    """种子库启用时：按文档顺序逐心所评估所缘相关度，再选 H。"""

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client

    def route(
        self,
        yuanwen: str,
        suoyuan_text: str,
        xinsuo_list: list[Xinsuo],
        seed_weights: dict[str, float],
    ) -> tuple[list[str], dict[str, float]]:
        """返回 (hint_names, relevance_scores)。调用方应仅在 SEED_LIBRARY_ENABLED 时使用。"""
        if not is_seed_library_enabled():
            raise RuntimeError("路由 H 仅在种子库启用时可用")

        xinsuo_by_name = {item.name: item for item in xinsuo_list}
        relevance: dict[str, float] = {name: 0.0 for name in xinsuo_by_name}

        for xinsuo in xinsuo_list:
            messages = build_router_single_xinsuo_messages(
                yuanwen,
                suoyuan_text,
                xinsuo,
            )
            with llm_call_context(stage="extraction.router"):
                response = self._llm.complete(messages, schema=_ROUTER_SINGLE_SCHEMA)
            relevance[xinsuo.name] = parse_router_single_response(response)

        hint_names = select_hint_xinsuo_names(relevance, seed_weights)
        return hint_names, relevance
