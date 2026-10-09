"""行为预测：有值心所综合 → 行为 + 有值清单 + 判断理由。"""

from dataclasses import dataclass

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.prediction.prompts import (
    BEHAVIOR_SCHEMA,
    build_behavior_messages,
)
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context


@dataclass(frozen=True)
class ActiveXinsuoItem:
    """单个有值心所及其分数。"""

    name: str
    score: float

    def to_dict(self) -> dict[str, str | float]:
        return {"name": self.name, "score": self.score}


@dataclass(frozen=True)
class BehaviorPrediction:
    """行为预测三件套：行为、有值心所及分数、判断理由。"""

    predicted_behavior: str = ""
    active_xinsuo: tuple[ActiveXinsuoItem, ...] = ()
    prediction_reason: str = ""

    def active_xinsuo_as_dicts(self) -> list[dict[str, str | float]]:
        return [item.to_dict() for item in self.active_xinsuo]


def _as_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def collect_active_xinsuo(
    scores_final: dict[str, float],
    xinsuo_by_name: dict[str, Xinsuo],
) -> list[tuple[str, float, Xinsuo]]:
    """筛出有值心所，按分数降序。"""
    active: list[tuple[str, float, Xinsuo]] = [
        (name, float(score), xinsuo_by_name[name])
        for name, score in scores_final.items()
        if float(score) > 0 and name in xinsuo_by_name
    ]
    active.sort(key=lambda item: item[1], reverse=True)
    return active


class BehaviorPredictor:
    """在 40 维消解后，根据有值心所综合预测可能行为并给出理由。"""

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client

    def predict(
        self,
        context: ScoringContext,
        scores_final: dict[str, float],
        xinsuo_by_name: dict[str, Xinsuo],
    ) -> BehaviorPrediction:
        """返回三件套；无有值心所时不调 LLM，清单为空。"""
        active = collect_active_xinsuo(scores_final, xinsuo_by_name)
        active_items = tuple(
            ActiveXinsuoItem(name=name, score=score) for name, score, _ in active
        )
        if not active:
            return BehaviorPrediction(active_xinsuo=active_items)

        messages = build_behavior_messages(context, active)
        with llm_call_context(stage="behavior.predict"):
            response = self._llm.complete(messages, schema=BEHAVIOR_SCHEMA)
        return BehaviorPrediction(
            predicted_behavior=_as_text(response.get("predicted_behavior", "")),
            active_xinsuo=active_items,
            prediction_reason=_as_text(response.get("prediction_reason", "")),
        )
