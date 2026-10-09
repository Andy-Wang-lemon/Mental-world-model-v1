"""对治关系冲突消解。"""

from __future__ import annotations

import os

from mental_world_model.core.relations.relation import Relation
from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.reconciliation.results import ConflictResolution
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.prompts import build_conflict_messages
from mental_world_model.engine.validator.definition_validator import DefinitionValidator
from mental_world_model.engine.validator.validation_result import ScoreConflict
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.xinsuo_names import normalize_xinsuo_name

_RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string"},
    },
    "required": ["winner"],
}

# 比分差 ≥ 此阈值时直接 argmax，跳过 LLM；近平局才问模型
_DEFAULT_SCORE_MARGIN = 0.08


def _score_margin() -> float:
    raw = os.environ.get("MWM_DUIZHI_SCORE_MARGIN", str(_DEFAULT_SCORE_MARGIN)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = _DEFAULT_SCORE_MARGIN
    return max(0.0, min(1.0, value))


class DuizhiReconciler:
    """检测互斥冲突：比分明显时代码决胜；近平局由 LLM 消解。"""

    def __init__(
        self,
        llm_client: LlmClient,
        validator: DefinitionValidator | None = None,
        *,
        score_margin: float | None = None,
    ) -> None:
        self._llm = llm_client
        self._validator = validator or DefinitionValidator()
        self._score_margin = (
            _score_margin() if score_margin is None else max(0.0, min(1.0, score_margin))
        )

    def reconcile(
        self,
        scores: dict[str, float],
        relations: list[Relation],
        xinsuo_by_name: dict[str, Xinsuo],
        context: ScoringContext,
    ) -> tuple[dict[str, float], list[ConflictResolution]]:
        """循环检测并消解冲突，直到无冲突。"""
        current_scores = dict(scores)
        resolutions: list[ConflictResolution] = []
        max_rounds = len(relations) + 1

        for _ in range(max_rounds):
            conflicts = self._validator.check_score_conflicts(current_scores, relations)
            if not conflicts:
                break

            resolution = self._resolve_one(
                conflicts[0], current_scores, xinsuo_by_name, context
            )
            resolutions.append(resolution)
            current_scores = resolution.scores_after

        return current_scores, resolutions

    @staticmethod
    def _match_winner(winner_raw: str, name_a: str, name_b: str) -> tuple[str | None, str]:
        """解析 LLM 返回的 winner，支持名称规范化与 none。"""
        stripped = winner_raw.strip()
        if not stripped or stripped.lower() == "none":
            return None, ""

        candidates = [stripped, normalize_xinsuo_name(stripped)]
        for candidate in candidates:
            if candidate == name_a:
                return name_a, ""
            if candidate == name_b:
                return name_b, ""

        return None, f"（无效 winner：{winner_raw!r}，已对 {name_a}/{name_b} 双归零）"

    def _apply_winner(
        self,
        *,
        name_a: str,
        name_b: str,
        scores: dict[str, float],
        winner: str | None,
        note: str = "",
    ) -> ConflictResolution:
        scores_before = dict(scores)
        scores_after = dict(scores)
        loser: str | None
        if winner == name_a:
            loser = name_b
            scores_after[loser] = 0.0
        elif winner == name_b:
            loser = name_a
            scores_after[loser] = 0.0
        else:
            winner, loser = None, None
            scores_after[name_a] = 0.0
            scores_after[name_b] = 0.0
        return ConflictResolution(
            name_a=name_a,
            name_b=name_b,
            winner=winner,
            loser=loser,
            note=note,
            scores_before=scores_before,
            scores_after=scores_after,
        )

    def _resolve_by_score(
        self,
        conflict: ScoreConflict,
        scores: dict[str, float],
    ) -> ConflictResolution | None:
        """比分差足够大时直接取高分者，无需 LLM。"""
        name_a = conflict.name_a
        name_b = conflict.name_b
        score_a = float(scores.get(name_a, 0.0))
        score_b = float(scores.get(name_b, 0.0))
        delta = abs(score_a - score_b)
        if delta < self._score_margin:
            return None
        winner = name_a if score_a > score_b else name_b
        return self._apply_winner(
            name_a=name_a,
            name_b=name_b,
            scores=scores,
            winner=winner,
            note=f"比分决胜（Δ={delta:.3f}≥{self._score_margin:.3f}）",
        )

    def _resolve_one(
        self,
        conflict: ScoreConflict,
        scores: dict[str, float],
        xinsuo_by_name: dict[str, Xinsuo],
        context: ScoringContext,
    ) -> ConflictResolution:
        by_score = self._resolve_by_score(conflict, scores)
        if by_score is not None:
            return by_score

        name_a = conflict.name_a
        name_b = conflict.name_b
        xinsuo_a = xinsuo_by_name[name_a]
        xinsuo_b = xinsuo_by_name[name_b]

        with llm_call_context(stage="reconciliation.duizhi"):
            response = self._llm.complete(
                build_conflict_messages(
                    name_a,
                    name_b,
                    xinsuo_a,
                    xinsuo_b,
                    context,
                    conflict.description,
                ),
                schema=_RESOLVE_SCHEMA,
            )

        winner_raw = str(response.get("winner", "none"))
        matched_winner, invalid_note = self._match_winner(winner_raw, name_a, name_b)
        return self._apply_winner(
            name_a=name_a,
            name_b=name_b,
            scores=scores,
            winner=matched_winner,
            note=invalid_note or "近平局，LLM 裁决",
        )
