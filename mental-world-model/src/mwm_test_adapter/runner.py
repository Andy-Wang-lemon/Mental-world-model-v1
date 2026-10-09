"""批量跑分：MWM → 40 心所 scores_final。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from mental_world_model.llm.client import LlmClient
from mental_world_model.model import MentalWorldModel


def build_core_text(row: "TestInputRow") -> str:
    """构建传给 MWM 核心的原文（故事 + 问题 + 角色）。"""
    parts = [row.story.strip()]
    if row.question.strip():
        parts.append(f"Question: {row.question.strip()}")
    if row.character.strip():
        parts.append(f"Character: {row.character.strip()}")
    return "\n\n".join(parts)


@dataclass
class TestInputRow:
    story: str
    question: str
    id: str | None = None
    character: str = ""
    question_type: str = ""


@dataclass
class TestSampleResult:
    id: str | None
    text: str
    scores_final: dict[str, float]
    question_type: str = ""
    predicted_behavior: str | None = None
    aggregated_action_distribution: list[dict[str, Any]] | None = None
    active_xinsuo: list[dict[str, Any]] | None = None
    prediction_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "question_type": self.question_type,
            "scores_final": dict(self.scores_final),
        }
        if self.aggregated_action_distribution is not None:
            payload["aggregated_action_distribution"] = list(
                self.aggregated_action_distribution
            )
        # 旧单路仍可能有单句；多支主路径为 None
        if self.predicted_behavior is not None:
            payload["predicted_behavior"] = self.predicted_behavior
        if self.active_xinsuo is not None:
            payload["active_xinsuo"] = list(self.active_xinsuo)
        if self.prediction_reason is not None:
            payload["prediction_reason"] = self.prediction_reason
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class TestRunSummary:
    total: int
    succeeded: int
    failed: int
    results: list[TestSampleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "results": [item.to_dict() for item in self.results],
        }


def iter_input_rows(path: Path) -> Iterator[TestInputRow]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 第 {line_no} 行无效：{error}") from error
        story = str(row.get("story", "")).strip()
        question = str(row.get("question", "")).strip()
        if not story:
            raise ValueError(f"JSONL 第 {line_no} 行缺少 story")
        if not question:
            raise ValueError(f"JSONL 第 {line_no} 行缺少 question")
        yield TestInputRow(
            id=str(row["id"]) if row.get("id") is not None else None,
            story=story,
            question=question,
            character=str(row.get("character", "")).strip(),
            question_type=str(row.get("question_type", "")).strip(),
        )


class TestAdapterRunner:
    """外接批量跑分：调用 MWM 核心，输出 40 心所 scores_final。"""

    def __init__(
        self,
        model: MentalWorldModel,
        llm_client: LlmClient,
        *,
        update_seeds: bool = False,
    ) -> None:
        self._model = model
        self._llm = llm_client
        self._update_seeds = update_seeds

    def run_sample(self, row: TestInputRow) -> TestSampleResult:
        scores_final: dict[str, float] = {}
        core_text = build_core_text(row)

        try:
            evaluation = self._model.evaluate_from_yuanwen(
                core_text,
                self._llm,
                update_seeds=self._update_seeds,
            )
            scores_final = dict(evaluation.scores_final)
            action_dist = None
            if evaluation.aggregated_behaviors:
                action_dist = [
                    {
                        "behavior_id": item.behavior_id,
                        "text": item.canonical_text,
                        "probability": item.total_probability,
                        "confidence": item.confidence,
                    }
                    for item in evaluation.aggregated_behaviors
                ]
            return TestSampleResult(
                id=row.id,
                text=core_text,
                scores_final=scores_final,
                question_type=row.question_type,
                predicted_behavior=evaluation.predicted_behavior,
                aggregated_action_distribution=action_dist,
                active_xinsuo=(
                    list(evaluation.active_xinsuo)
                    if evaluation.active_xinsuo is not None
                    else None
                ),
                prediction_reason=evaluation.prediction_reason,
            )
        except Exception as error:  # noqa: BLE001
            return TestSampleResult(
                id=row.id,
                text=core_text,
                scores_final=scores_final,
                question_type=row.question_type,
                error=str(error),
            )

    def run_file(
        self,
        input_path: Path,
        *,
        limit: int | None = None,
    ) -> TestRunSummary:
        rows = list(iter_input_rows(input_path))
        if limit is not None:
            rows = rows[:limit]

        results: list[TestSampleResult] = []
        for row in rows:
            results.append(self.run_sample(row))

        succeeded = sum(1 for item in results if item.error is None)
        return TestRunSummary(
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            results=results,
        )


def write_results_jsonl(results: list[TestSampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
