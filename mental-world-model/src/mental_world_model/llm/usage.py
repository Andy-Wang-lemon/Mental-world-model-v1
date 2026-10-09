"""LLM 调用统计与阶段上下文。"""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class LlmCallRecord:
    stage: str
    duration_ms: float
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "ok": self.ok,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class LlmStageUsage:
    count: int = 0
    duration_ms: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors,
        }


@dataclass
class LlmUsageSummary:
    """一次评估的 LLM 调用汇总。"""

    total_calls: int
    total_duration_ms: float
    by_stage: dict[str, LlmStageUsage]
    wall_clock_ms: float | None = None
    records: list[LlmCallRecord] = field(default_factory=list)

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_calls": self.total_calls,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "by_stage": {
                stage: usage.to_dict() for stage, usage in sorted(self.by_stage.items())
            },
            "wall_clock_ms": (
                round(self.wall_clock_ms, 2) if self.wall_clock_ms is not None else None
            ),
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


class LlmUsageTracker:
    """线程安全的 LLM 调用记录器。"""

    def __init__(self) -> None:
        self._records: list[LlmCallRecord] = []
        self._lock = threading.Lock()
        self._wall_start: float | None = None
        self._wall_end: float | None = None

    def start_wall_clock(self) -> None:
        self._wall_start = time.perf_counter()

    def stop_wall_clock(self) -> None:
        self._wall_end = time.perf_counter()

    def record(self, record: LlmCallRecord) -> None:
        with self._lock:
            self._records.append(record)

    def total_calls_so_far(self) -> int:
        with self._lock:
            return len(self._records)

    def summary(self, *, include_records: bool = False) -> LlmUsageSummary:
        with self._lock:
            records = list(self._records)

        by_stage: dict[str, LlmStageUsage] = {}
        total_duration_ms = 0.0
        for item in records:
            bucket = by_stage.setdefault(item.stage, LlmStageUsage())
            bucket.count += 1
            bucket.duration_ms += item.duration_ms
            if not item.ok:
                bucket.errors += 1
            total_duration_ms += item.duration_ms

        wall_clock_ms: float | None = None
        if self._wall_start is not None and self._wall_end is not None:
            wall_clock_ms = (self._wall_end - self._wall_start) * 1000.0

        summary = LlmUsageSummary(
            total_calls=len(records),
            total_duration_ms=total_duration_ms,
            by_stage=by_stage,
            wall_clock_ms=wall_clock_ms,
            records=records if include_records else [],
        )
        return summary


_llm_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mwm_llm_stage",
    default="unknown",
)
_llm_tracker: contextvars.ContextVar[LlmUsageTracker | None] = contextvars.ContextVar(
    "mwm_llm_tracker",
    default=None,
)


def current_llm_stage() -> str:
    return _llm_stage.get()


def current_llm_tracker() -> LlmUsageTracker | None:
    return _llm_tracker.get()


@contextmanager
def llm_call_context(*, stage: str) -> Iterator[None]:
    """标记下一次 ``complete()`` 所属阶段。"""
    token = _llm_stage.set(stage)
    try:
        yield
    finally:
        _llm_stage.reset(token)


@contextmanager
def llm_evaluation_scope(tracker: LlmUsageTracker | None = None) -> Iterator[LlmUsageTracker]:
    """一次完整 evaluate 的统计作用域（含 wall-clock）。"""
    active = tracker or LlmUsageTracker()
    token = _llm_tracker.set(active)
    active.start_wall_clock()
    try:
        yield active
    finally:
        active.stop_wall_clock()
        _llm_tracker.reset(token)
