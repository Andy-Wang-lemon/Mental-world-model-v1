"""评分阶段并行执行（线程池 + 全局并发闸门）。

规则：
- 分支与分支之间可并行（container，不占 LLM 槽）。
- 分支内：所缘→生起→业→现代心理 各波串行；波内全部待评心所并行（leaf，占槽）。
- 嵌套时外层不占槽、内层占槽，避免「分支持槽等内心所」死锁与过订阅。
"""

from __future__ import annotations

import contextvars
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")

_parallel_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mwm_parallel_depth", default=0
)
_slot_sem: threading.BoundedSemaphore | None = None
_slot_lock = threading.Lock()


def is_parallel_enabled() -> bool:
    raw = os.environ.get("MWM_LLM_PARALLEL", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# 心所表 40；落地扩支最多 4 支。默认让同一评分波内全部心所同时请求
# （3 支 × 38 主波、4 支 × 40 均覆盖）。四列仍分波，不在此拉平。
XINSUO_TABLE_SIZE = 40
DEFAULT_SCORING_BRANCHES = 4
DEFAULT_LLM_MAX_CONCURRENCY = XINSUO_TABLE_SIZE * DEFAULT_SCORING_BRANCHES


def max_concurrency() -> int:
    raw = os.environ.get("MWM_LLM_MAX_CONCURRENCY", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_LLM_MAX_CONCURRENCY


def _llm_slots() -> threading.BoundedSemaphore:
    """进程内全局 LLM 并发槽（与连接池上限对齐）。"""
    global _slot_sem
    with _slot_lock:
        if _slot_sem is None:
            _slot_sem = threading.BoundedSemaphore(max_concurrency())
        return _slot_sem


def reset_llm_slots_for_tests() -> None:
    """测试用：按当前环境变量重建信号量。"""
    global _slot_sem
    with _slot_lock:
        _slot_sem = threading.BoundedSemaphore(max_concurrency())


def _run_with_slot(fn: Callable[[], T]) -> T:
    with _llm_slots():
        return fn()


def run_parallel(
    tasks: dict[str, Callable[[], T]],
    *,
    acquire_slot: bool | None = None,
) -> dict[str, T]:
    """并行执行任务；key 为任务标识，value 为无参 callable。

    ``acquire_slot``:
    - True：叶子任务（心所列评分、单支想等），占用全局并发槽。
    - False：容器任务（整支评分），不占槽，便于支内再并行心所。
    - None：视为叶子，占槽。

    每个任务使用独立的 ``contextvars.copy_context()``。
    收到 KeyboardInterrupt 时 ``shutdown(wait=False, cancel_futures=True)``。
    """
    if not tasks:
        return {}
    if not is_parallel_enabled() or len(tasks) == 1:
        take_slot = False if acquire_slot is False else True
        if not take_slot:
            return {key: fn() for key, fn in tasks.items()}
        return {
            key: (lambda f=fn: _run_with_slot(f))()
            for key, fn in tasks.items()
        }

    depth = _parallel_depth.get()
    # 默认占槽（心所/想等叶子）；分支容器须显式 acquire_slot=False
    take_slot = True if acquire_slot is None else acquire_slot

    def _wrap(fn: Callable[[], T]) -> Callable[[], T]:
        if not take_slot:
            return fn

        def _guarded() -> T:
            return _run_with_slot(fn)

        return _guarded

    wrapped = {key: _wrap(fn) for key, fn in tasks.items()}
    # 叶子：worker 数 ≤ 全局槽；容器（分支）：按支数开线程，槽由内层叶子占用
    if take_slot:
        workers = min(max_concurrency(), len(wrapped))
    else:
        workers = len(wrapped)

    def _submit(executor: ThreadPoolExecutor, fn: Callable[[], T]):
        ctx = contextvars.copy_context()

        def _run() -> T:
            token = _parallel_depth.set(depth + 1)
            try:
                return fn()
            finally:
                _parallel_depth.reset(token)

        return executor.submit(ctx.run, _run)

    executor = ThreadPoolExecutor(max_workers=workers)
    future_to_key: dict = {}
    abrupt = False
    try:
        future_to_key = {
            _submit(executor, fn): key for key, fn in wrapped.items()
        }
        results = {}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            results[key] = future.result()
        return results
    except BaseException:
        abrupt = True
        for future in future_to_key:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not abrupt:
            executor.shutdown(wait=True)
