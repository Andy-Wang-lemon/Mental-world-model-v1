"""并行执行单元测试。"""

import time

from mental_world_model.engine.scoring.parallel_exec import run_parallel
from mental_world_model.llm.usage import current_llm_tracker, llm_call_context, llm_evaluation_scope


class TestRunParallel:
    def test_serial_fallback_when_disabled(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_PARALLEL", "0")
        order: list[str] = []

        def make_task(key: str):
            def task():
                order.append(key)
                return key

            return task

        result = run_parallel({f"k{i}": make_task(f"k{i}") for i in range(3)})
        assert result == {"k0": "k0", "k1": "k1", "k2": "k2"}
        assert order == ["k0", "k1", "k2"]

    def test_parallel_runs_all_tasks(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_PARALLEL", "1")
        monkeypatch.setenv("MWM_LLM_MAX_CONCURRENCY", "4")
        result = run_parallel(
            {
                "a": lambda: 1,
                "b": lambda: 2,
                "c": lambda: 3,
            }
        )
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_parallel_context_not_shared_across_threads(self, monkeypatch):
        """回归：多任务并行时不能共用一个 Context（会 already entered）。"""
        monkeypatch.setenv("MWM_LLM_PARALLEL", "1")
        monkeypatch.setenv("MWM_LLM_MAX_CONCURRENCY", "6")

        def make_task(key: str):
            def task():
                with llm_call_context(stage=f"stage.{key}"):
                    time.sleep(0.05)
                    tracker = current_llm_tracker()
                    return (key, tracker is not None)

            return task

        with llm_evaluation_scope() as tracker:
            result = run_parallel(
                {f"k{i}": make_task(f"k{i}") for i in range(8)}
            )

        assert len(result) == 8
        assert all(ok for _, ok in result.values())
        assert tracker is not None

    def test_interrupt_shuts_down_without_waiting(self, monkeypatch):
        """Ctrl+C 路径应 cancel_futures，不等待卡住的任务。"""
        import pytest

        monkeypatch.setenv("MWM_LLM_PARALLEL", "1")
        monkeypatch.setenv("MWM_LLM_MAX_CONCURRENCY", "2")
        started = time.perf_counter()

        def hang():
            time.sleep(10)
            return "x"

        def boom():
            time.sleep(0.05)
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run_parallel({"a": hang, "b": boom})
        assert time.perf_counter() - started < 2.0

    def test_default_concurrency_covers_full_xinsuo_wave(self, monkeypatch):
        monkeypatch.delenv("MWM_LLM_MAX_CONCURRENCY", raising=False)
        from mental_world_model.engine.scoring.parallel_exec import (
            DEFAULT_LLM_MAX_CONCURRENCY,
            XINSUO_TABLE_SIZE,
            max_concurrency,
            reset_llm_slots_for_tests,
        )

        reset_llm_slots_for_tests()
        assert max_concurrency() == DEFAULT_LLM_MAX_CONCURRENCY
        assert DEFAULT_LLM_MAX_CONCURRENCY >= XINSUO_TABLE_SIZE * 3

    def test_full_xinsuo_wave_can_run_all_leaves(self, monkeypatch):
        """闸门 ≥ 心所数时，单波全部叶子可同时进入临界区。"""
        monkeypatch.setenv("MWM_LLM_PARALLEL", "1")
        monkeypatch.setenv("MWM_LLM_MAX_CONCURRENCY", "40")
        from mental_world_model.engine.scoring.parallel_exec import reset_llm_slots_for_tests

        reset_llm_slots_for_tests()
        n_xinsuo = 38
        active = {"n": 0, "peak": 0}
        lock = __import__("threading").Lock()
        started = __import__("threading").Barrier(n_xinsuo)

        def leaf():
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            started.wait(timeout=2)
            with lock:
                active["n"] -= 1
            return 1

        run_parallel({f"x{i}": leaf for i in range(n_xinsuo)}, acquire_slot=True)
        assert active["peak"] == n_xinsuo

    def test_container_plus_leaf_respects_global_slots(self, monkeypatch):
        """分支容器不占槽；内心所占槽 → 同时进入临界区的叶子 ≤ 并发上限。"""
        monkeypatch.setenv("MWM_LLM_PARALLEL", "1")
        monkeypatch.setenv("MWM_LLM_MAX_CONCURRENCY", "2")
        from mental_world_model.engine.scoring.parallel_exec import reset_llm_slots_for_tests

        reset_llm_slots_for_tests()
        active = {"n": 0, "peak": 0}
        lock = __import__("threading").Lock()

        def leaf():
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.08)
            with lock:
                active["n"] -= 1
            return 1

        def branch_job():
            return run_parallel(
                {f"x{i}": leaf for i in range(4)},
                acquire_slot=True,
            )

        run_parallel(
            {"b1": branch_job, "b2": branch_job},
            acquire_slot=False,
        )
        assert active["peak"] <= 2
