"""逐幕调用 MWM，并隔开每个 trial 的连续性文件。"""

from __future__ import annotations

from pathlib import Path

from mental_world_model.engine.continuity.config import (
    is_cognitive_continuity_enabled,
    is_mental_continuity_enabled,
)
from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.continuity.store import ContinuityStateStore
from mental_world_model.model import MentalWorldModel

from dyntom_adapter.loader import build_role_context, build_scenario_text
from dyntom_adapter.snapshot import freeze_snapshot


def continuity_enabled() -> bool:
    return is_cognitive_continuity_enabled() or is_mental_continuity_enabled()


def load_model(definitions: Path, runtime_root: Path) -> MentalWorldModel:
    """加载模型，并把默认连续性文件指到本次输出目录，避免写入 data/runtime。"""
    runtime_root.mkdir(parents=True, exist_ok=True)
    return MentalWorldModel.load(
        definitions,
        seed_state_path=runtime_root / "unused_seeds.json",
        use_runtime_seeds=False,
    )


def bind_fresh_continuity(model: MentalWorldModel, runtime_dir: Path, person_id: str) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    store = ContinuityStateStore(runtime_dir / "continuity_state.json")
    model.continuity_state_store = store
    model.continuity_state = ContinuityState(person_id=person_id or "default")
    store.save(model.continuity_state)


def bind_saved_continuity(model: MentalWorldModel, runtime_dir: Path) -> ContinuityState:
    store = ContinuityStateStore(runtime_dir / "continuity_state.json")
    if not store.path.is_file():
        raise FileNotFoundError(f"找不到可续跑的连续性文件：{store.path}")
    state = store.load()
    model.continuity_state_store = store
    model.continuity_state = state
    return state


def run_scenario(
    model: MentalWorldModel,
    llm: object,
    story,
    scenario,
) -> dict:
    """一个已发生 scenario：行动反馈与行为分布同时打开，并写入连续性。"""
    bundle = model.evaluate_text_with_cognition_branches(
        build_scenario_text(scenario),
        llm,
        role_context=build_role_context(story),
        update_seeds=False,
        update_continuity=True,
        use_action_feedback=True,
        predict_branch_behaviors=True,
        aggregate_behaviors=True,
        focus_person=story.main_character,
    )
    frozen = freeze_snapshot(bundle.evaluation_result, scenario.index)
    frozen["continuity_event_count"] = int(
        getattr(model.continuity_state, "event_count", 0) or 0
    )
    return frozen
