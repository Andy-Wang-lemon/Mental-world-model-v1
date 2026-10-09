"""对一道 RecTOM 题调用一次 MWM。"""

from __future__ import annotations

from pathlib import Path

from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.continuity.store import ContinuityStateStore
from mental_world_model.model import MentalWorldModel

from rectom_adapter.loader import RectomItem
from rectom_adapter.snapshot import freeze_snapshot


def load_model(definitions: Path, runtime_root: Path) -> MentalWorldModel:
    runtime_root.mkdir(parents=True, exist_ok=True)
    return MentalWorldModel.load(
        definitions,
        seed_state_path=runtime_root / "unused_seeds.json",
        use_runtime_seeds=False,
    )


def run_item(model: MentalWorldModel, llm: object, item: RectomItem, runtime_dir: Path) -> dict:
    """对话前文已经在本题文本里，这次不继承别的题的连续性。"""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    store = ContinuityStateStore(runtime_dir / "continuity_state.json")
    model.continuity_state_store = store
    model.continuity_state = ContinuityState(person_id=item.focus_person)
    store.save(model.continuity_state)
    bundle = model.evaluate_text_with_cognition_branches(
        item.utterance_context,
        llm,
        role_context=(
            f"要理解的人是 {item.focus_person}。"
            "对话里还有另一位谈话者。根据目前已经说出的话，更新这个人的所缘、想和下一步行为倾向。"
        ),
        update_seeds=False,
        update_continuity=False,
        use_action_feedback=True,
        predict_branch_behaviors=True,
        aggregate_behaviors=True,
        focus_person=item.focus_person,
    )
    return freeze_snapshot(bundle.evaluation_result)
