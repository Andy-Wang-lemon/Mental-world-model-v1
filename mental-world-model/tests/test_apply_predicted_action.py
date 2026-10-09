"""预测所缘保留，只接行动写 H/M。"""

from pathlib import Path

from mental_world_model.engine.cognition.predicted_action import is_usable_predicted_action
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel

ROOT = Path(__file__).resolve().parents[1]


def test_apply_predicted_action_keeps_alambana_and_writes_m(tmp_path):
    model = MentalWorldModel.load(
        ROOT / "definitions",
        seed_state_path=tmp_path / "seed_state.json",
    )
    llm = MockLlmClient(
        default_response={
            "score": 0.7,
            "theta": 0.8,
            "relevance": 0.5,
        }
    )
    text = "林可私下来找陈默，寻求帮助。"
    bundle = model.evaluate_text_with_cognition_branches(
        text,
        llm,
        update_seeds=False,
        update_continuity=False,
        use_action_feedback=False,
        aggregate_behaviors=True,
        predict_branch_behaviors=True,
        focus_person="陈默",
    )
    ids_before = [c.id for c in bundle.perception_state.alambana_candidates]
    texts_before = [c.text for c in bundle.perception_state.alambana_candidates]
    assert ids_before
    assert len(model.continuity_state.mental_episodes) == 0
    assert bundle.aggregated_behaviors

    action = "表面没有反应"
    assert is_usable_predicted_action(action)

    land = model.apply_predicted_action(
        text,
        llm,
        bundle,
        action,
        focus_person="陈默",
        action_probability=0.6,
    )
    split = land.perception_state.stimulus_action_split
    assert split is not None
    assert abs(split.confidence - 0.6) < 1e-9
    ids_after = [c.id for c in land.perception_state.alambana_candidates]
    texts_after = [c.text for c in land.perception_state.alambana_candidates]
    for aid in ids_before:
        assert aid in ids_after
    for t in texts_before:
        assert t in texts_after
    assert len(model.continuity_state.mental_episodes) == 1
    assert not any("落地" in n for n in bundle.perception_state.uncertainty_notes)
    assert any("落地" in n for n in land.perception_state.uncertainty_notes)
    assert land.evaluation_result.action_evidence
    assert land.evaluation_result.action_evidence.get("has_observable_action") is True

    land_low = model.apply_predicted_action(
        text,
        llm,
        bundle,
        action,
        focus_person="陈默",
        action_probability=0.12,
    )
    assert abs(land_low.perception_state.stimulus_action_split.confidence - 0.12) < 1e-9


def test_evaluate_from_yuanwen_lands_top1_once(tmp_path):
    model = MentalWorldModel.load(
        ROOT / "definitions",
        seed_state_path=tmp_path / "seed_state.json",
    )
    llm = MockLlmClient(default_response={"score": 0.7, "theta": 0.8, "relevance": 0.5})
    result = model.evaluate_from_yuanwen(
        "林可私下来找陈默，寻求帮助。",
        llm,
        update_seeds=False,
        focus_person="陈默",
    )
    assert result.aggregated_behaviors
    assert len(model.continuity_state.mental_episodes) == 1
    notes = result.perception_state.uncertainty_notes
    assert any("接入同一套所缘" in n for n in notes)
    assert (result.continuity or {}).get("landed") is True


def test_unusable_top1_skips_continuity_write(tmp_path, monkeypatch):
    import mental_world_model.engine.cognition.predicted_action as pa

    model = MentalWorldModel.load(
        ROOT / "definitions",
        seed_state_path=tmp_path / "seed_state.json",
    )
    llm = MockLlmClient(default_response={"score": 0.7})
    monkeypatch.setattr(
        pa,
        "top_aggregated_action",
        lambda result: ("（无行为预测）", 0.0, []),
    )
    result = model.evaluate_from_yuanwen(
        "林可私下来找陈默，寻求帮助。",
        llm,
        update_seeds=False,
        focus_person="陈默",
    )
    assert len(model.continuity_state.mental_episodes) == 0
    notes = result.perception_state.uncertainty_notes
    assert any("未写入 H/M" in n for n in notes)
