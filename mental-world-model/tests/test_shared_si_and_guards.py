"""共享思绑定与护栏。"""

from pathlib import Path

from mental_world_model.engine.cognition.xiang_si import bind_si_to_alambana
from mental_world_model.engine.cognition.models import ExtractedMentalState
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel


ROOT = Path(__file__).resolve().parents[1]


class TestSharedSiBinding:
    def test_bind_keeps_text_changes_alambana_id(self):
        si = ExtractedMentalState(
            text="压制",
            source="原文",
            confidence=0.9,
            alambana_id="a1",
        )
        bound = bind_si_to_alambana(si, "a2")
        assert bound.text == "压制"
        assert bound.alambana_id == "a2"
        assert si.alambana_id == "a1"


class TestSharedSiOnce:
    def test_action_path_binds_si_once_and_xiang_once(self, tmp_path):
        model = MentalWorldModel.load(
            ROOT / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )
        split_calls = {"n": 0}
        likelihood_calls = {"n": 0}
        xiang_calls = {"n": 0}

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            # L0：刺激/行动拆分（契约保证有行动，不判有无/置信度）
            if "stimulus_text" in props and "action_span" in props and "likelihoods" not in props:
                split_calls["n"] += 1
                return {
                    "stimulus_text": "员工指出风险后，[行动已隔离]。",
                    "action_span": "领导当场打断并否定",
                    "action_description": "压制式打断否定",
                    "reason_summary": "拆分",
                }
            # 认知层：只补 L（LLM raw compat）
            if "likelihoods" in props and "has_observable_action" not in props:
                likelihood_calls["n"] += 1
                return {
                    "likelihoods": [
                        {"alambana_id": "O1", "likelihood": 0.8, "support_notes": []},
                        {"alambana_id": "O2", "likelihood": 0.4, "support_notes": []},
                        {"alambana_id": "O_C1", "likelihood": 0.8, "support_notes": []},
                        {"alambana_id": "O_C2", "likelihood": 0.4, "support_notes": []},
                    ],
                    "reason_summary": "似然",
                }
            if "xiangs" in props:
                # 单支隔离：每次调用只生成一支想
                xiang_calls["n"] += 1
                content = str(messages)
                aid = "O2" if 'alambana_id=O2' in content or '"id": "O2"' in content else "O1"
                text = "这是提醒" if aid == "O2" else "这是挑衅"
                return {
                    "xiangs": [
                        {
                            "alambana_id": aid,
                            "text": text,
                            "source": "推理",
                            "confidence": 0.7,
                            "reason_summary": f"{aid}想",
                        }
                    ]
                }
            if "behavior_predictions" in props:
                return {
                    "behavior_predictions": [
                        {
                            "behavior_id": "b1",
                            "text": "继续否定",
                            "conditional_probability": 1.0,
                            "confidence": 0.6,
                            "supporting_mental_factors": [],
                            "opposing_mental_factors": [],
                            "supporting_scores": {},
                            "reason_summary": "mock",
                        }
                    ]
                }
            if "winner" in props:
                return {"winner": "none"}
            return {"score": 0.5}

        yuanwen = "员工指出风险后，领导当场打断并否定。"
        bundle = model.evaluate_text_with_cognition_branches(
            yuanwen,
            MockLlmClient(handler=handler),
            update_seeds=False,
            use_action_feedback=True,
            aggregate_behaviors=False,
        )
        result = bundle.evaluation_result
        assert 1 <= len(result.cognition_branches) <= 3
        assert split_calls["n"] == 1
        assert likelihood_calls["n"] == 1
        assert xiang_calls["n"] == len(result.cognition_branches)
        texts = {b.si.text for b in result.cognition_branches if b.si}
        assert texts == {"压制式打断否定"}
        nonempty_xiang = [b for b in result.cognition_branches if b.xiang and b.xiang.text]
        assert len(nonempty_xiang) == len(result.cognition_branches)
        assert result.action_evidence is not None
        assert result.action_evidence["has_observable_action"] is True
