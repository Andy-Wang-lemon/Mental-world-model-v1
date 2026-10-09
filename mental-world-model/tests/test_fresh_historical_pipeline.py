"""Fresh/Historical 双路径、alignment 硬门闸、行为去重回归。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.cognition.behavior import (
    canonicalize_behavior_texts,
    observable_behavior_key,
    predict_behaviors_for_branch,
)
from mental_world_model.engine.cognition.models import CognitionBranch, ExtractedMentalState
from mental_world_model.engine.perception.builder import build_perception_state
from mental_world_model.engine.perception.grasping_paths import (
    dedup_grasping_list,
    merge_and_dedup_graspings,
)
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.llm.mock_client import MockLlmClient


S11_TEXT = (
    "CEO要求陈默和林可48小时内共同提交挽救方案，两人被安排在同一会议室协作。"
)


class TestCase1StrongNewStimulus:
    def test_s11_fresh_competes_historical_zero_alignment_not_p1(self):
        state = build_perception_state(
            S11_TEXT,
            MockLlmClient(),
            cognitive_bias_frames=[
                {
                    "id": "bf_1",
                    "text": "过去的方法失效，需要寻找新的方法；身份因此困惑",
                    "mass": 0.8,
                }
            ],
            expect_action=False,
            focus_person="陈默",
        )
        obj = state.objective_event
        assert obj is not None
        blob = " ".join(
            obj.constraint_changes
            + obj.role_changes
            + obj.resource_changes
            + [obj.action]
        )
        assert "共同提交" in blob or "48" in blob or "会议室" in blob

        trace = state.pipeline_trace
        assert trace["cognition_candidates"]["fresh_count_raw"] >= 1
        assert trace["cognition_candidates"]["historical_count_raw"] == 0
        assert trace["h_prior"]["triggered_ids"]
        assert "bf_1" in trace["h_prior"]["triggered_ids"]
        assert trace["cognition_candidates"]["after_semantic_dedup"] >= 2
        assert trace["cognition_candidates"]["valid_after_alignment_gate"] >= 1
        assert trace["used_neutral_fallback"] is False
        assert trace["accommodation"]["triggered"] is False

        fresh_texts = [
            h.text
            for h in (
                state.self_grasping_hypotheses + state.dharma_grasping_hypotheses
            )
            if h.source_type == "fresh"
        ]
        hist_ids = [
            h.id
            for h in (
                state.self_grasping_hypotheses + state.dharma_grasping_hypotheses
            )
            if h.source_type == "historical"
        ]
        stamped = [
            h
            for h in (
                state.self_grasping_hypotheses + state.dharma_grasping_hypotheses
            )
            if "bf_1" in (h.source_bias_frame_ids or [])
        ]
        assert fresh_texts
        assert any(
            "责任" in text or "CEO" in text or "共同" in text for text in fresh_texts
        )
        assert not hist_ids
        assert stamped

        combos = state.grasping_combos
        assert combos
        for combo in combos:
            assert float(combo.get("activation_alignment") or 0) >= 0.15
            assert combo.get("source_type") != "historical"
        assert any(combo.get("reactivated_h_ids") for combo in combos)

        for alb in state.alambana_candidates:
            assert "林可说" not in alb.text
            assert "林可建议" not in alb.text
            assert "林可告诉" not in alb.text
            assert alb.grounding_valid is True
            if alb.activation_alignment <= 0:
                assert alb.source_type == "neutral"

    def test_s11_behavior_candidates_are_distinct_before_probability(self):
        branch = CognitionBranch(
            id="b1",
            alambana=SubjectiveAlambanaCandidate(
                id="O1",
                text="陈默把CEO的安排理解为自己重新纳入共同责任",
                confidence=0.7,
            ),
            xiang=ExtractedMentalState(
                text="想", source="推理", confidence=0.6, alambana_id="O1"
            ),
        )
        xinsuo = {
            "瞋": Xinsuo(
                name="瞋", suoyuan="s", shengqi_nian="n", ye="y", xiandai_xinli="x"
            ),
        }
        preds = predict_behaviors_for_branch(
            S11_TEXT,
            branch,
            {"瞋": 0.8},
            xinsuo,
            MockLlmClient(),
        )
        assert branch.behavior_diagnostics["distinct_candidate_count"] >= 2
        assert branch.behavior_diagnostics["candidate_collapsed"] is False
        texts = [p.canonical_text or p.text for p in preds]
        assert len(set(texts)) == len(texts)


class TestCase2DuplicateCognition:
    def test_duplicate_dharma_merged_before_combo(self):
        dharma = [
            GraspingHypothesis(
                id="dg1",
                type="dharma_grasping",
                text="过去的方法失效，需要寻找新的方法",
                confidence=0.8,
                source_type="fresh",
            ),
            GraspingHypothesis(
                id="dg2",
                type="dharma_grasping",
                text="过去的方法失效，需要寻找新的方法",
                confidence=0.7,
                source_type="historical",
                source_bias_frame_ids=["bf_1"],
            ),
        ]
        self_g = [
            GraspingHypothesis(
                id="sg1",
                type="self_grasping",
                text="自己重新成为这项任务责任的一部分",
                confidence=0.8,
                source_type="fresh",
            )
        ]
        _merged_self, merged_dharma = merge_and_dedup_graspings([self_g], [dharma])
        assert len(merged_dharma) == 1
        kept = dedup_grasping_list(dharma)
        assert len(kept) == 1
        assert kept[0].id == "dg1"

    def test_fresh_kept_over_higher_confidence_historical(self):
        dharma = [
            GraspingHypothesis(
                id="dg_hist",
                type="dharma_grasping",
                text="过去的方法失效，需要寻找新的方法",
                confidence=0.95,
                source_type="historical",
                source_bias_frame_ids=["bf_1"],
            ),
            GraspingHypothesis(
                id="dg_fresh",
                type="dharma_grasping",
                text="过去的方法失效，需要寻找新的方法",
                confidence=0.55,
                source_type="fresh",
            ),
        ]
        kept = dedup_grasping_list(dharma)
        assert len(kept) == 1
        assert kept[0].id == "dg_fresh"


class TestCase3DuplicateBehavior:
    def test_silence_paraphrases_canonicalize(self):
        clustered = canonicalize_behavior_texts(
            ["保持沉默", "继续保持沉默", "没有回应"],
            llm=MockLlmClient(),
        )
        canons = {canonical for _, canonical in clustered}
        assert len(canons) == 1
        assert next(iter(canons)) == "保持沉默"

    def test_branch_does_not_inflate_silence_to_fake_certainty(self):
        class Client:
            def complete(self, messages, schema=None):
                props = (schema or {}).get("properties") or {}
                if "behavior_candidates" in props:
                    return {
                        "behavior_candidates": [
                            {"text": "保持沉默"},
                            {"text": "继续保持沉默"},
                            {"text": "没有回应"},
                        ]
                    }
                if "groups" in props:
                    return {
                        "groups": [
                            {
                                "canonical_action": "保持沉默",
                                "surface_forms": [
                                    "保持沉默",
                                    "继续保持沉默",
                                    "没有回应",
                                ],
                            }
                        ]
                    }
                if "behavior_probabilities" in props:
                    return {
                        "behavior_probabilities": [
                            {
                                "canonical_action": "保持沉默",
                                "conditional_probability": 1.0,
                                "confidence": 0.4,
                            }
                        ]
                    }
                if "behavior_predictions" in props:
                    return {
                        "behavior_predictions": [
                            {"text": "保持沉默", "conditional_probability": 0.4},
                            {"text": "继续保持沉默", "conditional_probability": 0.3},
                            {"text": "没有回应", "conditional_probability": 0.3},
                        ]
                    }
                return {"items": []}

        branch = CognitionBranch(
            id="b1",
            alambana=SubjectiveAlambanaCandidate(
                id="O1", text="陈默把CEO安排理解成重新承担责任", confidence=0.7
            ),
            xiang=ExtractedMentalState(
                text="想", source="推理", confidence=0.6, alambana_id="O1"
            ),
        )
        xinsuo = {
            "瞋": Xinsuo(
                name="瞋", suoyuan="s", shengqi_nian="n", ye="y", xiandai_xinli="x"
            ),
        }
        preds = predict_behaviors_for_branch(
            S11_TEXT,
            branch,
            {"瞋": 0.8},
            xinsuo,
            Client(),  # type: ignore[arg-type]
        )
        assert len(preds) == 1
        assert preds[0].canonical_text == "保持沉默"
        assert branch.behavior_diagnostics["raw_candidate_count"] == 3
        assert branch.behavior_diagnostics["distinct_candidate_count"] == 1
        assert branch.behavior_diagnostics["candidate_collapsed"] is True
        assert abs(preds[0].conditional_probability - 1.0) < 1e-9

    def test_breaking_silence_is_not_merged_into_keeping_silence(self):
        assert observable_behavior_key("打破沉默发言") != observable_behavior_key(
            "保持沉默"
        )
        assert observable_behavior_key("沉默地点头") != observable_behavior_key(
            "保持沉默"
        )
        clustered = canonicalize_behavior_texts(
            ["保持沉默", "打破沉默发言", "沉默地点头"],
            llm=None,
        )
        canons = {canonical for _, canonical in clustered}
        assert len(canons) == 3

    def test_silence_only_phrases_still_merge_without_llm(self):
        clustered = canonicalize_behavior_texts(
            ["保持沉默", "继续保持沉默", "没有回应"],
            llm=None,
        )
        canons = {canonical for _, canonical in clustered}
        assert len(canons) == 1

    def test_single_raw_candidate_skips_diversity_retry(self):
        class Client:
            def complete(self, messages, schema=None):
                props = (schema or {}).get("properties") or {}
                if "behavior_candidates" in props:
                    return {"behavior_candidates": [{"text": "当场提交方案"}]}
                if "groups" in props:
                    return {
                        "groups": [
                            {
                                "canonical_action": "当场提交方案",
                                "surface_forms": ["当场提交方案"],
                            }
                        ]
                    }
                if "behavior_probabilities" in props:
                    return {
                        "behavior_probabilities": [
                            {
                                "canonical_action": "当场提交方案",
                                "conditional_probability": 1.0,
                                "confidence": 0.7,
                            }
                        ]
                    }
                if "behavior_predictions" in props:
                    return {
                        "behavior_predictions": [
                            {
                                "text": "当场提交方案",
                                "conditional_probability": 1.0,
                            }
                        ]
                    }
                return {"items": []}

        branch = CognitionBranch(
            id="b1",
            alambana=SubjectiveAlambanaCandidate(
                id="O1", text="陈默把CEO安排理解成重新承担责任", confidence=0.7
            ),
            xiang=ExtractedMentalState(
                text="想", source="推理", confidence=0.6, alambana_id="O1"
            ),
        )
        xinsuo = {
            "瞋": Xinsuo(
                name="瞋", suoyuan="s", shengqi_nian="n", ye="y", xiandai_xinli="x"
            ),
        }
        preds = predict_behaviors_for_branch(
            S11_TEXT,
            branch,
            {"瞋": 0.8},
            xinsuo,
            Client(),  # type: ignore[arg-type]
        )
        assert len(preds) == 1
        assert branch.behavior_diagnostics["raw_candidate_count"] == 1
        assert branch.behavior_diagnostics["diversity_regenerated"] is False
        assert branch.behavior_diagnostics["candidate_collapsed"] is True
