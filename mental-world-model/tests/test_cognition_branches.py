"""多认知分支集成测试（Mock LLM）。"""

from pathlib import Path

import mental_world_model.engine.perception.config as perception_config
from mental_world_model.engine.cognition.adapters import branch_to_extracted_context
from mental_world_model.engine.cognition.models import CognitionBranch, ExtractedMentalState
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate
from mental_world_model.engine.scoring.weights import compute_source_modifiers
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel


ROOT = Path(__file__).resolve().parents[1]


def _load_model(tmp_path):
    return MentalWorldModel.load(
        ROOT / "definitions",
        seed_state_path=tmp_path / "seed_state.json",
    )


class TestCognitionBranchIntegration:
    def test_evaluate_keeps_multiple_branches(self, tmp_path):
        model = _load_model(tmp_path)
        llm = MockLlmClient(
            default_response={
                "score": 0.8,
                "fact_text": "员工在会议中指出方案存在风险",
                "fact_span": "员工在会议中指出方案存在风险",
            }
        )
        yuanwen = "员工在会议中指出方案存在风险。"
        result = model.evaluate_from_yuanwen(yuanwen, llm, update_seeds=False)
        assert result.cognition_branches is not None
        assert 1 <= len(result.cognition_branches) <= 3
        assert result.perception_state is not None
        assert result.aggregated_behaviors is not None
        assert result.expected_mental_factor_scores is not None
        assert abs(
            sum(b.branch_effective_probability for b in result.cognition_branches)
            - 1.0
        ) < 1e-6
        report = result.to_report_dict()
        assert "events" in report
        assert "pipeline_trace" in report
        assert "perception_inference" not in report
        assert "cognition_branches" not in report
        assert "aggregated_results" not in report
        trace = report["pipeline_trace"]
        assert "objective_event" in trace
        assert "self_grasping" in trace
        assert "dharma_grasping" in trace
        assert "subjective_alambanas" in trace
        assert "action_evidence" in trace
        assert "branches" in trace
        assert "aggregated" in trace
        assert "evaluation_result" in trace
        assert "action_detection" not in trace
        assert "objective_facts" not in trace
        assert "uncertainty_notes" not in trace
        assert len(trace["branches"]) == len(result.cognition_branches)
        for item in trace["subjective_alambanas"]:
            assert "action_compat" in item
            assert "action_likelihood" in item
            assert "action_informed" in item
        for branch in trace["branches"]:
            assert "xiang" in branch
            assert "si" in branch
            assert "mental_factor_scores_final" in branch
            assert "behavior_predictions" in branch
            # 精简报告只保留 >0 心所
            assert all(v > 0 for v in branch["mental_factor_scores_final"].values())
            al = branch.get("alambana") or {}
            assert "action_compat" in al
            assert "action_likelihood" in al
        assert result.scores_final == result.expected_mental_factor_scores
        assert result.mental_factors_event == result.expected_mental_factor_scores
        # 预测事件默认跨分支融合行为概率（不再生成 prediction_reason 文案）
        assert result.prediction_reason is None
        assert result.predicted_behavior is None
        assert result.aggregated_behaviors
        assert len(result.aggregated_behaviors) >= 1
        # 多支模式：不再指定摘要支 / 主所缘
        assert not hasattr(result, "summary_branch_id")
        assert result.perception_state is not None
        assert "selected_alambana" not in result.perception_state.to_dict()
        assert result.action_detection is not None
        # 顶层 extracted 并列各支
        assert result.extracted is not None
        assert "a=" in result.extracted.suoyuan.text
        assert result.stage_results == []
        assert "action_evidence" in trace
        verbose = result.to_report_dict(verbose=True)
        assert "cognition_branches" in verbose
        assert "perception_inference" in verbose
        assert "aggregated_results" in verbose
        assert "aggregated_action_distribution" in verbose["aggregated_results"]
        assert len(verbose["aggregated_results"]["aggregated_action_distribution"]) >= 1
        assert "action_distribution" in verbose["cognition_branches"][0]
        assert 1 <= len(verbose["cognition_branches"][0]["action_distribution"]) <= 3
        snap = result.to_event_snapshot()
        assert "aggregated_behaviors" in snap
        assert "stimulus_action_split" in snap or "stimulus_text" in snap
        assert "grasping_combos" in snap or snap.get("self_grasping") is not None
        assert "predicted_behavior" not in snap
        assert "mental_state_distribution" not in snap
        assert "action_detection" not in snap
        assert "activation_probability" in snap["branches"][0]
        assert "mental_factors_nonzero" in snap["branches"][0]
        assert "behavior_contribution_probability" not in snap["branches"][0]
        assert "mental_factors_event_nonzero" in snap
        # 预测事件：无跨支心所 final，看各支 mental_factors_nonzero
        assert snap.get("mental_report_mode") == "per_branch_only"
        assert "mental_factors_final_nonzero" not in snap
        assert result.active_xinsuo is None
        contrib0 = snap["aggregated_behaviors"][0]["branch_contributions"][0]
        assert "activation_probability" in contrib0
        assert "contribution" in contrib0
        assert "behavior_contribution_probability" not in contrib0
        for item in snap["subjective_alambanas"]:
            assert "action_compat" in item
            assert "action_likelihood" in item
            assert "action_informed" in item
            assert "prior_probability" in item
            assert "posterior_probability" in item
        al = snap["branches"][0].get("alambana") or {}
        assert "action_compat" in al
        assert "action_likelihood" in al

    def test_branch_scoring_includes_yuanwen(self):
        from mental_world_model.engine.cognition.adapters import branch_to_scoring_context

        branch = CognitionBranch(
            id="branch_1",
            alambana=SubjectiveAlambanaCandidate(
                id="a1",
                text="挑战资格",
                posterior_probability=0.6,
                confidence=0.8,
                source="推理",
            ),
            xiang=ExtractedMentalState(
                text="这是挑衅",
                source="推理",
                confidence=0.7,
                alambana_id="a1",
            ),
            si=ExtractedMentalState(
                text="压制",
                source="原文",
                confidence=0.9,
                alambana_id="a1",
            ),
        )
        ctx = branch_to_scoring_context("领导当场打断并否定员工意见。", branch)
        block = ctx.base_context_block
        assert "领导当场打断并否定员工意见。" in block
        assert "挑战资格" in block
        assert "这是挑衅" in block
        assert "压制" in block

    def test_action_updates_posterior_and_binds_si(self, tmp_path):
        model = _load_model(tmp_path)
        yuanwen = "员工指出风险后，领导当场打断并否定。"
        llm = MockLlmClient(
            default_response={
                "score": 0.8,
                "has_explicit_action": True,
                "action_span": "领导当场打断并否定",
                "action_si_text": "压制",
                "fact_text": "领导当场打断并否定",
                "fact_span": "领导当场打断并否定",
            }
        )
        bundle = model.evaluate_text_with_cognition_branches(
            yuanwen,
            llm,
            update_seeds=False,
            use_action_feedback=True,
            aggregate_behaviors=False,
        )
        result = bundle.evaluation_result
        assert result.cognition_branches
        assert result.action_evidence is not None
        assert result.action_evidence["has_observable_action"] is True
        assert any(
            "行动证据已更新后验" in n
            for n in (result.perception_state.uncertainty_notes or [])
        )
        assert any(
            "compat/扩支/后验已完成" in n
            for n in (result.perception_state.uncertainty_notes or [])
        )
        assert any(
            "想已分支出成" in n
            for n in (result.perception_state.uncertainty_notes or [])
        )
        si_texts = {b.si.text for b in result.cognition_branches if b.si}
        assert si_texts == {"压制"}
        nonempty_xiang = [b for b in result.cognition_branches if b.xiang and b.xiang.text]
        assert len(nonempty_xiang) == len(result.cognition_branches)

    def test_multi_event_routing_uses_history_and_final_no_action_feedback(self, tmp_path):
        model = _load_model(tmp_path)
        yuanwen = (
            "事件1：员工指出风险后，领导当场打断并否定。\n"
            "事件2：会议结束后，员工私下再次沟通。"
        )
        llm = MockLlmClient(
            default_response={
                "score": 0.8,
                "has_explicit_action": True,
                "action_span": "领导当场打断并否定",
                "action_si_text": "压制",
                "fact_text": "领导当场打断并否定",
                "fact_span": "领导当场打断并否定",
            }
        )
        result = model.evaluate_from_yuanwen(yuanwen, llm, update_seeds=False)
        assert result.cognition_branches
        assert result.action_evidence is not None
        assert result.action_evidence["has_observable_action"] is False
        assert all(not (b.si and b.si.text) for b in result.cognition_branches)
        assert any(
            "事件路由：检测到 2 件事件" in n
            for n in (result.perception_state.uncertainty_notes or [])
        )
        assert result.event_reports is not None
        assert len(result.event_reports) == 2
        assert result.event_reports[0]["event"] == "事件1"
        assert result.event_reports[1]["event"] == "事件2"
        assert "yuanwen" in result.event_reports[0]
        assert "branches" in result.event_reports[0]
        assert "mental_factors_event_nonzero" in result.event_reports[0]
        # 前序事件仍有跨支 final；预测事件为 per_branch_only
        assert "mental_factors_final_nonzero" in result.event_reports[0]
        assert result.event_reports[1].get("mental_report_mode") == "per_branch_only"
        assert "mental_factors_final_nonzero" not in result.event_reports[1]
        # 默认事件快照不再附带 uncertainty notes
        assert "notes" not in result.event_reports[0]
        assert "yuanwen" in result.event_reports[0] or "branches" in result.event_reports[0]
        # 默认报告与 verbose 均以 events 存全部历史中间结果
        slim = result.to_event_report_dict()
        assert [e["event"] for e in slim["events"]] == ["事件1", "事件2"]
        verbose = result.to_report_dict(verbose=True)
        assert [e["event"] for e in verbose["events"]] == ["事件1", "事件2"]
        assert "branches" in verbose["events"][0]
        assert "cognition_branches" in verbose  # 最后一事件调试明细仍保留

    def test_no_action_keeps_prior_and_empty_si(self, tmp_path):
        model = _load_model(tmp_path)
        yuanwen = "员工在会议中指出方案存在风险。"
        llm = MockLlmClient(
            default_response={
                "score": 0.8,
                "has_explicit_action": False,
                "fact_text": "员工在会议中指出方案存在风险",
                "fact_span": "员工在会议中指出方案存在风险",
            }
        )
        result = model.evaluate_from_yuanwen(yuanwen, llm, update_seeds=False)
        assert result.cognition_branches
        # 单事件走 use_action_feedback=False 路由，notes 中为"待预测事件"或"思置空"
        notes = result.perception_state.uncertainty_notes or []
        assert any(
            "待预测" in n or "无行动" in n or "思置空" in n or "业列置空" in n
            for n in notes
        )
        assert all(not (b.si and b.si.text) for b in result.cognition_branches)
        nonempty_xiang = [b for b in result.cognition_branches if b.xiang and b.xiang.text]
        assert len(nonempty_xiang) == len(result.cognition_branches)

    def test_branch_scoring_modifiers_are_one(self):
        branch = CognitionBranch(
            id="branch_1",
            alambana=SubjectiveAlambanaCandidate(
                id="a1",
                text="挑战资格",
                posterior_probability=0.6,
                confidence=0.5,
                source="推理",
            ),
            xiang=ExtractedMentalState(
                text="这是挑衅",
                source="推理",
                confidence=0.5,
                alambana_id="a1",
            ),
            si=ExtractedMentalState(
                text="压制",
                source="推理",
                confidence=0.4,
                alambana_id="a1",
            ),
        )
        extracted = branch_to_extracted_context("原文", branch)
        modifiers = compute_source_modifiers(extracted)
        assert all(v == 1.0 for v in modifiers.values())
        assert extracted.xiang.confidence == 1.0
        assert extracted.si.confidence == 1.0

    def test_no_cross_branch_conflict_resolution(self, tmp_path):
        """各分支独立消解：分支 A 有嗔、分支 B 有无嗔，不得互相消解。"""
        model = _load_model(tmp_path)

        def handler(messages, schema):
            props = (schema or {}).get("properties", {})
            # L1 客观事件
            if "actors" in props and "outcome_changes" in props:
                return {
                    "actors": ["员工", "当事人"],
                    "action": "指出方案存在风险",
                    "object": "方案",
                    "scene": "会议",
                    "outcome_changes": [],
                }
            # L2 我执+法执
            if "self_grasping" in props and "dharma_grasping" in props:
                return {
                    "self_grasping": [
                        {
                            "id": "g1",
                            "text": "执取领导资格为我",
                            "confidence": 0.85,
                        },
                        {
                            "id": "g2",
                            "text": "把协作互助当成我的位置",
                            "confidence": 0.75,
                        },
                    ],
                    "dharma_grasping": [
                        {
                            "id": "d1",
                            "text": "指出风险等于挑战权威",
                            "confidence": 0.8,
                        },
                        {
                            "id": "d2",
                            "text": "指出风险等于帮助团队",
                            "confidence": 0.7,
                        },
                    ],
                }
            # L2.5 组合元数据 / 语义确认（均为 items）
            if "items" in props and "candidates" not in props:
                item_props = (
                    ((props.get("items") or {}).get("items") or {}).get("properties")
                    or {}
                )
                if "same_perspective" in item_props:
                    # 语义确认：默认同视角，保持既有去重行为
                    return {
                        "items": [
                            {"pair_id": "p0", "same_perspective": True},
                            {"pair_id": "p1", "same_perspective": True},
                            {"pair_id": "p2", "same_perspective": True},
                            {"pair_id": "p3", "same_perspective": True},
                        ]
                    }
                return {
                    "items": [
                        {
                            "combo_id": "C1",
                            "activation_alignment": 0.9,
                        },
                        {
                            "combo_id": "C2",
                            "activation_alignment": 0.85,
                        },
                        {
                            "combo_id": "C3",
                            "activation_alignment": 0.8,
                        },
                        {
                            "combo_id": "C4",
                            "activation_alignment": 0.75,
                        },
                    ]
                }
            # L3 所缘渲染（candidates + combo_id）
            if "candidates" in props:
                content = str(messages)
                # 按已给组合逐条渲染；若 prompt 含多个 combo，返回对应文本
                renders = []
                if "C1" in content:
                    renders.append(
                        {
                            "id": "O1",
                            "combo_id": "C1",
                            "text": "员工正在挑战我的领导资格",
                        }
                    )
                if "C2" in content:
                    renders.append(
                        {
                            "id": "O2",
                            "combo_id": "C2",
                            "text": "员工正在帮助团队发现风险",
                        }
                    )
                if "C3" in content:
                    renders.append(
                        {
                            "id": "O3",
                            "combo_id": "C3",
                            "text": "风险提醒被看成能力标准之争",
                        }
                    )
                if not renders:
                    renders = [
                        {
                            "id": "O1",
                            "combo_id": "C1",
                            "text": "员工正在挑战我的领导资格",
                        },
                        {
                            "id": "O2",
                            "combo_id": "C2",
                            "text": "员工正在帮助团队发现风险",
                        },
                    ]
                return {"candidates": renders}
            # 兼容旧 schema
            if "objective_facts" in props:
                return {
                    "objective_facts": [
                        {
                            "text": "员工在会议中指出方案存在风险",
                            "source_span": "员工在会议中指出方案存在风险",
                            "confidence": 1.0,
                        }
                    ]
                }
            if "hypotheses" in props:
                return {
                    "hypotheses": [
                        {
                            "id": "g1",
                            "type": "self_grasping",
                            "text": "执取领导资格为我",
                            "prior_probability": 0.7,
                            "evidence_likelihood": 0.7,
                            "posterior_probability": 0.8,
                            "activation_probability": 0.85,
                            "supporting_evidence": ["会议场景"],
                            "counter_evidence": [],
                            "missing_information": [],
                            "source": "推理",
                            "confidence": 0.75,
                        },
                        {
                            "id": "g2",
                            "type": "self_grasping",
                            "text": "把团队风险排查当成固定协作义务",
                            "prior_probability": 0.6,
                            "evidence_likelihood": 0.6,
                            "posterior_probability": 0.7,
                            "activation_probability": 0.75,
                            "supporting_evidence": ["开放态度"],
                            "counter_evidence": [],
                            "missing_information": [],
                            "source": "推理",
                            "confidence": 0.7,
                        },
                    ]
                }
            if "alambana_candidates" in props:
                return {
                    "alambana_candidates": [
                        {
                            "id": "alambana_a",
                            "text": "员工正在挑战我的领导资格",
                            "objective_basis": ["指出方案风险"],
                            "self_grasping_ids": ["g1"],
                            "dharma_grasping_ids": [],
                            "posterior_probability": 0.55,
                            "confidence": 0.6,
                            "added_meaning": ["挑战"],
                            "distortion_risk": 0.4,
                            "missing_conditions": [],
                            "source": "推理",
                        },
                        {
                            "id": "alambana_b",
                            "text": "员工正在帮助团队发现风险",
                            "objective_basis": ["指出方案风险"],
                            "self_grasping_ids": ["g2"],
                            "dharma_grasping_ids": [],
                            "posterior_probability": 0.45,
                            "confidence": 0.55,
                            "added_meaning": ["帮助"],
                            "distortion_risk": 0.3,
                            "missing_conditions": [],
                            "source": "推理",
                        },
                    ]
                }
            if (
                "text" in props
                and "distortion_risk" in props
                and "alambana_id" not in props
                and "alambana_candidates" not in props
            ):
                content = str(messages)
                # 仅按执 id / 执文本区分，勿匹配 prompt 里的「会帮助」字样
                if "[g2]" in content or "固定协作义务" in content:
                    return {
                        "text": "员工正在帮助团队发现风险",
                        "confidence": 0.55,
                        "objective_basis": ["指出方案风险"],
                        "added_meaning": ["帮助"],
                        "distortion_risk": 0.3,
                        "missing_conditions": [],
                        "source": "推理",
                    }
                return {
                    "text": "员工正在挑战我的领导资格",
                    "confidence": 0.6,
                    "objective_basis": ["指出方案风险"],
                    "added_meaning": ["挑战"],
                    "distortion_risk": 0.4,
                    "missing_conditions": [],
                    "source": "推理",
                }
            if "xiang_text" in props and "behavior_predictions" not in props:
                content = str(messages)
                if "挑战" in content:
                    return {
                        "text": "这是挑衅",
                        "source": "推理",
                        "confidence": 0.6,
                        "alambana_id": "alambana_a",
                        "xiang_text": "这是挑衅",
                        "supporting_grasping_ids": ["g1"],
                        "reason_summary": "挑战分支",
                    }
                return {
                    "text": "听取并检查",
                    "source": "推理",
                    "confidence": 0.55,
                    "alambana_id": "alambana_b",
                    "xiang_text": "这是有价值的提醒",
                    "supporting_grasping_ids": ["g2"],
                    "reason_summary": "帮助分支",
                }
            if "text" in props and "source" in props and "alambana_id" in props:
                content = str(messages)
                if "挑战" in content:
                    return {
                        "text": "这是挑衅",
                        "source": "推理",
                        "confidence": 0.6,
                        "alambana_id": "alambana_a",
                        "supporting_grasping_ids": ["g1"],
                        "reason_summary": "挑战想",
                    }
                return {
                    "text": "这是有价值的提醒",
                    "source": "推理",
                    "confidence": 0.55,
                    "alambana_id": "alambana_b",
                    "supporting_grasping_ids": ["g2"],
                    "reason_summary": "帮助想",
                }
            if "behavior_predictions" in props:
                return {
                    "behavior_predictions": [
                        {
                            "behavior_id": "act",
                            "text": "采取行动",
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
            # 评分：按内容区分嗔/无嗔倾向
            content = str(messages)
            if "无嗔" in content and "挑战" in content:
                return {"score": 0.1}
            if "嗔" in content and "挑战" in content:
                return {"score": 0.9}
            if "无嗔" in content and "帮助" in content:
                return {"score": 0.9}
            if "嗔" in content and "帮助" in content:
                return {"score": 0.1}
            return {"score": 0.5}

        result = model.evaluate_from_yuanwen(
            "员工在会议中指出方案存在风险。",
            MockLlmClient(handler=handler),
            update_seeds=False,
        )
        assert 2 <= len(result.cognition_branches) <= 3
        names_a = set(result.cognition_branches[0].mental_factor_scores_final)
        names_b = set(result.cognition_branches[1].mental_factor_scores_final)
        assert names_a == names_b
        assert len(names_a) == 40
        # 各分支独立持有分数字典（对治不得跨分支共享同一 dict）
        assert (
            result.cognition_branches[0].mental_factor_scores_final
            is not result.cognition_branches[1].mental_factor_scores_final
        )
        assert result.cognition_branches[0].alambana.text != (
            result.cognition_branches[1].alambana.text
        )

    def test_legacy_path_when_branching_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
        model = _load_model(tmp_path)
        llm = MockLlmClient(
            default_response={
                "score": 0.8,
                "has_explicit_xiang": False,
                "suoyuan_text": "mock所缘",
            }
        )
        result = model.evaluate_from_yuanwen(
            "员工在会议中指出方案存在风险。",
            llm,
            update_seeds=False,
        )
        assert result.cognition_branches is None
        assert result.extracted is not None
        assert result.extracted.scoring_path in ("explicit", "implicit")
