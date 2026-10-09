"""行动证据更新后验与业列绑定。"""

import pytest

from mental_world_model.engine.cognition.action_evidence import (
    ActionEvidence,
    ActionLikelihood,
    action_evidence_from_perception_split,
    action_skeleton_from_perception_split,
    bind_action_as_si,
    score_action_likelihoods_from_split,
    score_action_likelihoods_on_skeleton,
    update_alambana_posteriors_with_action,
)
from mental_world_model.engine.cognition.prompts import (
    build_action_likelihood_messages,
)
from mental_world_model.engine.perception.action_expand import (
    finalize_mixed_action_posteriors,
    maybe_expand_action_informed_branch,
    should_expand_alambana_from_action,
)
from mental_world_model.engine.perception.config import (
    COGNITION_PROMPT_VERSION,
    PERCEPTION_PROMPT_VERSION,
)
from mental_world_model.engine.perception.models import (
    ObjectiveEventStructure,
    PerceptionState,
    StimulusActionSplit,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import (
    build_grasping_pair_action_messages,
    build_stimulus_action_split_messages,
)


def _cand(cid: str, prior: float) -> SubjectiveAlambanaCandidate:
    return SubjectiveAlambanaCandidate(
        id=cid,
        text=f"所缘{cid}",
        prior_probability=prior,
        posterior_probability=prior,
        confidence=0.6,
    )


class TestActionEvidencePrompt:
    def test_l0_covers_inner_outer_and_faithful_description(self):
        system = build_stimulus_action_split_messages("他沉默不语。")[0]["content"]
        assert f"prompt_version={PERCEPTION_PROMPT_VERSION}" in system
        assert "内心造作" in system
        assert "外在" in system
        assert "明确无反应" in system or "未行动" in system
        assert "忠实改写" in system
        assert "动机" in system
        assert "情绪" in system
        assert "强度" in system

    def test_likelihood_prompt_requires_candidate_ids(self):
        msgs = build_action_likelihood_messages(
            "他沉默。",
            ObjectiveEventStructure(action="沉默"),
            [],
            [_cand("O1", 0.5), _cand("O2", 0.5)],
            ActionEvidence(
                has_observable_action=True,
                action_span="沉默",
                action_description="沉默",
            ),
            required_ids=["O1", "O2"],
            retry_missing_ids=["O2"],
        )
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        assert f"prompt_version={COGNITION_PROMPT_VERSION}" in system
        assert "必须覆盖的 alambana_id：['O1', 'O2']" in user
        assert "重试强提示" in user
        assert "O2" in user


class TestUpdatePosteriors:
    def test_no_action_keeps_prior(self):
        cands = [_cand("O1", 0.6), _cand("O2", 0.4)]
        notes: list[str] = []
        update_alambana_posteriors_with_action(
            cands,
            ActionEvidence(has_observable_action=False),
            notes,
        )
        assert cands[0].posterior_probability == 0.6
        assert cands[1].posterior_probability == 0.4
        assert any("无行动" in n for n in notes)

    def test_action_reweights_by_likelihood(self):
        # equal prior → a=0.5
        # O1 compat=0.8≥0.5 → h_up=0.75 → L=0.85
        # O2 compat=0.2<0.5 → h_down=0.375 → L=0.7
        # raw=0.425, 0.35 → posterior ≈ 0.548, 0.452
        cands = [_cand("O1", 0.5), _cand("O2", 0.5)]
        update_alambana_posteriors_with_action(
            cands,
            ActionEvidence(
                has_observable_action=True,
                action_span="沉默",
                likelihoods=[
                    ActionLikelihood("O1", 0.8),
                    ActionLikelihood("O2", 0.2),
                ],
            ),
        )
        assert abs(cands[0].action_compat - 0.8) < 1e-9
        assert abs(cands[1].action_compat - 0.2) < 1e-9
        assert abs(cands[0].action_likelihood - 0.85) < 1e-9
        assert abs(cands[1].action_likelihood - 0.7) < 1e-9
        assert abs(cands[0].posterior_probability - 0.425 / 0.775) < 1e-9
        assert abs(cands[1].posterior_probability - 0.35 / 0.775) < 1e-9
        assert cands[0].text == "所缘O1"  # 不改所缘正文

    def test_missing_likelihood_uses_neutral_one_not_half(self):
        cands = [_cand("O1", 0.5), _cand("O2", 0.5)]
        notes: list[str] = []
        update_alambana_posteriors_with_action(
            cands,
            ActionEvidence(
                has_observable_action=True,
                action_span="沉默",
                likelihoods=[ActionLikelihood("O1", 0.8)],
            ),
            notes,
        )
        # O2 compat=1 → L=1；O1 L=0.85
        assert abs(cands[0].action_compat - 0.8) < 1e-9
        assert abs(cands[0].action_likelihood - 0.85) < 1e-9
        assert abs(cands[1].action_likelihood - 1.0) < 1e-9
        assert any("非静默 0.5" in n for n in notes)
        assert abs(cands[0].posterior_probability - 0.425 / 0.925) < 1e-9
        assert abs(cands[1].posterior_probability - 0.5 / 0.925) < 1e-9

    def test_synthesize_asymmetric_gate(self):
        from mental_world_model.engine.cognition.action_evidence import (
            synthesize_action_likelihood,
        )

        # 矛盾侧：h_down；a=0 → h=0.25 → L=0.75（不再用满额打到 0.5）
        assert abs(synthesize_action_likelihood(0.0, 0.0) - 0.75) < 1e-9
        # 矛盾侧：a=1 → h_down=0.5 → L=0.6（旧式满额 h 会打到 0.2）
        assert abs(synthesize_action_likelihood(1.0, 0.2) - 0.6) < 1e-9
        # 支持侧：a=1 → h_up=1 → L=compat
        assert abs(synthesize_action_likelihood(1.0, 0.9) - 0.9) < 1e-9
        # 支持侧：a=0.5, compat=0.8 → h=0.75 → L=0.85
        assert abs(synthesize_action_likelihood(0.5, 0.8) - 0.85) < 1e-9

    def test_high_activation_low_compat_not_crushed(self):
        """高 prior + 低 compat：矛盾用 h_down，后验不被打穿。"""
        cands = [_cand("O1", 0.8), _cand("O2", 0.2)]
        update_alambana_posteriors_with_action(
            cands,
            ActionEvidence(
                has_observable_action=True,
                action_span="表面迎合",
                likelihoods=[
                    ActionLikelihood("O1", 0.1),  # 高激活但不贴行动
                    ActionLikelihood("O2", 0.9),
                ],
            ),
        )
        # a1=0.8 → h_down=0.45 → L1=1+0.45*(0.1-1)=0.595
        # a2=0.2 → h_up=0.6 → L2=1+0.6*(0.9-1)=0.94
        assert abs(cands[0].action_likelihood - 0.595) < 1e-9
        assert abs(cands[1].action_likelihood - 0.94) < 1e-9
        # O1 仍应保持明显优势（旧公式 L1≈0.28 会严重削弱）
        assert cands[0].posterior_probability > 0.55
        assert cands[0].posterior_probability > cands[1].posterior_probability


class TestBindActionAsSi:
    def test_empty_when_no_action(self):
        si = bind_action_as_si(ActionEvidence(has_observable_action=False), "O1")
        assert si.text == ""
        assert si.confidence == 0.0

    def test_summary_when_action(self):
        si = bind_action_as_si(
            ActionEvidence(
                has_observable_action=True,
                action_span="沉默",
                action_description="当事人沉默",
                confidence=0.9,
            ),
            "O1",
        )
        assert si.text == "当事人沉默"
        assert si.alambana_id == "O1"


class TestScoreLikelihoodGrounding:
    def test_semicolon_spans_partial_grounding(self):
        class FakeLlm:
            def complete(self, messages, schema=None):
                return {
                    "likelihoods": [{"alambana_id": "O1", "likelihood": 0.7}],
                    "reason_summary": "内心行动",
                }

        evidence, notes = score_action_likelihoods_from_split(
            "被老师说丢人，不想上课了。",
            ObjectiveEventStructure(action="说"),
            [],
            [_cand("O1", 0.5)],
            has_observable_action=True,
            action_span="不想上课了;编造的片段xyz",
            action_description="不想上课了",
            confidence=1.0,
            reason_summary="拆分",
            llm=FakeLlm(),
        )
        assert evidence.has_observable_action is True
        assert "不想上课了" in evidence.action_span
        assert "编造" not in evidence.action_span
        assert evidence.action_description == "不想上课了"

    def test_fully_ungrounded_span_keeps_action(self):
        class FakeLlm:
            def complete(self, messages, schema=None):
                return {
                    "likelihoods": [{"alambana_id": "O1", "likelihood": 0.6}],
                    "reason_summary": "保留",
                }

        evidence, notes = score_action_likelihoods_from_split(
            "被老师说丢人，不想上课了。",
            ObjectiveEventStructure(action="说"),
            [],
            [_cand("O1", 0.5)],
            has_observable_action=True,
            action_span="完全编造的片段xyz",
            action_description="完全编造的片段xyz",
            confidence=1.0,
            reason_summary="拆分",
            llm=FakeLlm(),
        )
        assert evidence.has_observable_action is True
        assert evidence.action_span == "完全编造的片段xyz"
        assert any("不作废" in n for n in notes)


class TestScoreLikelihoodIdStrict:
    def test_retries_then_neutral_fill_for_missing_ids(self):
        calls: list[str] = []

        class FakeLlm:
            def complete(self, messages, schema=None):
                user = messages[1]["content"]
                calls.append(user)
                if "重试强提示" not in user:
                    return {
                        "likelihoods": [
                            {"alambana_id": "O1", "likelihood": 0.9},
                            {"alambana_id": "WRONG", "likelihood": 0.1},
                        ],
                        "reason_summary": "first",
                    }
                return {
                    "likelihoods": [{"alambana_id": "O1", "likelihood": 0.85}],
                    "reason_summary": "retry still incomplete",
                }

        evidence, notes = score_action_likelihoods_from_split(
            "他沉默不语。",
            ObjectiveEventStructure(action="沉默"),
            [],
            [_cand("O1", 0.5), _cand("O2", 0.5)],
            has_observable_action=True,
            action_span="沉默不语",
            action_description="沉默不语",
            confidence=1.0,
            reason_summary="拆分",
            llm=FakeLlm(),
        )
        assert len(calls) == 2
        assert evidence.has_observable_action is True
        by_id = {x.alambana_id: x.likelihood for x in evidence.likelihoods}
        assert abs(by_id["O1"] - 0.85) < 1e-9
        assert abs(by_id["O2"] - 1.0) < 1e-9
        assert any("未知所缘 ID" in n for n in notes)
        assert any("重试后仍缺" in n for n in notes)
        assert any("非静默 0.5" in n for n in notes)


class TestNoFullTextFallback:
    def test_missing_split_raises(self):
        class FakeLlm:
            def complete(self, messages, schema=None):
                raise AssertionError("不应回退到全文抽取 LLM")

        with pytest.raises(ValueError, match="stimulus_action_split"):
            action_evidence_from_perception_split(
                "他沉默。",
                ObjectiveEventStructure(action="沉默"),
                [],
                [_cand("O1", 0.5)],
                None,
                FakeLlm(),
            )

    def test_skeleton_from_split_needs_no_llm(self):
        split = StimulusActionSplit(
            stimulus_text="老师说丢人",
            has_observable_action=True,
            action_span="不想上课了",
            action_description="不想上课了",
            confidence=1.0,
        )
        evidence, notes = action_skeleton_from_perception_split(
            "被老师说丢人，不想上课了。",
            split,
        )
        assert evidence.has_observable_action is True
        assert evidence.action_span == "不想上课了"
        assert evidence.likelihoods == []
        assert notes == [] or isinstance(notes, list)

    def test_score_on_skeleton_fills_likelihoods(self):
        class FakeLlm:
            def complete(self, messages, schema=None):
                return {
                    "likelihoods": [{"alambana_id": "O1", "likelihood": 0.8}],
                    "reason_summary": "ok",
                }

        skeleton = ActionEvidence(
            has_observable_action=True,
            action_span="不想上课了",
            action_description="不想上课了",
            confidence=1.0,
        )
        evidence, _ = score_action_likelihoods_on_skeleton(
            "被老师说丢人，不想上课了。",
            ObjectiveEventStructure(action="说"),
            [],
            [_cand("O1", 0.5)],
            skeleton,
            FakeLlm(),
        )
        assert evidence.likelihoods[0].likelihood == 0.8

    def test_with_split_scores_likelihood_only(self):
        class FakeLlm:
            def complete(self, messages, schema=None):
                return {
                    "likelihoods": [{"alambana_id": "O1", "likelihood": 0.75}],
                    "reason_summary": "ok",
                }

        split = StimulusActionSplit(
            stimulus_text="老师说丢人",
            has_observable_action=True,
            action_span="不想上课了",
            action_description="不想上课了",
            confidence=1.0,
        )
        evidence, _ = action_evidence_from_perception_split(
            "被老师说丢人，不想上课了。",
            ObjectiveEventStructure(action="说"),
            [],
            [_cand("O1", 0.5)],
            split,
            FakeLlm(),
        )
        assert evidence.has_observable_action is True
        assert evidence.action_span == "不想上课了"
        assert evidence.likelihoods[0].likelihood == 0.75


class TestActionExpandAlambana:
    def test_should_expand_when_max_compat_low(self):
        cands = [_cand("O1", 0.4), _cand("O2", 0.35), _cand("O3", 0.25)]
        evidence = ActionEvidence(
            has_observable_action=True,
            action_span="摔门离开",
            likelihoods=[
                ActionLikelihood("O1", 0.3),
                ActionLikelihood("O2", 0.25),
                ActionLikelihood("O3", 0.2),
            ],
        )
        assert should_expand_alambana_from_action(cands, evidence) is True

    def test_should_not_expand_when_compat_high(self):
        cands = [_cand("O1", 0.5), _cand("O2", 0.5)]
        evidence = ActionEvidence(
            has_observable_action=True,
            action_span="沉默",
            likelihoods=[
                ActionLikelihood("O1", 0.8),
                ActionLikelihood("O2", 0.2),
            ],
        )
        assert should_expand_alambana_from_action(cands, evidence) is False

    def test_should_not_expand_at_max_branches(self):
        cands = [_cand(f"O{i}", 0.25) for i in range(1, 5)]
        evidence = ActionEvidence(
            has_observable_action=True,
            action_span="摔门",
            likelihoods=[ActionLikelihood(c.id, 0.2) for c in cands],
        )
        assert should_expand_alambana_from_action(cands, evidence) is False

    def test_action_informed_skips_posterior_reweight(self):
        old = _cand("O1", 0.5)
        informed = _cand("O2", 0.5)
        informed.action_informed = True
        informed.action_likelihood = 1.0
        notes: list[str] = []
        update_alambana_posteriors_with_action(
            [old, informed],
            ActionEvidence(
                has_observable_action=True,
                action_span="摔门",
                likelihoods=[ActionLikelihood("O1", 0.2)],
            ),
            notes,
        )
        assert informed.action_likelihood == 1.0
        assert informed.posterior_probability == 0.5
        assert old.action_compat == 0.2
        assert any("action_informed" in n for n in notes)
        finalize_mixed_action_posteriors([old, informed], notes)
        assert abs(old.posterior_probability + informed.posterior_probability - 1.0) < 1e-9

    def test_expand_runs_perception_subpath(self):
        perception = PerceptionState(
            stimulus_text="他被批评后摔门离开。",
            objective_event=ObjectiveEventStructure(action="批评"),
            alambana_candidates=[
                _cand("O1", 0.4),
                _cand("O2", 0.35),
                _cand("O3", 0.25),
            ],
        )
        evidence = ActionEvidence(
            has_observable_action=True,
            action_span="摔门离开",
            action_description="摔门离开",
            likelihoods=[
                ActionLikelihood("O1", 0.3),
                ActionLikelihood("O2", 0.25),
                ActionLikelihood("O3", 0.2),
            ],
        )

        class FakeLlm:
            def complete(self, messages, schema=None):
                system = messages[0]["content"]
                if "行动解释向二执组合" in system:
                    return {
                        "items": [
                            {"combo_id": "C_act1", "activation_alignment": 0.9}
                        ]
                    }
                if "行动解释向」我执" in system:
                    return {
                        "self_grasping": [
                            {
                                "id": "sg_act1",
                                "text": "把自己当成被公开羞辱的人",
                                "confidence": 0.8,
                            }
                        ],
                        "dharma_grasping": [
                            {
                                "id": "dg_act1",
                                "text": "把批评当成人身攻击",
                                "confidence": 0.8,
                            }
                        ],
                    }
                if "主观所缘境渲染器" in system:
                    return {
                        "candidates": [
                            {
                                "id": "O_C_act1",
                                "combo_id": "C_act1",
                                "text": "把当众批评理解成对自己人格的否定",
                            }
                        ]
                    }
                return {
                    "items": [{"pair_id": "p0", "same_perspective": False}]
                }

        notes: list[str] = []
        expanded = maybe_expand_action_informed_branch(
            "他被批评后摔门离开。",
            perception,
            evidence,
            evidence,
            FakeLlm(),
            notes,
        )
        assert expanded is True
        assert len(perception.alambana_candidates) == 4
        new_cand = perception.alambana_candidates[-1]
        assert new_cand.action_informed is True
        assert new_cand.action_likelihood == 1.0
        assert any(h.id.startswith("sg_act") for h in perception.self_grasping_hypotheses)
        update_alambana_posteriors_with_action(
            perception.alambana_candidates, evidence, notes
        )
        assert new_cand.action_informed is True
        assert new_cand.action_likelihood == 1.0
        finalize_mixed_action_posteriors(perception.alambana_candidates, notes)
        assert abs(sum(c.posterior_probability for c in perception.alambana_candidates) - 1.0) < 1e-9
        assert any("跳过后验" in n or "action_informed" in n for n in notes)

    def test_expand_aborts_when_grounding_fails(self):
        perception = PerceptionState(
            stimulus_text="他被批评后摔门离开。",
            objective_event=ObjectiveEventStructure(action="批评"),
            alambana_candidates=[
                _cand("O1", 0.4),
                _cand("O2", 0.35),
                _cand("O3", 0.25),
            ],
        )
        evidence = ActionEvidence(
            has_observable_action=True,
            action_span="摔门离开",
            action_description="摔门离开",
            likelihoods=[
                ActionLikelihood("O1", 0.3),
                ActionLikelihood("O2", 0.25),
                ActionLikelihood("O3", 0.2),
            ],
        )

        class FakeLlm:
            def complete(self, messages, schema=None):
                system = messages[0]["content"]
                if "grounding 校验器" in system:
                    return {
                        "items": [
                            {
                                "alambana_id": f"O{i}",
                                "grounded": False,
                                "hallucinated_facts": True,
                                "preserves_critical_event": False,
                                "reason": "hallucinated",
                            }
                            for i in range(1, 12)
                        ]
                    }
                if "行动解释向二执组合" in system:
                    return {
                        "items": [
                            {"combo_id": "C_act1", "activation_alignment": 0.9}
                        ]
                    }
                if "行动解释向」我执" in system:
                    return {
                        "self_grasping": [
                            {
                                "id": "sg_act1",
                                "text": "把自己当成被公开羞辱的人",
                                "confidence": 0.8,
                            }
                        ],
                        "dharma_grasping": [
                            {
                                "id": "dg_act1",
                                "text": "把批评当成人身攻击",
                                "confidence": 0.8,
                            }
                        ],
                    }
                if "主观所缘境渲染器" in system or "主观所缘重写器" in system:
                    return {
                        "candidates": [
                            {
                                "id": "O_C_act1",
                                "combo_id": "C_act1",
                                "text": "把当众批评理解成对自己人格的否定",
                            }
                        ]
                    }
                return {
                    "items": [{"pair_id": "p0", "same_perspective": False}]
                }

        notes: list[str] = []
        expanded = maybe_expand_action_informed_branch(
            "他被批评后摔门离开。",
            perception,
            evidence,
            evidence,
            FakeLlm(),
            notes,
        )
        assert expanded is False
        assert len(perception.alambana_candidates) == 3
        assert any("grounding" in n for n in notes)

    def test_action_grasping_prompt_mentions_existing_alambana(self):
        msgs = build_grasping_pair_action_messages(
            "他摔门离开。",
            ObjectiveEventStructure(action="摔门"),
            action_span="摔门离开",
            action_description="摔门离开",
            existing_alambana_summaries=[{"id": "O1", "text": "所缘O1"}],
        )
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        assert "已有主观所缘" in user
        assert "行动证据" in user
        assert "我执" in system
        assert "与自己分不开" in system
        assert "不能互相替代" in system

