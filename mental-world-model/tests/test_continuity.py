"""连续事件：认知 H 与心理 M。"""

from mental_world_model.engine.continuity.cognitive import (
    build_belief_items_from_branches,
    frames_from_final_branches,
    merge_cognitive_frames,
    update_cognitive_bias,
)
from mental_world_model.engine.continuity.mental import (
    blend_mental_scores,
    compute_lambda_eff,
    weighted_average_mental_history,
)
from mental_world_model.engine.continuity.models import BiasFrame, MentalEpisode
from mental_world_model.engine.cognition.models import (
    CognitionBranch,
    ExtractedMentalState,
)
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    SubjectiveAlambanaCandidate,
)


class TestMentalWeightedAverage:
    def test_history_weights_are_theta_only(self):
        episodes = [
            MentalEpisode(
                event_id="e1",
                theta=0.9,
                timestamp="2026-07-01T00:00:00+00:00",
                scores={"嗔": 0.7, "喜": 0.1},
            ),
            MentalEpisode(
                event_id="e2",
                theta=0.2,
                timestamp="2026-07-29T00:00:00+00:00",
                scores={"嗔": 0.05, "喜": 0.6},
            ),
        ]
        hist, weights, sum_w = weighted_average_mental_history(
            episodes,
            factor_names=["嗔", "喜"],
        )
        assert abs(sum_w - 1.1) < 1e-9
        assert abs(weights[0] - 0.9) < 1e-9
        assert abs(weights[1] - 0.2) < 1e-9
        assert abs(hist["嗔"] - (0.9 * 0.7 + 0.2 * 0.05) / 1.1) < 1e-9
        assert weights[0] > weights[1]
        assert hist["嗔"] > 0.4

    def test_blend_and_lambda(self):
        lam = compute_lambda_eff(0.5)
        assert abs(lam - 0.15) < 1e-9
        assert abs(compute_lambda_eff(1.0) - 0.30) < 1e-9
        final = blend_mental_scores({"嗔": 0.05}, {"嗔": 0.7}, 0.2)
        assert abs(final["嗔"] - (0.8 * 0.05 + 0.2 * 0.7)) < 1e-9


class TestCognitiveFrames:
    def test_reinforce_similar_frames(self):
        existing = [
            BiasFrame(id="bf_1", text="组织资源正向新人倾斜", mass=0.5, support=1)
        ]
        incoming = [
            BiasFrame(id="n1", text="组织资源正向新人倾斜，位置受威胁", mass=0.6)
        ]
        merged = merge_cognitive_frames(existing, incoming)
        assert len(merged) == 1
        assert merged[0].mass > 0.5
        assert merged[0].support == 2

    def test_llm_rejects_rule_near_frames(self):
        from mental_world_model.llm.mock_client import MockLlmClient

        existing = [
            BiasFrame(
                id="bf_1",
                text="我认为自己很有能力完成任务",
                mass=0.5,
                support=1,
            )
        ]
        incoming = [
            BiasFrame(
                id="n1",
                text="我不认为自己很有能力完成任务",
                mass=0.6,
            )
        ]
        llm = MockLlmClient(
            default_response={
                "semantic_confirm_items": [
                    {"pair_id": "p0", "same_perspective": False}
                ]
            }
        )
        merged = merge_cognitive_frames(existing, incoming, llm=llm)
        assert len(merged) == 2

    def test_new_frame_keeps_own_mass(self):
        existing = [
            BiasFrame(id="bf_1", text="旧解释AAAAAAAA", mass=0.8, support=1)
        ]
        incoming = [
            BiasFrame(id="n1", text="完全不同的新解释BBBBBB", mass=0.7)
        ]
        merged = merge_cognitive_frames(existing, incoming)
        assert len(merged) == 2
        new_one = next(f for f in merged if "新解释" in f.text)
        assert abs(new_one.mass - 0.7) < 1e-9

    def test_update_does_not_decay_old_mass(self):
        existing = [
            BiasFrame(id="bf_1", text="完全无关的旧偏置ABCDEF", mass=0.2, support=1)
        ]
        incoming = [
            BiasFrame(id="n1", text="评价标准正在改变", mass=0.7)
        ]
        updated = update_cognitive_bias(existing, incoming)
        old = next(f for f in updated if "旧偏置" in f.text)
        assert abs(old.mass - 0.2) < 1e-9
        assert any("评价标准" in f.text for f in updated)

    def test_merge_keeps_low_mass_beyond_eight(self):
        seeds = [
            "组织资源正向新人倾斜ABCDEFGH",
            "旧方法必须用数据才能被否定IJKLMNOP",
            "公开认错是我仍有资格的证明QRSTUVWX",
            "帮助他人只是职责不是身份YZABCDEF",
            "年报点名过时等于宣判我这个人GHIJKLMN",
            "天气很好适合出门散步OPQRSTUV",
            "晚饭想吃面条不想吃米饭WXYZABCD",
            "这台电脑风扇声音有点大EFGHIJKL",
        ]
        existing = [
            BiasFrame(id=f"bf_{i}", text=text, mass=0.9)
            for i, text in enumerate(seeds, start=1)
        ]
        incoming = [
            BiasFrame(id="n1", text="搬家时把钥匙锁在门外完全另一件事MNOPQRST", mass=0.12)
        ]
        merged = merge_cognitive_frames(existing, incoming)
        assert len(merged) == 9
        assert any("钥匙" in f.text and abs(f.mass - 0.12) < 1e-9 for f in merged)

    def test_frames_from_final_branches_keep_low_posterior(self):
        items = [
            {
                "id": "O1",
                "alambana": "感到被忽视、话语权减少",
                "self_grasping": ["位置是我的"],
                "dharma_grasping": ["人事调整是针对我"],
                "posterior": 0.31,
            },
            {
                "id": "O2",
                "alambana": "要更系统化地改进工作方式",
                "self_grasping": ["我应不断改进"],
                "dharma_grasping": [],
                "posterior": 0.69,
            },
        ]
        frames = frames_from_final_branches(items)
        assert len(frames) == 2
        low = next(f for f in frames if "被忽视" in f.text)
        assert abs(low.mass - 0.31) < 1e-9
        assert "我执：" not in low.text
        assert "位置是我的" in low.text
        assert "从此" in low.text

    def test_injection_uses_mass_not_scene_relevance(self):
        from mental_world_model.engine.continuity.models import ContinuityState
        from mental_world_model.engine.continuity.service import bias_frames_for_injection

        state = ContinuityState(
            cognitive_frames=[
                BiasFrame(id="bf_1", text="要不断系统化改进工作方式ABCDEFGH", mass=0.92),
                BiasFrame(id="bf_2", text="感到被忽视、话语权被拿走、位置不再属于我", mass=0.31),
                BiasFrame(id="bf_3", text="天气很好适合出门散步XYZXYZXYZ", mass=0.8),
            ]
        )
        chosen = bias_frames_for_injection(state)
        assert chosen[0]["id"] == "bf_1"
        assert [item["id"] for item in chosen[:3]] == ["bf_1", "bf_3", "bf_2"]

    def test_belief_items_use_grasping_alambana_posterior_xiang_no_activation(self):
        cand = SubjectiveAlambanaCandidate(
            id="O1",
            text="感到被质疑资格",
            prior_probability=0.6,
            posterior_probability=0.75,
            used_self_ids=["sg1"],
            used_dharma_ids=["dg1"],
        )
        branch = CognitionBranch(
            id="branch_O1",
            alambana=cand,
            branch_prior_probability=0.6,
            activation_probability=0.6,
            xiang=ExtractedMentalState(
                text="这是在否定我", source="推理", confidence=0.7, alambana_id="O1"
            ),
        )
        self_g = [
            GraspingHypothesis(id="sg1", type="self_grasping", text="把资格当自我")
        ]
        dharma_g = [
            GraspingHypothesis(
                id="dg1", type="dharma_grasping", text="把异议当否定"
            )
        ]
        items = build_belief_items_from_branches(
            [cand],
            [branch],
            self_grasping=self_g,
            dharma_grasping=dharma_g,
        )
        assert len(items) == 1
        item = items[0]
        assert item["id"] == "O1"
        assert item["alambana"] == "感到被质疑资格"
        assert item["posterior"] == 0.75
        assert item["xiang"] == "这是在否定我"
        assert item["self_grasping"] == ["把资格当自我"]
        assert item["dharma_grasping"] == ["把异议当否定"]
        assert "activation" not in item
        assert "text" not in item  # 旧字段，已改为 alambana
        frames = frames_from_final_branches(items)
        assert len(frames) == 1
        assert "感到被质疑资格" in frames[0].text
        assert "从此" in frames[0].text
        assert "把资格当自我" in frames[0].text
        assert "这是在否定我" not in frames[0].text
        assert abs(frames[0].mass - 0.75) < 1e-9


class TestContinuityIntegration:
    def test_two_events_update_h_and_m(self, tmp_path):
        from pathlib import Path

        from mental_world_model.llm.mock_client import MockLlmClient
        from mental_world_model.model import MentalWorldModel

        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(
            root / "definitions",
            seed_state_path=tmp_path / "seed_state.json",
        )
        llm = MockLlmClient(
            default_response={
                "score": 0.7,
                "theta": 0.8,
                "relevance": 0.5,
            }
        )
        r1 = model.evaluate_from_yuanwen(
            "CEO在会上公开表扬新人林可。",
            llm,
            update_seeds=False,
        )
        assert r1.continuity is not None
        assert len(model.continuity_state.cognitive_frames) >= 1
        assert len(model.continuity_state.mental_episodes) == 1
        # 首事件无历史：期望分 = 最终分
        assert r1.scores_final == r1.expected_mental_factor_scores

        r2 = model.evaluate_from_yuanwen(
            "三天后，林可向陈默请教方案细节。陈默沉默。",
            llm,
            update_seeds=False,
        )
        assert len(model.continuity_state.mental_episodes) == 2
        assert r2.continuity is not None
        mental = r2.continuity.get("mental") or {}
        assert mental.get("episode_count", 0) >= 1
        assert abs(float(mental.get("relevance") or 0.0) - 0.5) < 1e-9
        assert r2.mental_factors_event is not None
        assert r2.expected_mental_factor_scores == r2.mental_factors_event
        assert "continuity" in (r2.build_pipeline_trace(slim=True) or {})
        assert (tmp_path / "continuity_state.json").is_file()
