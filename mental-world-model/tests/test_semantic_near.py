"""语义近邻召回门闸与二执 LLM 确认。"""

from unittest.mock import MagicMock

from mental_world_model.engine.perception.combo_selection import select_grasping_combinations
from mental_world_model.engine.perception.models import GraspingHypothesis
from mental_world_model.engine.perception.prompts import SEMANTIC_CONFIRM_SCHEMA
from mental_world_model.engine.perception.semantic_confirm import (
    PerspectiveNearCache,
    confirm_perspective_pairs,
    perspectives_same,
)
from mental_world_model.engine.perception.validators import (
    text_fingerprint,
    texts_semantically_near,
)
from mental_world_model.llm.mock_client import MockLlmClient


class TestTextsSemanticallyNearRecall:
    def test_exact_match(self):
        assert texts_semantically_near("把资格当成我", "把资格当成我")

    def test_short_substring_not_near(self):
        # 短边过短：不召回（默认 min_len=4）
        assert not texts_semantically_near("沉默", "保持沉默")

    def test_moderate_overlap_can_recall(self):
        # 放宽 Jaccard 后，中等字面重叠也可进召回（再交 LLM）
        a = "把别人的否定理解成对体面的威胁"
        b = "把公开否定理解成对体面的打击"
        assert texts_semantically_near(a, b)

    def test_long_substring_still_near(self):
        assert texts_semantically_near(
            "把领导资格当成我的核心",
            "在会议上把领导资格当成我的核心身份",
        )

    def test_symmetric_jaccard_recall(self):
        # 对称 Jaccard 弱门闸：字面高度重叠应召回
        a = "我认为自己很有能力处理这件事"
        b = "我认为自己很有能力处理这些事"
        assert texts_semantically_near(a, b)

    def test_fingerprint_stable(self):
        a = text_fingerprint("资格 受威胁")
        b = text_fingerprint("资格受威胁")
        assert a == b
        assert a


class TestPerspectivesSameGate:
    def test_rule_fail_skips_llm(self):
        llm = MagicMock()
        assert not perspectives_same(
            "能力受质疑",
            "关系被疏远了很久",
            llm=llm,
        )
        llm.complete.assert_not_called()

    def test_rule_pass_without_llm_keeps_near(self):
        assert perspectives_same(
            "把领导资格当成我的核心",
            "在会议上把领导资格当成我的核心身份",
            llm=None,
        )

    def test_rule_pass_llm_false_rejects(self):
        llm = MockLlmClient(
            default_response={
                "semantic_confirm_items": [
                        {
                            "pair_id": "p0",
                            "same_perspective": False,
                        }
                ]
            }
        )
        # 字面召回会过（高度重叠），LLM 否定
        a = "我认为自己很有能力完成任务"
        b = "我不认为自己很有能力完成任务"
        assert texts_semantically_near(a, b)
        assert not perspectives_same(a, b, llm=llm)

    def test_rule_pass_llm_true_accepts(self):
        llm = MockLlmClient()
        a = "把领导资格当成我的核心"
        b = "在会议上把领导资格当成我的核心身份"
        assert perspectives_same(a, b, llm=llm)

    def test_batch_confirm_failure_defaults_false(self):
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("boom")
        out = confirm_perspective_pairs(
            [("p1", "把资格当成自我身份的核心", "把资格当成自我身份的核心点")],
            llm,
        )
        assert out == {"p1": False}

    def test_batches_all_pairs_beyond_chunk(self):
        llm = MagicMock()

        def complete(messages, schema=None):
            blob = " ".join(str(m.get("content", "")) for m in messages)
            items = []
            for i in range(13):
                pid = f"p{i}"
                if f"'pair_id': '{pid}'" in blob or f'"pair_id": "{pid}"' in blob:
                    items.append(
                        {"pair_id": pid, "same_perspective": i < 12}
                    )
            return {"items": items}

        llm.complete.side_effect = complete
        base = "把领导资格当成我的核心身份标记"
        pairs = [(f"p{i}", base, base + str(i)) for i in range(13)]
        out = confirm_perspective_pairs(pairs, llm, max_pairs=12)
        assert llm.complete.call_count == 2
        assert len(out) == 13
        assert out["p0"] is True
        assert out["p12"] is False

    def test_cache_avoids_second_llm(self):
        llm = MockLlmClient()
        cache = PerspectiveNearCache()
        a = "把领导资格当成我的核心身份标记"
        b = "把领导资格当成我的核心身份记号"
        assert perspectives_same(a, b, llm=llm, cache=cache)
        # 第二次应走缓存；Mock 无 call 计数时用 MagicMock 包一层不方便，
        # 直接验证 cache 已写入且无 llm 也能命中
        assert cache.get(a, b) is True
        assert perspectives_same(a, b, llm=None, cache=cache)


class TestComboDedupWithLlmConfirm:
    def test_antonym_not_merged_when_llm_rejects(self):
        self_g = [
            GraspingHypothesis(
                id="s1",
                type="self_grasping",
                text="我认为自己很有能力完成任务",
                confidence=0.7,
            ),
            GraspingHypothesis(
                id="s2",
                type="self_grasping",
                text="我不认为自己很有能力完成任务",
                confidence=0.65,
            ),
        ]
        dharma_g = [
            GraspingHypothesis(
                id="d1",
                type="dharma_grasping",
                text="把求助理解成协作澄清请求",
                confidence=0.6,
            )
        ]
        llm = MockLlmClient(
            default_response={
                "semantic_confirm_items": [
                        {
                            "pair_id": "p0",
                            "same_perspective": False,
                        }
                ]
            }
        )
        selected = select_grasping_combinations(self_g, dharma_g, llm=llm)
        used_self = {sid for c in selected for sid in c.used_self_ids}
        # LLM 否认同视角 → 两侧我执都可进代表，进而可能多组合
        assert "s1" in used_self
        assert "s2" in used_self

    def test_schema_exported(self):
        assert "same_perspective" in str(SEMANTIC_CONFIRM_SCHEMA)
