"""行为预测文本清洗：由 LLM 判断，不按词表切句。"""

from mental_world_model.engine.cognition.behavior import sanitize_behavior_text
from mental_world_model.llm.mock_client import MockLlmClient


class TestSanitizeBehaviorText:
    def test_strips_karma_and_modals(self):
        raw = (
            "陈默在帮助林可解决项目困难时，可能会主动提出解决方案或提供支持，"
            "这符合他内心的无瞋业，即促成善行。他可能通过调整工作方式，"
            "帮助林可找到更好的解决办法。"
        )
        cleaned = sanitize_behavior_text(raw, llm=MockLlmClient())
        assert "无瞋业" not in cleaned
        assert "即促成善行" not in cleaned
        assert "内心" not in cleaned
        assert "可能会" not in cleaned
        assert "主动提出解决方案或提供支持" in cleaned
        assert "帮助林可" in cleaned

    def test_drops_pure_jargon_keeps_original_when_cleaned_empty(self):
        raw = "这符合他内心的疑业，即促成善行。"
        assert sanitize_behavior_text(raw, llm=MockLlmClient()) == raw

    def test_llm_empty_cleaned_keeps_original(self):
        class _Empty:
            def complete(self, messages, schema=None):
                return {"items": [{"index": 0, "cleaned_text": ""}]}

        raw = "陈默当场点头答应。"
        assert sanitize_behavior_text(raw, llm=_Empty()) == raw

    def test_inner_mixed_action_falls_back_to_stripped(self):
        cleaned = sanitize_behavior_text(
            "陈默内心决定当场拒绝继续帮忙。", llm=MockLlmClient()
        )
        assert "内心" not in cleaned
        assert "拒绝继续帮忙" in cleaned

    def test_keeps_plain_action(self):
        assert "追问陈默" in sanitize_behavior_text(
            "林可当众追问陈默为何未署名。", llm=MockLlmClient()
        )

    def test_without_llm_does_not_keyword_cut(self):
        raw = "林可不但当众追问陈默为何私下补数据，还把旧账翻了出来。"
        assert sanitize_behavior_text(raw) == raw
