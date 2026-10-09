"""分析对象 focus_person 解析与提示注入。"""

from mental_world_model.engine.perception.focus import (
    focus_person_instruction,
    parse_focus_person_from_text,
    resolve_focus_person,
)
from mental_world_model.engine.perception.prompts import (
    build_grasping_pair_messages,
    build_stimulus_action_split_messages,
)
from mental_world_model.engine.perception.models import ObjectiveEventStructure


class TestParseFocusPerson:
    def test_parse_分析对象_line(self):
        text = "分析对象：陈默\n\n事件1：林可提出方案。"
        assert parse_focus_person_from_text(text) == "陈默"

    def test_parse_with_extra_punctuation(self):
        text = "建模对象: 林可（主角）\n事件1：开会。"
        assert parse_focus_person_from_text(text) == "林可"

    def test_parse_missing(self):
        assert parse_focus_person_from_text("事件1：开会。") == ""

    def test_explicit_overrides_text(self):
        text = "分析对象：林可\n事件1：开会。"
        assert resolve_focus_person("陈默", text) == "陈默"
        assert resolve_focus_person(None, text) == "林可"
        assert resolve_focus_person("", text) == "林可"


class TestFocusPromptInjection:
    def test_instruction_mentions_name(self):
        block = focus_person_instruction("陈默")
        assert "陈默" in block
        assert "唯一建模对象" in block
        assert focus_person_instruction("") == ""

    def test_grasping_prompt_locks_person(self):
        msgs = build_grasping_pair_messages(
            "林可提出方案，陈默沉默。",
            ObjectiveEventStructure(actors=["林可", "陈默"], action="提出"),
            focus_person="陈默",
        )
        joined = "\n".join(m["content"] for m in msgs)
        assert "分析对象锁定" in joined
        assert "陈默" in joined
        assert "禁止改写为其他人物" in joined

    def test_split_prompt_locks_person(self):
        msgs = build_stimulus_action_split_messages(
            "林可提出方案。陈默要求数据。",
            focus_person="陈默",
        )
        joined = "\n".join(m["content"] for m in msgs)
        assert "分析对象（必须遵守）：陈默" in joined
