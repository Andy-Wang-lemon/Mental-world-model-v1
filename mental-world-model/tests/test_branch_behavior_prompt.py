"""分支行为预测提示：心所业注入与单支输入。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.cognition.prompts import build_branch_behavior_messages
from mental_world_model.engine.perception.config import COGNITION_PROMPT_VERSION
from mental_world_model.engine.perception.models import SubjectiveAlambanaCandidate


class TestBranchBehaviorPrompt:
    def test_includes_ye_and_fluent_context(self):
        xinsuo = Xinsuo(
            name="懈怠",
            suoyuan="s",
            shengqi_nian="n",
            ye="倾向于拖延、回避应做之事",
            xiandai_xinli="x",
        )
        system, user = build_branch_behavior_messages(
            "同学叫我去上课。",
            SubjectiveAlambanaCandidate(
                id="O1",
                text="课堂被经验为丢人现场",
                posterior_probability=0.4,
                confidence=0.8,
            ),
            "既不想挪步，又觉得会被推着走",
            "内心造作：回避上课",
            [("懈怠", 0.7, xinsuo)],
            branch_id="branch_O1",
        )
        assert f"prompt_version={COGNITION_PROMPT_VERSION}" in system["content"]
        assert "业：" in user["content"]
        assert "倾向于拖延、回避应做之事" in user["content"]
        assert "想：既不想挪步" in user["content"]
        assert "思：内心造作" in user["content"]
        assert "主观所缘（branch_O1）" in user["content"]
        # 单支隔离靠调用方只传本支；提示不强加「禁止其他分支」说教
        assert "其他分支" not in system["content"]
        assert "为主" not in system["content"]
        assert "次要" not in system["content"]
        assert "行为由所缘、想、思和心所共同约束，但不是把想翻译成动作" in system["content"]
        assert "对当前取境的回应" in system["content"]
        assert "不等于直接执行想" in system["content"]
        assert "也可以只实现一部分、延后实现或暂不实现" in system["content"]
        assert "不要只写最直接完成想的那条" in system["content"]
        assert "不要编造刺激里没有的人物或事实" in system["content"]
        assert "若原文已有动作" not in system["content"]
        assert "编造无关情节" not in system["content"]
        assert "最多 3" in user["content"] or "2～3" in user["content"]
        assert "禁止在 text 里写心所名" in system["content"]
        assert "可能会" in system["content"]
