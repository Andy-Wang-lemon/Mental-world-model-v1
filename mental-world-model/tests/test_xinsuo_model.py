"""测试心所核心模型。"""

from mental_world_model.core.xinsuo.model import Xinsuo


class TestXinsuoModel:
    def test_import(self):
        assert Xinsuo is not None

    def test_instantiate(self):
        obj = Xinsuo(
            name="信",
            suoyuan="所缘是……",
            shengqi_nian="心中出现……",
            ye="使人乐于亲近善法……",
            xiandai_xinli="对真实价值……",
        )
        assert obj.name == "信"

    def test_from_dict(self):
        data = {
            "name": "精进",
            "suoyuan": "所缘是具体应修的善法……",
            "shengqi_nian": "心中生起主动承担……",
            "ye": "使善法圆满完成……",
            "xiandai_xinli": "面向有价值目标的持续行动力……",
        }
        obj = Xinsuo.from_dict(data)
        assert obj.name == "精进"
        assert obj.to_dict() == data
