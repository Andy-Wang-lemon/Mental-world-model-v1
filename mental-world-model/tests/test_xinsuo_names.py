"""测试心所名称规范化。"""

from mental_world_model.engine.normalizer.definition_normalizer import DefinitionNormalizer
from mental_world_model.utils.xinsuo_names import (
    normalize_relation_type,
    normalize_xinsuo_name,
)


class TestXinsuoNameNormalization:
    def test_normalize_aliases(self):
        normalizer = DefinitionNormalizer()
        assert normalizer.normalize_name("痴") == "无明"
        assert normalizer.normalize_name("无嗔") == "无瞋"

    def test_normalize_identity(self):
        assert normalize_xinsuo_name("信") == "信"
        assert normalize_xinsuo_name("无瞋") == "无瞋"

    def test_normalize_relation_type(self):
        assert normalize_relation_type("对治") == "对治"
        assert normalize_relation_type("障碍") == "对治"
