"""测试推理引擎相关类的导入、实例化与方法存在性。"""

from mental_world_model.engine.inference.rule import Rule
from mental_world_model.engine.inference.rule_engine import RuleEngine


class TestRule:
    def test_import(self):
        assert Rule is not None

    def test_instantiate(self):
        rule = Rule()
        assert rule is not None


class TestRuleEngine:
    def test_import(self):
        assert RuleEngine is not None

    def test_instantiate(self):
        engine = RuleEngine()
        assert engine is not None

    def test_methods_exist(self):
        engine = RuleEngine()
        for method_name in ["register_rule", "evaluate", "infer"]:
            assert hasattr(engine, method_name)
            assert callable(getattr(engine, method_name))
