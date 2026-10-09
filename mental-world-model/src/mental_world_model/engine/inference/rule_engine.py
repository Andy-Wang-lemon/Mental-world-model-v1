"""推理引擎（占位，尚未实现）。"""

from mental_world_model.engine.inference.rule import Rule


class RuleEngine:
    """基于规则执行推理。"""

    def register_rule(self, rule):
        raise NotImplementedError("RuleEngine.register_rule 尚未实现")

    def evaluate(self, context):
        raise NotImplementedError("RuleEngine.evaluate 尚未实现")

    def infer(self, context):
        raise NotImplementedError("RuleEngine.infer 尚未实现")
