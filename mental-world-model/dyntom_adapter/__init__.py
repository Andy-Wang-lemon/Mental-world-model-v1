"""DynToM 评测适配层。

只调用 MentalWorldModel 的公开接口，不修改模型内部。
建模阶段只读人物信息、可选关系、以及逐幕 background / dialogue。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
