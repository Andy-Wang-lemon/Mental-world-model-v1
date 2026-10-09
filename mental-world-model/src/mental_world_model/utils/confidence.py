"""置信度规范化。"""


def as_unit_float(value: object, default: float = 0.0) -> float:
    """将任意输入安全转为 [0, 1]；非法类型回退 default。"""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            parsed = float(default)
        except (TypeError, ValueError):
            parsed = 0.0
    return max(0.0, min(1.0, parsed))


def clamp_unit_interval(value: float) -> float:
    """将数值限制在 [0, 1]。"""
    return as_unit_float(value, 0.0)
