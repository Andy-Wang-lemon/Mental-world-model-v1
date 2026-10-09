"""种子库运行时与提取路由参数。"""

# 种子库总开关：False 时关闭种子库、H/Top-6 路由、想/思心所提示与 EMA 更新（代码保留）
SEED_LIBRARY_ENABLED = False


def is_seed_library_enabled() -> bool:
    """运行时读取种子库总开关（勿在其他模块 import 快照）。"""
    return SEED_LIBRARY_ENABLED

# H / Top-6 路由（仅 SEED_LIBRARY_ENABLED=True 时执行）
# combined = α·relevance + (1-α)·W_seed
ROUTER_RELEVANCE_ALPHA = 0.75
ROUTER_TOP_K = 6
ROUTER_MIN_RELEVANCE = 0.35

# 交互后种子 EMA 更新：W_new = (1-α)·W_old + α·score_final
SEED_EMA_ALPHA = 0.3
SEED_ZERO_DECAY = 0.05

# 想/思推理置信度
INFERENCE_CONFIDENCE_DEFAULT = 0.6
# 推理来源置信度上限（护栏：无原文依据时不允许 confidence=1）
INFERENCE_CONFIDENCE_MAX = 0.8

HINT_FIELD_MAX_CHARS = 100

DEFAULT_SEED_STATE_FILENAME = "seed_state.json"
