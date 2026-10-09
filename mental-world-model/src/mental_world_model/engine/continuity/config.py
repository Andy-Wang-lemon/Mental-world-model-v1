"""连续事件：认知偏置 H 与心理惯性 M 配置。"""

# 总开关（可分别关）
COGNITIVE_CONTINUITY_ENABLED = True
MENTAL_CONTINUITY_ENABLED = True

# ---- 认知 H（沉淀偏置；入库按 mass；作用域决定本场资格）----
COGNITIVE_TOP_K_INJECT = 5
COGNITIVE_REINFORCE_ALPHA = 0.35
H_MODULATION_GAIN = 0.35
H_SCOPE_LEXICAL_FLOOR = 0.15

# ---- 心理 M（多事件按 θ 加权；λ = λ0 × R）----
# 与 seed EMA 互斥概念：心理惯性请用本模块；SEED_LIBRARY 默认关闭。
MENTAL_MAX_EPISODES = 20
MENTAL_LAMBDA0 = 0.30
MENTAL_LAMBDA_MAX = 0.50

DEFAULT_CONTINUITY_STATE_FILENAME = "continuity_state.json"


def is_cognitive_continuity_enabled() -> bool:
    return COGNITIVE_CONTINUITY_ENABLED


def is_mental_continuity_enabled() -> bool:
    return MENTAL_CONTINUITY_ENABLED
