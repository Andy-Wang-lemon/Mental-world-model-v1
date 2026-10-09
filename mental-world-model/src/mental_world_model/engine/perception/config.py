"""认知分支与前置感知层配置。"""

# 总开关：True 时 evaluate_from_yuanwen 走多分支认知路径
COGNITION_BRANCHING_ENABLED = True

# 所缘分支数：过阈值才开，最多 3，最少可为 1（不过阈值则只留最高 1 支）
MAX_ALAMBANA_BRANCHES = 3
ALAMBANA_MIN_PROBABILITY = 0.08
ALAMBANA_MIN_CONFIDENCE = 0.20

# 本地加权 Softmax（仅无组合先验的兼容路径 / 诊断 raw_score）
# Score = 0.4·Event + 0.3·Self + 0.3·World
ALAMBANA_SCORE_W_EVENT = 0.4
ALAMBANA_SCORE_W_SELF = 0.3
ALAMBANA_SCORE_W_WORLD = 0.3
ALAMBANA_SOFTMAX_TEMPERATURE = 1.0

# 执组合选择：confidence 门槛、次视角相对主组合强度、id Jaccard 去重上限
GRASPING_COMBO_ACTIVATION = 0.25
GRASPING_COMBO_SECONDARY_RATIO = 0.5
COMBO_SECONDARY_RATIO = GRASPING_COMBO_SECONDARY_RATIO
COMBO_ID_JACCARD_MAX = 0.5
COMBO_MAX_SELF_REPS = 3
COMBO_MAX_DHARMA_REPS = 3
# 主路径：所缘 prior = 组合激活 P(C)（不再与 Softmax 融合）
# 保留常量以免旧脚本/文档引用报错；已不参与计算
COMBO_PRIOR_BLEND_ALPHA = 0.7

# 文本近邻召回门闸（弱规则）：过闸后可再 LLM 确认；不过闸不调 LLM
# 宜偏松：宁多召回，由 LLM 否决反义/不同轴；对称字符 Jaccard；短边包含须 ≥ min_len
SEMANTIC_RECALL_MIN_LEN = 4
SEMANTIC_RECALL_JACCARD_MIN = 0.55
# 二执 / H 合并等：规则过闸后每批最多确认的 pair 数（控单次成本；多批问完，不截断）
SEMANTIC_CONFIRM_MAX_PAIRS = 12

# 所缘结构校验失败后最多额外重抽次数（不含首次）
ALAMBANA_VALIDATION_MAX_RETRIES = 1

# L2.5：alignment / 原始分硬门闸；未过闸的 combo 不参与归一化
MIN_ALIGNMENT = 0.15
MIN_COMBO_RAW_SCORE = 0.05
# 有效 combo 的最高 alignment 仍低于此则触发 accommodation
ACCOMMODATION_MIN_FIT = 0.25
ACCOMMODATION_MAX_RETRIES = 1

# 行为：先生成可区分动作再估概率；diversity 只补一次
BEHAVIOR_MIN_DISTINCT = 2
BEHAVIOR_DIVERSITY_RETRIES = 1

MENTAL_FACTOR_PRESENT_THRESHOLD = 0.30

BEHAVIOR_NORMALIZATION_ENABLED = True

# 每支行为分布保留 Top-K（按条件概率）；跨支融合后再取 Top-K 作为最终展示
BRANCH_ACTION_TOP_K = 3
AGGREGATED_ACTION_TOP_K = 5

# 提示词版本 — v19：L2 我执放宽为「与我分不开」+ 不可替代则拆开 + confidence 按点燃程度
PERCEPTION_PROMPT_VERSION = "v21"
# v14：想读取本支已绑定 H（id/text/mass），不再另判相关度
COGNITION_PROMPT_VERSION = "v14"

# 所缘未给 confidence 时的默认值（不再用 E/S/W 兜底）
ALAMBANA_DEFAULT_CONFIDENCE = 0.5

# 行动似然 L 不对称门控：compat≥中性用 h_up；compat<中性用更弱的 h_down（少压高激活）
# L = 1 + h·(compat−1)；h_up=0.5+0.5a；h_down=0.25+0.25a
ACTION_COMPAT_NEUTRAL = 0.5
ACTION_LIKELIHOOD_H_UP_BASE = 0.5
ACTION_LIKELIHOOD_H_UP_SLOPE = 0.5
ACTION_LIKELIHOOD_H_DOWN_BASE = 0.25
ACTION_LIKELIHOOD_H_DOWN_SLOPE = 0.25

# 行动扩支：max(compat) 偏低时走感知子路径（二执→组合→所缘）；新支不再做 L 矫正
ACTION_EXPAND_ALAMBANA_ENABLED = True
ACTION_EXPAND_MAX_COMPAT = 0.45
ACTION_EXPAND_MAX_BRANCHES = 4
ACTION_EXPAND_MIN_PRIOR = 0.08
ACTION_EXPAND_MAX_NEW_PRIOR = 0.35


def is_cognition_branching_enabled() -> bool:
    """运行时读取分支总开关。"""
    return COGNITION_BRANCHING_ENABLED
