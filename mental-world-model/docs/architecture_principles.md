# 心智世界模型 — 架构方案与计算原理

> 基于 `MentalWorldModel.evaluate_from_yuanwen()` 的当前实现，含全部硬编码常量与公式。  
> 默认：`SEED_LIBRARY_ENABLED = False`（种子库关闭）；下文分别说明「关/开」两种路径。

---

## 一、系统定位与分层

**总体数据流：**

原文 → [LLM] 所缘 S → [LLM×40 + 代码 Top-6] H（种子库开）→ [LLM] 想/思 + source/confidence → [代码] 护栏 → [代码] 算 m、ŵ → [LLM×≤160] 每心所四关 s → [代码] S_col = Σ ŵ·s，F_init = S_col → [代码+LLM] 对治消解 → scores_final → [代码] EMA 更新 W_runtime（种子库开）

**设计原则：**

| 层 | 职责 |
|----|------|
| LLM | 语义理解：所缘/想/思、路由相关度、四列匹配分、冲突消解 |
| 代码 | 纪律规则：护栏、门控、权重公式、链式依赖、冲突改写分 |
| W_seed | 个性化画像：路由 H、想/思提示、EMA 记忆；**不乘最终分** |

---

## 二、阶段 0：加载（无 LLM）

| 数据 | 路径 |
|------|------|
| 40 心所 | definitions/normalized/xinsuo.json |
| 40 对治（互斥） | definitions/normalized/xinsuo_duizhi.json |
| 所缘/想/思定义 | definitions/normalized/suoyuan_xiang_si.json |
| 默认种子 W_default | definitions/seeds/xinsuo_seeds.json |
| 运行时种子 W_runtime | data/runtime/seed_state.json |

**种子库开关 SEED_LIBRARY_ENABLED（默认 False）：**

- **关**：不读种子文件、不路由 H、不想/思 H 提示、不 EMA
- **开**：加载 W_runtime（无则复制 W_default），参与路由与 EMA

---

## 三、阶段 A：提取（所缘 → 路由 → 想/思）

### A1. 所缘境 S

| 项 | 说明 |
|----|------|
| 执行者 | LLM × 1 |
| 输出 | suoyuan.text |
| 代码 | 固定 source=原文, confidence=1.0 |

### A2. 路由心所集合 H（仅种子库开）

对每个心所 i ∈ {1..40}：

1. LLM 返回语义相关度 r_i ∈ [0,1]（40 次 LLM）
2. 代码融合：

```
combined_i = α · r_i + (1 - α) · W_runtime(i)
```

| 常量 | 值 |
|------|-----|
| α = ROUTER_RELEVANCE_ALPHA | 0.75 |
| ROUTER_MIN_RELEVANCE | 0.35（r_i 低于此丢弃） |
| ROUTER_TOP_K | 6 |

取 combined 降序 Top-6 → 集合 H。

**种子库关**：跳过，H = 空，0 次路由 LLM。

### A3. 想 / 思

| 项 | 说明 |
|----|------|
| 执行者 | LLM × 1 |
| 输入 | 原文 + S + 想/思定义 + summary；种子库开时 + H 心所提示（生起念头/业各截 100 字） |
| LLM 输出 | 各字段 {text, source, confidence} |

**LLM 自评规则（prompt 要求）：**

- 原文有直接依据 → source=原文, confidence=1.0
- 推断 → source=推理, confidence ∈ [0,1]

**代码护栏 normalize_xiang_si_field：**

| 条件 | 处理 |
|------|------|
| 空 text | confidence = 0 |
| confidence | clamp 到 [0, 1]；缺失默认 0.6 |
| source=原文 且 confidence<1 | confidence → 1.0 |
| source=原文 但无原文依据 | source→推理，confidence=0.8 |
| source=推理 且 confidence≥1 且无原文依据 | confidence→0.8 |

「无原文依据」= 子串匹配或字符重叠 ≥ 85%（is_grounded_in_yuanwen）

INFERENCE_CONFIDENCE_MAX = 0.8

### 提取阶段 LLM 次数

| 种子库 | 次数 |
|--------|------|
| 关（默认） | 1（所缘）+ 1（想/思）= 2 |
| 开 | 1 + 40（路由）+ 1 = 42 |

---

## 四、阶段 B：40 心所评分

### B1. 遍历顺序

按 xinsuo.json 文档顺序：信 → … → 散乱，共 40 个。

**忿→恨→恼链（代码）：**

```
忿 final ≤ 0  → 恨、恼 不评 LLM，= 0
忿 > 0，恨 ≤ 0 → 恼 不评，= 0
忿 > 0，恨 > 0 → 评恼
```

### B2. 每个心所的四关 LLM 评分

| 关 | 列 | 门控 | 说明 |
|----|-----|------|------|
| 1 | 所缘境 suoyuan | 是 | 返回 s_1 |
| 2 | 生起念头 shengqi_nian | 是 | 返回 s_2 |
| 3 | 业 ye | 否 | 返回 s_3 |
| 4 | 现代心理学 xiandai_xinli | 否 | 返回 s_4 |

门控阈值 GATE_THRESHOLD = 0.3：

- 第 1 关 s_1 < 0.3 → 立即退出，final = 0，不调后续 LLM
- 第 2 关 s_2 < 0.3 → 同上
- 通过前两关 → 继续 3、4 关并合成

### B3. 来源调节系数 m（由想/思决定，全 40 心所共用）

四列索引 i ∈ {1,2,3,4} 对应：所缘境、生起念头、业、现代心理学。

**基础列权重 w（固定，Σ w_i = 1）：**

| 列 | 符号 | w_i |
|----|------|-----|
| 所缘境 | w_1 | 0.35 |
| 生起念头 | w_2 | 0.30 |
| 业 | w_3 | 0.10 |
| 现代心理学 | w_4 | 0.25 |

**来源调节 m：**

```
m_1 = 1.0（恒）

m_2 = λ(想) = 1.0                    若 source=原文
              max(c_想, 0.5)         若 source=推理

m_3 = λ(思)（同想，用思的 source/confidence）

m_4 = 1.0（恒）
```

其中 c = 护栏后的 confidence；INFERENCE_CONFIDENCE_FLOOR = 0.5。

### B4. 归一化有效权重 ŵ

```
ŵ_i = (w_i · m_i) / Σ_j (w_j · m_j)
```

保证 Σ ŵ_i = 1。

### B5. 列合成 S_col 与初始分 F_init

```
S_col = Σ_i (ŵ_i · s_i)

F_init = S_col
```

**W_seed 不参与乘分（方案 A）。** scores_initial[name] = F_init。

报告中仍输出 seed_weight = W_runtime(name)，标注为个性化参考。

### B6. 数值例题（单心所「信」）

设四关 LLM 分：s = [0.8, 0.8, 0.6, 0.6]

想=原文，思=推理且 confidence=0.42（护栏后）：

- m = [1.0, 1.0, 0.5, 1.0]（思：max(0.42, 0.5)=0.5）
- w·m = [0.35, 0.30, 0.05, 0.25]，总和 = 0.95
- ŵ ≈ [0.368, 0.316, 0.053, 0.263]

```
S_col = 0.368×0.8 + 0.316×0.8 + 0.053×0.6 + 0.263×0.6 ≈ 0.738
F_init = 0.738
```

若思也是原文（m_3=1.0），则 ŵ=w，S_col=0.8。

---

## 五、阶段 C：对治冲突检测与消解

### C1. 冲突检测（代码）

- 40 条「对治」关系，每心所一条 source
- 互斥对 (A, B)：双方 score > 0（阈值 0）即冲突
- 输入：scores_initial

### C2. 冲突消解（LLM）

每对冲突 1 次 LLM，返回 winner ∈ {A, B, none}：

| winner | 结果 |
|--------|------|
| A | A 保留，B → 0 |
| B | B 保留，A → 0 |
| none / 无效 | 双归零 |

循环直到无冲突或达到 41 轮 → scores_final。

---

## 六、阶段 D：种子库 EMA 更新（仅种子库开）

前提：update_seeds=True（CLI 默认；REST API 固定 False）

对每个心所 i：

```
若 F_final(i) > 0:
    W_new(i) = (1 - α_s) · W_old(i) + α_s · F_final(i)
若 F_final(i) = 0:
    W_new(i) = (1 - β) · W_old(i)
```

| 常量 | 值 |
|------|-----|
| α_s = SEED_EMA_ALPHA | 0.3 |
| β = SEED_ZERO_DECAY | 0.05 |

EMA 输入是 scores_final（已是 S_col 消解后，未乘 W_seed），避免 W 与分互相拖低。

写入 seed_state.json，供下次路由 H 使用。

---

## 七、LLM / 代码分工一览

| 步骤 | LLM | 代码 |
|------|-----|------|
| 所缘境 | ✓ | 固定原文/1.0 |
| 路由 relevance | ✓（种子库开，40 次） | Top-K 融合 W_seed |
| 想/思 + confidence | ✓ | clamp、一致性护栏 |
| 四列匹配分 s | ✓ | 门控 0.3、忿恨恼链 |
| m, ŵ, S_col | | ✓ 公式 |
| F_init | | ✓ = S_col |
| 冲突检测 | | ✓ |
| 冲突消解 | ✓ | 改分 |
| W_seed 更新 | | ✓ EMA |

---

## 八、常量速查表

| 常量 | 值 | 文件 |
|------|-----|------|
| SEED_LIBRARY_ENABLED | False | engine/seeds/config.py |
| GATE_THRESHOLD | 0.3 | engine/scoring/weights.py |
| w_1..w_4 | 0.35 / 0.30 / 0.10 / 0.25 | engine/scoring/weights.py |
| INFERENCE_CONFIDENCE_FLOOR | 0.5 | engine/scoring/weights.py |
| ROUTER_RELEVANCE_ALPHA | 0.75 | engine/seeds/config.py |
| ROUTER_TOP_K | 6 | engine/seeds/config.py |
| ROUTER_MIN_RELEVANCE | 0.35 | engine/seeds/config.py |
| INFERENCE_CONFIDENCE_DEFAULT | 0.6 | engine/seeds/config.py |
| INFERENCE_CONFIDENCE_MAX | 0.8 | engine/seeds/config.py |
| SEED_EMA_ALPHA | 0.3 | engine/seeds/config.py |
| SEED_ZERO_DECAY | 0.05 | engine/seeds/config.py |
| HINT_FIELD_MAX_CHARS | 100 | engine/seeds/config.py |
| 冲突判定 | score > 0 | engine/validator/definition_validator.py |

---

## 九、核心公式汇总

**路由（种子库开）：**

```
combined_i = 0.75 · r_i + 0.25 · W_runtime(i)
```

**来源调节：**

```
m_2 = λ(想),  m_3 = λ(思),  m_1 = m_4 = 1

λ = 1.0              若 source=原文
    max(c, 0.5)      若 source=推理
```

**列合成：**

```
ŵ_i = (w_i · m_i) / Σ_j (w_j · m_j)
S_col = Σ_i (ŵ_i · s_i)
F_init = S_col
```

**种子 EMA（种子库开）：**

```
W_new = 0.7 · W_old + 0.3 · F_final     (F_final > 0)
W_new = 0.95 · W_old                    (F_final = 0)
```

---

## 十、报告主要字段

| 字段 | 含义 |
|------|------|
| extracted | S / 想 / 思 + routed_hint_names + router_relevance |
| scores_initial | 消解前，= S_col |
| scores_final | 消解后 |
| score_breakdowns | s_i, w_i, m_i, ŵ_i, 贡献, S_col, W_seed（参考） |
| seed_weights_before/after | 本次交互前后 W_runtime |

---

## 十一、关键源码索引

| 用途 | 路径 |
|------|------|
| 统一入口 | src/mental_world_model/model.py |
| 提取 | engine/extraction/extractor.py, prompts.py, router.py, validators.py |
| 评分 | engine/scoring/pipeline.py, weights.py, breakdown.py |
| 消解 | engine/reconciliation/reconciler.py |
| 种子 | engine/seeds/config.py, state_store.py, updater.py |
