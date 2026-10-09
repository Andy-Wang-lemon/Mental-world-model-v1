# 报告字段说明

默认 CLI/API 输出：`EvaluationResult.to_event_report_dict()`。

## 顶层

| 字段 | 说明 |
|------|------|
| `focus_person` | 本次唯一分析对象 |
| `events` | 按事件的中间结果列表 |
| `llm_usage` | 调用次数与耗时（有则附） |

## 单事件（流水线顺序）

| 字段 | 层 | 说明 |
|------|----|------|
| `event` / `yuanwen` | — | 标签与事件正文 |
| `stimulus_text` / `stimulus_action_split` | L0 | 刺激窗与行动 span/描述 |
| `objective_event` | L1 | 客观刺激结构 |
| `self_grasping` / `dharma_grasping` | L2 | 我执 / 法执（扩支成功时含行动条件下新增条目） |
| `grasping_combos` | L2.5 | 组合激活与 prior 对照（扩支成功时含 `C_act*`） |
| `subjective_alambanas` | L3/L5 | 所缘；含 prior、posterior、**action_compat**、**action_likelihood**(有效 L)、**action_informed**；无 E/S/W 诊断字段 |
| `action_evidence` | L5 | 行动 + 各所缘 LLM likelihoods（=compat raw；不含 `action_informed` 新支） |
| `branches` | 认知 | `activation_probability`(a)、所缘、想、思、非零心所、支内 `behaviors` |
| `aggregated_behaviors` | 预测 | 跨支融合行为分布（仅预测事件） |
| `mental_factors_event_nonzero` | 心所 | 本事件跨支期望（非零；落库用，未掺历史） |
| `mental_factors_final_nonzero` | 心所 | **仅前序事件**：跨支期望再与历史融合后的非零分 |
| `mental_report_mode` | 心所 | 预测事件为 `per_branch_only`：无顶层跨支 final，看 `branches[].mental_factors_nonzero` |
| `continuity` | 连续 | H 帧摘要 + M（`apply_mode` / `lambda_eff`） |
| `notes` | — | 护栏与路由说明 |

### 前序 vs 预测

| | 前序事件 | 预测（末）事件 |
|--|----------|----------------|
| 跨支心所期望 | 有，写入 history | 有 `mental_factors_event_nonzero`，不作为行为输入 |
| 跨支心所 final | 有 `mental_factors_final_nonzero` | **无**（`mental_report_mode=per_branch_only`） |
| 支级心所 | 未掺历史的支内分 | **已掺同一份 `M_hist`** |
| 行为 | 无 | 支内并行预测 + `aggregated_behaviors` |
| 顶层 `active_xinsuo` | 可取自期望分 | `None` |
| 行动扩支 | 现有支 `max(compat)<0.45` 时可 +1 支（最多 4） | 不做 |

`action_informed=true`：该所缘由「客观所缘 + 行动 → 新二执 → 组合 → 所缘」生成，**不再做** `p'∝p·L`；`action_compat`/`action_likelihood` 固定为 1。旧支仍走 compat 后验，再与新支 posterior 混合归一。

### 所缘字段（`subjective_alambanas[]` / `branches[].alambana`）

| 字段 | 说明 |
|------|------|
| `prior_probability` | 组合激活 P(C)；扩支新支为夹到 [0.08, 0.35] 的行动组合激活质量 |
| `posterior_probability` | 旧支：`p'∝p·L` 后再与新支混合归一；新支：= prior，再混合归一 |
| `action_compat` | 旧支：LLM 行动贴合 raw；新支：不打分，固定 1.0 |
| `action_likelihood` | 旧支：有效 L；新支：不参与 L，固定 1.0 |
| `action_informed` | `true` = 行动扩支子路径生成，跳过行动后验 |
| `used_self_ids` / `used_dharma_ids` | 该支绑定的我执/法执；新支指向扩支写入的新条目 |

`notes` 在触发扩支时会出现「启动行动扩支感知子路径」或「compat/扩支/后验已完成」。

## 刻意不放进默认报告

- 全量 40 心所零分表（只保留非零）
- `action_detection` 兼容壳（与 `action_evidence` 重复）
- `mental_state_distribution`（与 `branches` 重复）
- 完整 H 帧长文（`continuity` 已截断摘要）
- 所缘 E/S/W、`raw_score`、combo `reason`/`focus_axis`（已停采，不参与计算）

调试用：`evaluate --verbose` → `to_report_dict(verbose=True)`。
