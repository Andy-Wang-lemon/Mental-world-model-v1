# Mental World Model（心智世界模型）

将「心 / 心所」定义工程化为可加载、可评分、可连续建模的系统：一次评估只建模**一个分析对象**的内心世界。

## 当前主流程

`MentalWorldModel.evaluate_from_yuanwen(原文, llm, focus_person=...)`

1. **锁定分析对象** — CLI/API `--focus-person` 优先；否则解析原文 `分析对象：某某`
2. **多事件路由** — 按 `事件1/2/…` 切分；前序解释事件更新连续性，末事件做行为预测
3. **感知（刺激侧）** — L0 刺激/行动拆分 → L1 客观刺激 → L2 我执+法执 → L2.5 组合激活 → L3 主观所缘（prior）
4. **认知** — 有行动时先对旧支打 compat；若贴合均偏低则走行动扩支（二执→组合→所缘），旧支再 `p'∝p·L`，新支跳过 L；然后搭支生成想。思=L0 行动骨架（各支共享）；各支 40 心所
5. **连续性** — 认知偏置 H（偏置下一事件 L2）+ 心理惯性 M（历史心所加权，掺进当前心所）
6. **预测事件** — 各支先掺同一份历史心所，再并行预测支内行为，最后只跨支融合行为（不做跨支心所融合）

概率口径（保留）：

| 符号 | 含义 |
|------|------|
| **a** | 内心激活 = 归一 prior \(P(C)\)（支权重；融合心所/行为用 a；不再×支置信） |
| **compat** | 行动贴合（LLM likelihood raw） |
| **L** | 不对称：`compat≥0.5` 用 `h_up=0.5+0.5a`，否则 `h_down=0.25+0.25a`；`L=1+h·(compat−1)`；**只改旧支** `p'∝p·L` |
| **扩支** | `max(compat)<0.45` 且支数 `<4`：客观所缘+行动 → 新二执 → 组合 → 所缘（`action_informed`）；新支不再用行动矫正 |

详见 [`docs/execution_flow.md`](docs/execution_flow.md)。

## 定义数据

- 40 心所 — `definitions/normalized/xinsuo.json`
- 40 对治 — `definitions/normalized/xinsuo_duizhi.json`
- 所缘/想/思定义 — `definitions/normalized/suoyuan_xiang_si.json`

## LLM 配置

```bash
export MWM_LLM_BASE_URL=https://your-llm-host/v1
export MWM_LLM_MODEL=your-model-id
export MWM_LLM_API_KEY=your-api-key   # 可选
```

详见 [`docs/llm_adapter.md`](docs/llm_adapter.md)。

## CLI

```bash
pip install -e ".[dev]"

# 推荐：显式分析对象 + 多事件文件
python3 -m mental_world_model.cli evaluate \
  --focus-person 陈默 \
  --file data/陈默事件（负面版）.txt \
  -o reports/out.json

# Mock
python3 -m mental_world_model.cli --mock evaluate --text "分析对象：陈默\n事件1：…"

# API
python3 -m mental_world_model.cli serve --port 8765
```

原文也可写：

```text
分析对象：陈默

事件1：…
事件8：…（末事件默认做行为预测）
```

## 报告（默认 `to_event_report_dict`）

按事件输出分析用中间结果（去重、流水线顺序）：

| 字段 | 内容 |
|------|------|
| `focus_person` | 分析对象 |
| `stimulus_action_split` | L0 刺激 / 行动 |
| `objective_event` | L1 客观刺激 |
| `self_grasping` / `dharma_grasping` | L2 二执（扩支成功时含行动条件下新增条目） |
| `grasping_combos` | L2.5 组合激活（扩支成功时含 `C_act*`） |
| `subjective_alambanas` | L3 prior / posterior / compat / L / `action_informed` |
| `action_evidence` | 行动 + 旧支 likelihoods（不含 `action_informed` 新支） |
| `branches[]` | a、所缘、想、思、非零心所（预测事件为掺历史后的支级分）、支内行为 |
| `aggregated_behaviors` | 跨支行为分布（仅预测事件） |
| `mental_factors_event_nonzero` | 本事件跨支期望（落库用，未掺历史） |
| `mental_factors_final_nonzero` | 前序事件：期望再与历史融合后的跨支心所 |
| `mental_report_mode` | 预测事件为 `per_branch_only`（无顶层跨支 final） |
| `continuity` | H / M 摘要（含 `apply_mode`、`lambda_eff`） |

`--verbose` 另附调试明细（`pipeline_trace` 等）。

## REST

`POST /v1/evaluate`

```json
{
  "yuanwen": "分析对象：陈默\n事件1：…",
  "focus_person": "陈默",
  "explain": "none"
}
```

## 测试 / 批量

```bash
python3 -m pytest tests/ -q
```

外接批量跑分：[`docs/test_adapter.md`](docs/test_adapter.md)、[`docs/opentom_benchmark.md`](docs/opentom_benchmark.md)。

## 文档索引

| 文档 | 内容 |
|------|------|
| [模型架构与原理.docx](docs/模型架构与原理.docx) | **完整架构、流程、公式与权重** |
| [execution_flow.md](docs/execution_flow.md) | 执行步骤摘要 |
| [report.md](docs/report.md) | 报告字段 |
| [architecture.md](docs/architecture.md) | 模块分层 |
| [data_flow.md](docs/data_flow.md) | 数据流 |
| [concepts.md](docs/concepts.md) | 概念 |

## 边界

以「心 / 心所 / 二执 / 所缘」为核心；不是通用心理学 / ToM / BDI 框架。
