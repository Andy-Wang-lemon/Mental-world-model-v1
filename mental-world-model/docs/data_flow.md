# 数据流说明

完整步骤与常量见 **[执行流程](execution_flow.md)**。

## 评估主路径

```
原文 + focus_person
        │
        ▼
  多事件切分（可选）
        │
        ▼
  perception.build_perception_state
  L0 拆分 → L1 客观 → L2 二执 → L2.5 组合 → L3 所缘 prior
        │
        ▼
  cognition（单事件）
  [有行动] compat（旧支）
           → 必要时 L2'→combo'→L3' 扩 1 支（action_informed）
           → 旧支 p'∝p·L；新支跳过 L，再混合归一
           → 搭支 → 各支想
  思 = L0 行动骨架（各支共享）
  各支 40 心所
        │
        ├─ 前序：跨支融成 M_event → 再掺历史（报告）→ 落库 M_event
        └─ 预测：各支掺同一 M_hist → 并行支内行为 → 只跨支融行为
        │
        ▼
  continuity 更新（H 偏置下一事件 L2；M episode 列表）
        │
        ▼
  EvaluationResult.to_event_report_dict()
```

末事件：无行动反馈（后验=先验、思空）；各支先掺历史心所再预测行为，只跨支融合行为。

## 定义加载

```
definitions/raw → Parser → Validator → Indexer → Graph
                ↘ normalized JSON（raw 更新则同步）
```

## 阶段输入输出（摘要）

| 阶段 | 输入 | 输出 |
|------|------|------|
| focus | CLI / 原文标记 | `focus_person` |
| L0 | 事件正文 | stimulus / action |
| L1–L3 | 刺激文本 | 客观、二执、组合、所缘 prior |
| L5 | 行动 + 旧支所缘 | compat、L、旧支 posterior |
| L2'–L3' | 客观所缘 + 行动（仅触发时） | 新二执、新 combo、`action_informed` 所缘；posterior=prior |
| 混合归一 | 旧支 p' + 新支 prior | 全体 posterior 和为 1 |
| 分支 | 所缘 + 想/思 | a、心所；预测事件再支级掺 M 后预测行为 |
| M | 历史 episode | 前序跨支期望入 history；预测支级掺历史 |
| 报告 | EvaluationResult | `events[]` 中间结果 |
