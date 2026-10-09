# 架构说明

## 分层

```
definitions/     心所 / 对治 / 所缘想思定义
      ↓
engine/
  perception/    L0–L3 主观世界建构（刺激侧；含 action_expand 扩支子路径）
  cognition/     行动 L、分支想/思、概率、行为
  continuity/    认知偏置 H + 心理惯性 M
  scoring/       心所四列评分与报告
  extraction/    旧单路提取（分支关闭时）
      ↓
MentalWorldModel.evaluate_from_yuanwen
      ↓
CLI / REST → report JSON
```

## 引擎模块

| 模块 | 职责 |
|------|------|
| perception | 刺激/行动拆分、客观刺激、二执、组合、所缘；以及行动扩支子路径（L2'→combo'→L3'） |
| cognition | 旧支 compat/L/后验、扩支触发、想、思、分支概率、行为融合 |
| continuity | 跨事件 H（偏置 L2）与 M（历史心所加权） |
| scoring | 心所评分、对治消解、`EvaluationResult` / 报告 |
| loader / parser / normalizer / validator / indexer | 定义管线 |
| llm | 厂商无关 `complete(messages, schema=…)` |

## 数据目录

- `definitions/raw|normalized|schemas|seeds/` — 定义与种子
- `data/自建实验/纵向/` — 多事件原文
- `data/自建实验/单独事件/` — 单事件原文
- `data/fixtures/` — 测试用例与 OpenToM 样例
- `data/runtime/` — 运行时种子 / 连续性状态
- `reports/dyntom`、`reports/rectom` — 公开基准结果
- `reports/自建实验/` — 纵向与单独事件结果

## 契约要点

- **一人一心智**：`focus_person` 全链路锁定
- **刺激/行动隔离**：常规感知不见行动造作；行动进认知做 compat/思。扩支时行动作为**生成条件**再进感知子路径，不再二次当后验似然
- **a / compat / L / 扩支**：见 [execution_flow.md](execution_flow.md)
