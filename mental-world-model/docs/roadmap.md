# 路线图

## 已完成

- [x] 心所 + 对治关系定义对接
- [x] 所缘/想/思提取与四列评分
- [x] 权重合成与 ScoreBreakdown 可解释性
- [x] 对治互斥消解
- [x] LLM Adapter、CLI、REST API
- [x] Golden test cases

## 进行中 / 近期

- [x] W_seed 种子库与 `definitions/seeds/` 加载（`SEED_LIBRARY_ENABLED` 控制，默认关）
- [ ] 真实 LLM 端到端验证

## 待对接（需用户提供 docx）

- [ ] 心 / 境 / 所缘 / 行相 / 相应 / 条件 / 功能 定义
- [ ] 对接后扩展 Schema 与 core 模型

## 后续能力

- [ ] QueryEngine — 复杂图/关系查询（可先用 `model.graph` / indexer）
- [ ] Rule + RuleEngine — 除 LLM 外的规则推理（规则定义待定）

## 不在当前范围

- 对话分析 / 行为解释 / 代理模拟等应用层（心所稳定后再考虑）
