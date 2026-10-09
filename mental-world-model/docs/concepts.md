# 概念说明

定义语义以用户提供的原始文件为准；本文只列工程对应关系。

## 核心定义

| 概念 | 模块 |
|------|------|
| 心所 Xinsuo | `core/xinsuo/` |
| 对治关系 | `core/relations/` |
| 所缘/想/思定义 | `definitions/normalized/suoyuan_xiang_si.json` |

评分列：所缘境 / 生起念头 / 业 / 现代心理学。

## 运行时概念（多分支路径）

| 概念 | 模块 |
|------|------|
| 分析对象 focus_person | `engine/perception/focus.py` |
| 我执 / 法执 | `engine/perception/` |
| 主观所缘候选 | 同上 |
| 行动扩支（action_informed） | `engine/perception/action_expand.py` |
| a / compat / L | `engine/cognition/action_evidence.py`、`probability.py` |
| 认知分支 | `engine/cognition/` |
| 连续性 H（认知偏置）/ M（心理惯性 episode） | `engine/continuity/` |

## 工程管线

| 概念 | 模块 |
|------|------|
| Definition 加载链 | loader → parser → normalizer → validator → indexer |
| Graph | `core/relations/graph.py` |

## 边界

不以现代心理学 / ToM / BDI 术语替换上述概念。一次评估只建模一个 `focus_person` 的内心世界。
