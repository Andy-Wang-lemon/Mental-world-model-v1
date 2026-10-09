# MWM 批量跑分（外接）

**不属于 `mental_world_model` 核心。** 在 JSONL 输入上调用 `evaluate_from_yuanwen`，输出 **40 心所 scores_final**。

完整 300 条 benchmark 数据构建见 [`docs/opentom_benchmark.md`](opentom_benchmark.md)。

## 关系

```
mental_world_model (MWM 核心)
        ↑ evaluate_from_yuanwen → scores_final
mwm_test_adapter (外接批量跑分)
```

## 用法

```bash
# Demo（3 条）
mwm-test --mock run --input data/fixtures/benchmark_samples/opentom_demo.jsonl --limit 1

# 300 条（需先构建 opentom_300.jsonl，见 opentom_benchmark.md）
mwm-test run \
  --input data/fixtures/benchmark_samples/opentom_300.jsonl \
  --output reports/opentom_300_results.jsonl \
  --summary reports/opentom_300_summary.json
```

## 输入 JSONL

每行至少包含 `story` 与 `question`；`character`、`question_type`、`id` 可选：

```json
{
  "id": "s1_attitude_q3",
  "story": "Emma and Liam are in the kitchen...",
  "question": "What is Emma's attitude towards ...?",
  "character": "Emma",
  "question_type": "attitude"
}
```

## 输出

每条样本输出 `scores_final`（40 心所最终分）：

```json
{
  "id": "s1_attitude_q3",
  "text": "Emma and Liam...\n\nQuestion: ...\n\nCharacter: Emma",
  "question_type": "attitude",
  "scores_final": {
    "贪": 0.8163,
    "瞋": 0.8510,
    "...": 0.0
  }
}
```

默认 `update_seeds=False`。
