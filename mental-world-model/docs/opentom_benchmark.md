# OpenToM 论文级评测（~300 条）

## 1. 下载官方数据

从 [HuggingFace: SeacowX/OpenToM](https://huggingface.co/datasets/SeacowX/OpenToM) 下载 `opentom.json`（正常长度故事，约 13,708 题）。

```bash
# 示例：浏览器下载后放到 data/raw/opentom.json
mkdir -p data/raw
```

## 2. 构建 300 条分层 benchmark

按论文 7 类题型分层抽样（每类约 42–43 题）：

```bash
python scripts/opentom/build_opentom_benchmark.py \
  --input data/raw/opentom.json \
  --output data/fixtures/benchmark_samples/opentom_300.jsonl \
  --sample-size 300 \
  --seed 42
```

或：

```bash
python scripts/opentom/preprocess_opentom.py \
  --input data/raw/opentom.json \
  --output data/fixtures/benchmark_samples/opentom_300.jsonl \
  --sample-size 300
```

## 3. 配置 LLM

```bash
cp .env.example .env
# 编辑 MWM_LLM_BASE_URL / MWM_LLM_MODEL / MWM_LLM_API_KEY
```

## 4. 跑分

```bash
mwm-test run \
  --input data/fixtures/benchmark_samples/opentom_300.jsonl \
  --output reports/opentom_300_results.jsonl \
  --summary reports/opentom_300_summary.json
```

## 5. 输出

每条样本输出 **40 心所 scores_final**，详见 [`docs/test_adapter.md`](test_adapter.md)。

## 成本提示

300 题 × 每题完整 MWM 流水线（多轮 LLM）。建议先 `--limit 5` 试跑确认 API 正常。

```bash
mwm-test run --input data/fixtures/benchmark_samples/opentom_300.jsonl --limit 5
```
