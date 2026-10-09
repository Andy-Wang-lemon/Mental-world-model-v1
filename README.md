# Mental World Model (MWM) · Multi-Branch Mental Evolution Architecture (MMEA)

**Languages:** [English](#english) | [简体中文](#简体中文)

<a id="english"></a>

# English

## Overview

Mental World Model (MWM) makes definitions of the *mind and mental factors* (心 / 心所) loadable, scorable, and traceable across events. Each evaluation models the inner world of **one focal person**. The implementation uses a multi-branch mental evolution architecture (MMEA) to represent subjective interpretations, cognitive processes, mental factors, historical influences, and possible behaviors.

This is **not** a general-purpose psychology framework or a conventional Theory of Mind (ToM) / Belief–Desire–Intention (BDI) agent framework. Public benchmarks (DynToM, RecToM, SimpleToM) and custom longitudinal experiments are external evaluations; the core is the mental-factor model.

**Requirements:** Python 3.10+.

> **Repository paths:** All commands below run from `mental-world-model/`, unless otherwise specified. Some dataset paths and tests depend on the local directory layout; see [Known issues](#known-issues).

## Repository structure

```text
心智世界模型（代码）/
├── 公开数据集/                       # DynToM, RecToM, SimpleToM, CAREBench
├── 代跑命令说明.docx                 # Benchmark execution guide
├── 实验原始数据.docx                 # Original experimental data
├── 模型最新架构.docx                 # Latest architecture
└── mental-world-model/
    ├── src/mental_world_model/       # Core model
    ├── src/mwm_test_adapter/         # JSONL batch evaluation
    ├── dyntom_adapter/               # DynToM adapter
    ├── rectom_adapter/               # RecToM adapter
    ├── simpletom_adapter/            # SimpleToM adapter
    ├── inference_baselines/          # CoT, SimToM, TimeToM, PICTURE, Agentic-ToM
    ├── definitions/                  # Finalized mental-factor definitions
    ├── data/
    │   ├── 自建实验/纵向/             # Multi-event narratives
    │   ├── 自建实验/单独事件/         # Single-event narratives
    │   ├── 自建实验/陈默正面答案.docx
    │   ├── fixtures/                 # Test fixtures and OpenToM samples
    │   └── runtime/                  # Continuity state
    ├── reports/
    │   ├── dyntom/
    │   ├── rectom/
    │   └── 自建实验/纵向|单独事件实验二/
    ├── scripts/
    │   ├── experiments/             # Longitudinal, single-event, two-agent runs
    │   ├── docgen/                  # Documentation generation
    │   └── opentom/                 # OpenToM sampling
    ├── tests/
    └── docs/
```

**Important:** Keep the directory name `reports/dyntom/dyntom mwm` (including its space). The folder `公开数据集/CAREBench ` has a trailing space in the documented local layout.

## Installation

```bash
cd mental-world-model
python3 -m pip install -e ".[dev]"
```

PyYAML (`yaml`) is optional. Development dependencies include pytest.

| CLI | Entry point |
| --- | --- |
| `mwm` | `mental_world_model.cli:main` |
| `mwm-test` | `mwm_test_adapter.cli:main` |

You can also run `python3 -m mental_world_model.cli` without invoking the installed CLI shortcut.

## Model configuration

Copy `.env.example` to `.env`. Project scripts automatically load the local `.env` file.

```bash
export MWM_LLM_BASE_URL=https://your-llm-host/v1
export MWM_LLM_MODEL=your-model-id
export MWM_LLM_API_KEY=your-api-key
```

`MWM_LLM_API_KEY` may be empty. The backend uses an OpenAI-compatible `chat/completions` endpoint. Use `--mock` to run without a real LLM.

## Core pipeline

**API:** `MentalWorldModel.evaluate_from_yuanwen(原文, llm, focus_person=...)`

1. **Select the focal person.** `--focus-person` takes priority; otherwise parse `分析对象：姓名` from the source text.
2. **Split events.** Parse `事件1`, `事件2`, etc. Earlier events update continuity; the final event is used for behavior prediction.
3. **Perception (stimulus side).** L0 separates stimulus and action; L1 represents objective stimuli; L2 handles self-grasping and dharma-grasping; L2.5 activates combinations; L3 constructs subjective objects of attention.
4. **Cognition.** When an action is observed, first assess its compatibility with existing branches. If all branches fit poorly, expand the branches based on the action. Then generate branch-specific thoughts (想); volition/action intention (思) comes from the shared L0 action skeleton. Score all 40 mental factors for each branch.
5. **Continuity.** H represents cognitive bias affecting L2 in later events. M represents psychological inertia, incorporating historical mental factors into the current event.
6. **Prediction.** Incorporate the same historical mental-factor state into each branch, predict behaviors within branches, and aggregate **behaviors only** across branches. Do not aggregate mental factors across branches at the prediction stage.

**Input format:**

```text
分析对象：陈默

事件1：…
事件2：…
```

A single-event input must contain only one event label.

### Probabilities and branch updates

| Symbol | Meaning |
| --- | --- |
| `a` | Normalized inner-activation / branch weight; used when aggregating mental factors and behaviors where applicable |
| `compat` | Model-estimated likelihood of an observed action under a branch |
| `L` | Reweights existing branches: raise when `compat ≥ 0.5`, lower otherwise, then `p' ∝ p · L` |
| Branch expansion | Create a new action-informed branch when the maximum old-branch compatibility is below `0.45` and there are fewer than four branches; do not re-correct the new branch with that action |

See `docs/execution_flow.md`.

## Definition data

| File | Description |
| --- | --- |
| `definitions/normalized/xinsuo.json` | 40 mental factors |
| `definitions/normalized/xinsuo_duizhi.json` | 40 counteractive factors |
| `definitions/normalized/suoyuan_xiang_si.json` | Objects of attention / perception (想) / volition (思) |
| `definitions/seeds/xinsuo_seeds.json` | Seed library |
| `definitions/raw/*.docx` | Original source documents for the JSON definitions |
| `definitions/schemas/` | JSON Schema files |

The seed library is disabled by default (`SEED_LIBRARY_ENABLED`). The primary pipeline evaluates all 40 mental factors, then applies branch weights `a`; it does not filter mental factors using a user profile.

## Run a single evaluation

```bash
python3 -m mental_world_model.cli evaluate \
  --focus-person 陈默 \
  --file data/自建实验/纵向/陈默纵向7事件.txt \
  -o reports/out.json
```

Mock example:

```bash
python3 -m mental_world_model.cli --mock evaluate \
  --text "分析对象：陈默\n事件1：散会后有人来问意见。"
```

Without `-o`, output is saved as `reports/report_年月日_时分秒_毫秒.json`. `--verbose` adds debugging fields such as `pipeline_trace`.

| Longitudinal input | Person / scenario |
| --- | --- |
| `data/自建实验/纵向/陈默纵向7事件.txt` | Chen Mo, seven events |
| `data/自建实验/纵向/陈默正向纵向6事件.txt` | Chen Mo, positive scenario, six events |
| `data/自建实验/纵向/程野纵向7事件.txt` | Cheng Ye |
| `data/自建实验/纵向/林夏纵向6事件.txt` | Lin Xia |
| `data/自建实验/纵向/沈川纵向7事件.txt` | Shen Chuan |
| `data/自建实验/纵向/王磊纵向7事件.txt` | Wang Lei |

Two referenced inputs, `data/陈默事件（负面版）.txt` and `data/林可负面事件.txt`, are missing in the documented layout. Supply them or override `--chenmo-file` / `--linke-file` before running the two-agent loop.

## Custom experiments

### Longitudinal rollout

Predictions are committed into subsequent events; event count is determined by the input file.

```bash
python3 scripts/experiments/chenmo_longitudinal_7.py --mock
python3 scripts/experiments/chenmo_longitudinal_7.py \
  -o reports/自建实验/纵向/chenmo_longitudinal_7
python3 scripts/experiments/chenmo_longitudinal_7.py \
  --event-file data/自建实验/纵向/程野纵向7事件.txt \
  -o reports/自建实验/纵向/chengye_longitudinal_7
```

To resume, use the same `-o` path together with `--resume`.

### Text-history LLM baseline

This baseline does **not** use the mental-factor engine.

```bash
python3 scripts/experiments/llm_baseline_longitudinal_rollout.py --mock
python3 scripts/experiments/llm_baseline_longitudinal_rollout.py \
  --event-file data/自建实验/纵向/程野纵向7事件.txt \
  -o reports/自建实验/纵向/llm_baseline_chengye_7
```

### Single-event prediction

Single-event runs predict only: they do not write H/M state or commit predicted outcomes. Each `.txt` contains one event; run one person per command.

```bash
python3 scripts/experiments/single_event_experiment_2.py \
  --event-file data/自建实验/单独事件/实验3苏宁.txt

python3 scripts/experiments/single_event_experiment_2_llm.py \
  --event-file data/自建实验/单独事件/实验3苏宁.txt
```

Outputs:
- MWM: `reports/自建实验/单独事件实验二/<文件名>/`
- Baseline: `reports/自建实验/单独事件实验二/<文件名>/llm结果/`

Other inputs include `实验3林舟.txt`, `实验3许哲.txt`, and `实验3陈溪.txt`.

### Two-agent closed loop

The external script alternates events between two agents. Its default negative-scenario input files are currently missing.

```bash
python3 scripts/experiments/dual_agent_closed_loop.py --mock \
  -o reports/自建实验/dual_loop
```

Existing longitudinal results are organized under `reports/自建实验/纵向/<人物>/`, with subfolders `mwm`, `gpt结果`, `gemini结果`, `llm结果`, and `grok结果`.

## Public benchmarks

Datasets are documented under `../公开数据集/` relative to `mental-world-model/`.

| Path | Dataset |
| --- | --- |
| `../公开数据集/DynToM.json` | DynToM |
| `../公开数据集/RecTOM/3_pred_rec.json` | RecToM recommender |
| `../公开数据集/RecTOM/4_pred_seeker.json` | RecToM seeker |
| `../公开数据集/SimpleToM/test.jsonl` | SimpleToM |
| `../公开数据集/CAREBench /third_person.json` | CAREBench third-person task (trailing space in folder name) |

### MWM benchmark runs

```bash
python3 dyntom_adapter/__main__.py baseline \
  --input "../公开数据集/DynToM.json" \
  --output "reports/dyntom/dyntom mwm"

python3 rectom_adapter/__main__.py baseline \
  --input "../公开数据集/RecTOM/3_pred_rec.json" \
  --output "reports/rectom/mwm" \
  --index 0
```

For seeker prediction, use `4_pred_seeker.json`; `--index` selects the question index.

### Inference-framework baselines

The five methods supported by `run_benchmark.py` are `cot`, `simtom`, `timetom`, `picture`, and `agentictom`.

```bash
python3 run_benchmark.py --dataset dyntom --method cot --limit 1
python3 run_benchmark.py --dataset rectom --method picture --limit 1
```

**Path caveat:** The default loader looks for `公开数据集/` one level above the code folder. The source README also describes a local arrangement where datasets are inside the code folder. Explicitly set `--data-dir` to the actual location. For the documented parent-folder layout:

```bash
python3 run_benchmark.py \
  --dataset dyntom \
  --method cot \
  --data-dir "../公开数据集" \
  --limit 1
```

Direct-answer LLM baselines (DeepSeek, Gemini, Grok, GPT) use their corresponding adapters' `baseline` commands.

| Model / method | DynToM results | RecToM results |
| --- | --- | --- |
| MWM / MMEA | `reports/dyntom/dyntom mwm` | `reports/rectom/mwm` |
| CoT | `reports/dyntom/cot` | `reports/rectom/cot` |
| SimToM | `reports/dyntom/simtom` | `reports/rectom/simtom` |
| TimeToM-style | `reports/dyntom/timetom` | `reports/rectom/timetom` |
| PICTURE | `reports/dyntom/picture` | `reports/rectom/picture` |
| Agentic-ToM | `reports/dyntom/agentictom` | `reports/rectom/agentictom` |
| DeepSeek | `reports/dyntom/dyntom_deepseek` | `reports/rectom/rectom_deepseek` |
| Gemini | `reports/dyntom/dyntom_gemini` | `reports/rectom/rectom_gemini` |
| Grok | `reports/dyntom/dyntom_grok` | `reports/rectom/rectom_grok` |
| GPT | `reports/dyntom/dyntom_gpt` | `reports/rectom/rectom_gpt` |

Summary files: `reports/dyntom/summary.txt` and `reports/rectom/summary.txt`. MWM runs update their own summary sections; other methods do not automatically update the aggregate summaries. See `../代跑命令说明.docx` and `scripts/docgen/build_benchmark_run_docx.py`.

### SimpleToM

```bash
python3 simpletom_adapter/__main__.py --help
```

### OpenToM

A demo sample is available at `data/fixtures/benchmark_samples/opentom_demo.jsonl`. To construct a 300-item sample, first obtain the official `opentom.json`:

```bash
python3 scripts/opentom/build_opentom_benchmark.py \
  --input data/raw/opentom.json \
  --output data/fixtures/benchmark_samples/opentom_300.jsonl \
  --sample-size 300 \
  --seed 42
```

To batch-score all 40 mental factors using `mwm-test`:

```bash
mwm-test --mock run \
  --input data/fixtures/benchmark_samples/opentom_demo.jsonl \
  --limit 1
```

Output contains `scores_final` for all 40 mental factors per record. See `docs/opentom_benchmark.md` and `docs/test_adapter.md`.

## Output report schema

The default `to_event_report_dict` includes intermediate results for each event.

| Field | Description |
| --- | --- |
| `focus_person` | Focal person |
| `stimulus_action_split` | L0 stimulus/action split |
| `objective_event` | L1 objective stimulus |
| `self_grasping` / `dharma_grasping` | L2 grasping; includes action-conditioned entries when branch expansion succeeds |
| `grasping_combos` | L2.5 combinations |
| `subjective_alambanas` | L3 prior, posterior, compat, L, and `action_informed` |
| `action_evidence` | Observed action and likelihood under existing branches |
| `branches[]` | Branch weight `a`, object, thought, volition, nonzero mental factors, within-branch behavior |
| `aggregated_behaviors` | Cross-branch behavior distribution; prediction events only |
| `mental_factors_event_nonzero` | Cross-branch expectation for this event, without historical blending; used for persistence |
| `mental_factors_final_nonzero` | Cross-branch mental factors after blending history in earlier events |
| `mental_report_mode` | `per_branch_only` for prediction events |
| `continuity` | H/M summary |

See `docs/report.md`.

## HTTP API

```bash
python3 -m mental_world_model.cli serve --port 8765
```

**Endpoint:** `POST /v1/evaluate`

```json
{
  "yuanwen": "分析对象：陈默\n事件1：…",
  "focus_person": "陈默",
  "explain": "none"
}
```

## Code architecture

```text
definitions/         Mental factors, counteractions, object/thought/volition
      ↓
engine/
  perception/       L0–L3, including action-informed branch expansion
  cognition/        Compatibility, branch thoughts/volition, probabilities, behavior
  continuity/       Historical cognitive bias (H) and mental inertia (M)
  scoring/          Mental-factor scoring and reports
  extraction/       Legacy extraction when branching is disabled
      ↓
MentalWorldModel.evaluate_from_yuanwen
      ↓
CLI / REST
```

Additional modules handle definition loading, parsing, normalization, validation, indexing, graphs, rule reasoning, prediction, querying, counteractive resolution, seed updates, and vendor-neutral LLM clients in `llm/`. Implemented parts of `core/` include `xinsuo` and `relations`.

The default runtime continuity state is stored at `data/runtime/continuity_state.json`. Longitudinal scripts redirect their state into their own output folders to avoid overwriting this default file.

## Tests

```bash
python3 -m pytest tests/ -q
```

Real-API tests are marked `integration`.

<a id="known-issues"></a>

### Known issues

- Two-agent closed-loop tests expect missing files: `data/陈默事件（负面版）.txt` and `data/林可负面事件.txt`.
- Some DynToM / RecToM / SimpleToM tests hardcode a `公开数据集/` folder **outside** the code directory; the original description also mentions data being **inside** it. Verify your checkout and adjust paths.
- Two assertions in `tests/test_summary_txt.py` use earlier sample counts (DynToM: 142; RecToM recommender: 20) and may fail against the larger current reports.

## Documentation

| File | Description |
| --- | --- |
| `docs/execution_flow.md` | Execution flow |
| `docs/architecture.md` | Module architecture |
| `docs/data_flow.md` | Data flow |
| `docs/concepts.md` | Concepts |
| `docs/report.md` | Report fields |
| `docs/llm_adapter.md` | LLM adapters |
| `docs/architecture_principles.md` | Detailed principles |
| `docs/roadmap.md` | Completed and planned work |
| `docs/test_adapter.md` | `mwm-test` |
| `docs/opentom_benchmark.md` | OpenToM 300-item benchmark |
| `../模型最新架构.docx` | Architecture, equations, weights; generated by `scripts/docgen/build_architecture_docx.py` |
| `../代跑命令说明.docx` | DynToM / RecToM execution |
| `../实验原始数据.docx` | Experimental source descriptions |

The previously indexed `docs/模型架构与原理.docx` is not present in the documented repository. Its content is represented by `../模型最新架构.docx`.

---

<a id="简体中文"></a>

# 简体中文

# Mental World Model（心智世界模型）

把「心 / 心所」定义做成可加载、可评分、可跨事件连续建模的系统。一次评估只建模**一个分析对象**的内心世界。

它不是通用心理学框架，也不是 ToM / BDI 智能体框架。公开基准（DynToM、RecToM、SimpleToM）和自建纵向实验都是外接跑分，核心仍是心所模型。

要求 Python 3.10 及以上。

## 目录

在 `mental-world-model/` 里执行下面的命令。公开数据在上一级的 `公开数据集/`。

```text
心智世界模型（代码）/
├── 公开数据集/                      # DynToM、RecToM、SimpleToM、CAREBench
├── 代跑命令说明.docx
├── 实验原始数据.docx
├── 模型最新架构.docx
└── mental-world-model/
    ├── src/mental_world_model/      # 模型本体
    ├── src/mwm_test_adapter/        # 外接 JSONL 批量跑分
    ├── dyntom_adapter/              # DynToM
    ├── rectom_adapter/              # RecToM
    ├── simpletom_adapter/           # SimpleToM
    ├── inference_baselines/         # CoT、SimToM、TimeToM、PICTURE、Agentic-ToM
    ├── definitions/                 # 心所定稿
    ├── data/
    │   ├── 自建实验/纵向/           # 多事件原文
    │   ├── 自建实验/单独事件/       # 单事件原文
    │   ├── 自建实验/陈默正面答案.docx
    │   ├── fixtures/                # 测试用例、OpenToM 样例
    │   └── runtime/                 # 连续性状态
    ├── reports/
    │   ├── dyntom/                  # 公开基准结果，目录名保持不动
    │   ├── rectom/
    │   └── 自建实验/纵向|单独事件实验二/
    ├── scripts/
    │   ├── experiments/             # 纵向、单独事件、双人闭环
    │   ├── docgen/                  # 生成说明文档
    │   └── opentom/                 # OpenToM 抽样
    ├── tests/
    └── docs/
```

`reports/dyntom/dyntom mwm` 这个目录名中间有空格，命令和测试都按这个名字写，不要改。`公开数据集/CAREBench ` 文件夹名末尾有一个空格。

## 安装

```bash
cd mental-world-model
python3 -m pip install -e ".[dev]"
```

可选依赖：`yaml`（PyYAML）。开发依赖带 pytest。

命令行入口：

| 命令 | 模块 |
|------|------|
| `mwm` | `mental_world_model.cli:main` |
| `mwm-test` | `mwm_test_adapter.cli:main` |

也可以不安装入口，直接 `python3 -m mental_world_model.cli`。

## 配置模型

复制 `.env.example` 为 `.env`。脚本会自动读项目里的 `.env`。

```bash
export MWM_LLM_BASE_URL=https://your-llm-host/v1
export MWM_LLM_MODEL=your-model-id
export MWM_LLM_API_KEY=your-api-key
```

`MWM_LLM_API_KEY` 可空。接口是 OpenAI 兼容的 `chat/completions`。不接真实模型时，命令加 `--mock`。

## 主流程

`MentalWorldModel.evaluate_from_yuanwen(原文, llm, focus_person=...)`

1. **锁定分析对象。** `--focus-person` 优先；否则读原文里的 `分析对象：某某`。
2. **按事件切分。** `事件1`、`事件2`……前序事件更新连续性，最后一件做行为预测。
3. **感知（刺激侧）。** L0 刺激/行动拆分，L1 客观刺激，L2 我执与法执，L2.5 组合激活，L3 主观所缘。
4. **认知。** 有行动时先看旧支是否贴合。贴合都偏低就按行动扩支。然后生成想；思来自 L0 行动骨架，各支共用；每支评 40 个心所。
5. **连续性。** H 是认知偏置，影响下一事件的 L2。M 是心理惯性，把历史心所掺进当前心所。
6. **预测。** 各支先掺同一份历史心所，再在支内预测行为，最后只把行为跨支融合。心所不做跨支融合。

原文格式：

```text
分析对象：陈默

事件1：…
事件2：…
```

单独事件文件里只能有一个事件标签。

### 概率

| 符号 | 含义 |
|------|------|
| **a** | 内心激活，归一后的支权重。融合心所和行为时用 a |
| **compat** | 行动贴合，模型给出的似然 |
| **L** | 只改旧支。`compat≥0.5` 时抬高，否则压低，然后 `p'∝p·L` |
| **扩支** | 旧支最大贴合低于 0.45，且支数少于 4 时，按行动开新支。新支不再用行动矫正 |

细节见 `docs/execution_flow.md`。

## 定义数据

| 文件 | 内容 |
|------|------|
| `definitions/normalized/xinsuo.json` | 40 心所 |
| `definitions/normalized/xinsuo_duizhi.json` | 40 对治 |
| `definitions/normalized/suoyuan_xiang_si.json` | 所缘 / 想 / 思 |
| `definitions/seeds/xinsuo_seeds.json` | 种子库 |
| `definitions/raw/*.docx` | 上述 JSON 的原文 |
| `definitions/schemas/` | JSON Schema |

种子库默认关闭（`SEED_LIBRARY_ENABLED`）。主路径是 40 心所全评，再用分支权重 a，不用画像筛心所。

## 跑一次评估

```bash
python3 -m mental_world_model.cli evaluate \
  --focus-person 陈默 \
  --file data/自建实验/纵向/陈默纵向7事件.txt \
  -o reports/out.json
```

Mock：

```bash
python3 -m mental_world_model.cli --mock evaluate \
  --text "分析对象：陈默\n事件1：散会后有人来问意见。"
```

不写 `-o` 时，结果落到 `reports/report_年月日_时分秒_毫秒.json`。`--verbose` 会附上 `pipeline_trace` 等调试字段。

纵向原文：

| 文件 | 人物 |
|------|------|
| `data/自建实验/纵向/陈默纵向7事件.txt` | 陈默，7 个事件 |
| `data/自建实验/纵向/陈默正向纵向6事件.txt` | 陈默，正向，6 个事件 |
| `data/自建实验/纵向/程野纵向7事件.txt` | 程野 |
| `data/自建实验/纵向/林夏纵向6事件.txt` | 林夏 |
| `data/自建实验/纵向/沈川纵向7事件.txt` | 沈川 |
| `data/自建实验/纵向/王磊纵向7事件.txt` | 王磊 |

`data/陈默事件（负面版）.txt` 和 `data/林可负面事件.txt` 不在目录里。README、双人闭环脚本和对应测试仍指向这两个路径，跑双人闭环前要自己补上原文，或改 `--chenmo-file` / `--linke-file`。

## 自建实验

都在 `mental-world-model` 目录下执行。

纵向，预测后落地，事件数由文件决定：

```bash
python3 scripts/experiments/chenmo_longitudinal_7.py --mock
python3 scripts/experiments/chenmo_longitudinal_7.py \
  -o reports/自建实验/纵向/chenmo_longitudinal_7
python3 scripts/experiments/chenmo_longitudinal_7.py \
  --event-file data/自建实验/纵向/程野纵向7事件.txt \
  -o reports/自建实验/纵向/chengye_longitudinal_7
```

续跑必须带同一个 `-o`，并加上 `--resume`。

纯文本历史的模型对照，不经过心所引擎：

```bash
python3 scripts/experiments/llm_baseline_longitudinal_rollout.py --mock
python3 scripts/experiments/llm_baseline_longitudinal_rollout.py \
  --event-file data/自建实验/纵向/程野纵向7事件.txt \
  -o reports/自建实验/纵向/llm_baseline_chengye_7
```

单独事件只预测，不写 H/M，不落地。每个 txt 一个事件，一条命令跑一个人：

```bash
python3 scripts/experiments/single_event_experiment_2.py \
  --event-file data/自建实验/单独事件/实验3苏宁.txt

python3 scripts/experiments/single_event_experiment_2_llm.py \
  --event-file data/自建实验/单独事件/实验3苏宁.txt
```

默认结果：

- 心智世界模型：`reports/自建实验/单独事件实验二/<文件名>/`
- 对照模型：`reports/自建实验/单独事件实验二/<文件名>/llm结果/`

单事件原文还有 `实验3林舟.txt`、`实验3许哲.txt`、`实验3陈溪.txt`。

双人闭环在模型外交替喂事件。默认读的两个负面原文目前缺失：

```bash
python3 scripts/experiments/dual_agent_closed_loop.py --mock \
  -o reports/自建实验/dual_loop
```

已有纵向结果在 `reports/自建实验/纵向/<人物>/` 下，每个目录里是 `mwm`、`gpt结果`、`gemini结果`、`llm结果`、`grok结果`。

## 公开基准

数据目录是 `mental-world-model` 的上一级 `公开数据集/`：

| 路径 | 内容 |
|------|------|
| `../公开数据集/DynToM.json` | DynToM |
| `../公开数据集/RecTOM/3_pred_rec.json` | RecToM 推荐者 |
| `../公开数据集/RecTOM/4_pred_seeker.json` | RecToM 寻求者 |
| `../公开数据集/SimpleToM/test.jsonl` | SimpleToM |
| `../公开数据集/CAREBench /third_person.json` | CAREBench 第三人称。目录名末尾有空格 |

心智世界模型在 DynToM / RecToM 上的默认输出：

```bash
python3 dyntom_adapter/__main__.py baseline \
  --input "../公开数据集/DynToM.json" \
  --output "reports/dyntom/dyntom mwm"

python3 rectom_adapter/__main__.py baseline \
  --input "../公开数据集/RecTOM/3_pred_rec.json" \
  --output "reports/rectom/mwm" \
  --index 0
```

RecToM 寻求者把 `--input` 换成 `4_pred_seeker.json`。`--index` 是题号。

五个推理框架走 `run_benchmark.py`，方法名是 `cot`、`simtom`、`timetom`、`picture`、`agentictom`：

```bash
python3 run_benchmark.py --dataset dyntom --method cot --limit 1
python3 run_benchmark.py --dataset rectom --method picture --limit 1
```

不写 `--data-dir` 时，加载器会到**代码文件夹的上一级**找 `公开数据集`。在本机目录里，数据其实在代码文件夹里面。跑这五个框架时要显式写上：

```bash
python3 run_benchmark.py \
  --dataset dyntom \
  --method cot \
  --data-dir "../公开数据集" \
  --limit 1
```

直接作答（DeepSeek、Gemini、Grok、GPT）用各自适配器的 `baseline`，输出目录是：

| 模型 | DynToM | RecToM |
|------|--------|--------|
| 心智世界模型 | `reports/dyntom/dyntom mwm` | `reports/rectom/mwm` |
| CoT | `reports/dyntom/cot` | `reports/rectom/cot` |
| SimToM | `reports/dyntom/simtom` | `reports/rectom/simtom` |
| TimeToM-style | `reports/dyntom/timetom` | `reports/rectom/timetom` |
| PICTURE | `reports/dyntom/picture` | `reports/rectom/picture` |
| Agentic-ToM | `reports/dyntom/agentictom` | `reports/rectom/agentictom` |
| DeepSeek | `reports/dyntom/dyntom_deepseek` | `reports/rectom/rectom_deepseek` |
| Gemini | `reports/dyntom/dyntom_gemini` | `reports/rectom/rectom_gemini` |
| Grok | `reports/dyntom/dyntom_grok` | `reports/rectom/rectom_grok` |
| GPT | `reports/dyntom/dyntom_gpt` | `reports/rectom/rectom_gpt` |

总表在 `reports/dyntom/summary.txt` 和 `reports/rectom/summary.txt`。心智世界模型跑完会改自己那一段；其他方法不会自动改总表。完整命令见上一级的 `代跑命令说明.docx`，生成脚本是 `scripts/docgen/build_benchmark_run_docx.py`。

SimpleToM：

```bash
python3 simpletom_adapter/__main__.py --help
```

OpenToM 样例在 `data/fixtures/benchmark_samples/opentom_demo.jsonl`。论文规模的 300 条要先有官方 `opentom.json`，再抽样：

```bash
python3 scripts/opentom/build_opentom_benchmark.py \
  --input data/raw/opentom.json \
  --output data/fixtures/benchmark_samples/opentom_300.jsonl \
  --sample-size 300 \
  --seed 42
```

批量跑心所分数用 `mwm-test`，输出是每条的 40 心所 `scores_final`：

```bash
mwm-test --mock run \
  --input data/fixtures/benchmark_samples/opentom_demo.jsonl \
  --limit 1
```

说明在 `docs/opentom_benchmark.md` 和 `docs/test_adapter.md`。

## 报告里有什么

默认是 `to_event_report_dict`，按事件给出中间结果。

| 字段 | 内容 |
|------|------|
| `focus_person` | 分析对象 |
| `stimulus_action_split` | L0 刺激 / 行动 |
| `objective_event` | L1 客观刺激 |
| `self_grasping` / `dharma_grasping` | L2 二执。扩支成功时含行动条件下的新条目 |
| `grasping_combos` | L2.5 组合 |
| `subjective_alambanas` | L3：prior、posterior、compat、L、是否 `action_informed` |
| `action_evidence` | 行动和旧支似然 |
| `branches[]` | 支权重 a、所缘、想、思、非零心所、支内行为 |
| `aggregated_behaviors` | 跨支行为分布，只在预测事件出现 |
| `mental_factors_event_nonzero` | 本事件跨支期望，落库用，未掺历史 |
| `mental_factors_final_nonzero` | 前序事件掺历史之后的跨支心所 |
| `mental_report_mode` | 预测事件为 `per_branch_only` |
| `continuity` | H / M 摘要 |

字段说明见 `docs/report.md`。

## HTTP 接口

```bash
python3 -m mental_world_model.cli serve --port 8765
```

`POST /v1/evaluate`

```json
{
  "yuanwen": "分析对象：陈默\n事件1：…",
  "focus_person": "陈默",
  "explain": "none"
}
```

## 代码怎么分层

```text
definitions/          心所、对治、所缘想思
        ↓
engine/
  perception/         L0–L3，含行动扩支
  cognition/          贴合、分支想/思、概率、行为
  continuity/         H 与 M
  scoring/            心所评分与报告
  extraction/         分支关闭时的旧提取
        ↓
MentalWorldModel.evaluate_from_yuanwen
        ↓
CLI / REST
```

旁边还有：定义的加载、解析、归一、校验、索引；图；规则推理；预测；查询；对治消解；种子更新；`llm/` 里的厂商无关客户端。`core/` 里实际有代码的是 `xinsuo`（心所）和 `relations`（关系图）。

运行中的连续性写在 `data/runtime/continuity_state.json`。纵向脚本会把本次连续性指到自己的输出目录，避免覆盖这份默认文件。

## 测试

```bash
python3 -m pytest tests/ -q
```

带真实接口的测试标了 `integration`。有两组测试现在会失败，原因在数据不在默认路径，不是整理目录时改坏的：

- 双人闭环测试要 `data/陈默事件（负面版）.txt` 和 `data/林可负面事件.txt`
- DynToM / RecToM / SimpleToM 的部分测试写死了代码文件夹**外面**的 `公开数据集/`。文件在代码文件夹里面

`tests/test_summary_txt.py` 里有两条对照的是早期题量（DynToM 142、RecToM 推荐者 20）。现在 `reports/` 里的结果更多，这两条也会失败。

## 文档

| 文件 | 内容 |
|------|------|
| `docs/execution_flow.md` | 执行步骤 |
| `docs/architecture.md` | 模块分层 |
| `docs/data_flow.md` | 数据流 |
| `docs/concepts.md` | 概念 |
| `docs/report.md` | 报告字段 |
| `docs/llm_adapter.md` | 模型适配 |
| `docs/architecture_principles.md` | 原理长文 |
| `docs/roadmap.md` | 已完成和未做的部分 |
| `docs/test_adapter.md` | `mwm-test` |
| `docs/opentom_benchmark.md` | OpenToM 300 条 |
| 上一级 `模型最新架构.docx` | 架构、公式、权重。由 `scripts/docgen/build_architecture_docx.py` 生成 |
| 上一级 `代跑命令说明.docx` | DynToM / RecToM 代跑 |
| 上一级 `实验原始数据.docx` | 实验原文说明 |

`docs/模型架构与原理.docx` 在 README 旧索引里出现过，仓库里没有这个文件。对应内容在上一级的 `模型最新架构.docx`。
