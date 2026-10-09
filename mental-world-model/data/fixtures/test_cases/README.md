# Golden Test Cases

存放固定原文、Mock 配置与断言，用于回归测试。

## 文件格式

```json
{
  "id": "case_01_xin",
  "description": "说明",
  "yuanwen": "原文",
  "mock": {
    "extraction": { "suoyuan": {...}, "xiang": {...}, "si": {...} },
    "elevated_scores": { "信": 0.85, "不信": 0.4 },
    "stage_scores": { "信": { "ye": 0.7, "xiandai": 0.7, "xiandai_confidence": 0.8 } },
    "reconcile": { "信|不信": "信" },
    "default_score": 0.0
  },
  "assertions": {
    "nonzero_final": ["信"],
    "zero_final": ["不信"],
    "skipped": ["恨"],
    "extracted_suoyuan_confidence": 1.0
  }
}
```

`mock.elevated_scores` 驱动所缘境/生起念头阶段分；`stage_scores` 可覆盖业与现代心理学列。
`mock.reconcile` 的 key 为 `心所A|心所B`（顺序无关），value 为消解赢家。

## 现有用例

| 文件 | 对治/逻辑 |
|------|-----------|
| `case_01_xin.json` | 信 vs 不信 |
| `case_02_jingjin.json` | 精进 vs 懈怠 |
| `case_03_wu_tan.json` | 无贪 vs 贪 |
| `case_04_yi.json` | 疑 vs 信 |
| `case_05_can.json` | 惭 vs 无惭 |
| `case_06_fen_chain.json` | 忿=0 → 恨/恼依序跳过 |

## 运行

```bash
python3 -m pytest tests/test_golden_cases.py -q
```

真实 LLM 集成测试（需配置 BASE_URL 与 MODEL）：

```bash
export MWM_LLM_BASE_URL=https://your-llm-host/v1
export MWM_LLM_MODEL=your-model
export MWM_LLM_API_KEY=your-key   # 若需要
python3 -m pytest tests/test_llm_adapter.py -m integration -q
```
