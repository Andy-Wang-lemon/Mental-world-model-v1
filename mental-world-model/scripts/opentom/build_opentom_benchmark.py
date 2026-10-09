#!/usr/bin/env python3
"""构建 OpenToM 论文级 benchmark 子集（默认 300 条分层抽样）。

用法：
  python scripts/opentom/build_opentom_benchmark.py \\
    --input /path/to/opentom.json \\
    --output data/fixtures/benchmark_samples/opentom_300.jsonl

跑分：
  mwm-test run \\
    --input data/fixtures/benchmark_samples/opentom_300.jsonl \\
    --output reports/opentom_300_results.jsonl \\
    --summary reports/opentom_300_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "opentom"))

from preprocess_opentom import convert_opentom  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 OpenToM 300 条分层 benchmark")
    parser.add_argument("--input", required=True, help="官方 opentom.json 路径")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "fixtures" / "benchmark_samples" / "opentom_300.jsonl"),
        help="输出 JSONL",
    )
    parser.add_argument("--sample-size", type=int, default=300, help="抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    count = convert_opentom(
        Path(args.input),
        Path(args.output),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"OpenToM benchmark：{count} 条 → {args.output}")
    print("下一步：mwm-test run --input", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
