"""批量跑分 CLI：输出 40 心所 scores_final。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.model import MentalWorldModel

from mwm_test_adapter.runner import TestAdapterRunner, write_results_jsonl

load_project_dotenv()


def _default_definitions_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "definitions"


def cmd_run(args: argparse.Namespace) -> int:
    model = MentalWorldModel.load(Path(args.definitions))
    llm = create_llm_client(use_mock=args.mock)
    runner = TestAdapterRunner(
        model,
        llm,
        update_seeds=args.update_seeds,
    )
    summary = runner.run_file(Path(args.input), limit=args.limit)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".jsonl":
            write_results_jsonl(summary.results, output_path)
        else:
            output_path.write_text(
                json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"完成：{summary.succeeded}/{summary.total}（失败 {summary.failed}）")
    if not args.output and not args.summary:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mwm-test",
        description="MWM 批量跑分：输出 40 心所 scores_final",
    )
    parser.add_argument(
        "--definitions",
        default=str(_default_definitions_dir()),
        help="definitions 目录路径",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 MockLlmClient（不调用真实 API）",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="在 JSONL 输入上批量跑分")
    run.add_argument("--input", required=True, help="输入 JSONL（每行含 story、question）")
    run.add_argument("--output", "-o", help="输出 .jsonl 或 summary JSON")
    run.add_argument("--summary", help="另存 summary JSON")
    run.add_argument("--limit", type=int, help="最多处理条数")
    run.add_argument(
        "--update-seeds",
        action="store_true",
        help="交互后更新运行时种子库（默认关闭）",
    )
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
