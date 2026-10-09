"""python run_benchmark.py --dataset dyntom --method cot --limit 1"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mental_world_model.llm.config import LlmConfig
from mental_world_model.llm.env_file import load_project_dotenv

from inference_baselines.items import EVAL_RECTOM_END, EVAL_TRIALS, load_split
from inference_baselines.methods import METHODS
from inference_baselines.runner import result_dir, run_items
from inference_baselines.text_client import TextChatClient

load_project_dotenv()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="用 DeepSeek V4 Flash 跑 CoT、SimToM、TimeToM-style、PICTURE 和 Agentic-ToM。",
    )
    parser.add_argument("--dataset", required=True, choices=("dyntom", "rectom"))
    parser.add_argument("--method", required=True, choices=tuple(METHODS))
    parser.add_argument("--resume", action="store_true", help="重试上次失败的题，已完成的仍跳过")
    parser.add_argument("--start-index", type=int, default=0, help="从第几条开始，从 0 计。已完成的文件仍会跳过")
    parser.add_argument("--limit", type=int, help="最多跑几条。不写则跑完所选范围")
    parser.add_argument(
        "--trials",
        default="",
        help=f"DynToM 的 trial，逗号分隔。不写则只跑 {', '.join(EVAL_TRIALS)}",
    )
    parser.add_argument(
        "--rectom-end",
        type=int,
        default=EVAL_RECTOM_END,
        help=f"RecToM 每个文件跑到这个下标（不含）。默认 {EVAL_RECTOM_END}，即之前的 0–{EVAL_RECTOM_END - 1}",
    )
    parser.add_argument(
        "--output",
        default=str(project_root() / "reports"),
        help="结果根目录。实际写入 reports/<dataset>/<method>/",
    )
    parser.add_argument("--data-dir", default="", help="公开数据集目录。默认用项目旁的公开数据集")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = os.environ.get("MWM_LLM_TEMPERATURE", "0").strip()
    if raw not in {"", "0", "0.0"}:
        print(f"警告：环境温度是 {raw}。这次请求仍使用 temperature=0", flush=True)
    data_dir = Path(args.data_dir) if args.data_dir else None
    if args.rectom_end < 1:
        raise SystemExit("--rectom-end 至少为 1")
    trials = tuple(part.strip() for part in args.trials.split(",") if part.strip())
    items = load_split(
        args.dataset,
        data_dir=data_dir,
        trials=trials or None,
        rectom_end=args.rectom_end,
    )
    start = max(0, args.start_index)
    chosen = items[start:]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit 至少为 1")
        chosen = chosen[: args.limit]
    if not chosen:
        raise SystemExit("这个范围内没有条目。")
    out_dir = result_dir(Path(args.output), args.dataset, args.method)
    print(
        f"{args.dataset}/{args.method} 所选范围共 {len(items)} 条，这次从 {start} 起跑 {len(chosen)} 条",
        flush=True,
    )
    print(f"第一条：{chosen[0].id}", flush=True)
    print(f"写入：{out_dir}", flush=True)
    client = TextChatClient(LlmConfig.from_env())
    failed = run_items(
        chosen,
        method=args.method,
        client=client,
        out_dir=out_dir,
        resume=args.resume,
    )
    return 1 if failed else 0
