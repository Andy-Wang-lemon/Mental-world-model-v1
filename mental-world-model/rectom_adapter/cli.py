"""RecTOM 适配层命令行。每次只跑一道题。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client

from rectom_adapter.loader import load_one
from rectom_adapter.rollout import load_model
from rectom_adapter.runner import ItemExistsError, run_baseline_item, run_mwm_item
from rectom_adapter.summary_txt import update_summary

load_project_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rectom_adapter",
        description="用 MWM 或 LLM 回答 RecTOM 的一道题。不会一次跑完整份数据。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="MWM 读对话，再给每个策略选项打分")
    _add_common(
        run,
        default_output=str(_project_root() / "reports" / "rectom" / "mwm"),
    )
    run.add_argument(
        "--definitions",
        default=str(_project_root() / "definitions"),
        help="心所定义目录",
    )
    run.set_defaults(func=cmd_run)

    baseline = sub.add_parser("baseline", help="LLM 直接看对话和选项，不经过 MWM")
    _add_common(baseline)
    baseline.set_defaults(func=cmd_baseline)
    return parser


def _add_common(parser: argparse.ArgumentParser, *, default_output: str = "") -> None:
    parser.add_argument("--input", required=True, help="一份 RecTOM JSON，例如 3_pred_rec.json")
    if default_output:
        parser.add_argument(
            "--output",
            default=default_output,
            help="结果目录。这一题单独一个子目录。默认 reports/rectom/mwm",
        )
    else:
        parser.add_argument("--output", required=True, help="结果目录。这一题单独一个子目录。")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="只跑这一题，从 0 开始。默认 0。",
    )
    parser.add_argument("--force", action="store_true", help="覆盖这一题已有结果")


def cmd_run(args: argparse.Namespace) -> int:
    _warn_temperature()
    path = Path(args.input)
    item = load_one(path, args.index)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = create_llm_client()
    model = load_model(Path(args.definitions), out_dir / "_runtime")
    try:
        row = run_mwm_item(
            item,
            model=model,
            llm=llm,
            out_dir=out_dir,
            dataset_name=path.stem,
            force=args.force,
        )
    except ItemExistsError as error:
        print(error, flush=True)
        return 1
    _print_row(row)
    written = update_summary(out_dir)
    if written is None:
        print("还没有可汇总的答题记录，不改 summary.txt", flush=True)
    else:
        print(f"已更新 {written}", flush=True)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    _warn_temperature()
    path = Path(args.input)
    item = load_one(path, args.index)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = create_llm_client()
    try:
        row = run_baseline_item(
            item,
            llm=llm,
            out_dir=out_dir,
            dataset_name=path.stem,
            force=args.force,
        )
    except ItemExistsError as error:
        print(error, flush=True)
        return 1
    _print_row(row)
    return 0


def _print_row(row: dict) -> None:
    mark = "对" if row.get("correct") else "错"
    if row.get("selector_correct") is False and row.get("correct") is True:
        mark = "选题不在标准答案里，核对后改对"
    scores = row.get("option_scores") or {}
    chosen = str(row.get("predicted_answer") or "")
    score_text = f" {scores[chosen]}" if chosen in scores else ""
    print(
        f"[{row.get('dataset')}] #{row.get('index')} {row.get('item_id')} "
        f"→ {chosen or '-'}{score_text} {mark}",
        flush=True,
    )
    print(f"标准：{row.get('gold_output')}", flush=True)
    print(f"结果目录：{row.get('dataset')} 下的这一题", flush=True)


def _warn_temperature() -> None:
    raw = os.environ.get("MWM_LLM_TEMPERATURE", "0").strip()
    if raw not in {"", "0", "0.0"}:
        print(f"警告：当前 MWM_LLM_TEMPERATURE={raw}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
