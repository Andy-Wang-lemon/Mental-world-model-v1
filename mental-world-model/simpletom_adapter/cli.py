"""SimpleToM 行为预测适配层命令行。每次只跑一道题。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client

from simpletom_adapter.loader import load_one
from simpletom_adapter.rollout import load_model
from simpletom_adapter.runner import ItemExistsError, run_mwm_item

load_project_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simpletom_adapter",
        description="用 MWM 回答 SimpleToM 行为预测的一道题。不会一次跑完整份数据。",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="MWM 读故事，再按下一步行动给每个选项加总概率")
    run.add_argument("--input", required=True, help="SimpleToM 的 jsonl，例如 test.jsonl")
    run.add_argument("--output", required=True, help="结果目录。这一题单独一个子目录。")
    run.add_argument("--index", type=int, default=0, help="只跑这一题，从 0 开始。默认 0。")
    run.add_argument("--force", action="store_true", help="覆盖这一题已有结果")
    run.add_argument(
        "--definitions",
        default=str(_project_root() / "definitions"),
        help="心所定义目录",
    )
    run.set_defaults(func=cmd_run)
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    raw = os.environ.get("MWM_LLM_TEMPERATURE", "0").strip()
    if raw not in {"", "0", "0.0"}:
        print(f"警告：当前 MWM_LLM_TEMPERATURE={raw}", flush=True)
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
    mark = "对" if row.get("correct") else "错"
    if row.get("selector_correct") is False and row.get("correct") is True:
        mark = "选题不在标准答案里，核对后改对"
    scores = row.get("option_scores") or {}
    chosen = str(row.get("predicted_answer") or "")
    score_text = f" {scores[chosen]}" if chosen in scores else ""
    print(
        f"[simpletom] #{row.get('index')} {row.get('item_id')} → {chosen or '-'}{score_text} {mark}",
        flush=True,
    )
    print(f"标准：{row.get('gold_output')}", flush=True)
    print(out_dir / path.stem / f"{item.index:04d}_{item.item_id}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
