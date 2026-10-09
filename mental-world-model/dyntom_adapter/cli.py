"""DynToM 适配层命令行。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.utils.atomic_io import atomic_write_json

from dyntom_adapter.loader import iter_trials
from dyntom_adapter.rollout import continuity_enabled, load_model
from dyntom_adapter.runner import (
    TrialExistsError,
    audit_saved_answers,
    needs_model,
    run_baseline_trial,
    run_trial,
)
from dyntom_adapter.score import load_answer_rows, summarize
from dyntom_adapter.summary_txt import mwm_root, update_summary

load_project_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_definitions() -> Path:
    return _project_root() / "definitions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dyntom_adapter",
        description=(
            "用 MWM 逐幕阅读 DynToM，再把冻结状态映射成选项。"
            "建模阶段不读标注、标准答案和选项。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="逐幕更新 MWM，投影后答题")
    _add_input_args(
        run,
        default_output=str(_project_root() / "reports" / "dyntom" / "dyntom mwm"),
    )
    run.add_argument(
        "--definitions",
        default=str(_default_definitions()),
        help="心所定义目录",
    )
    run.add_argument("--mock", action="store_true", help="MWM 用 Mock LLM；投影和选题使用占位结果")
    run.add_argument("--resume", action="store_true", help="沿用已有 state 与 answers.jsonl")
    run.add_argument("--limit", type=int, help="最多跑几个 trial")
    run.add_argument("--question-limit", type=int, help="每个 trial 最多答几题")
    run.add_argument("--trial", action="append", help="只跑这些 trial id，可重复")
    run.add_argument(
        "--no-control",
        action="store_true",
        help="不做「有题目和选项、但没有心理状态」的对照",
    )
    run.set_defaults(func=cmd_run)

    baseline = sub.add_parser("baseline", help="LLM 直接看故事和选项，不经过 MWM")
    _add_input_args(baseline)
    baseline.add_argument("--mock", action="store_true")
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--limit", type=int)
    baseline.add_argument("--question-limit", type=int)
    baseline.add_argument("--trial", action="append")
    baseline.set_defaults(func=cmd_baseline)

    score = sub.add_parser("score", help="汇总 answers.jsonl")
    score.add_argument("--input", required=True, help="answers.jsonl 或结果目录")
    score.add_argument("--output", help="写入 summary JSON")
    score.set_defaults(func=cmd_score)

    audit = sub.add_parser("audit", help="只核对选题判错的题：MWM 与标准是同一件事才改为对")
    audit.add_argument("--input", required=True, help="trial 目录，或其中的 answers.jsonl")
    audit.add_argument("--mock", action="store_true", help="不调用模型，错题全部保持错")
    audit.set_defaults(func=cmd_audit)
    return parser


def _add_input_args(parser: argparse.ArgumentParser, *, default_output: str = "") -> None:
    parser.add_argument(
        "--input",
        required=True,
        help="DynToM 的 data/script/data 目录、单个 trial 目录，或合并 JSON",
    )
    if default_output:
        parser.add_argument(
            "--output",
            default=default_output,
            help="结果目录。每个 trial 一个子目录。默认 reports/dyntom/dyntom mwm",
        )
    else:
        parser.add_argument(
            "--output",
            required=True,
            help="结果目录。每个 trial 一个子目录。",
        )


def cmd_run(args: argparse.Namespace) -> int:
    _warn_temperature()
    if args.mock:
        print(
            "mock：MWM 走 Mock LLM；选题不打分，准确率不能当结果。",
            flush=True,
        )
    trials = _select_trials(args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = create_llm_client(use_mock=args.mock)
    model = None
    failed = 0
    for index, (story, questions) in enumerate(trials, start=1):
        print(f"[{index}/{len(trials)}] {story.trial_id}", flush=True)
        if model is None and needs_model(out_dir, story.trial_id, len(story.scenarios)):
            model = load_model(Path(args.definitions), out_dir / "_runtime")
            if not continuity_enabled():
                print("警告：连续性开关是关的，后续 scenario 不会继承 H/M。", flush=True)
        try:
            run_trial(
                story,
                questions,
                model=model,
                llm=llm,
                out_dir=out_dir,
                resume=args.resume,
                question_limit=args.question_limit,
                run_control=not args.no_control,
                placeholder_adapter=args.mock,
            )
        except (TrialExistsError, RuntimeError, FileNotFoundError, ValueError) as error:
            failed += 1
            print(f"  失败：{error}", flush=True)
    _refresh_summary(out_dir)
    print(f"完成。失败 trial={failed}", flush=True)
    return 1 if failed else 0


def cmd_baseline(args: argparse.Namespace) -> int:
    _warn_temperature()
    trials = _select_trials(args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = create_llm_client(use_mock=args.mock)
    failed = 0
    for index, (story, questions) in enumerate(trials, start=1):
        print(f"[baseline {index}/{len(trials)}] {story.trial_id}", flush=True)
        try:
            run_baseline_trial(
                story,
                questions,
                llm=llm,
                out_dir=out_dir,
                resume=args.resume,
                question_limit=args.question_limit,
            )
        except (TrialExistsError, RuntimeError, FileNotFoundError, ValueError) as error:
            failed += 1
            print(f"  失败：{error}", flush=True)
    rows = []
    for file in sorted(out_dir.rglob("baseline.jsonl")):
        rows.extend(load_answer_rows(file))
    summary = summarize(rows)
    atomic_write_json(out_dir / "baseline_summary.json", summary)
    print(f"baseline accuracy={summary['total']['accuracy']} 失败 trial={failed}", flush=True)
    return 1 if failed else 0


def cmd_audit(args: argparse.Namespace) -> int:
    _warn_temperature()
    llm = _ScriptedWrongAudit() if args.mock else create_llm_client()
    summary = audit_saved_answers(Path(args.input), llm, placeholder=args.mock)
    total = summary["total"]
    print(
        f"核对 {summary['checked']} 道错题，改对 {summary['rescued']} 道。"
        f"最终 {summary['correct']}/{total}",
        flush=True,
    )
    _refresh_summary(mwm_root(Path(args.input)))
    return 0


class _ScriptedWrongAudit:
    def complete(self, messages, schema=None):
        return {"same": False, "reason": "占位核对，不改判。"}


def cmd_score(args: argparse.Namespace) -> int:
    rows = load_answer_rows(Path(args.input))
    summary = summarize(rows)
    text = __import__("json").dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _select_trials(args: argparse.Namespace) -> list:
    chosen = set(args.trial or [])
    trials = []
    for story, questions in iter_trials(Path(args.input)):
        if chosen and story.trial_id not in chosen:
            continue
        trials.append((story, questions))
        if args.limit is not None and len(trials) >= args.limit:
            break
    if not trials:
        raise SystemExit("没有匹配的 trial。")
    return trials


def _refresh_summary(out_dir: Path) -> None:
    written = update_summary(out_dir)
    if written is None:
        print("还没有可汇总的答题记录，不改 summary.txt", flush=True)
        return
    print(f"已更新 {written}", flush=True)


def _warn_temperature() -> None:
    raw = os.environ.get("MWM_LLM_TEMPERATURE", "0").strip()
    if raw not in {"", "0", "0.0"}:
        print(
            f"警告：Adapter 需要 temperature=0，当前 MWM_LLM_TEMPERATURE={raw}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
