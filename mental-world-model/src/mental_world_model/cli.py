"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from mental_world_model.llm.env_file import get_project_root, load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.model import MentalWorldModel

load_project_dotenv()


def _default_definitions_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "definitions"


def _allocate_report_path(reports_dir: Path) -> Path:
    """在 reports 目录下分配唯一报告文件名。"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    milliseconds = now.strftime("%f")[:3]
    path = reports_dir / f"report_{stamp}_{milliseconds}.json"
    suffix = 0
    while path.exists():
        suffix += 1
        path = reports_dir / f"report_{stamp}_{milliseconds}_{suffix}.json"
    return path


def _display_report_path(path: Path, project_root: Path) -> str:
    """优先显示相对项目根的路径，便于终端阅读。"""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _print_report_saved_banner(display_path: str) -> None:
    separator = "=" * 41
    print(separator)
    print("评估完成")
    print("结果已保存：")
    print(display_path)
    print(separator)


def _load_yuanwen(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise SystemExit("请提供 --text 或 --file")


def cmd_evaluate(args: argparse.Namespace) -> int:
    definitions_dir = Path(args.definitions)
    model = MentalWorldModel.load(definitions_dir)
    llm = create_llm_client(use_mock=args.mock)
    yuanwen = _load_yuanwen(args)
    focus_person = str(getattr(args, "focus_person", "") or "").strip() or None
    if not focus_person:
        from mental_world_model.engine.perception.focus import parse_focus_person_from_text

        parsed = parse_focus_person_from_text(yuanwen)
        if parsed:
            focus_person = parsed
            print(f"已从原文解析分析对象：{focus_person}")
        else:
            print(
                "警告：未指定 --focus-person，原文也未标明「分析对象：…」；"
                "我执/法执/所缘可能串人。建议加上 --focus-person 或在原文开头写明。"
            )
    result = model.evaluate_from_yuanwen(
        yuanwen, llm, focus_person=focus_person
    )

    if args.explain == "nonzero":
        explanation = result.explain_all_scores(only_nonzero=True)
    elif args.explain == "all":
        explanation = result.explain_all_scores(only_nonzero=False)
    else:
        explanation = None

    if args.verbose:
        output = result.to_report_dict(
            include_explanations=args.explain in ("all", "nonzero"),
            verbose=True,
        )
        if explanation is not None:
            output["selected_explanation"] = explanation
    else:
        output = result.to_event_report_dict()

    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text, encoding="utf-8")
    else:
        project_root = get_project_root()
        output_path = _allocate_report_path(project_root / "reports")
        output_path.write_text(text, encoding="utf-8")
        _print_report_saved_banner(_display_report_path(output_path, project_root))
    if result.llm_usage is not None:
        usage = result.llm_usage
        print(
            f"LLM 调用：{usage.total_calls} 次，"
            f"API 累计 {usage.total_duration_ms:.0f} ms，"
            f"总耗时 {usage.wall_clock_ms:.0f} ms"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from mental_world_model.api.server import run_server

    run_server(
        host=args.host,
        port=args.port,
        definitions_dir=Path(args.definitions),
        use_mock=args.mock,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mwm", description="心智世界模型 CLI")
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

    evaluate = subparsers.add_parser("evaluate", help="从原文评估心所")
    evaluate.add_argument("--text", help="原文文本")
    evaluate.add_argument("--file", help="原文文件路径")
    evaluate.add_argument(
        "--focus-person",
        default="",
        help="分析对象姓名（强烈建议）。未传时尝试从原文「分析对象：某某」解析",
    )
    evaluate.add_argument(
        "--explain",
        choices=["none", "all", "nonzero"],
        default="none",
        help="是否输出合并解释文本",
    )
    evaluate.add_argument(
        "--verbose",
        action="store_true",
        help="输出完整调试报告（含重复明细字段）；默认仅 pipeline_trace 精简报告",
    )
    evaluate.add_argument(
        "--output",
        "-o",
        help="结果 JSON 输出路径（未指定时自动保存到 reports/report_YYYYMMDD_HHMMSS_mmm.json）",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    serve = subparsers.add_parser("serve", help="启动 REST API 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
