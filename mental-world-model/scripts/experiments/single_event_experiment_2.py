#!/usr/bin/env python3
"""单独事件实验二：每个 txt 一个事件，一次命令跑一次 MWM。

只预测：输出各支想 + 跨支行为分布，不落地、不写 H/M。

用法（在 mental-world-model 目录）：
    python3 scripts/experiments/single_event_experiment_2.py --event-file data/自建实验/单独事件/实验3苏宁.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mental_world_model.engine.cognition.predicted_action import top_aggregated_action
from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.llm.prefixed_client import StagePrefixLlmClient
from mental_world_model.llm.tracing_client import ensure_tracing_client
from mental_world_model.model import MentalWorldModel
from mental_world_model.utils.atomic_io import atomic_write_json, atomic_write_text

load_project_dotenv()

EXPERIMENT = "单独事件实验二"
_EVENT_RE = re.compile(
    r"(?m)^\s*事件\s*([0-9]+|[一二三四五六七八九十百千]+|[A-Za-z]+)\s*[：:]"
)
_FOCUS_RE = re.compile(r"(?m)^\s*分析对象\s*[：:]\s*(.+?)\s*$")


def parse_event_file(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    focus_match = _FOCUS_RE.search(raw)
    focus = focus_match.group(1).strip() if focus_match else ""
    if not focus:
        raise ValueError(f"未找到分析对象：{path}")
    matches = list(_EVENT_RE.finditer(raw))
    if not matches:
        raise ValueError(f"未找到事件标签：{path}")
    if len(matches) != 1:
        raise ValueError(f"{path} 应只有 1 个事件，实际 {len(matches)} 个")
    body = raw[matches[0].end() :].strip()
    if not body:
        raise ValueError(f"事件正文为空：{path}")
    return focus, body


def collect_xiangs(result) -> list[dict]:
    rows = []
    for branch in getattr(result, "cognition_branches", None) or []:
        xiang = getattr(branch, "xiang", None)
        text = str(getattr(xiang, "text", "") or "").strip() if xiang else ""
        alambana = getattr(branch, "alambana", None)
        rows.append(
            {
                "branch_id": getattr(branch, "id", ""),
                "activation": float(
                    getattr(branch, "activation_probability", 0.0)
                    or getattr(branch, "branch_effective_probability", 0.0)
                    or 0.0
                ),
                "alambana": (
                    str(getattr(alambana, "text", "") or "").strip() if alambana else ""
                ),
                "xiang": text,
            }
        )
    rows.sort(key=lambda r: r["activation"], reverse=True)
    return rows


def render_txt(
    *,
    experiment: str,
    agent: str,
    stimulus: str,
    xiangs: list[dict],
    top_k: list[dict],
) -> str:
    lines = [
        f"实验：{experiment}",
        f"分析对象：{agent}",
        "",
        "原文：",
        stimulus,
        "",
        "想（按支激活降序）：",
    ]
    if not xiangs:
        lines.append("（无）")
    else:
        for i, row in enumerate(xiangs, start=1):
            lines.append(f"{i}. [{row['branch_id']}] p={row['activation']:.4f}")
            lines.append(row["xiang"] or "（空）")
            lines.append("")
    lines.append("预测行动：")
    if not top_k:
        lines.append("（无行为预测）")
    else:
        for i, item in enumerate(top_k, start=1):
            lines.append(
                f"{i}. {item['text']}  p={float(item['probability']):.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--definitions", type=Path, default=_ROOT / "definitions")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    event_file = args.event_file if args.event_file.is_absolute() else _ROOT / args.event_file
    agent, stimulus = parse_event_file(event_file)
    out_dir = args.output_dir or (
        _ROOT / "reports" / "自建实验" / EXPERIMENT / event_file.stem
    )
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = StagePrefixLlmClient(
        ensure_tracing_client(create_llm_client(use_mock=args.mock)),
        "single_event_2",
    )
    model = MentalWorldModel.load(args.definitions)
    bundle = model.evaluate_text_with_cognition_branches(
        stimulus,
        llm,
        role_context=f"分析对象：{agent}",
        update_seeds=False,
        update_continuity=False,
        use_action_feedback=False,
        aggregate_behaviors=True,
        predict_branch_behaviors=True,
        focus_person=agent,
    )
    result = bundle.evaluation_result
    result.focus_person = agent
    _, _, top_k = top_aggregated_action(result)
    xiangs = collect_xiangs(result)

    payload = result.to_event_report_dict()
    payload["_experiment"] = {
        "name": EXPERIMENT,
        "event_file": str(event_file),
        "agent": agent,
        "stimulus": stimulus,
        "xiangs": xiangs,
        "predicted_actions": top_k,
    }
    atomic_write_json(out_dir / "result.json", payload)
    atomic_write_text(
        out_dir / "result.txt",
        render_txt(
            experiment=EXPERIMENT,
            agent=agent,
            stimulus=stimulus,
            xiangs=xiangs,
            top_k=top_k,
        ),
    )
    print(f"{EXPERIMENT} 完成：{out_dir}")
    if top_k:
        print(f"Top-1：{top_k[0]['text']}  p={float(top_k[0]['probability']):.4f}")
    else:
        print("无行为预测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
