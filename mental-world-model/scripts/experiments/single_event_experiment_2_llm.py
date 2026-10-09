#!/usr/bin/env python3
"""单独事件实验二 · LLM 对照：每个 txt 一个事件，一次命令跑一次。

不加载 MentalWorldModel，无历史、无 H/M。
任务：预测分析对象此刻最可能的想法（不是可观察行动）。

用法（在 mental-world-model 目录）：
    python3 scripts/experiments/single_event_experiment_2_llm.py --event-file data/自建实验/单独事件/实验3苏宁.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mental_world_model.llm.config import LlmConfig
from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.utils.atomic_io import atomic_write_json, atomic_write_text

load_project_dotenv()

EXPERIMENT = "单独事件实验二"
PROTOCOL = "llm_single_event_thought"
_EVENT_RE = re.compile(
    r"(?m)^\s*事件\s*([0-9]+|[一二三四五六七八九十百千]+|[A-Za-z]+)\s*[：:]"
)
_FOCUS_RE = re.compile(r"(?m)^\s*分析对象\s*[：:]\s*(.+?)\s*$")

THOUGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "probability": {"type": "number"},
                },
                "required": ["text", "probability"],
            },
        }
    },
    "required": ["thoughts"],
}


class ThoughtAbort(RuntimeError):
    """模型没给出可用想法。"""


class ThoughtMockLlmClient:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del messages, schema
        return {
            "thoughts": [
                {"text": "mock想法甲", "probability": 0.5},
                {"text": "mock想法乙", "probability": 0.3},
                {"text": "mock想法丙", "probability": 0.2},
            ]
        }


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


def build_prompt(event: str, *, agent: str) -> str:
    return (
        f"分析对象：{agent}\n\n"
        "当前事件：\n\n"
        f"{(event or '').strip()}\n\n"
        f"请预测{agent}此刻最可能的想法。\n"
        "\n"
        "要求：\n"
        "- 输出最多3个不同的想法\n"
        "- 每个想法给出概率\n"
        "- 概率总和为1\n"
        "- 只写此刻的想法，不解释原因，不预测行为"
    )


def normalize_thoughts(raw: Any) -> list[dict[str, Any]]:
    payload = raw if isinstance(raw, dict) else {}
    items = payload.get("thoughts")
    if not isinstance(items, list):
        raise ThoughtAbort("模型未返回 thoughts 列表")
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            prob = float(item.get("probability") or 0.0)
        except (TypeError, ValueError):
            prob = 0.0
        if prob < 0:
            prob = 0.0
        cleaned.append({"text": text, "probability": prob})
    if not cleaned:
        raise ThoughtAbort("无可用想法")
    cleaned.sort(key=lambda row: float(row["probability"]), reverse=True)
    cleaned = cleaned[:3]
    total = sum(float(row["probability"]) for row in cleaned)
    if total <= 0:
        raise ThoughtAbort("概率全为 0")
    for row in cleaned:
        row["probability"] = float(row["probability"]) / total
    return cleaned


def llm_run_meta(*, use_mock: bool) -> dict[str, Any]:
    if use_mock:
        return {"client": "mock", "model": "mock"}
    cfg = LlmConfig.from_env_optional()
    if cfg is None:
        return {
            "client": "http",
            "model": os.environ.get("MWM_LLM_MODEL", "").strip(),
        }
    return {
        "client": "http",
        "model": cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "seed": cfg.seed,
    }


def render_txt(
    *,
    experiment: str,
    agent: str,
    stimulus: str,
    thoughts: list[dict[str, Any]],
) -> str:
    lines = [
        f"实验：{experiment}",
        f"协议：{PROTOCOL}",
        f"分析对象：{agent}",
        "",
        "原文：",
        stimulus,
        "",
        "预测想法：",
    ]
    if not thoughts:
        lines.append("（无）")
    else:
        for i, item in enumerate(thoughts, start=1):
            lines.append(
                f"{i}. {item['text']}  p={float(item['probability']):.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT} · LLM 对照")
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    event_file = args.event_file if args.event_file.is_absolute() else _ROOT / args.event_file
    agent, stimulus = parse_event_file(event_file)
    out_dir = args.output_dir or (
        _ROOT / "reports" / "自建实验" / EXPERIMENT / event_file.stem / "llm结果"
    )
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mock:
        key = os.environ.get("MWM_LLM_API_KEY", "").strip()
        model = os.environ.get("MWM_LLM_MODEL", "").strip()
        base = os.environ.get("MWM_LLM_BASE_URL", "").strip()
        if not base or not model:
            print(
                "缺少 MWM_LLM_BASE_URL / MWM_LLM_MODEL。"
                "请在 mental-world-model/.env 填写后重试。",
                file=sys.stderr,
            )
            return 2
        if not key or key == "your-api-key-here":
            print(
                "缺少有效 MWM_LLM_API_KEY。请写入 mental-world-model/.env 后重试。",
                file=sys.stderr,
            )
            return 2

    llm = ThoughtMockLlmClient() if args.mock else create_llm_client(use_mock=False)
    print(
        f"{EXPERIMENT} LLM  分析对象={agent}  模型={llm_run_meta(use_mock=args.mock).get('model')}\n"
        f"输出={out_dir}",
        flush=True,
    )
    try:
        raw = llm.complete(
            [{"role": "user", "content": build_prompt(stimulus, agent=agent)}],
            schema=THOUGHT_SCHEMA,
        )
        thoughts = normalize_thoughts(raw)
    except ThoughtAbort as exc:
        atomic_write_json(
            out_dir / "abort.json",
            {"message": str(exc), "agent": agent},
        )
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "event_file": str(event_file),
        "agent": agent,
        "stimulus": stimulus,
        "thoughts": thoughts,
        "top1": thoughts[0]["text"],
        "llm": llm_run_meta(use_mock=args.mock),
    }
    atomic_write_json(out_dir / "result.json", payload)
    atomic_write_text(
        out_dir / "result.txt",
        render_txt(
            experiment=EXPERIMENT,
            agent=agent,
            stimulus=stimulus,
            thoughts=thoughts,
        ),
    )
    print(f"{EXPERIMENT} LLM 完成：{out_dir}")
    print(f"Top-1：{thoughts[0]['text']}  p={float(thoughts[0]['probability']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
