#!/usr/bin/env python3
"""LLM 原始 baseline（纯文本历史 open rollout）。

对照 chenmo_longitudinal_7.py：同一底层模型、同一事件、同一「预测可观察行动」任务。
本脚本不加载 MentalWorldModel，不做 H/M、摘要、原因或人设提示。
分析对象与事件数由事件文件决定。

状态：
    History_t = History_{t-1} + Event_t + Action_t^{top1}

用法（在 mental-world-model 目录）：
    python3 scripts/experiments/llm_baseline_longitudinal_rollout.py --mock
    python3 scripts/experiments/llm_baseline_longitudinal_rollout.py -o reports/自建实验/纵向/llm_baseline_chenmo_7
    python3 scripts/experiments/llm_baseline_longitudinal_rollout.py --event-file data/自建实验/纵向/程野纵向7事件.txt -o reports/自建实验/纵向/llm_baseline_chengye_7
    python3 scripts/experiments/llm_baseline_longitudinal_rollout.py -o reports/自建实验/纵向/llm_baseline_chenmo_7 --resume

概率只用于记录候选分布与不确定性，不和 MWM 比较「谁更准」。人工对照看轨迹。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mental_world_model.llm.config import LlmConfig
from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.llm.message_utils import user_message_content
from mental_world_model.utils.atomic_io import atomic_write_json, atomic_write_text

load_project_dotenv()

PROTOCOL = "llm_raw_history_open_rollout"

_EVENT_RE = re.compile(
    r"(?m)^\s*事件\s*([0-9]+|[一二三四五六七八九十百千]+|[A-Za-z]+)\s*[：:]"
)
_FOCUS_RE = re.compile(r"(?m)^\s*分析对象\s*[：:]\s*(.+?)\s*$")


def requirements(agent: str) -> str:
    return (
        f"请预测{agent}接下来最可能出现的可观察行动。\n"
        "\n"
        "要求：\n"
        "- 输出最多3个不同的行为\n"
        "- 每个行为给出概率\n"
        "- 概率总和为1\n"
        "- 只描述行为，不解释原因"
    )
BEHAVIOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "behaviors": {
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
    "required": ["behaviors"],
}


class BaselineAbort(RuntimeError):
    """空/烂 Top-1，实验中止。"""


class BaselineMockLlmClient:
    """不走 MWM 分阶段 mock，只返回 1–3 条行为。"""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        content = user_message_content(messages)
        step = (
            sum(
                1
                for line in content.splitlines()
                if line.endswith("行动：") and "的行动：" not in line
            )
            + 1
        )
        return {
            "behaviors": [
                {"text": f"mock行动{step}甲", "probability": 0.5},
                {"text": f"mock行动{step}乙", "probability": 0.3},
                {"text": f"mock行动{step}丙", "probability": 0.2},
            ]
        }


def parse_event_file(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    matches = list(_EVENT_RE.finditer(raw))
    if not matches:
        raise ValueError(f"未找到事件标签：{path}")
    bodies: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        for marker in ("请你预测", "请预测"):
            cut = body.find(marker)
            if cut >= 0:
                body = body[:cut].rstrip("。；; \n")
        if body:
            bodies.append(body)
    return bodies


def parse_focus_person(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    match = _FOCUS_RE.search(raw)
    name = match.group(1).strip() if match else ""
    if not name:
        raise ValueError(f"未找到分析对象：{path}")
    return name


def build_prompt(history: str, event: str, *, agent: str) -> str:
    parts = [f"分析对象：{agent}", ""]
    hist = (history or "").strip()
    if hist:
        parts.extend(
            [
                f"以下是此前已经发生的事件和{agent}的行动：",
                "",
                hist,
                "",
            ]
        )
    parts.extend(
        [
            "当前事件：",
            "",
            (event or "").strip(),
            "",
            requirements(agent),
        ]
    )
    return "\n".join(parts)


def append_history(
    history: str, event_index: int, event: str, action: str, *, agent: str
) -> str:
    block = (
        f"事件{event_index}：\n{(event or '').strip()}\n\n"
        f"{agent}行动：\n{(action or '').strip()}"
    )
    hist = (history or "").rstrip()
    if not hist:
        return block
    return hist + "\n\n" + block


def normalize_behaviors(raw: Any) -> list[dict[str, Any]]:
    """整理候选分布。概率只表示相对不确定性，不作准确率比较。"""
    payload = raw if isinstance(raw, dict) else {}
    items = payload.get("behaviors")
    if not isinstance(items, list):
        raise BaselineAbort("模型未返回 behaviors 列表")
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
        raise BaselineAbort("无可用行为")
    cleaned.sort(key=lambda row: float(row["probability"]), reverse=True)
    cleaned = cleaned[:3]
    total = sum(float(row["probability"]) for row in cleaned)
    if total <= 0:
        raise BaselineAbort("概率全为 0")
    for row in cleaned:
        row["probability"] = float(row["probability"]) / total
    return cleaned


def step_id(index: int) -> str:
    return f"S{index:02d}"


def _short(text: str, limit: int = 72) -> str:
    body = (text or "").replace("\n", " ").strip()
    if len(body) <= limit:
        return body
    return body[: limit - 1] + "…"


def _elapsed(started: float) -> str:
    sec = max(0, int(time.monotonic() - started))
    if sec < 60:
        return f"{sec}s"
    return f"{sec // 60}m{sec % 60:02d}s"


def llm_run_meta(*, use_mock: bool) -> dict[str, Any]:
    if use_mock:
        return {
            "client": "mock",
            "model": "mock",
            "temperature": None,
            "top_p": None,
            "seed": None,
        }
    cfg = LlmConfig.from_env_optional()
    if cfg is None:
        return {
            "client": "http",
            "model": os.environ.get("MWM_LLM_MODEL", "").strip(),
            "temperature": None,
            "top_p": None,
            "seed": None,
        }
    return {
        "client": "http",
        "model": cfg.model,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "seed": cfg.seed,
    }


class LlmBaselineRollout:
    def __init__(
        self,
        *,
        event_file: Path,
        out_dir: Path,
        use_mock: bool = False,
        resume: bool = False,
        client: Any | None = None,
    ) -> None:
        self.bodies = parse_event_file(event_file)
        if not self.bodies:
            raise ValueError(f"{event_file} 未解析到任何事件")
        self.agent = parse_focus_person(event_file)
        self.total_events = len(self.bodies)

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps_dir = self.out_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock
        self.resume = resume
        self.progress_path = self.out_dir / "progress.json"
        self.llm_meta = llm_run_meta(use_mock=use_mock)

        if self.progress_path.is_file() and not resume:
            raise FileExistsError(
                "输出目录已有 progress.json。续跑请加 --resume，或换一个 -o。"
            )

        self.event_file = str(event_file)

        if client is not None:
            self.llm = client
        elif use_mock:
            self.llm = BaselineMockLlmClient()
        else:
            self.llm = create_llm_client(use_mock=False)

        self.completed: list[str] = []
        self.trajectory: list[dict[str, Any]] = []
        self.history_text = ""
        self._started = time.monotonic()

        if resume:
            self._load_resume()
        self._save_progress()

    def _load_resume(self) -> None:
        if not self.progress_path.is_file():
            raise FileNotFoundError("--resume 需要已有 progress.json")
        raw = json.loads(self.progress_path.read_text(encoding="utf-8"))
        self.completed = list(raw.get("completed_steps") or [])
        self.trajectory = list(raw.get("trajectory") or [])
        self.history_text = str(raw.get("history_text") or "")
        expected = len(self.completed)
        if expected != len(self.trajectory):
            raise ValueError(
                f"completed_steps={expected} 与 trajectory={len(self.trajectory)} 不一致，拒绝续跑。"
            )
        print(f"续跑：已完成 {expected} 步", flush=True)

    def _save_progress(self) -> None:
        atomic_write_json(
            self.progress_path,
            {
                "protocol": PROTOCOL,
                "agent": self.agent,
                "completed_steps": list(self.completed),
                "trajectory": list(self.trajectory),
                "history_text": self.history_text,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _write_abort(self, message: str, *, sid: str) -> None:
        atomic_write_json(
            self.out_dir / "abort.json",
            {
                "aborted_at": datetime.now().isoformat(timespec="seconds"),
                "message": message,
                "step_id": sid,
                "completed_steps": list(self.completed),
            },
        )
        self._save_progress()

    def _write_trajectories(self) -> None:
        atomic_write_json(
            self.out_dir / "behavior_trajectory.json",
            {
                "protocol": PROTOCOL,
                "agent": self.agent,
                "events": self.trajectory,
            },
        )
        lines = [
            f"{item['event_id']} → {item['predicted_action']}"
            for item in self.trajectory
        ]
        atomic_write_text(
            self.out_dir / "behavior_trajectory.txt",
            "\n".join(lines) + ("\n" if lines else ""),
        )
        atomic_write_text(
            self.out_dir / "history_final.txt",
            self.history_text + ("\n" if self.history_text else ""),
        )

    def run_step(self, index: int) -> None:
        sid = step_id(index)
        if sid in self.completed:
            left = max(0, self.total_events - len(self.completed))
            print(f"[跳过] E{index} 已完成  还剩 {left} 个任务", flush=True)
            return

        current_event = self.bodies[index - 1]
        history_before = self.history_text
        prompt = build_prompt(history_before, current_event, agent=self.agent)
        n = index
        left = self.total_events - n + 1
        print(
            f"[{n}/{self.total_events}] E{index} 预测进行中  还剩 {left} 个任务"
            f"  已用{_elapsed(self._started)}",
            flush=True,
        )

        try:
            raw = self.llm.complete(
                [{"role": "user", "content": prompt}],
                schema=BEHAVIOR_SCHEMA,
            )
            behaviors = normalize_behaviors(raw)
            top1 = str(behaviors[0]["text"]).strip()
            if not top1:
                raise BaselineAbort(f"闭环中止：{sid} Top-1 不可用（{top1!r}）")
        except BaselineAbort as exc:
            self._write_abort(str(exc), sid=sid)
            raise
        except Exception as exc:
            self._write_abort(f"{type(exc).__name__}: {exc}", sid=sid)
            raise

        history_after = append_history(
            history_before, index, current_event, top1, agent=self.agent
        )
        payload = {
            "event_id": sid,
            "history_before": history_before,
            "current_event": current_event,
            "behaviors": behaviors,
            "top1": top1,
            "history_after": history_after,
        }
        atomic_write_json(self.steps_dir / f"{sid}.json", payload)

        self.history_text = history_after
        self.completed.append(sid)
        self.trajectory.append(
            {
                "event_id": f"E{index}",
                "step_id": sid,
                "current_event": current_event,
                "predicted_action": top1,
                "behaviors": behaviors,
            }
        )
        self._save_progress()
        self._write_trajectories()
        print(
            f"[{n}/{self.total_events}] E{index} 预测完成  还剩 {self.total_events - n} 个任务"
            f"  已用{_elapsed(self._started)}  {_short(top1)}",
            flush=True,
        )

    def write_summary(self) -> dict[str, Any]:
        summary = {
            "protocol": PROTOCOL,
            "agent": self.agent,
            "total_events": self.total_events,
            "event_file": self.event_file,
            "llm": dict(self.llm_meta),
            "completed_steps": list(self.completed),
            "behavior_trajectory": [
                f"{item['event_id']} → {item['predicted_action']}"
                for item in self.trajectory
            ],
            "steps": [
                {
                    "event_id": item["event_id"],
                    "step_id": item["step_id"],
                    "top1": item["predicted_action"],
                    "behaviors": item.get("behaviors") or [],
                }
                for item in self.trajectory
            ],
        }
        atomic_write_json(self.out_dir / "summary.json", summary)
        self._write_trajectories()
        return summary

    def run(self) -> dict[str, Any]:
        done = len(self.completed)
        left = self.total_events - done
        model = self.llm_meta.get("model") or "（未读到模型名）"
        print(
            f"{self.agent}纵向 {self.total_events} 事件 LLM baseline  协议={PROTOCOL}\n"
            f"模型={model}  输出={self.out_dir}\n"
            f"共 {self.total_events} 个任务  已完成 {done}  还剩 {left} 个任务",
            flush=True,
        )
        for index in range(1, self.total_events + 1):
            self.run_step(index)
        summary = self.write_summary()
        print(
            f"全部完成  {self.total_events}/{self.total_events}  还剩 0 个任务  "
            f"已用{_elapsed(self._started)}\n"
            f"汇总：{self.out_dir / 'summary.json'}",
            flush=True,
        )
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM 原始 baseline（纯文本历史 rollout；事件数与分析对象由文件决定）"
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=_ROOT / "data" / "自建实验" / "纵向" / "陈默纵向7事件.txt",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认 reports/自建实验/纵向/llm_baseline_chenmo_7）",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --output-dir 的 progress.json 续跑",
    )
    args = parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error("--resume 需要同时指定 --output-dir")

    out_dir = args.output_dir or (
        _ROOT / "reports" / "自建实验" / "纵向" / "llm_baseline_chenmo_7"
    )
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

    runner = LlmBaselineRollout(
        event_file=args.event_file,
        out_dir=out_dir,
        use_mock=args.mock,
        resume=args.resume,
    )
    try:
        runner.run()
    except BaselineAbort as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
