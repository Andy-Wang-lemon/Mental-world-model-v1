#!/usr/bin/env python3
"""单实例纵向心智世界模型测试（开放 rollout）。

协议：
  一个 MentalWorldModel 实例，从 Event 1 初始化。
  分析对象与事件数由事件文件决定（文件头「分析对象：」，以及「事件N：」条数）。
  每一步：只输入当前事件刺激 → 预测（不写 H/M）→ Top-1 落地写 H/M。
  不得每步重跑 Event 1…t-1，不得每步重新初始化。

用法（在 mental-world-model 目录）：
    python3 scripts/experiments/chenmo_longitudinal_7.py --mock
    python3 scripts/experiments/chenmo_longitudinal_7.py -o reports/自建实验/纵向/chenmo_longitudinal_7
    python3 scripts/experiments/chenmo_longitudinal_7.py --event-file data/自建实验/纵向/程野纵向7事件.txt -o reports/自建实验/纵向/chengye_longitudinal_7
    python3 scripts/experiments/chenmo_longitudinal_7.py -o reports/自建实验/纵向/chenmo_longitudinal_7 --resume
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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mental_world_model.engine.cognition.predicted_action import (
    is_usable_predicted_action,
    top_aggregated_action,
)
from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.continuity.store import ContinuityStateStore
from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.llm.prefixed_client import StagePrefixLlmClient
from mental_world_model.llm.tracing_client import ensure_tracing_client
from mental_world_model.model import MentalWorldModel
from mental_world_model.utils.atomic_io import atomic_write_json, atomic_write_text

load_project_dotenv()

PROTOCOL = "single_agent_longitudinal_open_rollout"
TASKS_PER_EVENT = 2  # 预测 + 落地


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


def task_no(event_index: int, phase: str) -> int:
    """phase: predict | land。"""
    offset = 0 if phase == "predict" else 1
    return (event_index - 1) * TASKS_PER_EVENT + 1 + offset


def remaining_tasks(current_task: int, *, done: bool, total_tasks: int) -> int:
    """当前任务进行中含当前；完成后不含当前。"""
    if done:
        return max(0, total_tasks - current_task)
    return max(0, total_tasks - current_task + 1)


def log_progress(
    event_index: int,
    phase: str,
    *,
    total_tasks: int,
    done: bool = False,
    extra: str = "",
    started: float | None = None,
) -> None:
    n = task_no(event_index, phase)
    left = remaining_tasks(n, done=done, total_tasks=total_tasks)
    label = "预测" if phase == "predict" else "落地"
    state = "完成" if done else "进行中"
    clock = f"  已用{_elapsed(started)}" if started is not None else ""
    tail = f"  {extra}" if extra else ""
    print(
        f"[{n}/{total_tasks}] E{event_index} {label}{state}"
        f"  还剩 {left} 个任务{clock}{tail}",
        flush=True,
    )

_EVENT_RE = re.compile(
    r"(?m)^\s*事件\s*([0-9]+|[一二三四五六七八九十百千]+|[A-Za-z]+)\s*[：:]"
)
_FOCUS_RE = re.compile(r"(?m)^\s*分析对象\s*[：:]\s*(.+?)\s*$")
_MAX_HIST_EVENTS = 3
_MAX_BODY_CHARS = 600


class LongitudinalAbort(RuntimeError):
    """空/烂 Top-1，实验中止。"""


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


def build_role_context(prior_sealed: list[str], *, agent: str) -> str:
    """原文备忘（不是 H/M，也不触发重算已完成事件）。"""
    parts = [f"分析对象：{agent}"]
    if not prior_sealed:
        return parts[0]
    start = max(0, len(prior_sealed) - _MAX_HIST_EVENTS)
    for i, body in enumerate(prior_sealed[start:], start=start + 1):
        text = (body or "").strip()
        truncated = text[:_MAX_BODY_CHARS] + ("…" if len(text) > _MAX_BODY_CHARS else "")
        parts.append(f"[事件{i}] {truncated}")
    return "\n".join(parts)


def continuity_brief(state: ContinuityState | None) -> dict:
    if state is None:
        return {}
    return {
        "person_id": state.person_id,
        "event_count": int(state.event_count or 0),
        "mental_episode_count": len(state.mental_episodes or []),
        "cognitive_frame_count": len(state.cognitive_frames or []),
    }


def collapse_note(diagnostics: dict | None) -> str:
    data = diagnostics if isinstance(diagnostics, dict) else {}
    if bool(data.get("candidate_collapsed")):
        return "当前支内只产生一个有效行为类别，概率 1.0 不表示高确信。"
    return ""


def step_id(index: int) -> str:
    return f"S{index:02d}"


def _grasping_counts(result) -> dict:
    payload = result.to_event_snapshot() if result is not None else {}
    candidates = payload.get("cognition_candidates") or {}
    self_g = payload.get("self_grasping") or []
    dharma_g = payload.get("dharma_grasping") or []
    fresh = 0
    historical = 0
    for item in list(self_g) + list(dharma_g):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_type") or "")
        if source == "fresh":
            fresh += 1
        elif source == "historical":
            historical += 1
    return {
        "fresh_count_raw": candidates.get("fresh_count_raw", fresh),
        "historical_count_raw": candidates.get("historical_count_raw", historical),
        "fresh_branch_count": fresh,
        "historical_branch_count": historical,
        "after_semantic_dedup": candidates.get("after_semantic_dedup"),
        "valid_after_alignment_gate": candidates.get("valid_after_alignment_gate"),
    }


class ChenmoLongitudinalRunner:
    def __init__(
        self,
        *,
        definitions_dir: Path,
        event_file: Path,
        out_dir: Path,
        use_mock: bool = False,
        resume: bool = False,
    ) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps_dir = self.out_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = self.out_dir / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock
        self.resume = resume
        self.progress_path = self.out_dir / "progress.json"

        if self.progress_path.is_file() and not resume:
            raise FileExistsError(
                f"输出目录已有 progress.json。续跑请加 --resume，或换一个 -o。"
            )

        self.bodies = parse_event_file(event_file)
        if not self.bodies:
            raise ValueError(f"{event_file} 未解析到任何事件")
        self.agent = parse_focus_person(event_file)
        self.total_events = len(self.bodies)
        self.total_tasks = self.total_events * TASKS_PER_EVENT

        inner = ensure_tracing_client(create_llm_client(use_mock=use_mock))
        self.llm = StagePrefixLlmClient(inner, "chenmo")
        self.model = MentalWorldModel.load(definitions_dir)
        store = ContinuityStateStore(self.runtime_dir / "continuity_chenmo.json")
        self.model.continuity_state_store = store

        self.completed: list[str] = []
        self.trajectory: list[dict] = []
        self.sealed_history: list[str] = []
        self._started = time.monotonic()

        if resume:
            self._load_resume(store)
        else:
            self.model.continuity_state = ContinuityState(person_id=self.agent)
            store.save(self.model.continuity_state)
        self._save_progress()

    def _load_resume(self, store: ContinuityStateStore) -> None:
        if not self.progress_path.is_file():
            raise FileNotFoundError("--resume 需要已有 progress.json")
        raw = json.loads(self.progress_path.read_text(encoding="utf-8"))
        self.completed = list(raw.get("completed_steps") or [])
        self.trajectory = list(raw.get("trajectory") or [])
        self.sealed_history = list(raw.get("sealed_history") or [])
        state = store.load()
        if not state.person_id or state.person_id == "default":
            state.person_id = self.agent
        expected = len(self.completed)
        if int(state.event_count or 0) != expected:
            raise ValueError(
                f"continuity.event_count={state.event_count} "
                f"与已完成步骤 {expected} 不一致，拒绝续跑。"
            )
        self.model.continuity_state = state
        print(
            f"续跑：已完成 {expected} 步，"
            f"M={len(state.mental_episodes)} H={len(state.cognitive_frames)}"
        )

    def _save_progress(self) -> None:
        atomic_write_json(
            self.progress_path,
            {
                "protocol": PROTOCOL,
                "agent": self.agent,
                "completed_steps": list(self.completed),
                "trajectory": list(self.trajectory),
                "sealed_history": list(self.sealed_history),
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
        lines = []
        for item in self.trajectory:
            lines.append(f"{item['event_id']} → {item['predicted_action']}")
        atomic_write_text(
            self.out_dir / "behavior_trajectory.txt",
            "\n".join(lines) + ("\n" if lines else ""),
        )

    def _step_diagnostics(self, result) -> dict:
        snapshot = result.to_event_snapshot() if result is not None else {}
        accommodation = snapshot.get("accommodation") or {}
        diagnostics = snapshot.get("behavior_diagnostics") or {}
        counts = _grasping_counts(result)
        collapsed = bool(diagnostics.get("candidate_collapsed"))
        note = collapse_note(diagnostics)
        return {
            "fresh_count_raw": counts.get("fresh_count_raw"),
            "historical_count_raw": counts.get("historical_count_raw"),
            "fresh_branch_count": counts.get("fresh_branch_count"),
            "historical_branch_count": counts.get("historical_branch_count"),
            "accommodation_triggered": bool(accommodation.get("triggered")),
            "neutral_fallback": bool(snapshot.get("used_neutral_fallback")),
            "behavior_candidate_collapsed": collapsed,
            "collapse_note": note,
            "behavior_diagnostics": dict(diagnostics),
            "accommodation": dict(accommodation),
        }

    def run_step(self, index: int) -> None:
        sid = step_id(index)
        if sid in self.completed:
            left = remaining_tasks(
                task_no(index, "land"),
                done=True,
                total_tasks=self.total_tasks,
            )
            print(
                f"[跳过] E{index} 已完成  还剩 {left} 个任务",
                flush=True,
            )
            return
        stimulus = self.bodies[index - 1]
        role_context = build_role_context(self.sealed_history, agent=self.agent)
        hm_before = continuity_brief(self.model.continuity_state)
        injected_h = [
            {"id": f.id, "text": f.text, "mass": round(float(f.mass), 4)}
            for f in (self.model.continuity_state.cognitive_frames or [])
        ]

        log_progress(
            index, "predict", total_tasks=self.total_tasks, started=self._started
        )
        try:
            bundle = self.model.evaluate_text_with_cognition_branches(
                stimulus,
                self.llm,
                role_context=role_context,
                update_seeds=False,
                update_continuity=False,
                use_action_feedback=False,
                aggregate_behaviors=True,
                predict_branch_behaviors=True,
                focus_person=self.agent,
            )
        except Exception as exc:
            self._write_abort(f"{type(exc).__name__}: {exc}", sid=sid)
            raise

        result = bundle.evaluation_result
        result.focus_person = self.agent
        action, prob, top_k = top_aggregated_action(result)
        if not is_usable_predicted_action(action):
            msg = f"闭环中止：{sid} Top-1 不可用（{action!r}）"
            self._write_abort(msg, sid=sid)
            raise LongitudinalAbort(msg)

        diag = self._step_diagnostics(result)
        pred_payload = result.to_event_report_dict()
        pred_payload["_longitudinal"] = {
            "protocol": PROTOCOL,
            "phase": "prediction",
            "step_id": sid,
            "event_index": index,
            "stimulus": stimulus,
            "top_action": action,
            "top_probability": prob,
            "top_k": top_k,
            "continuity_before": hm_before,
            "cognitive_frames_before": injected_h,
            **diag,
        }
        atomic_write_json(self.steps_dir / f"{sid}_prediction.json", pred_payload)
        if diag["collapse_note"]:
            print(f"  说明：{diag['collapse_note']}", flush=True)
        log_progress(
            index,
            "predict",
            total_tasks=self.total_tasks,
            done=True,
            extra=f"p={prob:.2f}  {_short(action)}",
            started=self._started,
        )

        log_progress(
            index, "land", total_tasks=self.total_tasks, started=self._started
        )
        try:
            landed = self.model.apply_predicted_action(
                stimulus,
                self.llm,
                bundle,
                action,
                role_context=role_context,
                focus_person=self.agent,
                action_probability=prob,
                update_seeds=False,
            )
        except Exception as exc:
            self._write_abort(f"{type(exc).__name__}: {exc}", sid=sid)
            raise

        hm_after = continuity_brief(self.model.continuity_state)
        land_result = landed.evaluation_result
        land_result.focus_person = self.agent
        land_payload = land_result.to_event_report_dict()
        land_payload["_longitudinal"] = {
            "protocol": PROTOCOL,
            "phase": "landed",
            "step_id": sid,
            "event_index": index,
            "stimulus": stimulus,
            "landed_action": action,
            "landed_probability": prob,
            "continuity_before": hm_before,
            "continuity_after": hm_after,
            "cognitive_frames_after": [
                {"id": f.id, "text": f.text, "mass": round(float(f.mass), 4)}
                for f in (self.model.continuity_state.cognitive_frames or [])
            ],
            "mental_episodes_after": [
                {
                    "event_id": e.event_id,
                    "theta": e.theta,
                    "text_summary": e.text_summary,
                }
                for e in (self.model.continuity_state.mental_episodes or [])
            ],
        }
        atomic_write_json(self.steps_dir / f"{sid}_landed.json", land_payload)
        if self.model.continuity_state_store is not None:
            self.model.continuity_state_store.save(self.model.continuity_state)

        sealed = f"{stimulus}\n{self.agent}随后的行动：{action}"
        self.sealed_history.append(sealed)
        self.completed.append(sid)
        self.trajectory.append(
            {
                "event_id": f"E{index}",
                "step_id": sid,
                "stimulus": stimulus,
                "predicted_action": action,
                "probability": prob,
                "top_k": top_k,
                "behavior_candidate_collapsed": diag["behavior_candidate_collapsed"],
                "collapse_note": diag["collapse_note"],
                "fresh_branch_count": diag["fresh_branch_count"],
                "historical_branch_count": diag["historical_branch_count"],
                "accommodation_triggered": diag["accommodation_triggered"],
                "neutral_fallback": diag["neutral_fallback"],
            }
        )
        self._save_progress()
        self._write_trajectories()
        log_progress(
            index,
            "land",
            total_tasks=self.total_tasks,
            done=True,
            extra=f"H={hm_after.get('cognitive_frame_count')} "
            f"M={hm_after.get('mental_episode_count')}",
            started=self._started,
        )

    def write_summary(self) -> dict:
        brief = continuity_brief(self.model.continuity_state)
        summary = {
            "protocol": PROTOCOL,
            "agent": self.agent,
            "total_events": self.total_events,
            "completed_steps": list(self.completed),
            "behavior_trajectory": [
                f"{item['event_id']} → {item['predicted_action']}"
                for item in self.trajectory
            ],
            "steps": [
                {
                    "event_id": item["event_id"],
                    "predicted_action": item["predicted_action"],
                    "probability": item["probability"],
                    "behavior_candidate_collapsed": item["behavior_candidate_collapsed"],
                    "collapse_note": item["collapse_note"],
                    "fresh_branch_count": item["fresh_branch_count"],
                    "historical_branch_count": item["historical_branch_count"],
                    "accommodation_triggered": item["accommodation_triggered"],
                    "neutral_fallback": item["neutral_fallback"],
                }
                for item in self.trajectory
            ],
            "final_continuity": brief,
        }
        atomic_write_json(self.out_dir / "summary.json", summary)
        if self.model.continuity_state_store is not None:
            self.model.continuity_state_store.save(self.model.continuity_state)
        final_copy = self.runtime_dir / "continuity_chenmo_final.json"
        atomic_write_json(final_copy, self.model.continuity_state.to_dict())
        return summary

    def run(self) -> dict:
        done = len(self.completed)
        left = self.total_tasks - done * TASKS_PER_EVENT
        model = os.environ.get("MWM_LLM_MODEL", "").strip() or "（未读到 MWM_LLM_MODEL）"
        print(
            f"{self.agent}纵向 {self.total_events} 事件  协议={PROTOCOL}\n"
            f"模型={model}  输出={self.out_dir}\n"
            f"共 {self.total_tasks} 个任务（每事件：预测 + 落地）  "
            f"已完成 {done * TASKS_PER_EVENT}  还剩 {left} 个任务",
            flush=True,
        )
        for index in range(1, self.total_events + 1):
            self.run_step(index)
        summary = self.write_summary()
        print(
            f"全部完成  {self.total_tasks}/{self.total_tasks}  还剩 0 个任务  "
            f"已用{_elapsed(self._started)}\n"
            f"汇总：{self.out_dir / 'summary.json'}",
            flush=True,
        )
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单实例纵向测试（预测后落地；事件数由文件决定）"
    )
    parser.add_argument(
        "--definitions",
        type=Path,
        default=_ROOT / "definitions",
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
        help="输出目录（默认 reports/自建实验/纵向/chenmo_longitudinal_7）",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --output-dir 的 progress.json 与 continuity 续跑",
    )
    parser.add_argument(
        "--verbose-llm",
        action="store_true",
        help="打开引擎内部 [MWM] 阶段日志（默认关闭，只保留本脚本进度）",
    )
    args = parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error("--resume 需要同时指定 --output-dir")

    if not args.verbose_llm:
        os.environ["MWM_LLM_PROGRESS"] = "0"

    out_dir = args.output_dir or (_ROOT / "reports" / "自建实验" / "纵向" / "chenmo_longitudinal_7")
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
    runner = ChenmoLongitudinalRunner(
        definitions_dir=args.definitions,
        event_file=args.event_file,
        out_dir=out_dir,
        use_mock=args.mock,
        resume=args.resume,
    )
    try:
        runner.run()
    except LongitudinalAbort as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
