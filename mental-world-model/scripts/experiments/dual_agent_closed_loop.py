#!/usr/bin/env python3
"""双人心智世界模型闭环测试编排。

协议（A=陈默，B=林可）：
  1. 两人各自把初始前序当解释事件跑一遍，留下 H/M（只算一次）
  2. A：末事件「林可求助」→ 用已有 H/M 预测 A1 → 把 A1 接到同一套所缘上写 H/M
  3. B：末事件=A1 的可见投影 → 预测 B1 → 接入行动更新 H/M
  4–5. 交叉：B1→A2→B2→A3；A3 不给 B
  6. 外部→B「发现私下补数据」→ B3→A4→B4→A5→B5；B5 不给 A
  7. 外部→A「CEO 48h 共案」→ A6→B6→A7→B7 结束

转交规则：己方账本落地完整 Top-1（含内心/私下）；
给对方当前末事件前，经脚本层 LLM 滤成观察者可见刺激（不套「做出了如下行动」）；
投影须以行动者为主语，失败则不把含内心/黑话的原文交给对方（改为无可察行动）。
外部事件原文直给，不经过滤。A3/B5/B7 不转交。
默认增量：不清空 H/M、不重跑前序；预测本身不落库，再把 Top-1 接到预测所缘上写 H/M。
空/烂 Top-1（「（无行为预测）」等）中止闭环；「没表现」是合法行动。
陈默 / 林可 / 可见性过滤使用带 stage 前缀的独立 LLM 包装。
--rebuild-prologue 可退回「每次整篇重跑并重置」。
--resume 配合 --output-dir：读 progress.json 与 continuity，不清空 H/M。
--mock 或过滤 LLM 失败时：干净行动仅补主语；含内心/黑话则改为无可察行动。
输出目录不要放 iCloud，避免半写入导致无法续跑。

用法（在 mental-world-model 目录）：
  python3 scripts/experiments/dual_agent_closed_loop.py --mock
  python3 scripts/experiments/dual_agent_closed_loop.py -o reports/自建实验/dual_loop
  python3 scripts/experiments/dual_agent_closed_loop.py -o reports/自建实验/dual_loop --resume
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 保证可从 scripts/experiments/ 直接运行
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.cognition.predicted_action import (
    is_usable_predicted_action,
    top_aggregated_action,
)
from mental_world_model.llm.env_file import load_project_dotenv
from mental_world_model.utils.atomic_io import atomic_write_json
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.llm.prefixed_client import StagePrefixLlmClient
from mental_world_model.llm.tracing_client import ensure_tracing_client
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.model import MentalWorldModel

load_project_dotenv()

_EVENT_RE = re.compile(
    r"(?m)^\s*事件\s*([0-9]+|[一二三四五六七八九十百千]+|[A-Za-z]+)\s*[：:]"
)

# —— 协议中的外部刺激 ——
EXTERNAL_LINKE_HIDDEN_HELP = (
    "林可发现陈默曾私下补数据、未署名。陈默尚未表态。"
)
EXTERNAL_CHENMO_JOINT_TASK = (
    "CEO 要求两人 48h 共交挽救方案，同一会议室。"
)

CHENMO_FIRST_FINAL = (
    "林可负责的项目遇到了很大的困难。开会讨论时，陈默也在场。"
    "林可对大家说：“我认为以前老旧的手段和经验在这不适用。”"
    "但会议继续进行后，林可仍然没有找到更好的解决办法。"
    "如果问题继续无法解决，项目可能失败，林可也可能因此被降薪、话语权减少。"
    "会议结束后，林可私下来找陈默，寻求帮助。"
)

_INNER_MARKERS = (
    "内心",
    "心里",
    "心中",
    "暗自",
    "其实",
    "实则",
    "实际上",
)
_JARGON_CLAUSE_MARKERS = (
    "即促成",
    "这符合他内心",
    "这符合她内心",
    "这符合其内心",
    "符合他内心",
    "符合她内心",
)
_JARGON_FRAGMENTS = (
    "即促成善行",
    "无瞋业",
    "疑业",
)
_OBSERVER_PERCEIVE_VERBS = ("看到", "看见", "听见", "听闻")
_WRONG_TASK_VERBS = ("需要处理", "必须处理", "需要负责", "必须负责")
_SCENE_ONLY_KEYWORDS = (
    "补数据",
    "未署名",
    "48h",
    "48小时",
    "共交",
    "会议室",
    "挽救方案",
)
_KNOWN_AGENTS = ("陈默", "林可")

OBSERVABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "observable_event": {"type": "string"},
        "dropped": {"type": "array", "items": {"type": "string"}},
        "visibility": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["observable_event", "visibility"],
}


class DualLoopAbort(RuntimeError):
    """空/烂 Top-1，闭环中止。"""


@dataclass
class ObservableProjection:
    observable_event: str
    dropped: list[str] = field(default_factory=list)
    visibility: str = "visible"
    reason: str = ""
    source: str = "llm"


def _none_visible_event(actor: str, observer: str) -> str:
    return f"{actor}没有做出可被{observer}察觉的行动。"


def _ensure_period(text: str) -> str:
    body = text.strip()
    if not body:
        return body
    if body[-1] not in "。！？":
        body += "。"
    return body


def _is_none_visible_event(actor: str, observer: str, text: str) -> bool:
    body = (text or "").strip().rstrip("。！？")
    expected = _none_visible_event(actor, observer).rstrip("。！？")
    return body == expected


def observable_violations(
    actor: str,
    observer: str,
    text: str,
    *,
    raw_action: str = "",
) -> list[str]:
    """投影不合格原因（结构校验，触发 LLM 重试）；空列表表示可通过。"""
    body = (text or "").strip()
    if not body:
        return ["empty"]
    if _is_none_visible_event(actor, observer, body):
        return []
    hits: list[str] = []
    if not body.startswith(actor):
        hits.append("subject_not_actor")
    if body.startswith(observer):
        hits.append("observer_as_subject")
    for verb in _OBSERVER_PERCEIVE_VERBS:
        if f"{observer}{verb}" in body:
            hits.append("observer_perceives")
            break
    for person in _KNOWN_AGENTS:
        if person == actor:
            continue
        if any(f"{person}{verb}" in body for verb in _WRONG_TASK_VERBS):
            hits.append("wrong_person_task")
            break
        if body.startswith(person):
            hits.append("wrong_person_subject")
            break
    if any(frag in body for frag in _JARGON_FRAGMENTS) or any(
        m in body for m in _JARGON_CLAUSE_MARKERS
    ):
        hits.append("jargon_fragment")
    if any(m in body for m in _INNER_MARKERS):
        hits.append("inner_leaked")
    source = raw_action or ""
    invented = [kw for kw in _SCENE_ONLY_KEYWORDS if kw in body and kw not in source]
    if invented:
        hits.append("invented:" + ",".join(invented))
    return hits


def heuristic_observable_projection(
    actor: str, observer: str, raw_action: str
) -> "ObservableProjection":
    """LLM 不可用时的最后手段：补主语 + 原文，不按词表删情节。"""
    text = (raw_action or "").strip()
    if not text:
        return ObservableProjection(
            observable_event=_none_visible_event(actor, observer),
            visibility="none",
            reason="原文为空",
            source="heuristic",
        )
    body = text
    if not body.startswith(actor):
        body = f"{actor}{body}"
    return ObservableProjection(
        observable_event=_ensure_period(body),
        dropped=[],
        visibility="visible",
        reason="启发式：仅补行动者主语，可见性由 LLM 判断",
        source="heuristic",
    )


def _valid_or_none(
    proj: "ObservableProjection",
    actor: str,
    observer: str,
    raw_action: str,
) -> "ObservableProjection | None":
    if _is_none_visible_event(actor, observer, proj.observable_event):
        return proj
    if observable_violations(
        actor, observer, proj.observable_event, raw_action=raw_action
    ):
        return None
    return proj


def _fallback_observable(
    actor: str,
    observer: str,
    raw_action: str,
    *,
    source: str,
    reason: str,
) -> "ObservableProjection":
    heur = heuristic_observable_projection(actor, observer, raw_action)
    heur.source = source
    heur.reason = reason
    if _is_none_visible_event(actor, observer, heur.observable_event):
        heur.visibility = "none"
        return heur
    if not observable_violations(
        actor, observer, heur.observable_event, raw_action=raw_action
    ):
        return heur
    return ObservableProjection(
        observable_event=_none_visible_event(actor, observer),
        dropped=[],
        visibility="none",
        reason=reason + "；启发式原文仍不合格，改为无可察行动",
        source="fallback_none",
    )


def _projection_from_llm_resp(
    resp: object,
    *,
    source: str,
) -> "ObservableProjection | None":
    data = resp if isinstance(resp, dict) else {}
    text = str(data.get("observable_event") or "").strip()
    if not text:
        return None
    vis = str(data.get("visibility") or "").strip().lower()
    if vis not in {"visible", "partial", "none"}:
        vis = "visible"
    raw_dropped = data.get("dropped") or []
    dropped = [str(x).strip() for x in raw_dropped if str(x).strip()]
    reason = str(data.get("reason") or "").strip()
    return ObservableProjection(
        observable_event=_ensure_period(text),
        dropped=dropped,
        visibility=vis,
        reason=reason,
        source=source,
    )


def _call_observable_llm(llm, messages: list[dict[str, str]]):
    with llm_call_context(stage="script.observable_filter"):
        return llm.complete(messages, schema=OBSERVABLE_SCHEMA)


def _retry_observable_messages(
    messages: list[dict[str, str]],
    *,
    actor: str,
    violations: list[str],
) -> list[dict[str, str]]:
    hint = "、".join(violations) if violations else "输出为空或不完整"
    extra = {
        "role": "user",
        "content": (
            f"上次输出不合格：{hint}。\n"
            f"请重写 observable_event，必须以「{actor}」开头；"
            "不要写观察者「看到/听见」；不要心所残句。"
            "回家后的私下行为可去掉，但当众提及「私下补数据」等公开言行要保留。"
        ),
    }
    return list(messages) + [extra]


def project_observable_event(
    *,
    llm,
    use_mock: bool,
    actor: str,
    observer: str,
    raw_action: str,
    scene: str = "",
) -> "ObservableProjection":
    """把完整预测投影为观察者可见末事件。脚本层，不进引擎。"""
    del use_mock
    if llm is None:
        return _fallback_observable(
            actor,
            observer,
            raw_action,
            source="heuristic",
            reason="无过滤 LLM，仅补主语",
        )

    scene_trunc = (scene or "").replace("\n", " ").strip()[:400]
    messages = build_observable_messages(
        actor=actor,
        observer=observer,
        raw_action=raw_action,
        scene=scene_trunc,
    )

    try:
        resp = _call_observable_llm(llm, messages)
    except Exception as exc:  # noqa: BLE001
        return _fallback_observable(
            actor,
            observer,
            raw_action,
            source="heuristic_fallback",
            reason=f"LLM 失败，改用启发式：{exc}",
        )

    candidate = _projection_from_llm_resp(resp, source="llm")
    if candidate is None:
        return _fallback_observable(
            actor,
            observer,
            raw_action,
            source="llm_empty_fallback",
            reason="LLM 未返回 observable_event",
        )

    ok = _valid_or_none(candidate, actor, observer, raw_action)
    if ok is not None:
        return ok

    violations = observable_violations(
        actor, observer, candidate.observable_event, raw_action=raw_action
    )
    retry_messages = _retry_observable_messages(
        messages, actor=actor, violations=violations
    )
    try:
        resp2 = _call_observable_llm(llm, retry_messages)
    except Exception as exc:  # noqa: BLE001
        return _fallback_observable(
            actor,
            observer,
            raw_action,
            source="llm_rejected_heuristic",
            reason=f"LLM 投影未通过校验且重试失败：{exc}",
        )
    retry = _projection_from_llm_resp(resp2, source="llm_retry")
    if retry is not None:
        ok_retry = _valid_or_none(retry, actor, observer, raw_action)
        if ok_retry is not None:
            return ok_retry
    return _fallback_observable(
        actor,
        observer,
        raw_action,
        source="llm_rejected_heuristic",
        reason=candidate.reason or "LLM 投影未通过主语/残句校验",
    )


def build_observable_messages(
    *,
    actor: str,
    observer: str,
    raw_action: str,
    scene: str,
) -> list[dict[str, str]]:
    system = (
        "你是双人互动的「观察者可见性」过滤器，不是心理分析器。\n"
        f"行动者是{actor}，观察者是{observer}。\n"
        f"任务：把{actor}的完整预测行动改写成{observer}当场能感知的客观事件。\n"
        "规则：\n"
        "1. 去掉内心、动机、心所、盘算、未表现出来的情绪。\n"
        "2. 去掉观察者不在场的私下行为（回家、夜里、事后、单独联系第三人等）。\n"
        "3. 去掉「表面 / 其实 / 实则」等叙述口吻；可见情绪写成可见事实"
        "（「表面开心」→「陈默很开心」，不要写「表面开心」）。\n"
        "4. 不要替观察者下判断（禁止「口是心非」「看穿了」「在装」）。\n"
        "5. 几乎全不可见时，observable_event 写成"
        f"「{actor}没有做出可被{observer}察觉的行动。」visibility=none。\n"
        "6. 不编造原文没有的可见动作。\n"
        "7. 输出第三人称、一两句完整末事件正文；不要套「做出了如下行动」。\n"
        "8. 不要写观察者自己的行动或「某某看到」。\n"
        f"9. observable_event 必须以「{actor}」开头。\n"
        f"10. 禁止「{observer}看到 / 听见」；禁止把{observer}或其他人写成行动主语。\n"
        "11. 不把场景里的外部事实改写成行动者没做的任务。\n"
        "12. 不要输出「即促成善行」等心所解释残句。\n"
        "13. 回家后、观察者不在场的行为可去掉；当众说出「为何私下补数据」等公开言行必须保留。\n"
    )
    user = (
        f"行动者：{actor}\n"
        f"观察者：{observer}\n\n"
        f"行动者当时面对的场景（截断，仅供判断是否同场）：\n{scene or '（无）'}\n\n"
        f"行动者的完整预测行动：\n{raw_action}\n\n"
        "返回 JSON：\n"
        '{"observable_event":"","dropped":[],'
        '"visibility":"visible|partial|none","reason":""}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _strip_prediction_prompt(body: str) -> str:
    text = body.strip()
    for marker in ("请你预测", "请预测"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].rstrip("。；; \n")
    return text.strip()


def parse_event_file(path: Path) -> tuple[str, list[str]]:
    """返回 (分析对象行后的前言忽略, 事件正文列表按序号)。"""
    raw = path.read_text(encoding="utf-8")
    matches = list(_EVENT_RE.finditer(raw))
    if not matches:
        raise ValueError(f"未找到事件标签：{path}")
    bodies: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        body = _strip_prediction_prompt(raw[start:end].strip())
        if body:
            bodies.append(body)
    return raw, bodies


# 与引擎 evaluate_from_yuanwen 一致：背景只带最近 3 条、各 600 字
_MAX_HIST_EVENTS = 3
_MAX_BODY_CHARS = 600


def build_role_context(focus_person: str, prior_events: list[str]) -> str:
    """单事件评估时的原文备忘（不是 H/M）。"""
    parts: list[str] = [f"分析对象：{focus_person}"]
    if not prior_events:
        return parts[0]
    start = max(0, len(prior_events) - _MAX_HIST_EVENTS)
    for i, body in enumerate(prior_events[start:], start=start + 1):
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
        "mental_episodes": len(state.mental_episodes or []),
        "cognitive_frames": len(state.cognitive_frames or []),
    }


@dataclass
class AgentLedger:
    """单个分析对象的事件账本（脚本层状态，不进模型代码）。"""

    focus_person: str
    history: list[str] = field(default_factory=list)
    pending_final: str | None = None
    predict_count: int = 0
    last_action: str | None = None
    last_action_prob: float | None = None
    prologue_warmed: bool = False

    def build_yuanwen(self) -> str:
        if not self.pending_final:
            raise ValueError(f"{self.focus_person}：缺少末事件，无法预测")
        if not self.history:
            raise ValueError(f"{self.focus_person}：前序为空")
        lines = [f"分析对象：{self.focus_person}", ""]
        for i, body in enumerate(self.history, start=1):
            lines.append(f"事件{i}：{body}")
            lines.append("")
        lines.append(f"事件{len(self.history) + 1}：{self.pending_final}")
        return "\n".join(lines).strip() + "\n"

    def sealed_event(self, action: str) -> str:
        if not self.pending_final:
            raise ValueError(f"{self.focus_person}：无末事件可落地")
        return (
            f"{self.pending_final.rstrip('。')}。"
            f"{self.focus_person}随后的行动：{action}"
        )

    def seal_own_action(self, action: str) -> str:
        """把当前末事件 + 己方行动落成前序。返回落地正文。"""
        sealed = self.sealed_event(action)
        self.history.append(sealed)
        self.pending_final = None
        return sealed

    def set_stimulus(self, stimulus: str) -> None:
        """设置新的末事件（对方行动或外部事件，不含己方行动）。"""
        self.pending_final = stimulus.strip()


@dataclass
class StepRecord:
    step_id: str
    agent: str
    stimulus: str
    action: str
    probability: float
    top_k: list[dict]
    give_to_other: bool
    note: str = ""
    report_path: str = ""
    observable_event: str = ""
    observable_dropped: list[str] = field(default_factory=list)
    observable_visibility: str = ""
    observable_reason: str = ""
    observable_source: str = ""


def top_behavior(result) -> tuple[str, float, list[dict]]:
    """从评估结果取跨支融合行为 Top-1 与 Top-K。"""
    return top_aggregated_action(result)


def _other_agent(who: str) -> str:
    return "林可" if who == "陈默" else "陈默"


def select_step_action(
    who: str,
    top_k: list[dict],
    *,
    step_id: str = "",
) -> tuple[str, float, dict]:
    """从 Top-K 选第一条可用、主语正确的行动。"""
    skipped: list[dict] = []
    candidates = list(top_k or [])
    other = _other_agent(who)
    for idx, item in enumerate(candidates):
        text = str(item.get("text") or "").strip()
        prob = float(item.get("probability") or 0.0)
        if not is_usable_predicted_action(text):
            skipped.append({"rank": idx, "reason": "unusable", "text": text})
            continue
        if text.startswith(other) and not text.startswith(who):
            skipped.append({"rank": idx, "reason": "wrong_subject", "text": text})
            continue
        return text, prob, {"selected_rank": idx, "skipped": skipped}
    raise DualLoopAbort(
        f"[{step_id}] {who} Top-K 全部不可用，闭环中止"
    )


@dataclass(frozen=True)
class LoopStep:
    step_id: str
    who: str
    give_to_other: bool
    note: str = ""
    external: tuple[str, str] | None = None


LOOP_STEPS: tuple[LoopStep, ...] = (
    LoopStep("S01_A1", "陈默", True, "首轮：林可求助场面"),
    LoopStep("S02_B1", "林可", True, "末事件=陈默 A1 的可见投影"),
    LoopStep("S03_A2", "陈默", True),
    LoopStep("S04_B2", "林可", True),
    LoopStep("S05_A3", "陈默", False, "陈默第3次预测，不给林可"),
    LoopStep(
        "S06_B3",
        "林可",
        True,
        "外部：发现私下补数据",
        ("林可", EXTERNAL_LINKE_HIDDEN_HELP),
    ),
    LoopStep("S07_A4", "陈默", True),
    LoopStep("S08_B4", "林可", True),
    LoopStep("S09_A5", "陈默", True),
    LoopStep("S10_B5", "林可", False, "林可本轮预测不给陈默"),
    LoopStep(
        "S11_A6",
        "陈默",
        True,
        "外部：CEO 48h 共案",
        ("陈默", EXTERNAL_CHENMO_JOINT_TASK),
    ),
    LoopStep("S12_B6", "林可", True),
    LoopStep("S13_A7", "陈默", True),
    LoopStep("S14_B7", "林可", False, "结束"),
)


class DualLoopRunner:
    """编排双实例 evaluate。"""

    def __init__(
        self,
        *,
        definitions_dir: Path,
        chenmo_file: Path,
        linke_file: Path,
        out_dir: Path,
        use_mock: bool = False,
        rebuild_prologue: bool = False,
        resume: bool = False,
    ) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rebuild_prologue = rebuild_prologue
        self.use_mock = use_mock
        self.resume = resume
        inner = ensure_tracing_client(create_llm_client(use_mock=use_mock))
        self.llm_chenmo = StagePrefixLlmClient(inner, "chenmo")
        self.llm_linke = StagePrefixLlmClient(inner, "linke")
        self.llm_filter = StagePrefixLlmClient(inner, "filter")
        self.model_a = MentalWorldModel.load(definitions_dir)
        self.model_b = MentalWorldModel.load(definitions_dir)
        # 分离 continuity 落盘，避免两人互相污染默认 runtime 文件
        runtime = self.out_dir / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        from mental_world_model.engine.continuity.store import ContinuityStateStore

        store_a = ContinuityStateStore(runtime / "continuity_chenmo.json")
        store_b = ContinuityStateStore(runtime / "continuity_linke.json")
        self.model_a.continuity_state_store = store_a
        self.model_b.continuity_state_store = store_b

        _, chenmo_events = parse_event_file(chenmo_file)
        _, linke_events = parse_event_file(linke_file)
        if len(chenmo_events) < 7:
            raise ValueError("陈默负面文件至少需要事件1–7")
        if len(linke_events) < 8:
            raise ValueError("林可负面文件至少需要事件1–8（仅前序，不要末事件）")

        self._prologue_bodies = {
            "陈默": list(chenmo_events[:7]),
            "林可": list(linke_events[:8]),
        }
        self._completed_steps: list[str] = []
        self._landed: dict | None = None
        self.warmup_brief: dict = {}
        self.log: list[StepRecord] = []

        # 陈默：1–7 前序；首末事件用求助场面（不用原文件事件8里的预测提示）
        self.agent_a = AgentLedger(
            focus_person="陈默",
            history=list(self._prologue_bodies["陈默"]),
            pending_final=CHENMO_FIRST_FINAL,
        )
        # 林可：1–8 纯前序；末事件等 A1
        self.agent_b = AgentLedger(
            focus_person="林可",
            history=list(self._prologue_bodies["林可"]),
            pending_final=None,
        )

        if resume and rebuild_prologue:
            raise ValueError("--resume 与 --rebuild-prologue 不能同时使用")

        if resume:
            progress_path = self._progress_path()
            if not progress_path.is_file():
                raise FileNotFoundError(
                    f"--resume 需要已有进度文件：{progress_path}"
                )
            self.model_a.continuity_state = store_a.load()
            self.model_b.continuity_state = store_b.load()
            if not (self.model_a.continuity_state.person_id or "").strip():
                self.model_a.continuity_state.person_id = "陈默"
            if not (self.model_b.continuity_state.person_id or "").strip():
                self.model_b.continuity_state.person_id = "林可"
            self._load_progress()
            mismatch = self._continuity_mismatch()
            if mismatch:
                raise ValueError(mismatch)
            print(
                f"续跑：已完成 {self._completed_steps or '（尚无步骤）'}；"
                f"下一步 {self._next_step_id() or '（已全部完成）'}"
            )
        else:
            if self._progress_path().is_file():
                raise FileExistsError(
                    f"输出目录已有 progress.json。续跑请加 --resume，"
                    f"换目录再开新跑：{self._progress_path()}"
                )
            self.model_a.continuity_state = ContinuityState(person_id="陈默")
            self.model_b.continuity_state = ContinuityState(person_id="林可")
            self.model_a.continuity_state_store.save(self.model_a.continuity_state)
            self.model_b.continuity_state_store.save(self.model_b.continuity_state)
            self._save_progress()

    def _progress_path(self) -> Path:
        return self.out_dir / "progress.json"

    def _next_step_id(self) -> str | None:
        done = set(self._completed_steps)
        for step in LOOP_STEPS:
            if step.step_id not in done:
                return step.step_id
        return None

    def _expected_event_count(self, who: str) -> int:
        warmup_n = len(self._prologue_bodies[who])
        n = warmup_n
        for step in LOOP_STEPS:
            if step.step_id in self._completed_steps and step.who == who:
                n += 1
        if self._landed and self._landed.get("who") == who:
            n += 1
        return n

    def _continuity_mismatch(self) -> str | None:
        for who, model in (("陈默", self.model_a), ("林可", self.model_b)):
            ledger = self._ledger(who)
            actual = int(getattr(model.continuity_state, "event_count", 0) or 0)
            warmed = bool(ledger.prologue_warmed)
            did_steps = any(
                step.who == who and step.step_id in self._completed_steps
                for step in LOOP_STEPS
            )
            landed_who = (self._landed or {}).get("who")
            if warmed or did_steps or landed_who == who:
                warmup_n = len(self._prologue_bodies[who])
                if actual == 0:
                    return (
                        f"{who} 的 continuity 为空（event_count=0），无法续跑。"
                        "请确认 runtime/continuity_*.json 未被清空。"
                    )
                if warmed and actual < warmup_n:
                    return (
                        f"{who} 已标记预热完成，但 continuity event_count={actual}，"
                        f"期望至少 {warmup_n}。"
                    )
                expected = self._expected_event_count(who)
                if warmed and actual < expected:
                    return (
                        f"{who} continuity event_count={actual}，"
                        f"与进度不符（期望至少 {expected}）。"
                    )
        return None

    def _ledger_to_dict(self, ledger: AgentLedger) -> dict:
        return {
            "focus_person": ledger.focus_person,
            "history": list(ledger.history),
            "pending_final": ledger.pending_final,
            "predict_count": ledger.predict_count,
            "last_action": ledger.last_action,
            "last_action_prob": ledger.last_action_prob,
            "prologue_warmed": ledger.prologue_warmed,
        }

    def _apply_ledger_dict(self, ledger: AgentLedger, data: dict) -> None:
        if not isinstance(data, dict):
            return
        history = data.get("history")
        if isinstance(history, list):
            ledger.history = [str(x) for x in history]
        if "pending_final" in data:
            pending = data.get("pending_final")
            ledger.pending_final = None if pending is None else str(pending)
        if "predict_count" in data:
            ledger.predict_count = int(data.get("predict_count") or 0)
        if "last_action" in data:
            action = data.get("last_action")
            ledger.last_action = None if action is None else str(action)
        if "last_action_prob" in data:
            prob = data.get("last_action_prob")
            ledger.last_action_prob = None if prob is None else float(prob)
        ledger.prologue_warmed = bool(data.get("prologue_warmed"))

    def _record_to_dict(self, record: StepRecord) -> dict:
        return {
            "step_id": record.step_id,
            "agent": record.agent,
            "stimulus": record.stimulus,
            "action": record.action,
            "probability": record.probability,
            "top_k": record.top_k,
            "give_to_other": record.give_to_other,
            "observable_event": record.observable_event,
            "observable_dropped": record.observable_dropped,
            "observable_visibility": record.observable_visibility,
            "observable_reason": record.observable_reason,
            "observable_source": record.observable_source,
            "note": record.note,
            "report_path": record.report_path,
        }

    def _record_from_dict(self, data: dict) -> StepRecord:
        return StepRecord(
            step_id=str(data.get("step_id") or ""),
            agent=str(data.get("agent") or ""),
            stimulus=str(data.get("stimulus") or ""),
            action=str(data.get("action") or ""),
            probability=float(data.get("probability") or 0.0),
            top_k=list(data.get("top_k") or []),
            give_to_other=bool(data.get("give_to_other")),
            note=str(data.get("note") or ""),
            report_path=str(data.get("report_path") or ""),
            observable_event=str(data.get("observable_event") or ""),
            observable_dropped=list(data.get("observable_dropped") or []),
            observable_visibility=str(data.get("observable_visibility") or ""),
            observable_reason=str(data.get("observable_reason") or ""),
            observable_source=str(data.get("observable_source") or ""),
        )

    def _save_progress(self) -> None:
        payload = {
            "protocol": "dual_agent_closed_loop_v4",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "resume": self.resume,
            "completed_steps": list(self._completed_steps),
            "next_step_id": self._next_step_id(),
            "landed": self._landed,
            "warmup": self.warmup_brief,
            "agent_a": self._ledger_to_dict(self.agent_a),
            "agent_b": self._ledger_to_dict(self.agent_b),
            "log": [self._record_to_dict(r) for r in self.log],
        }
        atomic_write_json(self._progress_path(), payload)

    def _load_progress(self) -> None:
        path = self._progress_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"进度文件损坏：{path}。请勿删 continuity；"
                "若同目录有 progress.json.tmp 且完整，可先备份再改名后 --resume。"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(f"进度文件无效：{self._progress_path()}")
        self._apply_ledger_dict(self.agent_a, raw.get("agent_a") or {})
        self._apply_ledger_dict(self.agent_b, raw.get("agent_b") or {})
        self._completed_steps = [
            str(x) for x in (raw.get("completed_steps") or []) if str(x)
        ]
        landed = raw.get("landed")
        self._landed = landed if isinstance(landed, dict) and landed.get("step_id") else None
        warmup = raw.get("warmup") or {}
        self.warmup_brief = warmup if isinstance(warmup, dict) else {}
        log_raw = raw.get("log") or []
        self.log = [
            self._record_from_dict(item)
            for item in log_raw
            if isinstance(item, dict)
        ]

    def _write_abort(self, message: str, *, step_id: str = "") -> None:
        payload = {
            "aborted_at": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "step_id": step_id or self._next_step_id() or "",
            "completed_steps": list(self._completed_steps),
            "next_step_id": self._next_step_id(),
        }
        atomic_write_json(self.out_dir / "abort.json", payload)
        self._save_progress()

    def _action_of(self, step_id: str) -> str:
        for record in self.log:
            if record.step_id == step_id:
                return record.action
        return ""

    def _llm_for(self, who: str):
        return self.llm_chenmo if who == "陈默" else self.llm_linke

    def _model_for(self, who: str) -> MentalWorldModel:
        return self.model_a if who == "陈默" else self.model_b

    def _ledger(self, who: str) -> AgentLedger:
        return self.agent_a if who == "陈默" else self.agent_b

    def ingest_prologue(self, who: str) -> dict:
        """把初始前序当解释事件跑一遍，写入 H/M。只应调用一次。"""
        ledger = self._ledger(who)
        model = self._model_for(who)
        bodies = self._prologue_bodies[who]
        if ledger.prologue_warmed:
            brief = continuity_brief(model.continuity_state)
            self.warmup_brief[who] = brief
            return brief
        done = int(getattr(model.continuity_state, "event_count", 0) or 0)
        if done >= len(bodies):
            ledger.prologue_warmed = True
            brief = continuity_brief(model.continuity_state)
            self.warmup_brief[who] = brief
            self._save_progress()
            return brief
        print(
            f"\n前序预热 {who} … {len(bodies)} 条（只算一次）"
            + (f"，从第 {done + 1} 条续" if done else "")
        )
        for idx in range(done, len(bodies)):
            body = bodies[idx]
            role_context = build_role_context(who, bodies[:idx])
            model.evaluate_text_with_cognition_branches(
                body,
                self._llm_for(who),
                role_context=role_context,
                update_seeds=False,
                update_continuity=True,
                use_action_feedback=True,
                aggregate_behaviors=False,
                predict_branch_behaviors=False,
                focus_person=who,
            )
            self._save_progress()
        ledger.prologue_warmed = True
        brief = continuity_brief(model.continuity_state)
        self.warmup_brief[who] = brief
        print(
            f"前序预热 {who} 完成：M episodes={brief.get('mental_episodes')} "
            f"H frames={brief.get('cognitive_frames')}"
        )
        self._save_progress()
        return brief

    def _pick_and_guard_action(
        self,
        *,
        who: str,
        step_id: str,
        result,
    ) -> tuple[str, float, list[dict], dict]:
        action, prob, top_k = top_behavior(result)
        ranked = list(top_k or [])
        if not ranked:
            ranked = [{"text": action, "probability": prob}]
        action, prob, pick_meta = select_step_action(
            who, ranked, step_id=step_id
        )
        return action, prob, ranked, pick_meta

    def _land_predicted(
        self,
        who: str,
        bundle,
        action: str,
        *,
        stimulus: str,
        role_context: str,
        probability: float,
    ) -> dict:
        """把 Top-1 接到预测所缘上写 H/M，不重抽 L2/L3。"""
        model = self._model_for(who)
        model.apply_predicted_action(
            stimulus,
            self._llm_for(who),
            bundle,
            action,
            role_context=role_context,
            focus_person=who,
            action_probability=probability,
            update_seeds=False,
        )
        return continuity_brief(model.continuity_state)

    def _finish_step(
        self,
        *,
        who: str,
        step_id: str,
        give_to_other: bool,
        note: str,
        stimulus: str,
        yuanwen: str,
        result,
        action: str,
        prob: float,
        top_k: list[dict],
        extra_meta: dict | None = None,
    ) -> StepRecord:
        ledger = self._ledger(who)
        ledger.predict_count += 1
        ledger.last_action = action
        ledger.last_action_prob = prob

        projection: ObservableProjection | None = None
        other = "林可" if who == "陈默" else "陈默"
        if give_to_other:
            projection = project_observable_event(
                llm=self.llm_filter,
                use_mock=self.use_mock,
                actor=who,
                observer=other,
                raw_action=action,
                scene=stimulus,
            )

        report_path = self.out_dir / f"{step_id}_{who}.json"
        if result is not None and hasattr(result, "to_event_report_dict"):
            payload = result.to_event_report_dict()
        elif report_path.is_file():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = {}
            payload = loaded if isinstance(loaded, dict) else {}
        else:
            payload = {}
        payload["_dual_loop"] = {
            "step_id": step_id,
            "focus_person": who,
            "stimulus": stimulus,
            "top_action": action,
            "top_probability": prob,
            "give_to_other": give_to_other,
            "note": note,
            "yuanwen": yuanwen,
            "mode": "rebuild" if self.rebuild_prologue else "incremental",
        }
        if extra_meta:
            payload["_dual_loop"].update(extra_meta)
        if projection is not None:
            payload["_dual_loop"]["observable"] = {
                "event": projection.observable_event,
                "dropped": projection.dropped,
                "visibility": projection.visibility,
                "reason": projection.reason,
                "source": projection.source,
            }

        record = StepRecord(
            step_id=step_id,
            agent=who,
            stimulus=stimulus,
            action=action,
            probability=prob,
            top_k=top_k,
            give_to_other=give_to_other,
            note=note,
            report_path=str(report_path),
            observable_event=projection.observable_event if projection else "",
            observable_dropped=list(projection.dropped) if projection else [],
            observable_visibility=projection.visibility if projection else "",
            observable_reason=projection.reason if projection else "",
            observable_source=projection.source if projection else "",
        )
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log.append(record)
        if give_to_other and projection is not None:
            self.feed_other_action(other, record)
        if step_id not in self._completed_steps:
            self._completed_steps.append(step_id)
        self._landed = None
        self._save_progress()
        shown = action if len(action) <= 120 else action[:120] + "…"
        print(
            f"[{step_id}] {who} Top-1 p={prob:.3f} "
            f"{'→给对方' if give_to_other else '✗不给对方'}\n  {shown}"
        )
        if projection is not None:
            obs = projection.observable_event
            obs_shown = obs if len(obs) <= 120 else obs[:120] + "…"
            print(
                f"[{step_id}] 可见投影 →{other} "
                f"[{projection.visibility}/{projection.source}]\n  {obs_shown}"
            )
        return record

    def predict(
        self,
        who: str,
        *,
        step_id: str,
        give_to_other: bool,
        note: str = "",
    ) -> StepRecord:
        if self.rebuild_prologue:
            return self._predict_rebuild(
                who, step_id=step_id, give_to_other=give_to_other, note=note
            )
        return self._predict_incremental(
            who, step_id=step_id, give_to_other=give_to_other, note=note
        )

    def _predict_incremental(
        self,
        who: str,
        *,
        step_id: str,
        give_to_other: bool,
        note: str = "",
    ) -> StepRecord:
        ledger = self._ledger(who)
        model = self._model_for(who)
        if self._landed and self._landed.get("step_id") == step_id:
            print(f"\n[{step_id}] 已写入 H/M，只补过滤/转交 …")
            return self._finish_from_landed()
        if not ledger.prologue_warmed:
            self.ingest_prologue(who)
        if not ledger.pending_final:
            raise ValueError(f"{who}：缺少末事件，无法预测")
        yuanwen = ledger.build_yuanwen()
        stimulus = ledger.pending_final
        role_context = build_role_context(who, ledger.history)
        hm_before = continuity_brief(model.continuity_state)

        print(
            f"\n[{step_id}] 增量评估 {who} …"
            f"（已有前序 {len(ledger.history)}，只评末事件；"
            f"M={hm_before.get('mental_episodes')}）"
        )
        try:
            bundle = model.evaluate_text_with_cognition_branches(
                stimulus,
                self._llm_for(who),
                role_context=role_context,
                update_seeds=False,
                update_continuity=False,
                use_action_feedback=False,
                aggregate_behaviors=True,
                predict_branch_behaviors=True,
                focus_person=who,
            )
            result = bundle.evaluation_result
            result.focus_person = who
            action, prob, top_k, pick_meta = self._pick_and_guard_action(
                who=who,
                step_id=step_id,
                result=result,
            )
        except DualLoopAbort as exc:
            self._write_abort(str(exc), step_id=step_id)
            raise
        except Exception as exc:
            self._write_abort(f"{type(exc).__name__}: {exc}", step_id=step_id)
            raise

        sealed = ledger.sealed_event(action)
        hm_after = self._land_predicted(
            who,
            bundle,
            action,
            stimulus=stimulus,
            role_context=role_context,
            probability=prob,
        )
        extra_meta = {
            "continuity_before": hm_before,
            "continuity_after_land": hm_after,
            "landed_prologue": sealed,
            "land_mode": "apply_predicted_action",
            "selected_rank": pick_meta.get("selected_rank"),
            "skipped_candidates": pick_meta.get("skipped") or [],
        }
        self._landed = {
            "step_id": step_id,
            "who": who,
            "action": action,
            "prob": prob,
            "stimulus": stimulus,
            "yuanwen": yuanwen,
            "give_to_other": give_to_other,
            "note": note,
            "top_k": top_k,
            "sealed": sealed,
            "extra_meta": extra_meta,
        }
        self._save_progress()
        ledger.history.append(sealed)
        ledger.pending_final = None

        return self._finish_step(
            who=who,
            step_id=step_id,
            give_to_other=give_to_other,
            note=note,
            stimulus=stimulus,
            yuanwen=yuanwen,
            result=result,
            action=action,
            prob=prob,
            top_k=top_k,
            extra_meta=extra_meta,
        )

    def _finish_from_landed(self) -> StepRecord:
        landed = self._landed or {}
        who = str(landed.get("who") or "")
        ledger = self._ledger(who)
        sealed = str(landed.get("sealed") or "")
        extra = landed.get("extra_meta") or {}
        if not sealed and isinstance(extra, dict):
            sealed = str(extra.get("landed_prologue") or "")
        if sealed:
            if not ledger.history or ledger.history[-1] != sealed:
                ledger.history.append(sealed)
            ledger.pending_final = None
        return self._finish_step(
            who=who,
            step_id=str(landed.get("step_id") or ""),
            give_to_other=bool(landed.get("give_to_other")),
            note=str(landed.get("note") or ""),
            stimulus=str(landed.get("stimulus") or ""),
            yuanwen=str(landed.get("yuanwen") or ""),
            result=None,
            action=str(landed.get("action") or ""),
            prob=float(landed.get("prob") or 0.0),
            top_k=list(landed.get("top_k") or []),
            extra_meta=dict(extra) if isinstance(extra, dict) else {},
        )

    def _predict_rebuild(
        self,
        who: str,
        *,
        step_id: str,
        give_to_other: bool,
        note: str = "",
    ) -> StepRecord:
        ledger = self._ledger(who)
        model = self._model_for(who)
        yuanwen = ledger.build_yuanwen()
        stimulus = ledger.pending_final or ""

        model.continuity_state = ContinuityState(person_id=who)
        if model.continuity_state_store is not None:
            model.continuity_state_store.save(model.continuity_state)

        print(f"\n[{step_id}] 整篇重跑 {who} …（前序 {len(ledger.history)} + 末事件）")
        try:
            result = model.evaluate_from_yuanwen(
                yuanwen,
                self._llm_for(who),
                focus_person=who,
                update_seeds=False,
            )
            action, prob, top_k, _pick_meta = self._pick_and_guard_action(
                who=who,
                step_id=step_id,
                result=result,
            )
        except DualLoopAbort as exc:
            self._write_abort(str(exc), step_id=step_id)
            raise
        except Exception as exc:
            self._write_abort(f"{type(exc).__name__}: {exc}", step_id=step_id)
            raise
        ledger.seal_own_action(action)
        return self._finish_step(
            who=who,
            step_id=step_id,
            give_to_other=give_to_other,
            note=note,
            stimulus=stimulus,
            yuanwen=yuanwen,
            result=result,
            action=action,
            prob=prob,
            top_k=top_k,
        )

    def feed_other_action(self, receiver: str, record: StepRecord) -> None:
        """用已过滤的可见事件作为对方当前末事件。"""
        event = (record.observable_event or "").strip()
        if not event:
            raise ValueError(f"{record.step_id}：缺少可见投影，不能转交")
        self._ledger(receiver).set_stimulus(event)

    def feed_external(self, who: str, text: str) -> None:
        self._ledger(who).set_stimulus(text)

    def run(self) -> dict:
        try:
            return self._run_loop()
        except DualLoopAbort:
            raise
        except Exception as exc:
            if not (self.out_dir / "abort.json").is_file():
                self._write_abort(f"{type(exc).__name__}: {exc}")
            else:
                self._save_progress()
            raise

    def _run_loop(self) -> dict:
        if not self.rebuild_prologue:
            self.ingest_prologue("陈默")
            self.ingest_prologue("林可")

        for step in LOOP_STEPS:
            if step.step_id in self._completed_steps:
                print(f"[{step.step_id}] 已完成，跳过")
                continue
            already_landed = bool(
                self._landed and self._landed.get("step_id") == step.step_id
            )
            if step.external is not None and not already_landed:
                ext_who, ext_text = step.external
                self.feed_external(ext_who, ext_text)
                self._save_progress()
            self.predict(
                step.who,
                step_id=step.step_id,
                give_to_other=step.give_to_other,
                note=step.note,
            )

        return self._write_summary()

    def _write_summary(self) -> dict:
        summary = {
            "protocol": "dual_agent_closed_loop_v4",
            "mode": "rebuild" if self.rebuild_prologue else "incremental",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "warmup": self.warmup_brief,
            "final_continuity": {
                "陈默": continuity_brief(self.model_a.continuity_state),
                "林可": continuity_brief(self.model_b.continuity_state),
            },
            "externals": {
                "after_A3": EXTERNAL_LINKE_HIDDEN_HELP,
                "after_B5": EXTERNAL_CHENMO_JOINT_TASK,
            },
            "observable_filter": "script-layer; own ledger keeps raw Top-1",
            "steps": [self._record_to_dict(r) for r in self.log],
            "final_history_chenmo_events": len(self.agent_a.history),
            "final_history_linke_events": len(self.agent_b.history),
            "unused_prediction_A3_not_fed_to_B": self._action_of("S05_A3"),
            "unused_prediction_B5_not_fed_to_A": self._action_of("S10_B5"),
            "final_B7": self._action_of("S14_B7"),
        }
        summary_path = self.out_dir / "dual_loop_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.out_dir / "final_yuanwen_chenmo.txt").write_text(
            "\n".join(
                f"事件{i}：{b}" for i, b in enumerate(self.agent_a.history, 1)
            ),
            encoding="utf-8",
        )
        (self.out_dir / "final_yuanwen_linke.txt").write_text(
            "\n".join(
                f"事件{i}：{b}" for i, b in enumerate(self.agent_b.history, 1)
            ),
            encoding="utf-8",
        )
        print(f"\n汇总已写：{summary_path}")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="双人心智世界模型闭环测试（模型外）")
    parser.add_argument(
        "--definitions",
        type=Path,
        default=_ROOT / "definitions",
        help="定义目录",
    )
    parser.add_argument(
        "--chenmo-file",
        type=Path,
        default=_ROOT / "data" / "陈默事件（负面版）.txt",
    )
    parser.add_argument(
        "--linke-file",
        type=Path,
        default=_ROOT / "data" / "林可负面事件.txt",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认 reports/自建实验/dual_loop_<timestamp>）",
    )
    parser.add_argument("--mock", action="store_true", help="使用 Mock LLM")
    parser.add_argument(
        "--rebuild-prologue",
        action="store_true",
        help="旧行为：每次整篇重跑前序并重置 H/M（默认增量，前序只算一次）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --output-dir 的 progress.json 与 continuity 续跑，不清空 H/M",
    )
    args = parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error("--resume 需要同时指定 --output-dir")
    if args.resume and args.rebuild_prologue:
        parser.error("--resume 与 --rebuild-prologue 不能同时使用")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (_ROOT / "reports" / "自建实验" / f"dual_loop_{stamp}")

    runner = DualLoopRunner(
        definitions_dir=args.definitions,
        chenmo_file=args.chenmo_file,
        linke_file=args.linke_file,
        out_dir=out_dir,
        use_mock=args.mock,
        rebuild_prologue=args.rebuild_prologue,
        resume=args.resume,
    )
    try:
        runner.run()
    except DualLoopAbort as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
