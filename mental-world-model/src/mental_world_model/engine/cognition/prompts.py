"""分支想/思与分支行为预测提示词。"""

from __future__ import annotations

from mental_world_model.engine.perception.config import COGNITION_PROMPT_VERSION
from mental_world_model.engine.perception.focus import (
    focus_person_instruction,
    focus_person_user_line,
)
from mental_world_model.engine.perception.models import (
    GraspingHypothesis,
    ObjectiveFact,
    SubjectiveAlambanaCandidate,
)


def _grasping_block(items: list[GraspingHypothesis]) -> str:
    if not items:
        return "（无）"
    return "\n".join(f"- [{h.id}] {h.text}" for h in items)


def _reactivated_h_block(items: list[dict] | None) -> str:
    if not items:
        return "（本支未绑定历史总执）"
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        hid = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not hid or not text:
            continue
        try:
            mass = float(item.get("mass") or 0.0)
        except (TypeError, ValueError):
            mass = 0.0
        lines.append(f"- [{hid}] mass={mass:.3f} {text}")
    return "\n".join(lines) if lines else "（本支未绑定历史总执）"


def _facts_block(facts: list[ObjectiveFact]) -> str:
    if not facts:
        return "（无）"
    return "\n".join(f"- {f.text}（片段：{f.source_span}）" for f in facts)


def build_branch_behavior_messages(
    yuanwen: str,
    alambana: SubjectiveAlambanaCandidate,
    xiang_text: str,
    si_text: str,
    active_items: list[tuple[str, float, object]],
    *,
    role_context: str = "",
    branch_id: str = "",
    focus_person: str = "",
) -> list[dict[str, str]]:
    """组装单分支行为预测消息。

    调用方只传入本支数据（代码隔离）；提示中自然融合心所·业与所缘/想/思。
    ``active_items`` 为 ``(name, score, Xinsuo)``，建议已按分数降序。
    """
    lines: list[str] = []
    for idx, (name, score, xinsuo) in enumerate(active_items, start=1):
        ye = getattr(xinsuo, "ye", "") or ""
        lines.append(
            f"{idx}. {name}（强度 {float(score):.3f}）\n"
            f"   业：{ye}"
        )
    xinsuo_block = "\n".join(lines) if lines else "（无有值心所）"
    branch_tag = f"（{branch_id}）" if branch_id else ""
    who = str(focus_person or "").strip() or "此人"
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={COGNITION_PROMPT_VERSION}] 你是行为预测器。\n"
        f"{focus_block}"
        f"根据下方给定的心理状态，预测「{who}」在该状态下可能做出的行为。\n"
        f"{_behavior_response_instruction(who)}"
        "有值心所附带「业」说明（该类心所常见的造作趋向），"
        "并与原文、所缘、想、思一起理解；"
        "行为由所缘、想、思和心所共同约束，但不是把想翻译成动作。\n"
        "text 必须是第三人称、已经做出的可观察动作（说话、点头、离开、提问、沉默等）。\n"
        "禁止在 text 里写心所名、「业」、「内心」、「这符合」；这些只能出现在 supporting_mental_factors。\n"
        "禁止写「可能会」「可能通过」「开始思考」等未发生或纯内心措辞。\n"
        "禁止预测其他人物的行为。\n"
        "conditional_probability 是在该心理状态下的行为条件概率。"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"原文：\n{yuanwen}\n\n"
        f"主观所缘{branch_tag}：{alambana.text}\n"
        f"想：{xiang_text or '（无）'}\n"
        f"思：{si_text or '（无）'}\n\n"
        f"有值心所及业（按强度）：\n{xinsuo_block}\n\n"
        f"角色：{role_context or '（无）'}\n\n"
        "任务：给出 2～3 个最可能行为（按可能性排序，最多 3 条）；"
        "supporting_mental_factors 只能从上方有值心所名称中选取。\n"
        '返回 JSON：{"behavior_predictions":[{"behavior_id":"","text":"",'
        '"conditional_probability":0.0,"confidence":0.0,'
        '"supporting_mental_factors":[]}]}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


BEHAVIOR_SANITIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "cleaned_text": {"type": "string"},
                },
                "required": ["index", "cleaned_text"],
            },
        }
    },
    "required": ["items"],
}


def build_behavior_sanitize_messages(
    texts: list[str],
    *,
    extra_names: set[str] | frozenset[str] | None = None,
) -> list[dict[str, str]]:
    """把行为预测改写成已发生的可观察动作，由模型判断而非词表。"""
    names = "、".join(sorted(n for n in (extra_names or ()) if len(n) >= 2)) or "（无）"
    payload = [{"index": i, "text": t} for i, t in enumerate(texts)]
    system = (
        "你把行为预测改写成「已经做出的可观察动作」。\n"
        "去掉心所名、业解释、「这符合…」、「可能会/可能通过」等未发生或纯内心措辞。\n"
        "保留「不可能」等否定；不要把「不但」拆开；「当众质问为何私下…」这类公开言行要留。\n"
        "可观察不等于刺激最后一镜已经伸出的手。"
        "对当前取境的回应，只要可被看见或听见即可，"
        "不要因为刺激没写出该动作的准备过程就清空。\n"
        "不要编造刺激里没有的人物或事实。"
        "没有可观察造作则 cleaned_text 为空字符串。"
    )
    user = (
        f"心所名（仅供识别，不要写进 cleaned_text）：{names}\n\n"
        f"待清洗：\n{payload}\n\n"
        '返回 JSON：{"items":[{"index":0,"cleaned_text":""}]}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


BRANCH_BEHAVIOR_SCHEMA = {
    "type": "object",
    "properties": {
        "behavior_predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "behavior_id": {"type": "string"},
                    "text": {"type": "string"},
                    "conditional_probability": {"type": "number"},
                    "confidence": {"type": "number"},
                    "supporting_mental_factors": {"type": "array"},
                },
                "required": ["text", "conditional_probability"],
            },
        }
    },
    "required": ["behavior_predictions"],
}

BEHAVIOR_CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "behavior_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "behavior_id": {"type": "string"},
                    "text": {"type": "string"},
                    "supporting_mental_factors": {"type": "array"},
                },
                "required": ["text"],
            },
        }
    },
    "required": ["behavior_candidates"],
}

BEHAVIOR_PROBABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "behavior_probabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_action": {"type": "string"},
                    "conditional_probability": {"type": "number"},
                    "confidence": {"type": "number"},
                    "supporting_mental_factors": {"type": "array"},
                },
                "required": ["canonical_action", "conditional_probability"],
            },
        }
    },
    "required": ["behavior_probabilities"],
}

BEHAVIOR_CANONICALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_action": {"type": "string"},
                    "surface_forms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["canonical_action", "surface_forms"],
            },
        }
    },
    "required": ["groups"],
}


def _behavior_context_block(
    yuanwen: str,
    alambana,
    xiang_text: str,
    si_text: str,
    active_items: list[tuple[str, float, object]],
    *,
    role_context: str = "",
    branch_id: str = "",
    focus_person: str = "",
) -> tuple[str, str, str]:
    lines: list[str] = []
    for idx, (name, score, xinsuo) in enumerate(active_items, start=1):
        ye = getattr(xinsuo, "ye", "") or ""
        lines.append(
            f"{idx}. {name}（强度 {float(score):.3f}）\n"
            f"   业：{ye}"
        )
    xinsuo_block = "\n".join(lines) if lines else "（无有值心所）"
    branch_tag = f"（{branch_id}）" if branch_id else ""
    who = str(focus_person or "").strip() or "此人"
    focus_block = focus_person_instruction(focus_person)
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"原文：\n{yuanwen}\n\n"
        f"主观所缘{branch_tag}：{alambana.text}\n"
        f"想：{xiang_text or '（无）'}\n"
        f"思：{si_text or '（无）'}\n\n"
        f"有值心所及业（按强度）：\n{xinsuo_block}\n\n"
        f"角色：{role_context or '（无）'}\n\n"
    )
    return who, focus_block, user


def _behavior_response_instruction(who: str) -> str:
    """行为预测共用：想的落法，不等于直接执行想。"""
    return (
        f"行为是「{who}」对当前取境的回应，必须是可被看见或听见的造作。\n"
        "行为是想在当前条件下的实际落法，不等于直接执行想。\n"
        "同一个想可以直接实现，也可以只实现一部分、延后实现或暂不实现；都属于有效回应。\n"
        "有几种不能互相替代的落法，就分别写几条；一条只表达一种落法。"
        "不要只写最直接完成想的那条，也不要只写最克制的那条。\n"
        "不要把刺激最后停住的接触当成已经回答了所缘。\n"
        "不要编造刺激里没有的人物或事实。"
        "人物不必此刻在场；对取境中已出现对象的回应，只要可被看见或听见即可。\n"
    )


def build_behavior_candidate_messages(
    yuanwen: str,
    alambana: SubjectiveAlambanaCandidate,
    xiang_text: str,
    si_text: str,
    active_items: list[tuple[str, float, object]],
    *,
    role_context: str = "",
    branch_id: str = "",
    focus_person: str = "",
    max_candidates: int = 4,
) -> list[dict[str, str]]:
    who, focus_block, user = _behavior_context_block(
        yuanwen,
        alambana,
        xiang_text,
        si_text,
        active_items,
        role_context=role_context,
        branch_id=branch_id,
        focus_person=focus_person,
    )
    system = (
        f"[prompt_version={COGNITION_PROMPT_VERSION}] 你是行为候选生成器。\n"
        f"{focus_block}"
        f"根据给定心理状态，列出「{who}」可能做出的可观察行为。\n"
        f"{_behavior_response_instruction(who)}"
        "先只生成候选，不要给概率。\n"
        "text 必须是第三人称、已经做出的可观察动作。"
        "候选必须在回应方式上互异，不要用不同措辞重复同一回应。\n"
        "不要为了凑数量生成明显不符合当前心理状态的行为。\n"
        f"给出 1～{max_candidates} 个真正不同且合理的行为。\n"
        "禁止写心所名、「业」、「可能会」。禁止预测其他人物。\n"
    )
    user += (
        "任务：生成互异可观察行为候选；supporting_mental_factors 只能从上方有值心所中选取。\n"
        '返回 JSON：{"behavior_candidates":[{"behavior_id":"","text":"",'
        '"supporting_mental_factors":[]}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_behavior_diversity_messages(
    yuanwen: str,
    alambana: SubjectiveAlambanaCandidate,
    xiang_text: str,
    si_text: str,
    active_items: list[tuple[str, float, object]],
    existing: list[str],
    *,
    role_context: str = "",
    branch_id: str = "",
    focus_person: str = "",
    max_candidates: int = 4,
) -> list[dict[str, str]]:
    who, focus_block, user = _behavior_context_block(
        yuanwen,
        alambana,
        xiang_text,
        si_text,
        active_items,
        role_context=role_context,
        branch_id=branch_id,
        focus_person=focus_person,
    )
    system = (
        f"[prompt_version={COGNITION_PROMPT_VERSION}] 你是行为多样性补全器。\n"
        f"{focus_block}"
        f"{_behavior_response_instruction(who)}"
        f"当前候选缺乏对取境的不同回应。请为「{who}」生成真正不同的行为，"
        "不要只改写措辞，也不要只换一种眼前接触。"
        "不要为了凑数量生成明显不符合当前心理状态的行为。\n"
    )
    user += (
        f"已有候选：{existing}\n"
        f"请重新给出 1～{max_candidates} 个回应方式互异的行为。\n"
        '返回 JSON：{"behavior_candidates":[{"behavior_id":"","text":"",'
        '"supporting_mental_factors":[]}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_behavior_probability_messages(
    yuanwen: str,
    alambana: SubjectiveAlambanaCandidate,
    xiang_text: str,
    si_text: str,
    active_items: list[tuple[str, float, object]],
    canonical_actions: list[str],
    *,
    role_context: str = "",
    branch_id: str = "",
    focus_person: str = "",
) -> list[dict[str, str]]:
    who, focus_block, user = _behavior_context_block(
        yuanwen,
        alambana,
        xiang_text,
        si_text,
        active_items,
        role_context=role_context,
        branch_id=branch_id,
        focus_person=focus_person,
    )
    system = (
        f"[prompt_version={COGNITION_PROMPT_VERSION}] 你是行为条件概率估计器。\n"
        f"{focus_block}"
        f"只给下面已经去重后的行为分配 P(A|{who}当前心理状态)。不要新增行为，不要改写动作本质。\n"
        "conditional_probability 是该心理状态下的条件概率。\n"
        "不要因为某条更容易由现场完成，就打更高。\n"
        "不要默认「最直接完成想」的概率最高；"
        "直接、部分、延后、暂不实现，都按当前条件给概率。\n"
    )
    user += (
        f"已去重行为：{canonical_actions}\n"
        "canonical_action 必须原样使用上表中的文本。\n"
        '返回 JSON：{"behavior_probabilities":[{"canonical_action":"",'
        '"conditional_probability":0.0,"confidence":0.0,'
        '"supporting_mental_factors":[]}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_behavior_canonicalize_messages(texts: list[str]) -> list[dict[str, str]]:
    payload = [{"index": i, "text": t} for i, t in enumerate(texts)]
    system = (
        "你把可观察行为按「动作本质」聚类。"
        "措辞不同但同一动作（如保持沉默 / 继续沉默 / 没有回应）必须并入同一组。\n"
        "真正不同的动作分开。canonical_action 用最简短的可观察描述。\n"
    )
    user = (
        f"待聚类：\n{payload}\n\n"
        '返回 JSON：{"groups":[{"canonical_action":"","surface_forms":[]}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_action_likelihood_messages(
    yuanwen: str,
    objective_event,
    facts: list,
    candidates: list[SubjectiveAlambanaCandidate],
    evidence,
    *,
    required_ids: list[str] | None = None,
    retry_missing_ids: list[str] | None = None,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """已有行动骨架时：只对各所缘打 L(Act|O)。"""
    event_json = (
        objective_event.to_dict()
        if objective_event is not None and hasattr(objective_event, "to_dict")
        else {}
    )
    ids = required_ids if required_ids is not None else [c.id for c in candidates]
    cand_block = [
        {
            "alambana_id": c.id,
            "text": c.text,
            "prior_probability": round(c.prior_probability, 4),
        }
        for c in candidates
    ]
    action_block = (
        f"span={getattr(evidence, 'action_span', '')}\n"
        f"description={getattr(evidence, 'action_description', '')}"
    )
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={COGNITION_PROMPT_VERSION}] 你是行动似然评分器。\n"
        f"{focus_block}"
        "行动证据（思）已给定，勿改写、勿增删；行动证据须理解为分析对象侧。\n"
        "对每个候选主观所缘 Oi，likelihood∈[0,1] 只回答：\n"
        "「若分析对象内心所缘是 Oi，出现上述行动证据有多自然？」\n"
        "likelihoods 必须且只能使用下方列出的 alambana_id，禁止自造或省略。\n"
        "不要输出后验概率（由本地 p'=normalize(p·L) 计算）。\n"
        "禁止输出心所名或行为预测清单。\n"
    )
    retry_note = ""
    if retry_missing_ids:
        retry_note = (
            f"\n[重试强提示] 上次缺少这些 ID，本次必须全部返回："
            f"{retry_missing_ids}\n"
        )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"原文：\n{yuanwen}\n\n"
        f"客观刺激：\n{event_json}\n\n"
        f"客观事实：\n{_facts_block(facts)}\n\n"
        f"已给定行动证据（思）：\n{action_block}\n\n"
        f"候选主观所缘及先验：\n{cand_block}\n\n"
        f"必须覆盖的 alambana_id：{ids}\n"
        f"{retry_note}"
        "返回 JSON：\n"
        '{"likelihoods":[{"alambana_id":"O1","likelihood":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_branch_xiang_messages(
    yuanwen: str,
    objective_event,
    candidate: SubjectiveAlambanaCandidate,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    evidence,
    *,
    prefer_passed_grasping: bool = False,
    focus_person: str = "",
    reactivated_h: list[dict] | None = None,
) -> list[dict[str, str]]:
    """L6：单支隔离生成想（调用方并行多支，勿同框塞入其他候选）。

    ``prefer_passed_grasping=True``：直接使用传入二执列表（runner 已 resolve 的本支执）。
    否则按 used_*_ids 过滤；无匹配则空，禁止回退到「更大范围」的执。
    ``reactivated_h`` 为本支已绑定的历史总执（id/text/mass）；想只读，不再判相关度。
    """
    event_json = (
        objective_event.to_dict()
        if objective_event is not None and hasattr(objective_event, "to_dict")
        else {}
    )
    used_self = list(candidate.used_self_ids or candidate.self_grasping_ids)
    used_dharma = list(candidate.used_dharma_ids or candidate.dharma_grasping_ids)
    if prefer_passed_grasping:
        self_used = list(self_grasping)
        dharma_used = list(dharma_grasping)
    else:
        self_used = (
            [h for h in self_grasping if h.id in used_self] if used_self else list(self_grasping)
        )
        dharma_used = (
            [h for h in dharma_grasping if h.id in used_dharma]
            if used_dharma
            else list(dharma_grasping)
        )
    # 想依赖所缘正文 + 二执 + 本支已绑定 H + 行动骨架；不吃 prior/posterior
    cand_block = {
        "id": candidate.id,
        "text": candidate.text,
        "used_self_ids": used_self,
        "used_dharma_ids": used_dharma,
    }
    action_block = "（无）"
    if getattr(evidence, "has_observable_action", False):
        action_block = (
            f"span={evidence.action_span}\n"
            f"description={evidence.action_description}"
        )
    focus_block = focus_person_instruction(focus_person)
    who = str(focus_person or "").strip() or "主体"
    system = (
        f"[prompt_version={COGNITION_PROMPT_VERSION}] 你是「想」生成器。\n"
        f"{focus_block}"
        "本调用只处理一个主观所缘分支；根据该支所缘、相关二执，"
        "以及本支已绑定的历史总执 H 生成「想」。\n"
        "已给出的 H 就是本支要用的，不要再判断是否相关，不要另选 H。\n"
        "mass 越大，越应把该命题当作既有评价惯性来组织想；H 不另起所缘。\n"
        "有所缘就必须有想；禁止空字符串。\n"
        f"想 = 「{who}」在该所缘下的认知整理 + 评价 + 内心造作趋向。\n"
        "禁止输出心所名；禁止行为清单预测。\n"
        "禁止改写成其他人物的想。\n"
        "若有行动证据（内心造作 / 外在表现 / 明确无反应，以描述为准）："
        "想须与「该支所缘 + 该行动证据」相容。"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"原文：\n{yuanwen}\n\n"
        f"客观所缘境：\n{event_json}\n\n"
        f"相关我执：\n{_grasping_block(self_used)}\n\n"
        f"相关法执：\n{_grasping_block(dharma_used)}\n\n"
        f"本支已绑定的历史总执 H（只读；不要再做相关度判断）：\n"
        f"{_reactivated_h_block(reactivated_h)}\n\n"
        f"本支主观所缘：\n{cand_block}\n\n"
        f"行动证据（思）：\n{action_block}\n\n"
        f"为 alambana_id={candidate.id} 返回一条想。\n"
        f'返回：{{"xiangs":[{{"alambana_id":"{candidate.id}","text":"",'
        '"source":"推理","confidence":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


ACTION_LIKELIHOOD_SCHEMA = {
    "type": "object",
    "properties": {
        "likelihoods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "alambana_id": {"type": "string"},
                    "likelihood": {"type": "number"},
                },
                "required": ["alambana_id", "likelihood"],
            },
        },
    },
    "required": ["likelihoods"],
}

BRANCH_XIANGS_SCHEMA = {
    "type": "object",
    "properties": {
        "xiangs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "alambana_id": {"type": "string"},
                    "text": {"type": "string"},
                    "source": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["alambana_id", "text"],
            },
        }
    },
    "required": ["xiangs"],
}
