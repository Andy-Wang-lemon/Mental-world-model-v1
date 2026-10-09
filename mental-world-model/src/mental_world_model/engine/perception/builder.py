"""主观世界建构：刺激/行动拆分 → 客观刺激 → 我执+法执 → 组合 → 所缘。

感知只用 stimulus_text；行动骨架留给 cognition（思 / L / 想）。
本模块只产出先验分布。
"""

from __future__ import annotations

from mental_world_model.engine.continuity.prior import (
    bind_h_scope_to_fresh,
    h_texts_by_id,
    resolve_h_scope,
)
from mental_world_model.engine.extraction.validators import is_grounded_in_yuanwen
from mental_world_model.engine.perception.combo_selection import (
    enumerate_candidate_combinations,
    select_grasping_combinations,
)
from mental_world_model.engine.perception.config import (
    ACCOMMODATION_MIN_FIT,
    ALAMBANA_DEFAULT_CONFIDENCE,
    ALAMBANA_MIN_CONFIDENCE,
    ALAMBANA_MIN_PROBABILITY,
    ALAMBANA_SOFTMAX_TEMPERATURE,
    ALAMBANA_VALIDATION_MAX_RETRIES,
    MAX_ALAMBANA_BRANCHES,
)
from mental_world_model.engine.perception.grasping_paths import (
    extract_accommodation_grasping,
    extract_fresh_grasping,
    make_neutral_graspings,
    merge_and_dedup_graspings,
)
from mental_world_model.engine.perception.grounding import (
    apply_grounding_with_retry,
    dedup_subjective_alambanas,
)
from mental_world_model.engine.perception.semantic_confirm import PerspectiveNearCache
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
    ObjectiveEventStructure,
    ObjectiveFact,
    PerceptionState,
    StimulusActionSplit,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import (
    COMBO_META_SCHEMA,
    OBJECTIVE_EVENT_SCHEMA,
    STIMULUS_ACTION_SPLIT_SCHEMA,
    SUBJECTIVE_ALAMBANA_SCHEMA,
    build_combo_meta_messages,
    build_objective_event_messages,
    build_stimulus_action_split_messages,
    build_subjective_alambana_messages,
)
from mental_world_model.engine.perception.softmax import (
    apply_local_weighted_softmax,
)
from mental_world_model.engine.perception.validators import (
    parse_alambana_candidate_raw,
    parse_objective_event,
    validate_alambana_candidate,
    validate_objective_event,
)
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context


def split_stimulus_and_action(
    text: str,
    llm: LlmClient,
    role_context: str = "",
    *,
    expect_action: bool = True,
    focus_person: str = "",
) -> tuple[StimulusActionSplit, list[str]]:
    """方案 A：一次 LLM 拆出刺激文本与行动证据骨架。

    expect_action=True（历史/解释事件）：契约保证有行动；LLM 只拆内容，
    不判有无、不打置信度（本地 confidence=1.0）。
    expect_action=False（待预测事件）：不调用拆分 LLM，刺激=全文、无行动。
    """
    notes: list[str] = []
    if not expect_action:
        notes.append("待预测事件：跳过刺激/行动拆分 LLM，刺激=全文。")
        return (
            StimulusActionSplit(
                stimulus_text=text.strip(),
                has_observable_action=False,
                confidence=0.0,
                reason_summary="待预测：无行动拆分",
            ),
            notes,
        )

    with llm_call_context(stage="perception.stimulus_action_split"):
        response = llm.complete(
            build_stimulus_action_split_messages(
                text, role_context=role_context, focus_person=focus_person
            ),
            schema=STIMULUS_ACTION_SPLIT_SCHEMA,
        )
    raw = response if isinstance(response, dict) else {}
    stimulus = str(raw.get("stimulus_text") or "").strip()
    span = str(raw.get("action_span") or "").strip()
    desc = str(raw.get("action_description") or "").strip()
    reason = str(raw.get("reason_summary") or "").strip()

    if not stimulus:
        stimulus = text.strip()
        notes.append("刺激文本为空，降级为全文（可能含行动污染风险）。")

    if span:
        span_parts = [p.strip() for p in span.replace("；", ";").split(";") if p.strip()]
        grounded = [p for p in span_parts if is_grounded_in_yuanwen(p, text)]
        if span_parts and not grounded:
            # 契约：解释事件行动已干净分离；不强行整段作废
            notes.append(
                "行动片段未能严格匹配原文，按契约保留 span/description（不作废）。"
            )
        elif grounded and len(grounded) < len(span_parts):
            notes.append("部分行动片段未匹配原文，已剔除未落地段。")
            span = ";".join(grounded)
        elif grounded:
            span = ";".join(grounded)

    # 契约：解释事件必有行动；置信度本地固定，不经 LLM
    has_action = bool(span or desc)
    if not has_action:
        notes.append("行动 span/description 皆空，违反「必有行动」契约，降级为无行动。")
    else:
        has_action = True

    split = StimulusActionSplit(
        stimulus_text=stimulus,
        has_observable_action=has_action,
        action_span=span if has_action else "",
        action_description=desc if has_action else "",
        confidence=1.0 if has_action else 0.0,
        reason_summary=reason or ("已拆分刺激与行动" if has_action else "行动缺失"),
    )
    notes.append(
        "已拆分刺激/行动（本地认定有行动，confidence=1.0）。"
        if has_action
        else "拆分完成但行动字段为空。"
    )
    return split, notes


def parse_objective_event_structure(
    text: str,
    llm: LlmClient,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> tuple[ObjectiveEventStructure, list[str]]:
    notes: list[str] = []
    with llm_call_context(stage="perception.objective_event"):
        response = llm.complete(
            build_objective_event_messages(
                text, role_context=role_context, focus_person=focus_person
            ),
            schema=OBJECTIVE_EVENT_SCHEMA,
        )
    raw = response if isinstance(response, dict) else {}
    event = parse_objective_event(raw)
    validated, vnotes = validate_objective_event(event)
    notes.extend(vnotes)
    if validated is None:
        fallback = ObjectiveEventStructure(scene=text.strip()[:500], action="")
        notes.append("客观事件解析失败，降级为原文场景摘要。")
        return fallback, notes
    return validated, notes


def extract_grasping_pair(
    text: str,
    objective_event: ObjectiveEventStructure,
    llm: LlmClient,
    role_context: str = "",
    cognitive_bias_frames: list[dict] | None = None,
    *,
    focus_person: str = "",
) -> tuple[list[GraspingHypothesis], list[GraspingHypothesis], list[str]]:
    """兼容入口：仅 Fresh 路径，不再立刻插入 neutral。"""
    del cognitive_bias_frames
    return extract_fresh_grasping(
        text,
        objective_event,
        llm,
        role_context=role_context,
        focus_person=focus_person,
    )


def _fetch_combo_meta(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    candidates: list[GraspingCombination],
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[dict[str, dict], list[str]]:
    notes: list[str] = []
    if not candidates:
        return {}, notes
    try:
        with llm_call_context(stage="perception.combo_meta"):
            response = llm.complete(
                build_combo_meta_messages(
                    text,
                    objective_event,
                    self_grasping,
                    dharma_grasping,
                    candidates,
                    focus_person=focus_person,
                ),
                schema=COMBO_META_SCHEMA,
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"组合元数据评估失败，使用默认对齐分：{exc}")
        return {}, notes
    raw = response if isinstance(response, dict) else {}
    items = raw.get("items")
    if not isinstance(items, list):
        notes.append("组合元数据无 items，使用默认对齐分。")
        return {}, notes
    meta: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("combo_id") or "").strip()
        if not cid:
            continue
        meta[cid] = {
            "activation_alignment": item.get(
                "activation_alignment", item.get("event_alignment")
            ),
        }
    return meta, notes


def select_psychological_perspectives(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    llm: LlmClient | None = None,
    *,
    focus_person: str = "",
) -> tuple[list[GraspingCombination], list[str]]:
    """枚举组合 →（可选）LLM 元数据 → 按激活去同视角 → prior 归一。

    枚举只做一次；combo meta 直接打在这批候选上，避免 C1/C2 错位。
    """
    notes: list[str] = []
    cache = PerspectiveNearCache()
    candidates = enumerate_candidate_combinations(
        self_grasping,
        dharma_grasping,
        llm=llm,
        near_cache=cache,
        focus_person=focus_person,
    )
    meta: dict[str, dict] = {}
    if llm is not None and candidates:
        meta, mnotes = _fetch_combo_meta(
            text,
            objective_event,
            self_grasping,
            dharma_grasping,
            candidates,
            llm,
            focus_person=focus_person,
        )
        notes.extend(mnotes)
    selected = select_grasping_combinations(
        self_grasping,
        dharma_grasping,
        meta_by_combo_id=meta or None,
        llm=llm,
        focus_person=focus_person,
        near_cache=cache,
        candidates=candidates,
    )
    if not selected:
        notes.append("二执组合为空。")
    else:
        notes.append(f"二执组合保留 {len(selected)} 个心理视角。")
    return selected, notes


def select_valid_alambana_candidates(
    candidates: list[SubjectiveAlambanaCandidate],
    uncertainty_notes: list[str] | None = None,
) -> list[SubjectiveAlambanaCandidate]:
    """无组合先验时的兼容路径：Softmax 先验 + 阈值筛选。"""
    notes = uncertainty_notes if uncertainty_notes is not None else []
    return _select_after_softmax(candidates, notes)


def _resolve_combo_prob(
    cand: SubjectiveAlambanaCandidate,
    combo_probs: dict[str, float],
) -> float:
    key = (cand.combo_reason or "").strip()
    if key in combo_probs:
        return float(combo_probs[key])
    for cid, p in combo_probs.items():
        if cand.id.endswith(cid) or cid in cand.id:
            return float(p)
    return 0.0


def _select_after_softmax(
    candidates: list[SubjectiveAlambanaCandidate],
    uncertainty_notes: list[str],
    *,
    combo_probs: dict[str, float] | None = None,
) -> list[SubjectiveAlambanaCandidate]:
    """写入 prior 并截断。

    主路径（``combo_probs`` 非空）：prior = 组合激活 P(C)；不做 Softmax。
    兼容路径（无组合）：若候选仍带一致性字段则 Softmax，否则均匀 prior。
    """
    if not candidates:
        return []
    candidates = candidates[:MAX_ALAMBANA_BRANCHES]

    consistency_probs: list[float] | None = None
    if combo_probs is None:
        consistencies = [
            (c.event_consistency, c.self_consistency, c.world_consistency)
            for c in candidates
        ]
        # 全 0 表示未采集一致性：均匀分布
        if any(sum(t) > 0 for t in consistencies):
            raw_scores, consistency_probs = apply_local_weighted_softmax(
                consistencies, temperature=ALAMBANA_SOFTMAX_TEMPERATURE
            )
            for cand, raw in zip(candidates, raw_scores):
                cand.raw_score = raw
        else:
            n = len(candidates)
            consistency_probs = [1.0 / n] * n

    for idx, cand in enumerate(candidates):
        if combo_probs is not None:
            cand.prior_probability = max(_resolve_combo_prob(cand, combo_probs), 0.0)
        else:
            assert consistency_probs is not None
            cand.prior_probability = consistency_probs[idx]
        cand.posterior_probability = cand.prior_probability
        cand.action_compat = 1.0
        cand.action_likelihood = 1.0
        if cand.confidence <= 0:
            cand.confidence = ALAMBANA_DEFAULT_CONFIDENCE
        cand.sync_grasping_aliases()

    total = sum(c.prior_probability for c in candidates)
    if total > 0:
        for c in candidates:
            c.prior_probability /= total
            c.posterior_probability = c.prior_probability
    elif combo_probs is not None:
        uncertainty_notes.append("组合激活质量为零，跳过归一化，不把失败候选写成 P=1。")
        return []
    else:
        n = len(candidates)
        for c in candidates:
            c.prior_probability = 1.0 / n
            c.posterior_probability = c.prior_probability

    ranked = sorted(
        candidates,
        key=lambda c: (c.prior_probability, c.confidence),
        reverse=True,
    )

    if combo_probs is not None:
        kept = ranked[:MAX_ALAMBANA_BRANCHES]
        for c in kept:
            if c.confidence < ALAMBANA_MIN_CONFIDENCE:
                uncertainty_notes.append(
                    f"{c.id} 置信偏低({c.confidence:.2f})，仍保留（组合驱动不删支）。"
                )
            if c.prior_probability < ALAMBANA_MIN_PROBABILITY:
                uncertainty_notes.append(
                    f"{c.id} 组合激活偏低({c.prior_probability:.3f})，仍保留。"
                )
    else:
        kept = [
            c
            for c in ranked
            if c.prior_probability >= ALAMBANA_MIN_PROBABILITY
            and c.confidence >= ALAMBANA_MIN_CONFIDENCE
        ]
        if not kept:
            uncertainty_notes.append("无候选过 Softmax/置信阈值，保留最高分 1 条。")
            kept = ranked[:1]
        kept = kept[:MAX_ALAMBANA_BRANCHES]

    total = sum(c.prior_probability for c in kept)
    if total > 0:
        for c in kept:
            c.prior_probability /= total
            c.posterior_probability = c.prior_probability
    else:
        n = len(kept)
        for c in kept:
            c.prior_probability = 1.0 / n
            c.posterior_probability = c.prior_probability
    return kept


def _fallback_alambana_from_combo(
    text: str,
    objective_event: ObjectiveEventStructure,
    combo: GraspingCombination,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
) -> SubjectiveAlambanaCandidate:
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    bits = [objective_event.summary_text() or text.strip()[:300] or "事件按字面呈现"]
    for sid in combo.used_self_ids:
        if sid in self_by_id:
            bits.append(f"我执侧重：{self_by_id[sid].text}")
    for did in combo.used_dharma_ids:
        if did in dharma_by_id:
            bits.append(f"法执侧重：{dharma_by_id[did].text}")
    cand = SubjectiveAlambanaCandidate(
        id=f"O_{combo.combo_id}",
        text="；".join(bits),
        used_self_ids=list(combo.used_self_ids),
        used_dharma_ids=list(combo.used_dharma_ids),
        combo_reason=combo.combo_id,
        event_consistency=0.5,
        self_consistency=0.4,
        world_consistency=0.4,
        confidence=0.35,
        source_type=combo.source_type,
        source_bias_frame_ids=list(combo.source_bias_frame_ids),
        reactivated_h_ids=list(combo.reactivated_h_ids or combo.source_bias_frame_ids),
        raw_combo_score=combo.raw_combo_score or combo.score,
        activation_alignment=combo.activation_alignment,
    )
    persp = combo.perspective_text(self_by_id, dharma_by_id)
    if not combo.perspective_fingerprint:
        combo.stamp_perspective_fingerprint(self_by_id, dharma_by_id)
    cand.perspective_text = persp
    cand.perspective_fingerprint = combo.perspective_fingerprint
    cand.sync_grasping_aliases()
    return cand


def generate_subjective_alambanas(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    llm: LlmClient,
    role_context: str = "",
    combinations: list[GraspingCombination] | None = None,
    *,
    focus_person: str = "",
    h_lookup: dict[str, str] | None = None,
) -> tuple[list[SubjectiveAlambanaCandidate], list[str]]:
    """按已选二执组合渲染主观所缘；无组合时先本地选择。"""
    notes: list[str] = []
    combos = list(combinations or [])
    if not combos:
        combos, cnotes = select_psychological_perspectives(
            text,
            objective_event,
            self_grasping,
            dharma_grasping,
            llm=llm,
            focus_person=focus_person,
        )
        notes.extend(cnotes)
    if not combos:
        notes.append("无二执组合，使用中性降级所缘。")
        neutral = objective_event.summary_text() or text.strip()[:300] or "事件按字面呈现"
        parsed = [
            SubjectiveAlambanaCandidate(
                id="O_neutral",
                text=neutral,
                used_self_ids=[self_grasping[0].id] if self_grasping else [],
                used_dharma_ids=[dharma_grasping[0].id] if dharma_grasping else [],
                combo_reason="C_neutral",
                event_consistency=0.5,
                self_consistency=0.3,
                world_consistency=0.3,
                confidence=0.35,
            )
        ]
        parsed[0].sync_grasping_aliases()
        return _select_after_softmax(parsed, notes), notes

    combo_by_id = {c.combo_id: c for c in combos}
    messages = build_subjective_alambana_messages(
        text,
        objective_event,
        self_grasping,
        dharma_grasping,
        role_context=role_context,
        max_candidates=len(combos),
        combinations=combos,
        focus_person=focus_person,
        h_lookup=h_lookup,
    )

    parsed: list[SubjectiveAlambanaCandidate] = []
    for attempt in range(ALAMBANA_VALIDATION_MAX_RETRIES + 1):
        call_messages = messages
        if attempt > 0 and notes:
            call_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": messages[1]["content"]
                    + "\n\n上次校验失败原因：\n- "
                    + "\n- ".join(notes[-5:])
                    + "\n请修正后重新输出。",
                },
            ]
        with llm_call_context(stage="perception.subjective_alambana"):
            response = llm.complete(call_messages, schema=SUBJECTIVE_ALAMBANA_SCHEMA)
        raw = response if isinstance(response, dict) else {}
        items = raw.get("candidates")
        if not isinstance(items, list):
            items = raw.get("alambana_candidates")
        if not isinstance(items, list):
            notes.append(f"所缘生成第 {attempt + 1} 次无 candidates。")
            continue

        by_combo: dict[str, SubjectiveAlambanaCandidate] = {}
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            combo_id = str(item.get("combo_id") or "").strip()
            combo = combo_by_id.get(combo_id)
            if combo is None:
                notes.append(f"所缘引用未知 combo_id={combo_id!r}，已忽略。")
                continue
            # 强制绑定组合执取，忽略模型自填 ids
            forced = dict(item)
            forced["id"] = str(item.get("id") or f"O_{combo_id}").strip() or f"O_{combo_id}"
            forced["used_self_ids"] = list(combo.used_self_ids)
            forced["used_dharma_ids"] = list(combo.used_dharma_ids)
            forced["combo_reason"] = combo.combo_id
            cand, pnotes = parse_alambana_candidate_raw(
                forced,
                index=index,
                valid_self_ids={h.id for h in self_grasping},
                valid_dharma_ids={h.id for h in dharma_grasping},
            )
            notes.extend(pnotes)
            if cand is None:
                continue
            valid, vnotes = validate_alambana_candidate(cand)
            notes.extend(vnotes)
            if valid is None:
                continue
            if not valid.objective_basis:
                summary = objective_event.summary_text()
                if summary:
                    valid.objective_basis = [summary]
            valid.used_self_ids = list(combo.used_self_ids)
            valid.used_dharma_ids = list(combo.used_dharma_ids)
            valid.combo_reason = combo.combo_id
            valid.source_type = combo.source_type
            valid.source_bias_frame_ids = list(combo.source_bias_frame_ids)
            valid.reactivated_h_ids = list(
                combo.reactivated_h_ids or combo.source_bias_frame_ids
            )
            valid.raw_combo_score = combo.raw_combo_score or combo.score
            valid.activation_alignment = combo.activation_alignment
            valid.sync_grasping_aliases()
            by_combo[combo_id] = valid

        batch = [by_combo[c.combo_id] for c in combos if c.combo_id in by_combo]
        if len(batch) == len(combos):
            parsed = batch
            break
        if batch:
            # 缺的用降级补齐
            for combo in combos:
                if combo.combo_id not in by_combo:
                    notes.append(f"组合 {combo.combo_id} 所缘缺失，使用降级渲染。")
                    by_combo[combo.combo_id] = _fallback_alambana_from_combo(
                        text, objective_event, combo, self_grasping, dharma_grasping
                    )
            parsed = [by_combo[c.combo_id] for c in combos]
            break
        notes.append(f"所缘生成第 {attempt + 1} 次校验后为空。")

    if not parsed:
        notes.append("主观所缘生成失败，按组合降级渲染。")
        parsed = [
            _fallback_alambana_from_combo(
                text, objective_event, combo, self_grasping, dharma_grasping
            )
            for combo in combos
        ]

    combo_probs = {
        c.combo_id: (c.activation_probability or c.probability) for c in combos
    }
    selected = _select_after_softmax(parsed, notes, combo_probs=combo_probs)
    # prior=P(C) 后写入视角指纹
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    for cand in selected:
        combo = combo_by_id.get((cand.combo_reason or "").strip())
        if combo is not None:
            if not combo.perspective_fingerprint:
                combo.stamp_perspective_fingerprint(self_by_id, dharma_by_id)
            cand.perspective_text = combo.perspective_text(self_by_id, dharma_by_id)
            cand.perspective_fingerprint = combo.perspective_fingerprint
            cand.source_type = combo.source_type
            cand.source_bias_frame_ids = list(combo.source_bias_frame_ids)
            cand.reactivated_h_ids = list(
                combo.reactivated_h_ids or combo.source_bias_frame_ids
            )
            cand.raw_combo_score = combo.raw_combo_score or combo.score
            cand.activation_alignment = combo.activation_alignment
    return selected, notes


def _extract_fresh_with_h_prior(
    stimulus_text: str,
    objective_event: ObjectiveEventStructure,
    llm: LlmClient,
    *,
    role_context: str,
    h_scope: dict,
    focus_person: str,
) -> tuple[
    list[GraspingHypothesis],
    list[GraspingHypothesis],
    dict,
    list[str],
]:
    """Fresh L2 后用已缓存的作用域结果调制；不再重算 H 是否触发。"""
    notes: list[str] = []
    try:
        fresh_s, fresh_d, n_fresh = extract_fresh_grasping(
            stimulus_text,
            objective_event,
            llm,
            role_context=role_context,
            focus_person=focus_person,
        )
    except Exception as exc:  # noqa: BLE001
        fresh_s, fresh_d, n_fresh = [], [], [f"Fresh L2 异常：{exc}"]
    notes.extend(n_fresh)
    h_prior = bind_h_scope_to_fresh(h_scope, fresh_s, fresh_d, llm)
    working_n = len(h_prior.get("working_set") or [])
    triggered_n = len(h_prior.get("triggered_ids") or [])
    if working_n:
        notes.append(f"H 工作集 {working_n} 条，作用域触发 {triggered_n} 条。")
    return fresh_s, fresh_d, h_prior, notes


def build_perception_state(
    text: str,
    llm: LlmClient,
    role_context: str = "",
    cognitive_bias_frames: list[dict] | None = None,
    *,
    expect_action: bool = True,
    focus_person: str = "",
) -> PerceptionState:
    notes: list[str] = []
    source_text = text
    focus = str(focus_person or "").strip()
    if focus:
        notes.append(f"分析对象锁定：{focus}")

    try:
        split, n0 = split_stimulus_and_action(
            text,
            llm,
            role_context=role_context,
            expect_action=expect_action,
            focus_person=focus,
        )
        notes.extend(n0)
    except Exception as exc:  # noqa: BLE001
        split = StimulusActionSplit(
            stimulus_text=text.strip(),
            reason_summary=f"刺激/行动拆分失败，降级为全文：{exc}",
        )
        notes.append(split.reason_summary)

    stimulus_text = split.stimulus_text or text.strip()

    try:
        objective_event, n1 = parse_objective_event_structure(
            stimulus_text, llm, role_context=role_context, focus_person=focus
        )
        notes.extend(n1)
    except Exception as exc:  # noqa: BLE001
        objective_event = ObjectiveEventStructure(scene=stimulus_text.strip()[:500])
        notes.append(f"客观刺激异常，已降级：{exc}")

    near_cache = PerspectiveNearCache()
    h_scope = resolve_h_scope(stimulus_text, cognitive_bias_frames, llm)
    fresh_s, fresh_d, h_prior, l2_notes = _extract_fresh_with_h_prior(
        stimulus_text,
        objective_event,
        llm,
        role_context=role_context,
        h_scope=h_scope,
        focus_person=focus,
    )
    notes.extend(l2_notes)

    self_g, dharma_g = merge_and_dedup_graspings(
        [fresh_s],
        [fresh_d],
        llm=llm,
        near_cache=near_cache,
        focus_person=focus,
    )
    fresh_count_raw = len(fresh_s) + len(fresh_d)
    historical_count_raw = 0
    after_dedup = len(self_g) + len(dharma_g)

    def _select_combos(
        selves: list[GraspingHypothesis],
        dharmas: list[GraspingHypothesis],
    ) -> list[GraspingCombination]:
        selected, cnotes = select_psychological_perspectives(
            stimulus_text,
            objective_event,
            selves,
            dharmas,
            llm=llm,
            focus_person=focus,
        )
        notes.extend(cnotes)
        return selected

    combos: list[GraspingCombination] = []
    if self_g or dharma_g:
        try:
            combos = _select_combos(self_g, dharma_g)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"二执组合异常：{exc}")
            combos = select_grasping_combinations(
                self_g, dharma_g, llm=llm, focus_person=focus
            )

    def _max_fit(items: list[GraspingCombination]) -> float:
        if not items:
            return 0.0
        return max(float(c.activation_alignment or 0.0) for c in items)

    accommodation = {
        "triggered": False,
        "reason": "",
        "regeneration_success": False,
    }
    used_neutral = False

    need_accommodation = (not combos) or (_max_fit(combos) < ACCOMMODATION_MIN_FIT)
    if need_accommodation:
        accommodation["triggered"] = True
        accommodation["reason"] = (
            "valid_combos empty"
            if not combos
            else f"max_current_fit={_max_fit(combos):.3f}<{ACCOMMODATION_MIN_FIT}"
        )
        notes.append(f"触发 Cognitive Accommodation：{accommodation['reason']}")
        try:
            acc_s, acc_d, n_acc = extract_accommodation_grasping(
                stimulus_text,
                objective_event,
                llm,
                failed_reason=str(accommodation["reason"]),
                focus_person=focus,
            )
            notes.extend(n_acc)
            acc_s, acc_d = merge_and_dedup_graspings(
                [acc_s],
                [acc_d],
                llm=llm,
                near_cache=near_cache,
                focus_person=focus,
            )
            if acc_s or acc_d:
                self_g, dharma_g = acc_s, acc_d
                h_prior = bind_h_scope_to_fresh(h_scope, self_g, dharma_g, llm)
                combos = _select_combos(self_g, dharma_g)
                accommodation["regeneration_success"] = bool(combos)
            else:
                combos = []
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Accommodation 异常：{exc}")
            combos = []

    if not combos:
        used_neutral = True
        notes.append("所有真实候选失败，使用 neutral fallback。")
        self_g, dharma_g = make_neutral_graspings(focus_person=focus)
        combos = select_grasping_combinations(
            self_g, dharma_g, llm=None, focus_person=focus
        )

    h_lookup = h_texts_by_id(h_prior)
    alambanas: list[SubjectiveAlambanaCandidate] = []
    try:
        alambanas, n3 = generate_subjective_alambanas(
            stimulus_text,
            objective_event,
            self_g,
            dharma_g,
            llm,
            role_context=role_context,
            combinations=combos,
            focus_person=focus,
            h_lookup=h_lookup,
        )
        notes.extend(n3)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"主观所缘异常，已降级：{exc}")
        if combos:
            alambanas = [
                _fallback_alambana_from_combo(
                    stimulus_text, objective_event, combos[0], self_g, dharma_g
                )
            ]
            alambanas[0].prior_probability = 1.0
            alambanas[0].posterior_probability = 1.0

    if alambanas:
        grounded, gnotes = apply_grounding_with_retry(
            stimulus_text,
            objective_event,
            alambanas,
            combos,
            self_g,
            dharma_g,
            llm,
            focus_person=focus,
        )
        notes.extend(gnotes)
        if not grounded and not accommodation["triggered"] and not used_neutral:
            accommodation["triggered"] = True
            accommodation["reason"] = "all alambanas failed current-event grounding"
            notes.append("全部所缘 grounding 失败，触发 Accommodation 后重试 L3。")
            try:
                acc_s, acc_d, n_acc = extract_accommodation_grasping(
                    stimulus_text,
                    objective_event,
                    llm,
                    failed_reason=str(accommodation["reason"]),
                    focus_person=focus,
                )
                notes.extend(n_acc)
                acc_s, acc_d = merge_and_dedup_graspings(
                    [acc_s],
                    [acc_d],
                    llm=llm,
                    near_cache=near_cache,
                    focus_person=focus,
                )
                if acc_s or acc_d:
                    self_g, dharma_g = acc_s, acc_d
                    h_prior = bind_h_scope_to_fresh(h_scope, self_g, dharma_g, llm)
                    h_lookup = h_texts_by_id(h_prior)
                    combos = _select_combos(self_g, dharma_g)
                    if combos:
                        alambanas, n3b = generate_subjective_alambanas(
                            stimulus_text,
                            objective_event,
                            self_g,
                            dharma_g,
                            llm,
                            role_context=role_context,
                            combinations=combos,
                            focus_person=focus,
                            h_lookup=h_lookup,
                        )
                        notes.extend(n3b)
                        grounded, gnotes2 = apply_grounding_with_retry(
                            stimulus_text,
                            objective_event,
                            alambanas,
                            combos,
                            self_g,
                            dharma_g,
                            llm,
                            focus_person=focus,
                        )
                        notes.extend(gnotes2)
                        accommodation["regeneration_success"] = bool(grounded)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"grounding 后 Accommodation 异常：{exc}")
                grounded = []
        alambanas = grounded

    if not alambanas and not used_neutral:
        used_neutral = True
        notes.append("grounding / 组合均失败，使用 neutral fallback。")
        self_g, dharma_g = make_neutral_graspings(focus_person=focus)
        combos = select_grasping_combinations(
            self_g, dharma_g, llm=None, focus_person=focus
        )
        if combos:
            alambanas = [
                _fallback_alambana_from_combo(
                    stimulus_text, objective_event, combos[0], self_g, dharma_g
                )
            ]
            alambanas[0].prior_probability = 1.0
            alambanas[0].posterior_probability = 1.0
            alambanas[0].source_type = "neutral"
            alambanas[0].grounding_valid = True
            alambanas[0].grounding_reason = "neutral fallback"

    if alambanas:
        alambanas = dedup_subjective_alambanas(
            alambanas,
            llm=llm,
            focus_person=focus,
            max_keep=MAX_ALAMBANA_BRANCHES,
        )

    if combos and alambanas:
        by_cid = {c.combo_id: c for c in combos}
        for alb in alambanas:
            combo = by_cid.get(alb.combo_reason)
            if combo is not None:
                combo.alambana_prior_probability = alb.prior_probability

    facts = objective_event.as_facts()
    if not facts and stimulus_text.strip():
        facts = [
            ObjectiveFact(
                text=stimulus_text.strip()[:500],
                source_span=stimulus_text.strip()[:120],
            )
        ]

    pipeline_trace = {
        "cognition_candidates": {
            "fresh_count_raw": fresh_count_raw,
            "historical_count_raw": historical_count_raw,
            "after_semantic_dedup": after_dedup,
            "valid_after_alignment_gate": len(combos) if not used_neutral else 0,
        },
        "h_prior": dict(h_prior),
        "accommodation": dict(accommodation),
        "neutral_fallback": {"used": used_neutral},
        "used_neutral_fallback": used_neutral,
    }

    return PerceptionState(
        source_text=source_text,
        stimulus_text=stimulus_text,
        stimulus_action_split=split,
        objective_event=objective_event,
        objective_facts=facts,
        self_grasping_hypotheses=self_g,
        dharma_grasping_hypotheses=dharma_g,
        base_alambana_text=objective_event.summary_text(),
        grasping_combos=[c.to_dict() for c in combos],
        alambana_candidates=alambanas,
        uncertainty_notes=notes,
        pipeline_trace=pipeline_trace,
        h_prior=dict(h_prior),
    )
