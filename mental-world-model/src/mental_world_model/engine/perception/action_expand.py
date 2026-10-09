"""行动扩支：客观所缘 + 行动 → 新二执 → 组合 → 主观所缘（感知子路径）。

与常规 L2-L3 同构；生成时已纳入行动，故不再参与 compat 后验矫正。
"""

from __future__ import annotations

from mental_world_model.engine.perception.config import (
    ACTION_EXPAND_ALAMBANA_ENABLED,
    ACTION_EXPAND_MAX_BRANCHES,
    ACTION_EXPAND_MAX_COMPAT,
    ACTION_EXPAND_MAX_NEW_PRIOR,
    ACTION_EXPAND_MIN_PRIOR,
    MAX_ALAMBANA_BRANCHES,
)
from mental_world_model.engine.cognition.action_evidence import ActionEvidence
from mental_world_model.engine.continuity.prior import h_texts_by_id
from mental_world_model.engine.perception.builder import (
    generate_subjective_alambanas,
)
from mental_world_model.engine.perception.grounding import apply_grounding_with_retry
from mental_world_model.engine.perception.combo_selection import (
    apply_combo_meta,
    combo_passes_hard_gate,
    enumerate_candidate_combinations,
    filter_diverse_combinations,
    is_same_perspective,
    normalize_combination_probabilities,
)
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
    ObjectiveEventStructure,
    PerceptionState,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import (
    COMBO_META_SCHEMA,
    GRASPING_PAIR_SCHEMA,
    build_combo_meta_action_messages,
    build_grasping_pair_action_messages,
)
from mental_world_model.engine.perception.validators import (
    normalize_probability_distribution,
    parse_grasping_list,
)
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context


def should_expand_alambana_from_action(
    candidates: list[SubjectiveAlambanaCandidate],
    evidence: ActionEvidence,
    *,
    enabled: bool = ACTION_EXPAND_ALAMBANA_ENABLED,
    max_compat_threshold: float = ACTION_EXPAND_MAX_COMPAT,
    max_branches: int = ACTION_EXPAND_MAX_BRANCHES,
) -> bool:
    """现有所缘 max(compat) 低于阈值且未达支数上限时触发扩支。"""
    if not enabled:
        return False
    if not evidence.has_observable_action or not candidates:
        return False
    if len(candidates) >= max_branches:
        return False
    if not evidence.likelihoods:
        return False
    max_compat = max(item.likelihood for item in evidence.likelihoods)
    return max_compat < max_compat_threshold


def _next_alambana_id(candidates: list[SubjectiveAlambanaCandidate]) -> str:
    existing = {c.id for c in candidates}
    for index in range(1, 100):
        cid = f"O{index}"
        if cid not in existing:
            return cid
    return f"O_act_{len(candidates) + 1}"


def _dedupe_grasping_ids(
    new_items: list[GraspingHypothesis],
    existing_ids: set[str],
) -> list[GraspingHypothesis]:
    """避免与感知池 id 冲突。"""
    out: list[GraspingHypothesis] = []
    for hyp in new_items:
        base = str(hyp.id or "").strip() or "g"
        cid = base
        suffix = 1
        while cid in existing_ids:
            cid = f"{base}_{suffix}"
            suffix += 1
        existing_ids.add(cid)
        out.append(
            GraspingHypothesis(
                id=cid,
                type=hyp.type,
                text=hyp.text,
                confidence=hyp.confidence,
                source=hyp.source,
                supporting_evidence=list(hyp.supporting_evidence),
                counter_evidence=list(hyp.counter_evidence),
                missing_information=list(hyp.missing_information),
                source_type=hyp.source_type,
                source_bias_frame_ids=list(hyp.source_bias_frame_ids),
            )
        )
    return out


def _existing_perspective_combos(
    candidates: list[SubjectiveAlambanaCandidate],
    self_by_id: dict[str, GraspingHypothesis],
    dharma_by_id: dict[str, GraspingHypothesis],
) -> list[GraspingCombination]:
    combos: list[GraspingCombination] = []
    for cand in candidates:
        combo = GraspingCombination(
            combo_id=str(cand.combo_reason or cand.id),
            used_self_ids=list(cand.used_self_ids or cand.self_grasping_ids),
            used_dharma_ids=list(cand.used_dharma_ids or cand.dharma_grasping_ids),
        )
        combo.stamp_perspective_fingerprint(self_by_id, dharma_by_id)
        if cand.perspective_fingerprint:
            combo.perspective_fingerprint = cand.perspective_fingerprint
        combos.append(combo)
    return combos


def extract_grasping_pair_from_action(
    text: str,
    objective_event: ObjectiveEventStructure,
    action_evidence: ActionEvidence,
    existing_candidates: list[SubjectiveAlambanaCandidate],
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[list[GraspingHypothesis], list[GraspingHypothesis], list[str]]:
    """L2'：客观所缘 + 行动 → 新我法二执。"""
    notes: list[str] = []
    summaries = [
        {"id": c.id, "text": c.text, "combo_reason": c.combo_reason}
        for c in existing_candidates
    ]
    with llm_call_context(stage="perception.grasping_pair_action"):
        response = llm.complete(
            build_grasping_pair_action_messages(
                text,
                objective_event,
                action_span=action_evidence.action_span,
                action_description=action_evidence.action_description,
                existing_alambana_summaries=summaries,
                focus_person=focus_person,
            ),
            schema=GRASPING_PAIR_SCHEMA,
        )
    raw = response if isinstance(response, dict) else {}
    self_g, n_self = parse_grasping_list(
        raw.get("self_grasping"), grasping_type="self_grasping"
    )
    dharma_g, n_dharma = parse_grasping_list(
        raw.get("dharma_grasping"), grasping_type="dharma_grasping"
    )
    notes.extend(n_self)
    notes.extend(n_dharma)
    if not self_g and not dharma_g:
        notes.append("行动扩支二执皆空，放弃扩支。")
    return self_g, dharma_g, notes


def _fetch_combo_meta_action(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    candidates: list[GraspingCombination],
    action_evidence: ActionEvidence,
    llm: LlmClient,
    *,
    focus_person: str = "",
) -> tuple[dict[str, dict], list[str]]:
    notes: list[str] = []
    if not candidates:
        return {}, notes
    try:
        with llm_call_context(stage="perception.combo_meta_action"):
            response = llm.complete(
                build_combo_meta_action_messages(
                    text,
                    objective_event,
                    self_grasping,
                    dharma_grasping,
                    candidates,
                    action_span=action_evidence.action_span,
                    action_description=action_evidence.action_description,
                    focus_person=focus_person,
                ),
                schema=COMBO_META_SCHEMA,
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"行动扩支组合元数据失败，用默认对齐：{exc}")
        return {}, notes
    raw = response if isinstance(response, dict) else {}
    items = raw.get("items")
    if not isinstance(items, list):
        notes.append("行动扩支 combo meta 无 items。")
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


def _select_one_action_combo(
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    existing_combos: list[GraspingCombination],
    meta: dict[str, dict],
    llm: LlmClient,
    *,
    full_self_by_id: dict[str, GraspingHypothesis],
    full_dharma_by_id: dict[str, GraspingHypothesis],
    focus_person: str = "",
    candidates: list[GraspingCombination] | None = None,
) -> tuple[GraspingCombination | None, list[str]]:
    """从行动二执枚举组合，去重后取 Top-1。"""
    notes: list[str] = []
    if candidates is None:
        raw_combos = enumerate_candidate_combinations(
            self_grasping,
            dharma_grasping,
            llm=llm,
            focus_person=focus_person,
        )
        if not raw_combos:
            notes.append("行动扩支无可枚举组合。")
            return None, notes
        candidates = [
            GraspingCombination(
                combo_id=f"C_act{index}",
                used_self_ids=list(combo.used_self_ids),
                used_dharma_ids=list(combo.used_dharma_ids),
                self_confidence=combo.self_confidence,
                dharma_confidence=combo.dharma_confidence,
                activation_alignment=combo.activation_alignment,
                score=combo.score,
            )
            for index, combo in enumerate(raw_combos, start=1)
        ]
    else:
        candidates = list(candidates)
    if not candidates:
        notes.append("行动扩支无可枚举组合。")
        return None, notes

    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    scored_ids = set(meta) if meta else None

    for combo in candidates:
        meta_item = (meta or {}).get(combo.combo_id)
        if meta_item:
            apply_combo_meta(
                combo,
                activation_alignment=meta_item.get("activation_alignment"),
            )

    gated = [
        c
        for c in candidates
        if (scored_ids is None or c.combo_id in scored_ids)
        and combo_passes_hard_gate(c)
    ]
    if not gated:
        notes.append("行动扩支组合未过 alignment/raw_score 硬门闸。")
        return None, notes

    ranked = sorted(gated, key=lambda c: c.score, reverse=True)
    diverse = filter_diverse_combinations(
        ranked,
        self_by_id=self_by_id,
        dharma_by_id=dharma_by_id,
        max_keep=MAX_ALAMBANA_BRANCHES,
        llm=llm,
        focus_person=focus_person,
    )

    picked: GraspingCombination | None = None
    for cand in diverse:
        if any(
            is_same_perspective(
                cand,
                ex,
                self_by_id=full_self_by_id,
                dharma_by_id=full_dharma_by_id,
                llm=llm,
                focus_person=focus_person,
            )
            for ex in existing_combos
        ):
            continue
        picked = cand
        break
    if picked is None and diverse:
        notes.append("行动扩支组合均与已有视角过近，取最高分仍扩支。")
        picked = diverse[0]
    if picked is None:
        notes.append("行动扩支未选出有效组合。")
        return None, notes

    normalize_combination_probabilities([picked])
    notes.append(
        f"行动扩支选中组合 {picked.combo_id}，"
        f"activation={picked.activation_probability:.3f}。"
    )
    return picked, notes


def expand_action_informed_branch(
    yuanwen: str,
    perception: PerceptionState,
    action_evidence: ActionEvidence,
    existing_candidates: list[SubjectiveAlambanaCandidate],
    llm: LlmClient,
    notes: list[str] | None = None,
    *,
    focus_person: str = "",
) -> SubjectiveAlambanaCandidate | None:
    """跑 L2'→combo'→L3'，写入 perception 并返回新所缘（未 append 到 candidates）。"""
    notes = notes if notes is not None else []
    objective_event = perception.objective_event
    if objective_event is None:
        notes.append("缺少客观所缘境，无法行动扩支。")
        return None

    self_new, dharma_new, gnotes = extract_grasping_pair_from_action(
        yuanwen,
        objective_event,
        action_evidence,
        existing_candidates,
        llm,
        focus_person=focus_person,
    )
    notes.extend(gnotes)
    if not self_new and not dharma_new:
        return None

    existing_ids = {
        h.id
        for h in (
            perception.self_grasping_hypotheses
            + perception.dharma_grasping_hypotheses
        )
    }
    self_new = _dedupe_grasping_ids(self_new, set(existing_ids))
    existing_ids = existing_ids | {h.id for h in self_new}
    dharma_new = _dedupe_grasping_ids(dharma_new, set(existing_ids))

    perception.self_grasping_hypotheses.extend(self_new)
    perception.dharma_grasping_hypotheses.extend(dharma_new)

    self_pool = self_new
    dharma_pool = dharma_new
    enum_combos = enumerate_candidate_combinations(
        self_pool,
        dharma_pool,
        llm=llm,
        focus_person=focus_person,
    )
    if not enum_combos:
        notes.append("行动二执无法形成组合。")
        return None

    # 预编号供 meta 对齐：与筛选共用同一批候选，未打分者不默认 alignment=1
    prenumbered = [
        GraspingCombination(
            combo_id=f"C_act{index}",
            used_self_ids=list(c.used_self_ids),
            used_dharma_ids=list(c.used_dharma_ids),
            self_confidence=c.self_confidence,
            dharma_confidence=c.dharma_confidence,
            score=c.score,
        )
        for index, c in enumerate(enum_combos, start=1)
    ]
    meta, mnotes = _fetch_combo_meta_action(
        yuanwen,
        objective_event,
        self_pool,
        dharma_pool,
        prenumbered,
        action_evidence,
        llm,
        focus_person=focus_person,
    )
    notes.extend(mnotes)

    ex_self_by_id = {h.id: h for h in perception.self_grasping_hypotheses}
    ex_dharma_by_id = {h.id: h for h in perception.dharma_grasping_hypotheses}
    existing_combos = _existing_perspective_combos(
        existing_candidates,
        ex_self_by_id,
        ex_dharma_by_id,
    )

    combo, cnotes = _select_one_action_combo(
        self_pool,
        dharma_pool,
        existing_combos,
        meta,
        llm,
        full_self_by_id=ex_self_by_id,
        full_dharma_by_id=ex_dharma_by_id,
        focus_person=focus_person,
        candidates=prenumbered,
    )
    notes.extend(cnotes)
    if combo is None:
        return None

    alambanas, a_notes = generate_subjective_alambanas(
        perception.stimulus_text or yuanwen,
        objective_event,
        perception.self_grasping_hypotheses,
        perception.dharma_grasping_hypotheses,
        llm,
        combinations=[combo],
        focus_person=focus_person,
        h_lookup=h_texts_by_id(getattr(perception, "h_prior", None)),
    )
    notes.extend(a_notes)
    if not alambanas:
        notes.append("行动扩支所缘渲染失败。")
        return None

    new_cand = alambanas[0]
    new_cand.id = _next_alambana_id(existing_candidates)
    new_cand.action_informed = True
    new_cand.combo_reason = combo.combo_id
    # 单 combo 渲染会把 prior 归一成 1；扩支质量用组合激活分，再夹到配置区间
    raw_mass = float(combo.score or combo.activation_probability or 0.0)
    mass = raw_mass if raw_mass > 0 else ACTION_EXPAND_MIN_PRIOR
    mass = max(ACTION_EXPAND_MIN_PRIOR, min(ACTION_EXPAND_MAX_NEW_PRIOR, mass))
    new_cand.prior_probability = mass
    new_cand.posterior_probability = mass
    new_cand.action_compat = 1.0
    new_cand.action_likelihood = 1.0
    combo.alambana_prior_probability = new_cand.prior_probability
    combo.stamp_perspective_fingerprint(
        {h.id: h for h in perception.self_grasping_hypotheses},
        {h.id: h for h in perception.dharma_grasping_hypotheses},
    )
    new_cand.perspective_text = combo.perspective_text(
        {h.id: h for h in perception.self_grasping_hypotheses},
        {h.id: h for h in perception.dharma_grasping_hypotheses},
    )
    new_cand.perspective_fingerprint = combo.perspective_fingerprint
    new_cand.sync_grasping_aliases()

    grounded, gnotes = apply_grounding_with_retry(
        perception.stimulus_text or yuanwen,
        objective_event,
        [new_cand],
        [combo],
        perception.self_grasping_hypotheses,
        perception.dharma_grasping_hypotheses,
        llm,
        focus_person=focus_person,
    )
    notes.extend(gnotes)
    if not grounded:
        notes.append("行动扩支所缘未通过当前事件 grounding，放弃扩支。")
        return None
    new_cand = grounded[0]
    new_cand.action_informed = True
    new_cand.prior_probability = mass
    new_cand.posterior_probability = mass
    new_cand.action_compat = 1.0
    new_cand.action_likelihood = 1.0

    perception.grasping_combos.append(combo.to_dict())
    notes.append(
        f"行动扩支所缘 {new_cand.id} 已生成（prior={new_cand.prior_probability:.3f}，"
        "跳过后验 compat 矫正）。"
    )
    return new_cand


def finalize_mixed_action_posteriors(
    candidates: list[SubjectiveAlambanaCandidate],
    notes: list[str] | None = None,
) -> None:
    """旧支 action-corrected + 新支 action-informed prior 合并后归一化 posterior。"""
    notes = notes if notes is not None else []
    raw = [max(0.0, float(c.posterior_probability or 0.0)) for c in candidates]
    if sum(raw) <= 0:
        notes.append("混合后验质量为零，跳过归一化。")
        return
    norms = normalize_probability_distribution(raw)
    for cand, prob in zip(candidates, norms):
        cand.posterior_probability = prob
    notes.append(
        f"行动扩支后混合归一化：共 {len(candidates)} 支 posterior 和=1。"
    )


def maybe_expand_action_informed_branch(
    yuanwen: str,
    perception: PerceptionState,
    action_evidence: ActionEvidence,
    compat_evidence: ActionEvidence,
    llm: LlmClient,
    notes: list[str] | None = None,
    *,
    focus_person: str = "",
) -> bool:
    """判断并执行行动扩支；成功则 append 到 perception.alambana_candidates。"""
    notes = notes if notes is not None else []
    existing = [
        c for c in perception.alambana_candidates if not c.action_informed
    ]
    if not should_expand_alambana_from_action(existing, compat_evidence):
        return False

    max_compat = max(
        (item.likelihood for item in compat_evidence.likelihoods),
        default=1.0,
    )
    notes.append(
        f"行动 compat 最高={max_compat:.3f}<{ACTION_EXPAND_MAX_COMPAT}，"
        "启动行动扩支感知子路径（二执→组合→所缘）。"
    )
    new_cand = expand_action_informed_branch(
        yuanwen,
        perception,
        action_evidence,
        existing,
        llm,
        notes,
        focus_person=focus_person,
    )
    if new_cand is None:
        notes.append("行动扩支失败，保持原有所缘集合。")
        return False

    perception.alambana_candidates.append(new_cand)
    return True
