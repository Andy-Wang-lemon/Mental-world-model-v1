"""二执组合选择：生成互异潜在心理状态视角（先组合、后所缘）。

组合 = 主体在当前刺激下可能被激活的关注视角（非行为解释候选）。
保留只看 activation；行为融合在认知层用 a，不在感知层打分。
去重：id Jaccard +（弱规则召回 → 可选 LLM 语义确认）；宁 1 勿 3。
"""

from __future__ import annotations

from mental_world_model.engine.perception.config import (
    COMBO_ID_JACCARD_MAX,
    COMBO_MAX_DHARMA_REPS,
    COMBO_MAX_SELF_REPS,
    COMBO_SECONDARY_RATIO,
    GRASPING_COMBO_ACTIVATION,
    MAX_ALAMBANA_BRANCHES,
    MIN_ALIGNMENT,
    MIN_COMBO_RAW_SCORE,
)
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
    GraspingSourceType,
)
from mental_world_model.engine.perception.semantic_confirm import (
    PerspectiveNearCache,
    perspectives_same,
)
from mental_world_model.engine.perception.validators import (
    normalize_probability_distribution,
)
from mental_world_model.llm.client import LlmClient
from mental_world_model.utils.confidence import as_unit_float, clamp_unit_interval

# 可选调试枚举；不参与剪枝/主契约
FOCUS_AXES = frozenset(
    {
        "identity",
        "competence",
        "status",
        "relationship",
        "fairness",
        "control",
        "goal",
        "security",
        "other",
    }
)


def _combo_confidence(hyp: GraspingHypothesis) -> float:
    """组合入口强度：条目 confidence。"""
    return as_unit_float(hyp.confidence)


def infer_combo_source(
    selves: list[GraspingHypothesis],
    dharmas: list[GraspingHypothesis],
) -> tuple[GraspingSourceType, list[str]]:
    """由成员执取汇总组合来源；来源只影响 prior 信息，不决定名额。"""
    hyps = [h for h in (*selves, *dharmas) if h is not None]
    types = {str(h.source_type or "fresh") for h in hyps}
    frame_ids: list[str] = []
    seen: set[str] = set()
    for hyp in hyps:
        for fid in hyp.source_bias_frame_ids:
            key = str(fid).strip()
            if key and key not in seen:
                seen.add(key)
                frame_ids.append(key)
    if not types or types <= {"fresh"}:
        return "fresh", frame_ids
    if types <= {"historical"}:
        return "historical", frame_ids
    if types <= {"accommodation"}:
        return "accommodation", frame_ids
    if types <= {"neutral"}:
        return "neutral", frame_ids
    return "mixed", frame_ids


def stamp_combo_source(
    combo: GraspingCombination,
    *,
    self_by_id: dict[str, GraspingHypothesis],
    dharma_by_id: dict[str, GraspingHypothesis],
) -> None:
    selves = [self_by_id[i] for i in combo.used_self_ids if i in self_by_id]
    dharmas = [dharma_by_id[i] for i in combo.used_dharma_ids if i in dharma_by_id]
    source_type, frame_ids = infer_combo_source(selves, dharmas)
    combo.source_type = source_type
    combo.source_bias_frame_ids = frame_ids
    combo.reactivated_h_ids = list(frame_ids)


def combo_passes_hard_gate(
    combo: GraspingCombination,
    *,
    min_alignment: float = MIN_ALIGNMENT,
    min_raw_score: float = MIN_COMBO_RAW_SCORE,
) -> bool:
    """alignment / raw_score 硬门闸：未过闸者不参与归一化。"""
    alignment = as_unit_float(combo.activation_alignment)
    raw = max(float(combo.raw_combo_score or combo.score or 0.0), 0.0)
    return alignment >= float(min_alignment) and raw >= float(min_raw_score)


def select_object_representatives(
    hypotheses: list[GraspingHypothesis],
    *,
    activation_threshold: float,
    max_reps: int,
    llm: LlmClient | None = None,
    near_cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> list[GraspingHypothesis]:
    """过阈后按 confidence 排序，语义近的执只留最强一条。"""
    ranked = sorted(
        [h for h in hypotheses if _combo_confidence(h) >= activation_threshold],
        key=_combo_confidence,
        reverse=True,
    )
    reps: list[GraspingHypothesis] = []
    for hyp in ranked:
        if any(
            perspectives_same(
                hyp.text,
                kept.text,
                llm=llm,
                cache=near_cache,
                focus_person=focus_person,
            )
            for kept in reps
        ):
            continue
        reps.append(hyp)
        if len(reps) >= max_reps:
            break
    return reps


def _fallback_strongest_combination(
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
) -> GraspingCombination | None:
    """门槛下无人过线时：取 confidence 最高的我执/法执各一条凑 1 视角。"""
    s_best = max(self_grasping, key=_combo_confidence, default=None)
    d_best = max(dharma_grasping, key=_combo_confidence, default=None)
    if s_best is None and d_best is None:
        return None
    selves = [s_best] if s_best is not None else []
    dharmas = [d_best] if d_best is not None else []
    i_c = _combo_confidence(s_best) if s_best is not None else 0.0
    d_c = _combo_confidence(d_best) if d_best is not None else 0.0
    a_score = combo_activation_score(i_c, d_c, 1.0)
    source_type, frame_ids = infer_combo_source(selves, dharmas)
    return GraspingCombination(
        combo_id="C1",
        used_self_ids=[h.id for h in selves],
        used_dharma_ids=[h.id for h in dharmas],
        self_confidence=i_c,
        dharma_confidence=d_c,
        activation_alignment=1.0,
        score=a_score,
        raw_combo_score=a_score,
        source_type=source_type,
        source_bias_frame_ids=frame_ids,
        reactivated_h_ids=list(frame_ids),
        reason="降级：无过阈组合，取 confidence 最强执",
    )


def jaccard_ids(a: GraspingCombination, b: GraspingCombination) -> float:
    sa, sb = a.id_set(), b.id_set()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def combo_activation_score(
    self_confidence: float,
    dharma_confidence: float,
    activation_alignment: float,
    *,
    unilateral_fill: float = 0.5,
) -> float:
    """激活分 = I_c × D_c × activation_alignment；单侧用 fill。"""
    i = as_unit_float(self_confidence) if self_confidence > 0 else unilateral_fill
    d = as_unit_float(dharma_confidence) if dharma_confidence > 0 else unilateral_fill
    a = as_unit_float(activation_alignment)
    if self_confidence <= 0 and dharma_confidence <= 0:
        return 0.0
    return i * d * a


def enumerate_candidate_combinations(
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    *,
    activation_threshold: float = GRASPING_COMBO_ACTIVATION,
    max_self_reps: int = COMBO_MAX_SELF_REPS,
    max_dharma_reps: int = COMBO_MAX_DHARMA_REPS,
    llm: LlmClient | None = None,
    near_cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> list[GraspingCombination]:
    """枚举 (我执代表 × 法执代表) 及必要单侧组合。"""
    self_reps = select_object_representatives(
        self_grasping,
        activation_threshold=activation_threshold,
        max_reps=max_self_reps,
        llm=llm,
        near_cache=near_cache,
        focus_person=focus_person,
    )
    dharma_reps = select_object_representatives(
        dharma_grasping,
        activation_threshold=activation_threshold,
        max_reps=max_dharma_reps,
        llm=llm,
        near_cache=near_cache,
        focus_person=focus_person,
    )
    out: list[GraspingCombination] = []
    idx = 0

    def _add(
        selves: list[GraspingHypothesis],
        dharmas: list[GraspingHypothesis],
    ) -> None:
        nonlocal idx
        if not selves and not dharmas:
            return
        idx += 1
        i_c = max((_combo_confidence(h) for h in selves), default=0.0)
        d_c = max((_combo_confidence(h) for h in dharmas), default=0.0)
        a_score = combo_activation_score(i_c, d_c, 1.0)
        source_type, frame_ids = infer_combo_source(selves, dharmas)
        out.append(
            GraspingCombination(
                combo_id=f"C{idx}",
                used_self_ids=[h.id for h in selves],
                used_dharma_ids=[h.id for h in dharmas],
                self_confidence=i_c,
                dharma_confidence=d_c,
                activation_alignment=1.0,
                score=a_score,
                raw_combo_score=a_score,
                source_type=source_type,
                source_bias_frame_ids=frame_ids,
                reactivated_h_ids=list(frame_ids),
            )
        )

    if self_reps and dharma_reps:
        for s in self_reps:
            for d in dharma_reps:
                _add([s], [d])
    elif self_reps:
        for s in self_reps:
            _add([s], [])
    elif dharma_reps:
        for d in dharma_reps:
            _add([], [d])
    return out


def is_same_perspective(
    a: GraspingCombination,
    b: GraspingCombination,
    *,
    self_by_id: dict[str, GraspingHypothesis],
    dharma_by_id: dict[str, GraspingHypothesis],
    id_jaccard_max: float = COMBO_ID_JACCARD_MAX,
    llm: LlmClient | None = None,
    near_cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> bool:
    """同视角：id 过近，或（弱规则+可选 LLM）语义近。focus_axis 不参与决策。"""
    if jaccard_ids(a, b) >= id_jaccard_max:
        return True
    ta = a.perspective_text(self_by_id, dharma_by_id)
    tb = b.perspective_text(self_by_id, dharma_by_id)
    if ta and tb and perspectives_same(
        ta,
        tb,
        llm=llm,
        cache=near_cache,
        focus_person=focus_person,
    ):
        return True
    return False


def filter_diverse_combinations(
    ranked: list[GraspingCombination],
    *,
    self_by_id: dict[str, GraspingHypothesis],
    dharma_by_id: dict[str, GraspingHypothesis],
    max_keep: int = MAX_ALAMBANA_BRANCHES,
    secondary_ratio: float = COMBO_SECONDARY_RATIO,
    id_jaccard_max: float = COMBO_ID_JACCARD_MAX,
    llm: LlmClient | None = None,
    near_cache: PerspectiveNearCache | None = None,
    focus_person: str = "",
) -> list[GraspingCombination]:
    """按激活分保留 + 去同视角；次强须达主组合相对激活强度。"""
    if not ranked:
        return []
    ordered = sorted(ranked, key=lambda c: c.score, reverse=True)
    top = ordered[0].score
    min_secondary = top * secondary_ratio if top > 0 else 0.0
    selected: list[GraspingCombination] = []
    for cand in ordered:
        if len(selected) >= max_keep:
            break
        if selected and cand.score < min_secondary:
            break
        if any(
            is_same_perspective(
                cand,
                kept,
                self_by_id=self_by_id,
                dharma_by_id=dharma_by_id,
                id_jaccard_max=id_jaccard_max,
                llm=llm,
                near_cache=near_cache,
                focus_person=focus_person,
            )
            for kept in selected
        ):
            continue
        selected.append(cand)
    if not selected:
        selected = [ordered[0]]
    return selected


def normalize_combination_probabilities(
    combos: list[GraspingCombination],
) -> list[GraspingCombination]:
    """归一 activation；probability 兼容别名 = activation。

    只对过闸后的 valid_combos 归一。分数全为 0 时返回空列表，
    禁止把唯一失败候选归一成 P(C)=1。
    """
    if not combos:
        return []
    a_scores = [max(c.score, 0.0) for c in combos]
    if sum(a_scores) <= 0:
        return []
    a_norms = normalize_probability_distribution(a_scores)
    for combo, a_p in zip(combos, a_norms):
        combo.activation_probability = clamp_unit_interval(a_p)
        combo.probability = combo.activation_probability
    return combos


def apply_combo_meta(
    combo: GraspingCombination,
    *,
    activation_alignment: float | None = None,
    focus_axis: str | None = None,
    reason: str | None = None,
) -> None:
    """写入 LLM/外部元数据并重算激活分。"""
    if activation_alignment is not None:
        combo.activation_alignment = as_unit_float(activation_alignment)
    if focus_axis is not None:
        axis = str(focus_axis).strip().lower()
        combo.focus_axis = axis if axis in FOCUS_AXES else (axis or "other")
    if reason is not None:
        combo.reason = str(reason).strip()
    combo.score = combo_activation_score(
        combo.self_confidence,
        combo.dharma_confidence,
        combo.activation_alignment,
    )
    combo.raw_combo_score = combo.score


def select_grasping_combinations(
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    *,
    activation_threshold: float = GRASPING_COMBO_ACTIVATION,
    max_keep: int = MAX_ALAMBANA_BRANCHES,
    secondary_ratio: float = COMBO_SECONDARY_RATIO,
    id_jaccard_max: float = COMBO_ID_JACCARD_MAX,
    meta_by_combo_id: dict[str, dict] | None = None,
    llm: LlmClient | None = None,
    focus_person: str = "",
    near_cache: PerspectiveNearCache | None = None,
    candidates: list[GraspingCombination] | None = None,
) -> list[GraspingCombination]:
    """主入口：按 confidence 枚举 →（可选 meta）→ 硬门闸 → 按激活去同视角 → 归一。

    若传入 ``candidates``，不再重新枚举，以便 combo meta 与筛选共用同一批编号。
    当 ``meta_by_combo_id`` 非空时，未打分 combo 不进入竞争（禁止默认 alignment=1）。
    无人过阈或全部未过 alignment/raw_score 门闸时返回空列表。
    """
    cache = near_cache if near_cache is not None else PerspectiveNearCache()
    if candidates is None:
        candidates = enumerate_candidate_combinations(
            self_grasping,
            dharma_grasping,
            activation_threshold=activation_threshold,
            llm=llm,
            near_cache=cache,
            focus_person=focus_person,
        )
    else:
        candidates = list(candidates)
    if not candidates:
        return []

    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    scored_ids = set(meta_by_combo_id) if meta_by_combo_id else None

    for cand in candidates:
        stamp_combo_source(
            cand, self_by_id=self_by_id, dharma_by_id=dharma_by_id
        )
        meta = (meta_by_combo_id or {}).get(cand.combo_id) or {}
        if meta:
            act = meta.get("activation_alignment")
            if act is None:
                act = meta.get("event_alignment")  # 旧 LLM 输出一次性映射
            apply_combo_meta(
                cand,
                activation_alignment=act,
            )
        else:
            cand.raw_combo_score = cand.score

    valid: list[GraspingCombination] = []
    for cand in candidates:
        if scored_ids is not None and cand.combo_id not in scored_ids:
            continue
        if combo_passes_hard_gate(cand):
            valid.append(cand)
    if not valid:
        return []

    selected = filter_diverse_combinations(
        valid,
        self_by_id=self_by_id,
        dharma_by_id=dharma_by_id,
        max_keep=max_keep,
        secondary_ratio=secondary_ratio,
        id_jaccard_max=id_jaccard_max,
        llm=llm,
        near_cache=cache,
        focus_person=focus_person,
    )
    for i, combo in enumerate(selected, start=1):
        combo.combo_id = f"C{i}"
        combo.stamp_perspective_fingerprint(self_by_id, dharma_by_id)
        stamp_combo_source(
            combo, self_by_id=self_by_id, dharma_by_id=dharma_by_id
        )
    return normalize_combination_probabilities(selected)
