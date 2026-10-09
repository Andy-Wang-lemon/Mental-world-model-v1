"""分支内多行为条件预测。"""

from __future__ import annotations

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.cognition.models import (
    BranchBehaviorPrediction,
    CognitionBranch,
)
from mental_world_model.engine.cognition.probability import normalize_behavior_id
from mental_world_model.engine.cognition.prompts import (
    BEHAVIOR_CANDIDATES_SCHEMA,
    BEHAVIOR_CANONICALIZE_SCHEMA,
    BEHAVIOR_PROBABILITY_SCHEMA,
    BEHAVIOR_SANITIZE_SCHEMA,
    BRANCH_BEHAVIOR_SCHEMA,
    build_behavior_candidate_messages,
    build_behavior_canonicalize_messages,
    build_behavior_diversity_messages,
    build_behavior_probability_messages,
    build_behavior_sanitize_messages,
    build_branch_behavior_messages,
)
from mental_world_model.engine.prediction.predictor import collect_active_xinsuo
from mental_world_model.engine.perception.config import (
    BEHAVIOR_DIVERSITY_RETRIES,
    BEHAVIOR_MIN_DISTINCT,
    BRANCH_ACTION_TOP_K,
)
from mental_world_model.engine.perception.validators import (
    normalize_probability_distribution,
    texts_semantically_near,
)
from mental_world_model.engine.extraction.validators import normalize_match_text
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.confidence import as_unit_float


def sanitize_behavior_texts(
    texts: list[str],
    *,
    llm: LlmClient | None = None,
    extra_names: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """LLM 判断哪些是可观察动作；无 LLM 时原样返回（不按词表切句）。

    LLM 失败、漏答或 ``cleaned_text`` 为空时保留原文，避免 Top-K 被洗空。
    """
    stripped = [(t or "").strip() for t in texts]
    if not stripped:
        return []
    if llm is None:
        return stripped
    try:
        with llm_call_context(stage="cognition.behavior_sanitize"):
            response = llm.complete(
                build_behavior_sanitize_messages(
                    stripped, extra_names=extra_names
                ),
                schema=BEHAVIOR_SANITIZE_SCHEMA,
            )
    except Exception:  # noqa: BLE001
        return stripped
    data = response if isinstance(response, dict) else {}
    items = data.get("items")
    if not isinstance(items, list):
        return stripped
    by_index: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = str(item.get("cleaned_text") or "").strip()
    out: list[str] = []
    for i, raw in enumerate(stripped):
        if i in by_index:
            cleaned = by_index[i]
            out.append(cleaned if cleaned else raw)
        else:
            out.append(raw)
    return out


def sanitize_behavior_text(
    text: str,
    *,
    extra_names: set[str] | frozenset[str] | None = None,
    llm: LlmClient | None = None,
) -> str:
    """单条清洗；有 LLM 时由模型判断，无 LLM 时不改写。"""
    results = sanitize_behavior_texts(
        [text], llm=llm, extra_names=extra_names
    )
    return results[0] if results else ""


def _keep_top_k_behaviors(
    predictions: list[BranchBehaviorPrediction],
    *,
    top_k: int = BRANCH_ACTION_TOP_K,
) -> list[BranchBehaviorPrediction]:
    """按条件概率保留 Top-K，并在支内重新归一化。"""
    if not predictions:
        return []
    ranked = sorted(
        predictions,
        key=lambda p: p.conditional_probability,
        reverse=True,
    )[: max(1, top_k)]
    norms = normalize_probability_distribution(
        [p.conditional_probability for p in ranked]
    )
    for pred, norm in zip(ranked, norms):
        pred.conditional_probability = norm
    return ranked


_SILENCE_ONLY = frozenset(
    {
        "保持沉默",
        "沉默",
        "没有回应",
        "不回应",
        "未回应",
        "没有说话",
        "不说话",
        "没有开口",
        "继续沉默",
        "继续没有回应",
    }
)
_NOISE_TOKENS = ("继续", "暂时", "开始", "仍然", "还是", "再次")


def observable_behavior_key(text: str) -> str:
    """仅当整句就是「不作为/沉默」时才并入同一 canonical。

    「打破沉默」「沉默地点头」等含「沉默」的复合动作不得并入。
    """
    blob = normalize_match_text(text)
    for token in _NOISE_TOKENS:
        blob = blob.replace(token, "")
    if blob in _SILENCE_ONLY:
        return "canonical:保持沉默"
    return blob


def canonicalize_behavior_texts(
    texts: list[str],
    *,
    llm: LlmClient | None = None,
) -> list[tuple[str, str]]:
    """返回 [(surface, canonical), ...]；同一 observable action 共用 canonical。"""
    stripped = [(t or "").strip() for t in texts if (t or "").strip()]
    if not stripped:
        return []
    groups: list[tuple[str, list[str]]] | None = None
    if llm is not None:
        try:
            with llm_call_context(stage="cognition.behavior_canonicalize"):
                response = llm.complete(
                    build_behavior_canonicalize_messages(stripped),
                    schema=BEHAVIOR_CANONICALIZE_SCHEMA,
                )
            raw = response if isinstance(response, dict) else {}
            items = raw.get("groups")
            parsed: list[tuple[str, list[str]]] = []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    canonical = str(item.get("canonical_action") or "").strip()
                    surfaces = item.get("surface_forms")
                    forms = (
                        [str(x).strip() for x in surfaces if str(x).strip()]
                        if isinstance(surfaces, list)
                        else []
                    )
                    if canonical and forms:
                        parsed.append((canonical, forms))
            if parsed:
                groups = parsed
        except Exception:  # noqa: BLE001
            groups = None

    mapping: dict[str, str] = {}
    if groups:
        for canonical, forms in groups:
            for form in forms:
                mapping[form] = canonical

    clustered: list[tuple[str, str]] = []
    canon_by_key: dict[str, str] = {}
    for text in stripped:
        if text in mapping:
            clustered.append((text, mapping[text]))
            continue
        key = observable_behavior_key(text)
        assigned = canon_by_key.get(key)
        if assigned is None:
            for existing_key, existing_canon in canon_by_key.items():
                if texts_semantically_near(text, existing_canon):
                    assigned = existing_canon
                    break
        if assigned is None:
            assigned = text
            canon_by_key[key] = assigned
        clustered.append((text, assigned))
    return clustered


def empty_behavior_diagnostics() -> dict:
    return {
        "raw_candidate_count": 0,
        "distinct_candidate_count": 0,
        "diversity_regenerated": False,
        "candidate_collapsed": False,
    }


def _extract_behavior_items(response: object) -> list[dict]:
    if not isinstance(response, dict):
        return []
    for key in (
        "behavior_candidates",
        "behavior_probabilities",
        "behavior_predictions",
    ):
        raw = response.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _text_of_item(item: dict) -> str:
    return str(
        item.get("text")
        or item.get("canonical_action")
        or item.get("canonical_text")
        or ""
    ).strip()


def predict_behaviors_for_branch(
    yuanwen: str,
    branch: CognitionBranch,
    scores_final: dict[str, float],
    xinsuo_by_name: dict[str, Xinsuo],
    llm: LlmClient,
    *,
    role_context: str = "",
    top_k: int = BRANCH_ACTION_TOP_K,
    focus_person: str = "",
) -> list[BranchBehaviorPrediction]:
    """先生成互异行为 → 语义去重 → 可选多样性重试 → 再估 P(A|branch)。"""
    diagnostics = empty_behavior_diagnostics()
    active = collect_active_xinsuo(scores_final, xinsuo_by_name)
    active_names = {name for name, _, _ in active}
    score_map = {name: float(score) for name, score, _ in active}
    xiang_text = branch.xiang.text if branch.xiang else ""
    si_text = branch.si.text if branch.si else ""

    if not active:
        branch.behavior_diagnostics = diagnostics
        return []

    with llm_call_context(stage="cognition.behavior_candidates"):
        response = llm.complete(
            build_behavior_candidate_messages(
                yuanwen,
                branch.alambana,
                xiang_text,
                si_text,
                active,
                role_context=role_context,
                branch_id=branch.id,
                focus_person=focus_person,
            ),
            schema=BEHAVIOR_CANDIDATES_SCHEMA,
        )
    raw_items = _extract_behavior_items(response)
    if not raw_items:
        with llm_call_context(stage="cognition.behavior"):
            response = llm.complete(
                build_branch_behavior_messages(
                    yuanwen,
                    branch.alambana,
                    xiang_text,
                    si_text,
                    active,
                    role_context=role_context,
                    branch_id=branch.id,
                    focus_person=focus_person,
                ),
                schema=BRANCH_BEHAVIOR_SCHEMA,
            )
        raw_items = _extract_behavior_items(response)

    raw_texts = [_text_of_item(item) for item in raw_items]
    cleaned_texts = sanitize_behavior_texts(
        raw_texts, llm=llm, extra_names=active_names
    )
    paired = [
        (item, text)
        for item, text in zip(raw_items, cleaned_texts)
        if text
    ]
    diagnostics["raw_candidate_count"] = len(paired)

    def _dedup(pairs: list[tuple[dict, str]]) -> list[tuple[str, dict]]:
        clustered = canonicalize_behavior_texts(
            [text for _, text in pairs], llm=llm
        )
        by_canon: dict[str, dict] = {}
        order: list[str] = []
        for (item, surface), (_surface, canonical) in zip(pairs, clustered):
            key = canonical or surface
            if key not in by_canon:
                by_canon[key] = item
                order.append(key)
            else:
                # 同一 observable 只留一条，不把重复措辞的概率相加
                continue
        return [(key, by_canon[key]) for key in order]

    distinct = _dedup(paired)
    diversity_regenerated = False
    # 仅当模型给出了多个表面形式、却塌成同一动作时，才再要一次真正不同的行为。
    # 一开始就只生成 1 条，视为情境高度确定，不强制凑多样性。
    if (
        len(distinct) < BEHAVIOR_MIN_DISTINCT
        and int(diagnostics["raw_candidate_count"] or 0) >= BEHAVIOR_MIN_DISTINCT
        and BEHAVIOR_DIVERSITY_RETRIES > 0
        and paired
    ):
        diversity_regenerated = True
        with llm_call_context(stage="cognition.behavior_diversity"):
            regen = llm.complete(
                build_behavior_diversity_messages(
                    yuanwen,
                    branch.alambana,
                    xiang_text,
                    si_text,
                    active,
                    [text for text, _ in distinct],
                    role_context=role_context,
                    branch_id=branch.id,
                    focus_person=focus_person,
                ),
                schema=BEHAVIOR_CANDIDATES_SCHEMA,
            )
        regen_items = _extract_behavior_items(regen)
        regen_texts = sanitize_behavior_texts(
            [_text_of_item(item) for item in regen_items],
            llm=llm,
            extra_names=active_names,
        )
        regen_pairs = [
            (item, text)
            for item, text in zip(regen_items, regen_texts)
            if text
        ]
        if regen_pairs:
            merged_pairs = paired + regen_pairs
            distinct = _dedup(merged_pairs)

    diagnostics["diversity_regenerated"] = diversity_regenerated
    diagnostics["distinct_candidate_count"] = len(distinct)
    diagnostics["candidate_collapsed"] = len(distinct) == 1
    if not distinct:
        branch.behavior_diagnostics = diagnostics
        return []

    canonical_actions = [text for text, _ in distinct]
    with llm_call_context(stage="cognition.behavior_probability"):
        prob_response = llm.complete(
            build_behavior_probability_messages(
                yuanwen,
                branch.alambana,
                xiang_text,
                si_text,
                active,
                canonical_actions,
                role_context=role_context,
                branch_id=branch.id,
                focus_person=focus_person,
            ),
            schema=BEHAVIOR_PROBABILITY_SCHEMA,
        )
    prob_items = _extract_behavior_items(prob_response)
    prob_by_text: dict[str, dict] = {}
    for item in prob_items:
        key = _text_of_item(item)
        if key:
            prob_by_text[key] = item
            prob_by_text[observable_behavior_key(key)] = item

    predictions: list[BranchBehaviorPrediction] = []
    for canonical, seed in distinct:
        meta = prob_by_text.get(canonical) or prob_by_text.get(
            observable_behavior_key(canonical)
        ) or seed
        supporting_raw = meta.get("supporting_mental_factors", [])
        supporting = (
            [str(x).strip() for x in supporting_raw if str(x).strip() in active_names]
            if isinstance(supporting_raw, list)
            else []
        )
        supporting_scores = {name: score_map[name] for name in supporting}
        text = canonical
        behavior_id = str(meta.get("behavior_id") or seed.get("behavior_id") or "").strip()
        if not behavior_id:
            behavior_id = normalize_behavior_id(text)
        predictions.append(
            BranchBehaviorPrediction(
                behavior_id=behavior_id,
                text=text,
                canonical_text=canonical,
                conditional_probability=as_unit_float(
                    meta.get("conditional_probability"), 0.0
                ),
                confidence=as_unit_float(meta.get("confidence"), 0.5),
                supporting_mental_factors=supporting,
                opposing_mental_factors=[],
                supporting_scores=supporting_scores,
                alambana_id=branch.alambana.id,
                xiang_text=xiang_text,
                si_text=si_text,
                reason_summary="",
            )
        )

    if not any(p.conditional_probability > 0 for p in predictions):
        uniform = 1.0 / len(predictions)
        for pred in predictions:
            pred.conditional_probability = uniform
        if diagnostics["candidate_collapsed"]:
            # 单一有效类别时 1.0 只表示集合坍缩，不表示高确信
            predictions[0].reason_summary = "behavior_candidate_collapsed"

    norms = normalize_probability_distribution(
        [p.conditional_probability for p in predictions]
    )
    for pred, norm in zip(predictions, norms):
        pred.conditional_probability = norm
    kept = _keep_top_k_behaviors(predictions, top_k=top_k)
    branch.behavior_diagnostics = diagnostics
    return kept
