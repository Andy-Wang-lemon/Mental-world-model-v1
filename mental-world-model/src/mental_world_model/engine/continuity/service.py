"""连续事件在评估前后的编排。"""

from __future__ import annotations

from mental_world_model.engine.continuity.cognitive import (
    build_belief_items_from_branches,
    compress_belief_to_frames,
    update_cognitive_bias,
)
from mental_world_model.engine.continuity.config import (
    COGNITIVE_TOP_K_INJECT,
    MENTAL_MAX_EPISODES,
    is_cognitive_continuity_enabled,
    is_mental_continuity_enabled,
)
from mental_world_model.engine.continuity.mental import (
    blend_mental_scores,
    compute_lambda_eff,
    weighted_average_mental_history,
)
from mental_world_model.engine.continuity.meta import extract_event_meta
from mental_world_model.engine.continuity.models import (
    ContinuityState,
    MentalEpisode,
    utc_now_iso,
)
from mental_world_model.llm.client import LlmClient


def bias_frames_for_injection(state: ContinuityState) -> list[dict]:
    """按 mass 取出本场工作集；是否触发、绑哪条执由作用域/调制决定。"""
    if not is_cognitive_continuity_enabled():
        return []
    chosen = state.top_cognitive_frames(COGNITIVE_TOP_K_INJECT)
    return [
        {"id": f.id, "text": f.text, "mass": round(f.mass, 4)}
        for f in chosen
    ]


def ensure_event_meta(
    yuanwen: str,
    state: ContinuityState,
    llm: LlmClient,
    mental_trace: dict,
) -> dict:
    """保证 mental_trace 含 theta / relevance。

    ``resolve_mental_history`` 已写入这些键时直接复用，避免同事件抽两次。
    """
    needed = ("theta", "relevance")
    if all(k in mental_trace and mental_trace[k] is not None for k in needed):
        return mental_trace
    try:
        theta, relevance = extract_event_meta(yuanwen, state, llm)
    except Exception as exc:  # noqa: BLE001
        theta, relevance = 0.5, 0.0
        mental_trace.setdefault("meta_reason", f"元数据失败：{exc}")
    mental_trace.setdefault("theta", theta)
    mental_trace.setdefault("relevance", relevance)
    return mental_trace


def resolve_mental_history(
    *,
    yuanwen: str,
    state: ContinuityState,
    llm: LlmClient,
    factor_names: list[str],
    event_scores: dict[str, float] | None = None,
) -> tuple[dict[str, float], float, dict]:
    """解析历史心所与 λ；不写入融合结果。

    返回 (M_hist, lambda_eff, trace)。无历史或关闭时 hist={}、λ=0。
    """
    trace: dict = {
        "enabled": is_mental_continuity_enabled(),
        "mental_factors_event": dict(event_scores or {}),
    }
    if not is_mental_continuity_enabled() or not state.mental_episodes:
        trace["lambda_eff"] = 0.0
        trace["sum_w"] = 0.0
        trace["mental_factors_hist"] = {}
        return {}, 0.0, trace

    try:
        theta, relevance = extract_event_meta(yuanwen, state, llm)
    except Exception as exc:  # noqa: BLE001
        theta, relevance = 0.5, 0.0
        trace["meta_reason"] = f"元数据失败：{exc}"

    trace.update(
        {
            "theta": theta,
            "relevance": relevance,
        }
    )

    hist, weights, sum_w = weighted_average_mental_history(
        state.mental_episodes,
        factor_names=factor_names,
    )
    lam = compute_lambda_eff(relevance)
    trace.update(
        {
            "lambda_eff": lam,
            "sum_w": sum_w,
            "episode_weights": weights,
            "mental_factors_hist": {
                k: v for k, v in hist.items() if float(v) > 0
            },
            "episode_count": len(state.mental_episodes),
        }
    )
    return hist, lam, trace


def apply_mental_continuity(
    *,
    yuanwen: str,
    event_scores: dict[str, float],
    state: ContinuityState,
    llm: LlmClient,
    factor_names: list[str],
) -> tuple[dict[str, float], dict]:
    """前序事件：跨支期望心所再与历史融合。返回 (final_scores, trace)。"""
    hist, lam, trace = resolve_mental_history(
        yuanwen=yuanwen,
        state=state,
        llm=llm,
        factor_names=factor_names,
        event_scores=event_scores,
    )
    trace["apply_mode"] = "cross_branch_event"
    if lam <= 0.0 or not hist:
        return dict(event_scores), trace
    final = blend_mental_scores(event_scores, hist, lam)
    return final, trace


def blend_branches_with_mental_history(
    branches: list,
    *,
    hist_scores: dict[str, float],
    lambda_eff: float,
) -> None:
    """预测事件：同一 M_hist/λ 写入各支 mental_factor_scores_final（就地）。"""
    if lambda_eff <= 0.0 or not hist_scores:
        return
    for branch in branches:
        event_branch = dict(branch.mental_factor_scores_final or {})
        branch.mental_factor_scores_final = blend_mental_scores(
            event_branch, hist_scores, lambda_eff
        )


def update_continuity_after_event(
    *,
    yuanwen: str,
    state: ContinuityState,
    llm: LlmClient,
    candidates: list,
    branches: list,
    event_scores: dict[str, float],
    mental_trace: dict,
    self_grasping: list | None = None,
    dharma_grasping: list | None = None,
) -> tuple[ContinuityState, list[str]]:
    """事件结束后更新 H 与心理 episode 列表。返回 (state, notes)。"""
    state.event_count += 1
    event_id = f"e{state.event_count}"
    now = utc_now_iso()
    notes: list[str] = []

    if is_mental_continuity_enabled():
        mental_trace = ensure_event_meta(yuanwen, state, llm, dict(mental_trace))
        theta = float(mental_trace.get("theta") or 0.5)
        summary = yuanwen.strip().replace("\n", " ")[:120]
        state.mental_episodes.append(
            MentalEpisode(
                event_id=event_id,
                theta=theta,
                timestamp=now,
                scores={k: float(v) for k, v in event_scores.items()},
                text_summary=summary,
            )
        )
        if len(state.mental_episodes) > MENTAL_MAX_EPISODES:
            state.mental_episodes = state.mental_episodes[-MENTAL_MAX_EPISODES:]

    if is_cognitive_continuity_enabled():
        belief_items = build_belief_items_from_branches(
            candidates,
            branches,
            self_grasping=self_grasping,
            dharma_grasping=dharma_grasping,
        )
        incoming = compress_belief_to_frames(yuanwen, belief_items, llm)
        state.cognitive_frames = update_cognitive_bias(
            state.cognitive_frames,
            incoming,
            llm=llm,
            focus_person=str(getattr(state, "person_id", "") or ""),
        )
        if not incoming:
            notes.append("本轮无最终所缘支可登记 H。")

    state.updated_at = now
    return state, notes
