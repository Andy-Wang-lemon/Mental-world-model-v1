"""隐式百法路径：所缘门控 + 默认种子 + 换位激活 → 最终分。"""

from __future__ import annotations

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.implicit_formula import (
    IMPLICIT_SUOYUAN_BETA,
    compute_implicit_score,
)
from mental_world_model.engine.scoring.pipeline import XinsuoScoringPipeline
from mental_world_model.engine.scoring.prompts import build_suoyuan_messages
from mental_world_model.engine.scoring.results import (
    DEPENDENT_ON_FEN,
    FEN_NAME,
    HEN_NAME,
    NAO_NAME,
    FEN_HEN_NAO_CHAIN,
    StageResult,
    XinsuoScoreResult,
)
from mental_world_model.engine.scoring.weights import GATE_THRESHOLD, ScoringWeights
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context
from mental_world_model.utils.confidence import as_unit_float

_STAGE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
}

_ACTIVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "activations": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        }
    },
    "required": ["activations"],
}


def build_activation_messages(
    context: ScoringContext,
    candidates: list[tuple[str, float, float, Xinsuo]],
) -> list[dict[str, str]]:
    """换位激活：candidates 为 (name, suoyuan_score, seed_weight, xinsuo)。"""
    lines = [
        "【换位激活】",
        "你必须完全代入原文中的分析对象本人（第一人称「我」），带着我执"
        "（护持自我评价、地位、正确感、被承认感）来判断。",
        "禁止旁观者道德叙事（如「为了公司」「应该大度」）。",
        "",
        f"原文：\n{context.yuanwen}",
        "",
        f"所缘境：{context.extracted_suoyuan}",
        "",
        "下列心所所缘相似度均已 > 0.3。请判断在你身上哪些会真正生起、强度如何。",
        "对每个心所给出 activation ∈ [0,1]（0=不起，1=很强）。",
        "",
        "候选心所：",
    ]
    for name, suoyuan_score, seed_weight, xinsuo in candidates:
        lines.append(
            f"- {name}（所缘相似度={suoyuan_score:.4f}，大众种子敏感度={seed_weight:.4f}）\n"
            f"  所缘定义：{xinsuo.suoyuan}\n"
            f"  生起念头：{xinsuo.shengqi_nian}\n"
            f"  业：{xinsuo.ye}"
        )
    lines.extend(
        [
            "",
            '只返回 JSON：{"activations": {"心所名": 0.0, ...}}',
            "必须包含上面每一个候选心所名称。",
        ]
    )
    return [{"role": "user", "content": "\n".join(lines)}]


class ImplicitBaifaPipeline:
    """隐式路径评分：不做想/思推理；忿→恨→恼链保留。"""

    def __init__(
        self,
        llm_client: LlmClient,
        default_seed_weights: dict[str, float],
        *,
        gate_threshold: float = GATE_THRESHOLD,
        beta: float = IMPLICIT_SUOYUAN_BETA,
    ) -> None:
        self._llm = llm_client
        self._seeds = default_seed_weights
        self._gate = gate_threshold
        self._beta = beta
        # 复用所缘 batch 预取
        self._suoyuan_helper = XinsuoScoringPipeline(
            llm_client, weights=ScoringWeights(), gate_threshold=gate_threshold
        )

    def run(
        self,
        xinsuo_list: list[Xinsuo],
        context: ScoringContext,
    ) -> list[XinsuoScoreResult]:
        by_name = {x.name: x for x in xinsuo_list}
        suoyuan_prefetch = self._suoyuan_helper._prefetch_suoyuan_scores(
            by_name, context
        )

        results: dict[str, XinsuoScoreResult] = {
            name: XinsuoScoreResult(name=name, exit_at=None) for name in by_name
        }

        # 主集合（不含恨、恼）先写入所缘并门控
        main_names = [n for n in by_name if n not in DEPENDENT_ON_FEN]
        pending_activate: list[str] = []
        for name in main_names:
            score = float(suoyuan_prefetch.get(name, 0.0))
            result = results[name]
            result.stage_results.append(StageResult(stage="suoyuan", score=score))
            result.column_scores["suoyuan"] = score
            if score <= self._gate:
                result.exit_at = "suoyuan"
                result.final_score = 0.0
            else:
                pending_activate.append(name)

        # 对主集合（含忿）一次换位激活
        if pending_activate:
            self._apply_activations(by_name, context, results, pending_activate)

        # 忿 → 恨 → 恼
        self._score_fen_hen_nao_chain(by_name, context, results)

        self._ensure_placeholders(by_name, results)
        return [results[x.name] for x in xinsuo_list]

    def _apply_activations(
        self,
        by_name: dict[str, Xinsuo],
        context: ScoringContext,
        results: dict[str, XinsuoScoreResult],
        names: list[str],
    ) -> None:
        candidates: list[tuple[str, float, float, Xinsuo]] = []
        for name in names:
            xinsuo = by_name[name]
            suoyuan_score = float(results[name].column_scores.get("suoyuan", 0.0))
            seed = float(self._seeds.get(name, 0.5))
            candidates.append((name, suoyuan_score, seed, xinsuo))

        activations = self._fetch_activations(context, candidates)
        for name, suoyuan_score, seed, _ in candidates:
            a = float(activations.get(name, 0.0))
            final = compute_implicit_score(
                suoyuan_score,
                seed,
                a,
                beta=self._beta,
                gate_threshold=self._gate,
            )
            result = results[name]
            result.column_scores["activation"] = a
            result.column_scores["seed_weight"] = seed
            result.final_score = final
            if final <= 0:
                result.exit_at = result.exit_at or "suoyuan"

    def _fetch_activations(
        self,
        context: ScoringContext,
        candidates: list[tuple[str, float, float, Xinsuo]],
    ) -> dict[str, float]:
        if not candidates:
            return {}
        messages = build_activation_messages(context, candidates)
        with llm_call_context(stage="scoring.implicit_activation"):
            response = self._llm.complete(messages, schema=_ACTIVATION_SCHEMA)
        raw = response.get("activations", {})
        if not isinstance(raw, dict):
            raw = {}
        out: dict[str, float] = {}
        for name, _, _, _ in candidates:
            try:
                out[name] = float(raw.get(name, 0.0))
            except (TypeError, ValueError):
                out[name] = 0.0
        return out

    def _score_single_suoyuan(
        self,
        xinsuo: Xinsuo,
        context: ScoringContext,
    ) -> float:
        with llm_call_context(stage="scoring.suoyuan_single"):
            response = self._llm.complete(
                build_suoyuan_messages(xinsuo, context),
                schema=_STAGE_SCORE_SCHEMA,
            )
        return as_unit_float(response.get("score"), 0.0)

    def _score_fen_hen_nao_chain(
        self,
        by_name: dict[str, Xinsuo],
        context: ScoringContext,
        results: dict[str, XinsuoScoreResult],
    ) -> None:
        if FEN_NAME not in by_name:
            for name in FEN_HEN_NAO_CHAIN:
                if name not in results:
                    results[name] = self._skip(name, f"缺少 {FEN_NAME}，链未执行")
            return

        fen = results[FEN_NAME]
        if fen.final_score <= 0:
            results[HEN_NAME] = self._skip(HEN_NAME, f"{FEN_NAME} 为 0，依序不判断")
            results[NAO_NAME] = self._skip(NAO_NAME, f"{FEN_NAME} 为 0，依序不判断")
            return

        if HEN_NAME not in by_name:
            results[HEN_NAME] = self._skip(HEN_NAME, f"缺少 {HEN_NAME}")
            results[NAO_NAME] = self._skip(NAO_NAME, f"缺少 {HEN_NAME}")
            return

        hen_sy = self._score_single_suoyuan(by_name[HEN_NAME], context)
        hen_result = results.get(HEN_NAME) or XinsuoScoreResult(name=HEN_NAME, exit_at=None)
        hen_result.stage_results.append(StageResult(stage="suoyuan", score=hen_sy))
        hen_result.column_scores["suoyuan"] = hen_sy
        results[HEN_NAME] = hen_result

        if hen_sy <= self._gate:
            hen_result.exit_at = "suoyuan"
            hen_result.final_score = 0.0
            results[NAO_NAME] = self._skip(NAO_NAME, f"{HEN_NAME} 所缘未过门，依序不判断")
            return

        self._apply_activations(by_name, context, results, [HEN_NAME])
        if results[HEN_NAME].final_score <= 0:
            results[NAO_NAME] = self._skip(NAO_NAME, f"{HEN_NAME} 为 0，依序不判断")
            return

        if NAO_NAME not in by_name:
            results[NAO_NAME] = self._skip(NAO_NAME, f"缺少 {NAO_NAME}")
            return

        nao_sy = self._score_single_suoyuan(by_name[NAO_NAME], context)
        nao_result = results.get(NAO_NAME) or XinsuoScoreResult(name=NAO_NAME, exit_at=None)
        nao_result.stage_results.append(StageResult(stage="suoyuan", score=nao_sy))
        nao_result.column_scores["suoyuan"] = nao_sy
        results[NAO_NAME] = nao_result

        if nao_sy <= self._gate:
            nao_result.exit_at = "suoyuan"
            nao_result.final_score = 0.0
            return

        self._apply_activations(by_name, context, results, [NAO_NAME])

    @staticmethod
    def _skip(name: str, reason: str) -> XinsuoScoreResult:
        return XinsuoScoreResult(
            name=name,
            exit_at=None,
            skipped=True,
            skip_reason=reason,
            final_score=0.0,
        )

    @staticmethod
    def _ensure_placeholders(
        by_name: dict[str, Xinsuo],
        results: dict[str, XinsuoScoreResult],
    ) -> None:
        for name in FEN_HEN_NAO_CHAIN:
            if name not in results:
                results[name] = ImplicitBaifaPipeline._skip(
                    name, "链占位：未评分"
                )
        for name in by_name:
            if name not in results:
                results[name] = ImplicitBaifaPipeline._skip(name, "未评分")

    @staticmethod
    def to_scores_dict(results: list[XinsuoScoreResult]) -> dict[str, float]:
        return {item.name: item.final_score for item in results}
