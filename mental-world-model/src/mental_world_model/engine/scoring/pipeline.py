"""心所评分管线。"""

from __future__ import annotations

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.parallel_exec import run_parallel
from mental_world_model.engine.scoring.prompts import (
    build_group_suoyuan_messages,
    build_shengqi_nian_messages,
    build_suoyuan_messages,
    build_xiandai_xinli_messages,
    build_ye_messages,
)
from mental_world_model.engine.scoring.results import (
    DEPENDENT_ON_FEN,
    FEN_NAME,
    HEN_NAME,
    NAO_NAME,
    FEN_HEN_NAO_CHAIN,
    StageResult,
    XinsuoScoreResult,
)
from mental_world_model.engine.scoring.suoyuan_groups import (
    group_suoyuan_output_schema,
    parse_group_suoyuan_response,
    plan_suoyuan_scoring_groups,
)
from mental_world_model.engine.scoring.weights import (
    GATE_THRESHOLD,
    SUOYUAN_HARD_FLOOR,
    ScoringWeights,
    compute_normalized_column_weights,
    compute_score_breakdown,
    compute_source_modifiers,
)
from mental_world_model.utils.confidence import as_unit_float
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context

_STAGE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
}

StageScore = float


class XinsuoScoringPipeline:
    """按文档顺序遍历心所，分阶段门控评分。"""

    def __init__(
        self,
        llm_client: LlmClient,
        weights: ScoringWeights | None = None,
        gate_threshold: float = GATE_THRESHOLD,
    ) -> None:
        self._llm = llm_client
        self._weights = weights or ScoringWeights()
        self._gate_threshold = gate_threshold
        self._source_modifiers: dict[str, float] | None = None
        self._normalized_weights: dict[str, float] | None = None

    def run(
        self,
        xinsuo_list: list[Xinsuo],
        context: ScoringContext,
    ) -> list[XinsuoScoreResult]:
        """四波评分：所缘 batch → 生起念头 → 业 → 现代心理学。

        四列仍分波（不把所缘/生起/业/心理拍平）。每波内全部待评心所并行。
        所缘境：13 组合并 + 8 单独（约 21 次 LLM）。门控 0.3 不变。
        忿 / 恨 / 恼 链式依赖不变。
        """
        self._source_modifiers = compute_source_modifiers(context.extracted)
        self._normalized_weights = compute_normalized_column_weights(
            self._weights.column_weights,
            self._source_modifiers,
        )

        by_name = {xinsuo.name: xinsuo for xinsuo in xinsuo_list}
        suoyuan_prefetch = self._prefetch_suoyuan_scores(by_name, context)

        main_names = [
            xinsuo.name
            for xinsuo in xinsuo_list
            if xinsuo.name not in DEPENDENT_ON_FEN
        ]
        results_by_name = self._score_waves(
            by_name,
            main_names,
            context,
            suoyuan_prefetch=suoyuan_prefetch,
        )

        if FEN_NAME in results_by_name:
            self._score_fen_dependents(
                by_name,
                context,
                results_by_name[FEN_NAME],
                results_by_name,
            )

        self._ensure_fen_chain_placeholders(by_name, results_by_name)
        return [results_by_name[xinsuo.name] for xinsuo in xinsuo_list]

    def _ensure_fen_chain_placeholders(
        self,
        by_name: dict[str, Xinsuo],
        results_by_name: dict[str, XinsuoScoreResult],
    ) -> None:
        """补全忿/恨/恼链上缺失条目的跳过结果，避免 KeyError。"""
        if FEN_NAME not in by_name:
            for name in FEN_HEN_NAO_CHAIN:
                if name not in results_by_name:
                    results_by_name[name] = self._make_dependency_skip_result(
                        name,
                        f"心所表缺少 {FEN_NAME}，{FEN_HEN_NAO_CHAIN} 链未执行",
                    )
            return

        for name in (HEN_NAME, NAO_NAME):
            if name not in by_name and name not in results_by_name:
                results_by_name[name] = self._make_dependency_skip_result(
                    name,
                    f"心所表缺少 {name}，依序不判断",
                )
            elif name not in results_by_name:
                results_by_name[name] = self._make_dependency_skip_result(
                    name,
                    f"心所表缺少链前置条目，未评分",
                )

    def _score_fen_dependents(
        self,
        by_name: dict[str, Xinsuo],
        context: ScoringContext,
        fen_result: XinsuoScoreResult,
        results_by_name: dict[str, XinsuoScoreResult],
    ) -> None:
        """忿 评完后，按链式条件决定是否评恨、恼。"""
        if HEN_NAME not in by_name:
            results_by_name[HEN_NAME] = self._make_dependency_skip_result(
                HEN_NAME, f"心所表缺少 {HEN_NAME}，依序不判断"
            )
            results_by_name[NAO_NAME] = self._make_dependency_skip_result(
                NAO_NAME, f"心所表缺少 {HEN_NAME}，依序不判断"
            )
            return

        if fen_result.final_score <= 0:
            results_by_name[HEN_NAME] = self._make_dependency_skip_result(
                HEN_NAME, f"{FEN_NAME} 为 0，依序不判断"
            )
            results_by_name[NAO_NAME] = self._make_dependency_skip_result(
                NAO_NAME, f"{FEN_NAME} 为 0，依序不判断"
            )
            return

        hen_results = self._score_waves(
            by_name,
            [HEN_NAME],
            context,
            suoyuan_prefetch={},
        )
        hen_result = hen_results[HEN_NAME]
        results_by_name[HEN_NAME] = hen_result

        if hen_result.final_score <= 0:
            results_by_name[NAO_NAME] = self._make_dependency_skip_result(
                NAO_NAME, f"{HEN_NAME} 为 0，依序不判断"
            )
            return

        if NAO_NAME not in by_name:
            results_by_name[NAO_NAME] = self._make_dependency_skip_result(
                NAO_NAME, f"心所表缺少 {NAO_NAME}，依序不判断"
            )
            return

        nao_results = self._score_waves(
            by_name,
            [NAO_NAME],
            context,
            suoyuan_prefetch={},
        )
        results_by_name[NAO_NAME] = nao_results[NAO_NAME]

    @staticmethod
    def _make_dependency_skip_result(name: str, reason: str) -> XinsuoScoreResult:
        return XinsuoScoreResult(
            name=name,
            exit_at=None,
            skipped=True,
            skip_reason=reason,
            final_score=0.0,
        )

    def _prefetch_suoyuan_scores(
        self,
        xinsuo_by_name: dict[str, Xinsuo],
        context: ScoringContext,
    ) -> dict[str, StageScore]:
        """第 0 波：所缘境合并 batch 预评（约 21 次 LLM，波内全部并行）。"""
        batches = plan_suoyuan_scoring_groups(xinsuo_by_name)

        def fetch_batch(index: int):
            batch = batches[index]
            try:
                with llm_call_context(stage="scoring.suoyuan_batch"):
                    response = self._llm.complete(
                        build_group_suoyuan_messages(
                            batch.suoyuan_text,
                            batch.xinsuo_names,
                            context,
                        ),
                        schema=group_suoyuan_output_schema(),
                    )
                return parse_group_suoyuan_response(response, batch.xinsuo_names)
            except Exception:  # noqa: BLE001
                # 单批失败不废整支：该批心所所缘分记 0，后续仍可凭生起念头过门
                return {name: 0.0 for name in batch.xinsuo_names}

        batch_results = run_parallel(
            {str(index): (lambda i=index: fetch_batch(i)) for index in range(len(batches))}
        )

        prefetched: dict[str, StageScore] = {}
        for index in range(len(batches)):
            prefetched.update(batch_results[str(index)])
        return prefetched

    def _fetch_individual_suoyuan(
        self,
        by_name: dict[str, Xinsuo],
        names: list[str],
        context: ScoringContext,
    ) -> dict[str, StageScore]:
        """单心所所缘境（恨/恼等未纳入 batch 者），波内并行。"""

        def fetch_one(name: str) -> StageScore:
            try:
                with llm_call_context(stage="scoring.suoyuan_single"):
                    response = self._llm.complete(
                        build_suoyuan_messages(by_name[name], context),
                        schema=_STAGE_SCORE_SCHEMA,
                    )
                return as_unit_float(response.get("score"), 0.0)
            except Exception:  # noqa: BLE001
                return 0.0

        return run_parallel({name: (lambda n=name: fetch_one(n)) for name in names})

    def _parallel_column_scores(
        self,
        by_name: dict[str, Xinsuo],
        names: list[str],
        context: ScoringContext,
        stage: str,
    ) -> dict[str, StageScore]:
        """对一批心所并行请求同一列评分。"""

        stage_name = f"scoring.{stage}"

        def fetch_one(name: str) -> StageScore:
            xinsuo = by_name[name]
            if stage == "shengqi_nian":
                messages = build_shengqi_nian_messages(xinsuo, context)
            elif stage == "ye":
                messages = build_ye_messages(xinsuo, context)
            elif stage == "xiandai_xinli":
                messages = build_xiandai_xinli_messages(xinsuo, context)
            else:
                raise ValueError(f"未知评分列：{stage}")
            try:
                with llm_call_context(stage=stage_name):
                    response = self._llm.complete(messages, schema=_STAGE_SCORE_SCHEMA)
                return as_unit_float(response.get("score"), 0.0)
            except Exception:  # noqa: BLE001
                return 0.0

        return run_parallel({name: (lambda n=name: fetch_one(n)) for name in names})

    def _apply_suoyuan_wave(
        self,
        results: dict[str, XinsuoScoreResult],
        names: list[str],
        suoyuan_scores: dict[str, StageScore],
    ) -> list[str]:
        """写入所缘境分；所缘 < SUOYUAN_HARD_FLOOR 直接出局，其余进入生起念头。"""
        active: list[str] = []
        floor = SUOYUAN_HARD_FLOOR
        for name in names:
            score = suoyuan_scores[name]
            result = results[name]
            result.stage_results.append(
                StageResult(stage="suoyuan", score=score)
            )
            result.column_scores["suoyuan"] = score
            if float(score) < floor:
                result.exit_at = "suoyuan"
                result.final_score = 0.0
            else:
                active.append(name)
        return active

    def _apply_column_wave(
        self,
        results: dict[str, XinsuoScoreResult],
        names: list[str],
        column_scores: dict[str, StageScore],
        *,
        stage: str,
        gate: bool,
    ) -> list[str]:
        """写入某一列分数；若 gate=True 则应用单列门控并返回仍活跃者。"""
        active: list[str] = []
        for name in names:
            score = column_scores[name]
            result = results[name]
            result.stage_results.append(
                StageResult(stage=stage, score=score)
            )
            result.column_scores[stage] = score
            if gate and score < self._gate_threshold:
                result.exit_at = stage
                result.final_score = 0.0
            else:
                active.append(name)
        return active

    def _apply_suoyuan_shengqi_dual_gate(
        self,
        results: dict[str, XinsuoScoreResult],
        names: list[str],
    ) -> list[str]:
        """所缘与生起念头（想）双门控：仅当两者都低于阈值才排除。"""
        active: list[str] = []
        gate = self._gate_threshold
        for name in names:
            result = results[name]
            suoyuan = float(result.column_scores.get("suoyuan", 0.0))
            shengqi = float(result.column_scores.get("shengqi_nian", 0.0))
            if suoyuan < gate and shengqi < gate:
                result.exit_at = "shengqi_nian"
                result.final_score = 0.0
            else:
                result.exit_at = None
                active.append(name)
        return active

    def _finalize_results(
        self,
        results: dict[str, XinsuoScoreResult],
        names: list[str],
        context: ScoringContext,
    ) -> None:
        """对完成四列的心所计算 breakdown 与 final_score。"""
        for name in names:
            result = results[name]
            if result.exit_at is not None:
                continue
            breakdown = compute_score_breakdown(
                name,
                result.column_scores,
                self._weights,
                context.extracted,
                source_modifiers=self._source_modifiers,
                normalized_weights=self._normalized_weights,
            )
            result.score_breakdown = breakdown
            result.final_score = breakdown.final_score

    def _score_waves(
        self,
        by_name: dict[str, Xinsuo],
        names: list[str],
        context: ScoringContext,
        *,
        suoyuan_prefetch: dict[str, StageScore],
    ) -> dict[str, XinsuoScoreResult]:
        """四波：所缘 → 生起念头 → 双门控 → 业 → 现代心理学 → 合成。

        各波串行；波内全部待评心所并行。
        门控：所缘 < 0.1 出局；其余评生起后双门控，仅当所缘与想都未过才排除。
        """
        if not names:
            return {}

        results: dict[str, XinsuoScoreResult] = {
            name: XinsuoScoreResult(name=name, exit_at=None) for name in names
        }

        missing_suoyuan = [name for name in names if name not in suoyuan_prefetch]
        suoyuan_scores = dict(suoyuan_prefetch)
        if missing_suoyuan:
            suoyuan_scores.update(
                self._fetch_individual_suoyuan(by_name, missing_suoyuan, context)
            )

        # 所缘：<0.1 直接出局；其余进生起念头后再双门控
        candidates = self._apply_suoyuan_wave(results, names, suoyuan_scores)
        if not candidates:
            return results

        shengqi_scores = self._parallel_column_scores(
            by_name, candidates, context, "shengqi_nian"
        )
        self._apply_column_wave(
            results,
            candidates,
            shengqi_scores,
            stage="shengqi_nian",
            gate=False,
        )
        active = self._apply_suoyuan_shengqi_dual_gate(results, candidates)
        if not active:
            return results

        ye_scores = self._parallel_column_scores(by_name, active, context, "ye")
        active = self._apply_column_wave(
            results,
            active,
            ye_scores,
            stage="ye",
            gate=False,
        )
        if not active:
            return results

        xiandai_scores = self._parallel_column_scores(
            by_name, active, context, "xiandai_xinli"
        )
        self._apply_column_wave(
            results,
            active,
            xiandai_scores,
            stage="xiandai_xinli",
            gate=False,
        )
        self._finalize_results(results, names, context)
        return results

    def to_scores_dict(self, results: list[XinsuoScoreResult]) -> dict[str, float]:
        """提取心所名称到最终分的映射。"""
        return {item.name: item.final_score for item in results}
