"""心智世界模型统一入口。"""

import threading
from dataclasses import dataclass, field
from pathlib import Path
import re

from mental_world_model.core.relations.graph import RelationGraph
from mental_world_model.core.relations.relation import Relation
from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.extraction.extractor import ContextExtractor
from mental_world_model.engine.extraction.models import (
    ExtractedContext,
    SuoyuanXiangSiDefinitions,
)
from mental_world_model.engine.graph_builder.mental_graph_builder import MentalGraphBuilder
from mental_world_model.engine.indexer.definition_indexer import DefinitionIndexer
from mental_world_model.engine.loader.definition_loader import DefinitionLoader
from mental_world_model.engine.normalizer.definition_normalizer import DefinitionNormalizer
from mental_world_model.engine.parser.definition_parser import DefinitionParser
from mental_world_model.engine.prediction.predictor import BehaviorPredictor
from mental_world_model.engine.reconciliation.reconciler import DuizhiReconciler
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.implicit_pipeline import ImplicitBaifaPipeline
from mental_world_model.engine.scoring.pipeline import XinsuoScoringPipeline
from mental_world_model.engine.scoring.results import EvaluationResult
from mental_world_model.engine.scoring.weights import (
    ScoringWeights,
    build_default_seed_weights,
)
from mental_world_model.engine.loader.seed_loader import (
    SeedLoaderError,
    load_xinsuo_seed_weights,
)
from mental_world_model.engine.cognition.runner import (
    CognitionEvaluationBundle,
    apply_predicted_action as run_apply_predicted_action,
    evaluate_text_with_cognition_branches as run_cognition_branches,
)
from mental_world_model.engine.perception.config import is_cognition_branching_enabled
from mental_world_model.engine.seeds.config import is_seed_library_enabled
from mental_world_model.engine.seeds.state_store import SeedStateStore
from mental_world_model.engine.seeds.updater import update_seed_weights
from mental_world_model.engine.continuity.config import (
    is_cognitive_continuity_enabled,
    is_mental_continuity_enabled,
)
from mental_world_model.engine.continuity.models import ContinuityState
from mental_world_model.engine.continuity.store import ContinuityStateStore
from mental_world_model.engine.validator.definition_validator import DefinitionValidator
from mental_world_model.engine.validator.validation_result import DefinitionValidationError
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.env_file import get_project_root
from mental_world_model.llm.tracing_client import ensure_tracing_client
from mental_world_model.llm.usage import llm_evaluation_scope


@dataclass
class MentalWorldModel:
    """加载定义并执行评分与冲突消解的统一入口。"""

    xinsuo_list: list[Xinsuo]
    relations: list[Relation]
    suoyuan_xiang_si_definitions: SuoyuanXiangSiDefinitions
    indexer: DefinitionIndexer
    validator: DefinitionValidator
    graph: RelationGraph
    weights: ScoringWeights
    # 默认种子表（只读，供隐式路径；不启种子库子系统）
    default_seed_catalog: dict[str, float] = field(default_factory=dict)
    seed_state_store: SeedStateStore | None = None
    continuity_state_store: ContinuityStateStore | None = None
    continuity_state: ContinuityState = field(default_factory=ContinuityState)
    definitions_dir: Path = field(default_factory=Path)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(
        cls,
        definitions_dir: str | Path,
        weights: ScoringWeights | None = None,
        *,
        seed_state_path: str | Path | None = None,
        use_runtime_seeds: bool = True,
    ) -> "MentalWorldModel":
        """从 definitions 目录加载心所、对治关系及所缘境/想/思定义。

        三种定义均采用统一策略：normalized JSON 优先，缺失时从 raw docx 解析并写入 normalized。
        加载后：校验原始 dict → 规范化 → 解析 → 语义校验（含重复名与关系端点）。
        """
        root = Path(definitions_dir)
        schemas_dir = root / "schemas"

        loader = DefinitionLoader()
        normalizer = DefinitionNormalizer()
        parser = DefinitionParser()
        validator = DefinitionValidator(schemas_dir=schemas_dir)

        xinsuo_raw = loader.load_xinsuo_definitions(root)
        xinsuo_pre = validator.validate_xinsuo_list(xinsuo_raw)
        if not xinsuo_pre.valid:
            raise DefinitionValidationError(xinsuo_pre.errors)

        xinsuo_raw = normalizer.normalize(xinsuo_raw)
        xinsuo_list = parser.parse_xinsuo(xinsuo_raw)

        relations_raw = loader.load_duizhi_definitions(root)
        relations_pre = validator.validate_relation_list(relations_raw)
        if not relations_pre.valid:
            raise DefinitionValidationError(relations_pre.errors)

        relations_raw = normalizer.normalize(relations_raw)
        relations = parser.parse_relation(relations_raw)

        sx_raw = loader.load_suoyuan_xiang_si_definitions(root)
        sx_raw = normalizer.normalize(sx_raw)
        sx_validation = validator.validate_suoyuan_xiang_si(sx_raw)
        if not sx_validation.valid:
            raise DefinitionValidationError(sx_validation.errors)

        xinsuo_names = {xinsuo.name for xinsuo in xinsuo_list}
        relation_validation = validator.validate_relation_list(
            relations,
            xinsuo_names=xinsuo_names,
        )
        if not relation_validation.valid:
            raise DefinitionValidationError(relation_validation.errors)

        suoyuan_xiang_si_definitions = SuoyuanXiangSiDefinitions.from_dict(sx_raw)

        indexer = DefinitionIndexer()
        indexer.build_index({"xinsuo": xinsuo_list, "relations": relations})

        graph_builder = MentalGraphBuilder()
        graph = graph_builder.build_from_definitions(
            {"xinsuo": xinsuo_list, "relations": relations}
        )

        default_seed_weights = build_default_seed_weights(
            [x.name for x in xinsuo_list],
            definitions_dir=root,
        )
        try:
            default_seed_catalog = load_xinsuo_seed_weights(
                root, [x.name for x in xinsuo_list]
            )
        except (SeedLoaderError, FileNotFoundError, OSError, ValueError):
            default_seed_catalog = {
                name: 0.5 for name in [x.name for x in xinsuo_list]
            }

        scoring_weights = weights or ScoringWeights()
        seed_store: SeedStateStore | None = None

        if is_seed_library_enabled() and use_runtime_seeds:
            if seed_state_path is None:
                seed_store = SeedStateStore.default_path(get_project_root())
            else:
                seed_store = SeedStateStore(seed_state_path)
            scoring_weights.seed_weights = seed_store.load_or_initialize(
                default_seed_weights
            )
        elif not scoring_weights.seed_weights:
            scoring_weights.seed_weights = default_seed_weights

        continuity_store: ContinuityStateStore | None = None
        continuity_state = ContinuityState()
        if is_cognitive_continuity_enabled() or is_mental_continuity_enabled():
            if seed_state_path is not None:
                continuity_store = ContinuityStateStore(
                    Path(seed_state_path).parent / "continuity_state.json"
                )
            else:
                continuity_store = ContinuityStateStore.default_path(get_project_root())
            continuity_state = continuity_store.load()

        return cls(
            xinsuo_list=xinsuo_list,
            relations=relations,
            suoyuan_xiang_si_definitions=suoyuan_xiang_si_definitions,
            indexer=indexer,
            validator=validator,
            graph=graph,
            weights=scoring_weights,
            default_seed_catalog=default_seed_catalog,
            seed_state_store=seed_store,
            continuity_state_store=continuity_store,
            continuity_state=continuity_state,
            definitions_dir=root,
        )

    def extract(self, yuanwen: str, llm_client: LlmClient) -> ExtractedContext:
        """从原文提取所缘境、想、思（与心所评分使用同一 LLM）。"""
        extractor = ContextExtractor(
            llm_client,
            self.suoyuan_xiang_si_definitions,
            self.xinsuo_list,
            seed_weights=self.weights.seed_weights if is_seed_library_enabled() else {},
        )
        return extractor.extract(yuanwen)

    def evaluate(
        self,
        context: ScoringContext,
        llm_client: LlmClient,
        *,
        update_seeds: bool = True,
    ) -> EvaluationResult:
        """心所打分（显式四列 / 隐式百法）→ 对治消解 → 行为预测。"""
        seed_weights_before = dict(self.weights.seed_weights)

        if context.extracted.is_implicit_path:
            pipeline = ImplicitBaifaPipeline(
                llm_client,
                self.default_seed_catalog
                or {x.name: 0.5 for x in self.xinsuo_list},
            )
            stage_results = pipeline.run(self.xinsuo_list, context)
            scores_initial = ImplicitBaifaPipeline.to_scores_dict(stage_results)
        else:
            pipeline = XinsuoScoringPipeline(llm_client, weights=self.weights)
            stage_results = pipeline.run(self.xinsuo_list, context)
            scores_initial = pipeline.to_scores_dict(stage_results)

        conflicts_initial = self.validator.check_score_conflicts(
            scores_initial, self.relations
        )

        xinsuo_by_name = {x.name: x for x in self.xinsuo_list}
        reconciler = DuizhiReconciler(llm_client, validator=self.validator)
        scores_final, resolutions = reconciler.reconcile(
            scores_initial,
            self.relations,
            xinsuo_by_name,
            context,
        )
        conflicts_final = self.validator.check_score_conflicts(
            scores_final, self.relations
        )

        prediction = BehaviorPredictor(llm_client).predict(
            context,
            scores_final,
            xinsuo_by_name,
        )

        seed_weights_after = seed_weights_before
        if update_seeds and is_seed_library_enabled():
            seed_weights_after = update_seed_weights(
                seed_weights_before,
                scores_final,
            )
            self.weights.seed_weights = seed_weights_after
            if self.seed_state_store is not None:
                self.seed_state_store.save(seed_weights_after)

        return EvaluationResult(
            extracted=context.extracted,
            stage_results=stage_results,
            scores_initial=scores_initial,
            conflicts_initial=conflicts_initial,
            conflicts_final=conflicts_final,
            resolutions=resolutions,
            scores_final=scores_final,
            predicted_behavior=prediction.predicted_behavior,
            active_xinsuo=prediction.active_xinsuo_as_dicts(),
            prediction_reason=prediction.prediction_reason,
            seed_weights_before=seed_weights_before,
            seed_weights_after=seed_weights_after,
        )

    def evaluate_text_with_cognition_branches(
        self,
        text: str,
        llm_client: LlmClient,
        *,
        role_context: str = "",
        update_seeds: bool = False,
        update_continuity: bool | None = None,
        use_action_feedback: bool = False,
        aggregate_behaviors: bool = False,
        predict_branch_behaviors: bool = True,
        focus_person: str = "",
    ) -> CognitionEvaluationBundle:
        """多主观所缘认知分支评估（感知 → 分支想/思 → 评分 → 汇总）。"""
        llm = ensure_tracing_client(llm_client)
        do_continuity = (
            update_continuity
            if update_continuity is not None
            else (
                is_cognitive_continuity_enabled() or is_mental_continuity_enabled()
            )
        )
        with llm_evaluation_scope() as tracker:
            bundle = run_cognition_branches(
                text,
                llm,
                xinsuo_list=self.xinsuo_list,
                relations=self.relations,
                weights=self.weights,
                validator=self.validator,
                role_context=role_context,
                continuity_state=self.continuity_state,
                update_continuity=do_continuity,
                use_action_feedback=use_action_feedback,
                aggregate_behaviors=aggregate_behaviors,
                predict_branch_behaviors=predict_branch_behaviors,
                focus_person=focus_person,
            )
            with self._state_lock:
                if do_continuity and bundle.continuity_state is not None:
                    self.continuity_state = bundle.continuity_state
                    if self.continuity_state_store is not None:
                        self.continuity_state_store.save(self.continuity_state)
                if update_seeds and is_seed_library_enabled():
                    seed_before = dict(self.weights.seed_weights)
                    seed_scores = (
                        bundle.evaluation_result.mental_factors_event
                        or bundle.evaluation_result.expected_mental_factor_scores
                        or bundle.evaluation_result.scores_final
                    )
                    seed_after = update_seed_weights(
                        seed_before,
                        seed_scores,
                    )
                    self.weights.seed_weights = seed_after
                    if self.seed_state_store is not None:
                        self.seed_state_store.save(seed_after)
                    bundle.evaluation_result.seed_weights_before = seed_before
                    bundle.evaluation_result.seed_weights_after = seed_after
        bundle.evaluation_result.llm_usage = tracker.summary()
        return bundle

    def apply_predicted_action(
        self,
        text: str,
        llm_client: LlmClient,
        predicted: CognitionEvaluationBundle,
        action: str,
        *,
        role_context: str = "",
        focus_person: str = "",
        action_probability: float = 0.0,
        update_seeds: bool = False,
    ) -> CognitionEvaluationBundle:
        """在预测所缘上接入 Top-1 行动，compat/扩支后写入 H/M。不重抽 L2/L3。"""
        llm = ensure_tracing_client(llm_client)
        with llm_evaluation_scope() as tracker:
            bundle = run_apply_predicted_action(
                text,
                llm,
                predicted,
                action,
                xinsuo_list=self.xinsuo_list,
                relations=self.relations,
                weights=self.weights,
                validator=self.validator,
                role_context=role_context,
                continuity_state=self.continuity_state,
                focus_person=focus_person,
                action_probability=action_probability,
            )
            with self._state_lock:
                if bundle.continuity_state is not None:
                    self.continuity_state = bundle.continuity_state
                    if self.continuity_state_store is not None:
                        self.continuity_state_store.save(self.continuity_state)
                if update_seeds and is_seed_library_enabled():
                    seed_before = dict(self.weights.seed_weights)
                    seed_scores = (
                        bundle.evaluation_result.mental_factors_event
                        or bundle.evaluation_result.expected_mental_factor_scores
                        or bundle.evaluation_result.scores_final
                    )
                    seed_after = update_seed_weights(
                        seed_before,
                        seed_scores,
                    )
                    self.weights.seed_weights = seed_after
                    if self.seed_state_store is not None:
                        self.seed_state_store.save(seed_after)
                    bundle.evaluation_result.seed_weights_before = seed_before
                    bundle.evaluation_result.seed_weights_after = seed_after
        bundle.evaluation_result.llm_usage = tracker.summary()
        return bundle

    def _land_predicted_top1(
        self,
        bundle: CognitionEvaluationBundle,
        llm_client: LlmClient,
        *,
        event_text: str,
        role_context: str,
        focus: str,
        update_seeds: bool,
    ) -> CognitionEvaluationBundle:
        """预测不写 M；可用 Top-1 接到同一套所缘后再写 H/M。返回仍是预测结果。"""
        from mental_world_model.engine.cognition.predicted_action import (
            is_usable_predicted_action,
            top_aggregated_action,
        )

        result = bundle.evaluation_result
        action, prob, _ = top_aggregated_action(result)
        notes = []
        if result.perception_state is not None:
            notes = result.perception_state.uncertainty_notes
        if not is_usable_predicted_action(action):
            notes.append(f"末事件预测行动不可用（{action!r}），未写入 H/M。")
            return bundle
        land = self.apply_predicted_action(
            event_text,
            llm_client,
            bundle,
            action,
            role_context=role_context,
            focus_person=focus,
            action_probability=prob,
            update_seeds=update_seeds,
        )
        notes.append(
            f"末事件已按预测 Top-1 接入同一套所缘并写入 H/M：{action}"
        )
        bundle.continuity_state = land.continuity_state
        cont = dict(result.continuity or {})
        land_cont = land.evaluation_result.continuity or {}
        cont["landed_action"] = action
        cont["landed"] = True
        if "mental_episode_count" in land_cont:
            cont["mental_episode_count"] = land_cont["mental_episode_count"]
        result.continuity = cont
        return bundle

    _EVENT_LABEL_RE = re.compile(
        r"(?m)^\s*事件\s*([0-9]+|[一二三四五六七八九十百千]+|[A-Za-z]+)\s*[：:]"
    )

    @classmethod
    def _split_numbered_events(cls, yuanwen: str) -> list[tuple[str, str]]:
        """按「事件1/事件2/事件A...」切分。

        返回 ``[(label, body), ...]``，其中 *body* 已去掉"事件X："标签行。
        切不出或仅一个事件时返回空列表。
        """
        matches = list(cls._EVENT_LABEL_RE.finditer(yuanwen))
        if len(matches) <= 1:
            return []
        results: list[tuple[str, str]] = []
        empty_labels: list[str] = []
        for idx, match in enumerate(matches):
            label = f"事件{match.group(1)}"
            content_start = match.end()
            content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(yuanwen)
            body = yuanwen[content_start:content_end].strip()
            if body:
                results.append((label, body))
            else:
                empty_labels.append(label)
        if empty_labels:
            import warnings
            warnings.warn(
                f"以下事件标签正文为空，已跳过（可能导致路由错误）：{empty_labels}",
                stacklevel=3,
            )
        return results

    def reset_continuity_state(self) -> ContinuityState:
        """清空认知偏置与心理事件列表。"""
        self.continuity_state = ContinuityState(person_id=self.continuity_state.person_id)
        if self.continuity_state_store is not None:
            self.continuity_state_store.save(self.continuity_state)
        return self.continuity_state

    def evaluate_from_yuanwen(
        self,
        yuanwen: str,
        llm_client: LlmClient,
        *,
        update_seeds: bool = True,
        role_context: str = "",
        history: str = "",
        focus_person: str | None = None,
    ) -> EvaluationResult:
        """原文评估入口。

        默认（COGNITION_BRANCHING_ENABLED=True）走多认知分支；
        关闭时回退到旧的单一提取 → 评分路径。

        末事件按预测处理：不先写 H/M；若 Top-1 可用，接到同一套所缘上
        再写 H/M（不重抽 L2/L3）。返回仍是预测结果（含行为分布）。

        ``focus_person``：CLI/API 显式分析对象优先；未传则从原文
        「分析对象：某某」等标记解析。多事件全程锁定同一人。
        """
        from mental_world_model.engine.perception.focus import resolve_focus_person

        focus = resolve_focus_person(focus_person, yuanwen)
        if focus and self.continuity_state.person_id in ("", "default"):
            self.continuity_state.person_id = focus

        if is_cognition_branching_enabled():
            event_pairs = self._split_numbered_events(yuanwen)
            if len(event_pairs) > 1:
                # 提取事件标签前的前言（角色设定等），作为所有事件的冷背景
                first_match = self._EVENT_LABEL_RE.search(yuanwen)
                preamble = yuanwen[: first_match.start()].strip() if first_match else ""

                event_reports: list[dict] = []
                cumulative_context_parts: list[str] = []
                if preamble:
                    cumulative_context_parts.append(preamble)

                # 每个历史事件截取前 600 字，最多保留最近 3 个，避免 context 无限膨胀
                _MAX_HIST_EVENTS = 3
                _MAX_BODY_CHARS = 600

                for label, body in event_pairs[:-1]:
                    recent_parts = cumulative_context_parts[-_MAX_HIST_EVENTS:]
                    bg = "\n".join(recent_parts) if recent_parts else ""
                    combined_rc = f"{role_context}\n{bg}".strip() if (role_context or bg) else ""
                    hist_bundle = self.evaluate_text_with_cognition_branches(
                        body,
                        llm_client,
                        role_context=combined_rc,
                        update_seeds=False,
                        use_action_feedback=True,
                        aggregate_behaviors=False,
                        predict_branch_behaviors=False,
                        focus_person=focus,
                    )
                    event_reports.append(
                        hist_bundle.evaluation_result.to_event_snapshot(event_label=label)
                    )
                    truncated = body[:_MAX_BODY_CHARS] + ("…" if len(body) > _MAX_BODY_CHARS else "")
                    cumulative_context_parts.append(f"[{label}] {truncated}")

                final_label, final_body = event_pairs[-1]
                recent_parts = cumulative_context_parts[-_MAX_HIST_EVENTS:]
                bg = "\n".join(recent_parts) if recent_parts else ""
                combined_rc = f"{role_context}\n{bg}".strip() if (role_context or bg) else ""
                bundle = self.evaluate_text_with_cognition_branches(
                    final_body,
                    llm_client,
                    role_context=combined_rc,
                    update_seeds=False,
                    update_continuity=False,
                    use_action_feedback=False,
                    # 预测事件：跨支行为概率融合（非只取最高所缘）
                    aggregate_behaviors=True,
                    focus_person=focus,
                )
                bundle = self._land_predicted_top1(
                    bundle,
                    llm_client,
                    event_text=final_body,
                    role_context=combined_rc,
                    focus=focus,
                    update_seeds=update_seeds,
                )
                event_reports.append(
                    bundle.evaluation_result.to_event_snapshot(event_label=final_label)
                )
                bundle.evaluation_result.perception_state.uncertainty_notes.append(
                    f"事件路由：检测到 {len(event_pairs)} 件事件，前序事件已用于连续性更新。"
                )
                if focus:
                    bundle.evaluation_result.perception_state.uncertainty_notes.append(
                        f"分析对象锁定：{focus}（多事件全程统一）。"
                    )
                bundle.evaluation_result.event_reports = event_reports
                bundle.evaluation_result.focus_person = focus
                return bundle.evaluation_result
            bundle = self.evaluate_text_with_cognition_branches(
                yuanwen,
                llm_client,
                role_context=role_context,
                update_seeds=False,
                update_continuity=False,
                use_action_feedback=False,
                aggregate_behaviors=True,
                focus_person=focus,
            )
            bundle = self._land_predicted_top1(
                bundle,
                llm_client,
                event_text=yuanwen,
                role_context=role_context,
                focus=focus,
                update_seeds=update_seeds,
            )
            bundle.evaluation_result.event_reports = [
                bundle.evaluation_result.to_event_snapshot(event_label="事件1")
            ]
            bundle.evaluation_result.focus_person = focus
            return bundle.evaluation_result

        llm = ensure_tracing_client(llm_client)
        with llm_evaluation_scope() as tracker:
            extracted = self.extract(yuanwen, llm)
            context = ScoringContext.from_extracted(extracted)
            result = self.evaluate(context, llm, update_seeds=update_seeds)
        result.llm_usage = tracker.summary()
        result.focus_person = focus
        return result

    def reset_runtime_seeds(self) -> dict[str, float]:
        """将运行时种子库重置为 definitions 默认种子。"""
        default_weights = build_default_seed_weights(
            [x.name for x in self.xinsuo_list],
            definitions_dir=self.definitions_dir,
        )
        self.weights.seed_weights = default_weights
        if self.seed_state_store is not None:
            self.seed_state_store.save(default_weights)
        return dict(default_weights)

    def get_xinsuo(self, name: str) -> Xinsuo | None:
        """按名称获取心所定义。"""
        return self.indexer.find(name)
