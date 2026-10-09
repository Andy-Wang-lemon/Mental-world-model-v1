"""测试用 Mock LLM 客户端。"""

import re
from typing import Any, Callable

from mental_world_model.engine.scoring.prompts import (
    is_group_suoyuan_stage_content,
    is_xiandai_stage_content,
    is_ye_stage_content,
)
from mental_world_model.engine.scoring.suoyuan_groups import is_group_suoyuan_schema
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.message_utils import user_message_content

ResponseHandler = Callable[[list[dict[str, str]], dict[str, Any] | None], dict[str, Any]]


def _schema_properties(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {}
    properties = schema.get("properties", {})
    return properties if isinstance(properties, dict) else {}


def _is_suoyuan_only_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "suoyuan" in properties and "xiang" not in properties


def _is_router_single_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "relevance" in properties
        and "candidates" not in properties
        and "theta" not in properties
    )


def _is_xiang_si_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "xiang" in properties
        and "si" in properties
        and "suoyuan" not in properties
        and "has_explicit_xiang" not in properties
    )


def _is_xiang_evidence_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "has_explicit_xiang" in properties


def _is_activation_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "activations" in properties


def _group_xinsuo_names_from_content(content: str) -> list[str]:
    if "【心所列表】" in content:
        block = content.split("【心所列表】", 1)[1].split("\n\n", 1)[0].strip()
        return [name.strip() for name in block.split("、") if name.strip()]
    match = re.search(r"心所「(.+?)」", content)
    if match:
        return [match.group(1)]
    return []


def _group_suoyuan_mock_response(
    messages: list[dict[str, str]],
    default_response: dict[str, Any],
) -> dict[str, Any]:
    content = user_message_content(messages)
    names = _group_xinsuo_names_from_content(content)
    default_score = float(default_response.get("score", 1.0))
    return {
        "scores": {
            name: {"score": default_score} for name in names
        }
    }


def _is_objective_facts_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "objective_facts" in properties


def _is_objective_event_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "actors" in properties
        and "action" in properties
        and "scene" in properties
        and "outcome_changes" in properties
    )


def _is_grasping_pair_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "self_grasping" in properties and "dharma_grasping" in properties


def _is_grasping_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "hypotheses" in properties


def _is_stimulus_action_split_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "stimulus_text" in properties
        and "action_span" in properties
        and "action_description" in properties
        and "has_observable_action" not in properties
        and "likelihoods" not in properties
    )


def _is_action_likelihood_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "likelihoods" in properties
        and "has_observable_action" not in properties
        and "stimulus_text" not in properties
    )


def _is_branch_xiangs_schema(schema: dict[str, Any] | None) -> bool:
    """一次生成多支想：{xiangs: [{alambana_id, text, ...}, ...]}。"""
    properties = _schema_properties(schema)
    return "xiangs" in properties


def _is_final_xiang_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "text" in properties
        and "source" in properties
        and "confidence" in properties
        and "alambana_id" not in properties
        and "xiang_text" not in properties
        and "xiangs" not in properties
        and "behavior_predictions" not in properties
        and "has_observable_action" not in properties
        and "distortion_risk" not in properties
        and "candidates" not in properties
    )


def _is_self_representation_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "self_related_elements" in properties
        and "activated_self_identity" in properties
        and "self_owned_elements" in properties
    )


def _is_world_representation_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "entity_understanding" in properties
        and "relationship_understanding" in properties
        and "causal_understanding" in properties
    )


def _is_subjective_alambana_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "candidates" in properties


def _is_combo_meta_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    items = properties.get("items")
    if not isinstance(items, dict):
        return False
    item_props = ((items.get("items") or {}).get("properties") or {})
    return (
        "activation_alignment" in item_props
        or "event_alignment" in item_props
    )


def _is_alambana_grounding_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    items = properties.get("items")
    if not isinstance(items, dict):
        return False
    item_props = ((items.get("items") or {}).get("properties") or {})
    return "grounded" in item_props and "alambana_id" in item_props


def _is_behavior_candidates_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "behavior_candidates" in properties


def _is_behavior_probability_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "behavior_probabilities" in properties


def _is_behavior_canonicalize_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "groups" in properties


def _is_semantic_confirm_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    items = properties.get("items")
    if not isinstance(items, dict):
        return False
    item_props = ((items.get("items") or {}).get("properties") or {})
    return "same_perspective" in item_props and "pair_id" in item_props


def _is_h_scope_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    items = properties.get("items")
    if not isinstance(items, dict):
        return False
    item_props = ((items.get("items") or {}).get("properties") or {})
    return "frame_id" in item_props and "triggered" in item_props


def _is_h_bind_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "bindings" in properties


def _is_h_compress_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "propositions" in properties


def _is_behavior_sanitize_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    items = properties.get("items")
    if not isinstance(items, dict):
        return False
    item_props = ((items.get("items") or {}).get("properties") or {})
    return "cleaned_text" in item_props and "index" in item_props


def _is_observable_filter_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "observable_event" in properties and "visibility" in properties


def _mock_clean_behavior_text(text: str) -> str:
    """仅 Mock：去掉心所解释尾巴，供单测。生产路径走真实 LLM。"""
    cleaned = text or ""
    for phrase in (
        "这符合他内心的无瞋业",
        "这符合她内心的无瞋业",
        "这符合其内心的无瞋业",
        "这符合他内心的疑业",
        "这符合她内心的疑业",
        "即促成善行",
        "即促成",
        "可能会",
        "可能通过",
        "也许会",
        "或许会",
    ):
        cleaned = cleaned.replace(phrase, "")
    cleaned = cleaned.replace("内心", "")
    cleaned = re.sub(r"[，,]{2,}", "，", cleaned)
    cleaned = re.sub(r"[。；;]{2,}", "。", cleaned)
    cleaned = cleaned.strip("的地得 ，,。；; ")
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned):
        return ""
    return cleaned


def _mock_observable_from_action(actor: str, raw_action: str) -> dict[str, Any]:
    """仅 Mock：去掉明显内心/回家分句，保留公开场合里的「私下」。"""
    text = (raw_action or "").strip()
    if not text:
        return {
            "observable_event": f"{actor}没有做出可被察觉的行动。",
            "dropped": [],
            "visibility": "none",
            "reason": "mock empty",
        }
    parts = re.split(r"[。；;！!]+", text)
    kept: list[str] = []
    dropped: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if any(m in chunk for m in ("内心", "心里", "心中", "暗自", "其实", "实则")):
            dropped.append(chunk)
            continue
        if any(m in chunk for m in ("回到家", "回家后", "晚上独自")):
            dropped.append(chunk)
            continue
        if chunk.startswith("即") or "即促成" in chunk or "无瞋业" in chunk:
            dropped.append(chunk)
            continue
        chunk = chunk.replace("表面上", "").replace("可能会", "").replace("可能通过", "")
        chunk = chunk.strip("的地得 ，,")
        if chunk:
            kept.append(chunk)
    if not kept:
        return {
            "observable_event": f"{actor}没有做出可被察觉的行动。",
            "dropped": dropped,
            "visibility": "none",
            "reason": "mock none visible",
        }
    body = "，".join(kept)
    if not body.startswith(actor):
        body = f"{actor}{body}"
    if body[-1] not in "。！？":
        body += "。"
    return {
        "observable_event": body,
        "dropped": dropped,
        "visibility": "partial" if dropped else "visible",
        "reason": "mock observable",
    }


def _is_alambana_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "alambana_candidates" in properties


def _is_subjective_from_base_schema(schema: dict[str, Any] | None) -> bool:
    """执组合染色原来的所缘：单条 text，无 alambana_id / alambana_candidates。"""
    properties = _schema_properties(schema)
    return (
        "text" in properties
        and "distortion_risk" in properties
        and "alambana_id" not in properties
        and "alambana_candidates" not in properties
        and "hypotheses" not in properties
        and "objective_facts" not in properties
    )


def _is_branch_xiang_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "text" in properties
        and "source" in properties
        and "confidence" in properties
        and "alambana_id" in properties
        and "xiang_text" not in properties
        and "behavior_predictions" not in properties
    )


def _is_branch_si_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return (
        "text" in properties
        and "xiang_text" in properties
        and "behavior_predictions" not in properties
    )


def _is_branch_behavior_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "behavior_predictions" in properties


def _is_bias_frames_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "frames" in properties and "has_observable_action" not in properties


def _is_event_meta_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "theta" in properties and "relevance" in properties


def _is_explicit_action_schema(schema: dict[str, Any] | None) -> bool:
    properties = _schema_properties(schema)
    return "has_explicit_action" in properties

def _response_covers_schema(response: dict[str, Any], schema: dict[str, Any] | None) -> bool:
    """旧 handler 常对未知 schema 返回 {score:1}；对新感知/分支 schema 需回退内置。"""
    if not isinstance(response, dict):
        return False
    if _is_explicit_action_schema(schema):
        return "has_explicit_action" in response
    if _is_event_meta_schema(schema):
        return "theta" in response and "relevance" in response
    if _is_bias_frames_schema(schema):
        return "frames" in response
    if _is_objective_facts_schema(schema):
        return "objective_facts" in response
    if _is_objective_event_schema(schema):
        return "actors" in response and "action" in response
    if _is_grasping_schema(schema):
        return "hypotheses" in response
    if _is_grasping_pair_schema(schema):
        return "self_grasping" in response and "dharma_grasping" in response
    if _is_combo_meta_schema(schema):
        return "items" in response
    if _is_alambana_grounding_schema(schema):
        return "items" in response
    if _is_behavior_candidates_schema(schema):
        return "behavior_candidates" in response
    if _is_behavior_probability_schema(schema):
        return "behavior_probabilities" in response
    if _is_behavior_canonicalize_schema(schema):
        return "groups" in response
    if _is_semantic_confirm_schema(schema):
        return "items" in response
    if _is_h_scope_schema(schema):
        return "items" in response
    if _is_h_bind_schema(schema):
        return "bindings" in response
    if _is_h_compress_schema(schema):
        return "propositions" in response
    if _is_behavior_sanitize_schema(schema):
        return "items" in response
    if _is_observable_filter_schema(schema):
        return "observable_event" in response
    if _is_stimulus_action_split_schema(schema):
        return "stimulus_text" in response and "action_span" in response
    if _is_action_likelihood_schema(schema):
        return "likelihoods" in response
    if _is_branch_xiangs_schema(schema):
        return "xiangs" in response
    if _is_final_xiang_schema(schema):
        return "text" in response and "source" in response
    if _is_self_representation_schema(schema):
        return "self_related_elements" in response
    if _is_world_representation_schema(schema):
        return "entity_understanding" in response
    if _is_subjective_alambana_schema(schema):
        return "candidates" in response or "alambana_candidates" in response
    if _is_alambana_schema(schema):
        return "alambana_candidates" in response
    if _is_subjective_from_base_schema(schema):
        return "text" in response
    if _is_branch_behavior_schema(schema):
        return "behavior_predictions" in response
    if _is_branch_si_schema(schema) or _is_branch_xiang_schema(schema):
        return "text" in response and "source" in response and "confidence" in response
    return True


class MockLlmClient:
    """可配置响应的 LLM 客户端，用于测试与开发。"""

    def __init__(
        self,
        default_response: dict[str, Any] | None = None,
        handler: ResponseHandler | None = None,
    ) -> None:
        self.default_response = default_response or {"score": 1.0}
        self.handler = handler

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.handler is not None:
            response = self.handler(messages, schema)
            if _response_covers_schema(response, schema):
                return response
            # 未覆盖新 schema 时回退内置，避免旧测试 handler 打断分支路径

        return self._builtin_complete(messages, schema=schema)

    def _builtin_complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if _is_stimulus_action_split_schema(schema):
            content = user_message_content(messages)
            action_words = ("打断", "否定", "反驳", "离开", "沉默", "点头", "斥责", "未回应")
            span = str(self.default_response.get("action_span", ""))
            if not span:
                for w in action_words:
                    if w in content:
                        span = w
                        break
            if not span:
                span = "（行动）"
            desc = self.default_response.get("action_description")
            if not desc:
                desc = self.default_response.get("action_si_text")
            if not desc:
                desc = f"当事人表现出：{span}"
            stimulus = str(self.default_response.get("stimulus_text") or "").strip()
            if not stimulus:
                stimulus = content
                if "原文：" in content:
                    stimulus = content.split("原文：", 1)[1].strip()
                if span and span in stimulus:
                    stimulus = stimulus.replace(span, "[行动已隔离]", 1)
            return {
                "stimulus_text": stimulus,
                "action_span": span,
                "action_description": desc or "",
                "reason_summary": "mock刺激行动拆分",
            }

        if _is_action_likelihood_schema(schema):
            likes = self.default_response.get("likelihoods")
            if not isinstance(likes, list):
                likes = [
                    {"alambana_id": "O1", "likelihood": 0.8, "support_notes": ["内部消化"]},
                    {"alambana_id": "O2", "likelihood": 0.4, "support_notes": []},
                    {"alambana_id": "O_C1", "likelihood": 0.8, "support_notes": []},
                    {"alambana_id": "O_C2", "likelihood": 0.5, "support_notes": []},
                ]
            return {
                "likelihoods": likes,
                "reason_summary": "mock行动似然",
            }

        if is_group_suoyuan_schema(schema):
            return _group_suoyuan_mock_response(messages, self.default_response)

        if _is_explicit_action_schema(schema):
            content = user_message_content(messages)
            custom = self.default_response.get("has_explicit_action")
            # 默认：含「打断/否定/反驳」等动作词则视为有明确动作
            action_words = ("打断", "否定", "反驳", "离开", "沉默", "点头", "斥责")
            inferred = any(w in content for w in action_words)
            has_action = bool(custom) if custom is not None else inferred
            if "has_explicit_action" in self.default_response:
                has_action = bool(self.default_response["has_explicit_action"])
            span = self.default_response.get("action_span", "")
            if has_action and not span:
                for w in action_words:
                    if w in content:
                        span = w
                        break
                if not span and "原文：" in content:
                    span = content.split("原文：", 1)[1].strip().split("\n", 1)[0][:30]
            return {
                "has_explicit_action": has_action,
                "action_span": span if has_action else "",
                "si_text": self.default_response.get(
                    "action_si_text", "压制" if has_action else ""
                ),
                "confidence": 0.9 if has_action else 0.0,
                "reason_summary": "mock动作检测",
            }

        if _is_event_meta_schema(schema):
            return {
                "theta": float(self.default_response.get("theta", 0.6)),
                "relevance": float(self.default_response.get("relevance", 0.4)),
            }

        if _is_bias_frames_schema(schema):
            custom = self.default_response.get("frames")
            if isinstance(custom, list):
                return {"frames": custom}
            return {
                "frames": [
                    {
                        "text": "组织评价标准可能正在偏向新人",
                        "mass": 0.55,
                        "from_alambana_ids": ["O1"],
                    },
                    {
                        "text": "经验资历的权重可能下降",
                        "mass": 0.4,
                        "from_alambana_ids": ["O1", "O2"],
                    },
                ]
            }

        if _is_objective_facts_schema(schema):
            content = user_message_content(messages)
            # 尽量从原文摘一段可匹配片段
            span = content
            if "原文：" in content:
                span = content.split("原文：", 1)[1].strip().split("\n\n", 1)[0].strip()
            span = span[:40] if span else "测试"
            custom = self.default_response.get("objective_facts")
            if isinstance(custom, list):
                return {"objective_facts": custom}
            return {
                "objective_facts": [
                    {
                        "text": self.default_response.get("fact_text", span),
                        "source_span": self.default_response.get("fact_span", span),
                        "confidence": 1.0,
                    }
                ]
            }

        if _is_objective_event_schema(schema):
            content = user_message_content(messages)
            action = self.default_response.get("event_action", "发生交流")
            scene = self.default_response.get("event_scene", "当前场景")
            actor = self.default_response.get("event_actor", "当事人")
            obj = self.default_response.get("event_object", "")
            source = content
            if "刺激文本：" in content:
                source = content.split("刺激文本：", 1)[1].strip().split("\n\n", 1)[0].strip()
            elif "原文：" in content:
                source = content.split("原文：", 1)[1].strip().split("\n\n", 1)[0].strip()
            if source:
                scene = source[:40]
            role_changes = list(self.default_response.get("role_changes") or [])
            relationship_changes = list(
                self.default_response.get("relationship_changes") or []
            )
            constraint_changes = list(
                self.default_response.get("constraint_changes") or []
            )
            resource_changes = list(self.default_response.get("resource_changes") or [])
            outcome_changes = list(self.default_response.get("event_outcomes") or [])
            if any(k in source for k in ("48h", "48小时", "共同提交", "同一会议室")):
                action = "CEO要求陈默和林可共同提交挽救方案"
                actor = "CEO"
                obj = "陈默和林可"
                if not constraint_changes:
                    constraint_changes = [
                        "陈默和林可必须共同提交方案",
                        "方案需要在48小时内完成",
                        "双方被安排在同一会议室协作",
                    ]
                if not role_changes:
                    role_changes = ["陈默重新进入该项目的共同责任链"]
                if not resource_changes:
                    resource_changes = ["两人被安排在同一会议室"]
            return {
                "actors": [actor],
                "action": action,
                "object": obj,
                "scene": scene,
                "outcome_changes": outcome_changes,
                "role_changes": role_changes,
                "relationship_changes": relationship_changes,
                "constraint_changes": constraint_changes,
                "resource_changes": resource_changes,
            }

        if _is_grasping_schema(schema):
            content = user_message_content(messages)
            gtype = (
                "dharma_grasping" if "法执" in content else "self_grasping"
            )
            custom = self.default_response.get("hypotheses")
            if isinstance(custom, list):
                return {"hypotheses": custom}
            return {
                "hypotheses": [
                    {
                        "id": f"{gtype}_1",
                        "type": gtype,
                        "text": self.default_response.get(
                            "grasping_text", "把资格与地位当成自我的核心"
                        ),
                        "prior_probability": 0.7,
                        "evidence_likelihood": 0.7,
                        "posterior_probability": 0.8,
                        "activation_probability": 0.85,
                        "supporting_evidence": ["上下文线索"],
                        "counter_evidence": [],
                        "missing_information": [],
                        "source": "推理",
                        "confidence": 0.75,
                    },
                    {
                        "id": f"{gtype}_2",
                        "type": gtype,
                        "text": self.default_response.get(
                            "grasping_text_alt", "把协作成果当成固定属于团队的本质"
                        ),
                        "prior_probability": 0.55,
                        "evidence_likelihood": 0.55,
                        "posterior_probability": 0.65,
                        "activation_probability": 0.7,
                        "supporting_evidence": ["模糊线索"],
                        "counter_evidence": [],
                        "missing_information": ["更多上下文"],
                        "source": "推理",
                        "confidence": 0.65,
                    },
                ]
            }

        if _is_grasping_pair_schema(schema):
            custom = self.default_response.get("grasping_pair")
            if isinstance(custom, dict):
                return custom
            joined = "\n".join(str(m.get("content") or "") for m in messages)
            if "认知调节（accommodation）" in joined or "你是认知调节" in joined:
                return {
                    "self_grasping": [
                        {
                            "id": "sg_a1",
                            "text": "自己已被重新纳入必须共同负责的任务",
                            "confidence": 0.8,
                        }
                    ],
                    "dharma_grasping": [
                        {
                            "id": "dg_a1",
                            "text": "CEO把共同提交方案变成必须处理的工作要求",
                            "confidence": 0.8,
                        }
                    ],
                }
            return {
                "self_grasping": [
                    {
                        "id": "sg_f1",
                        "text": self.default_response.get(
                            "grasping_text",
                            "自己重新成为这项任务责任的一部分",
                        ),
                        "confidence": 0.8,
                    }
                ],
                "dharma_grasping": [
                    {
                        "id": "dg_f1",
                        "text": self.default_response.get(
                            "grasping_text_alt",
                            "CEO已经把共同提交方案变成必须处理的工作要求",
                        ),
                        "confidence": 0.75,
                    },
                    {
                        "id": "dg_f2",
                        "text": "把对方行为理解成协作澄清",
                        "confidence": 0.55,
                    },
                ],
            }

        if _is_combo_meta_schema(schema):
            custom = self.default_response.get("combo_meta_items")
            if isinstance(custom, list):
                return {"items": custom}
            content = user_message_content(messages)
            combo_ids = re.findall(
                r"['\"]combo_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            if not combo_ids:
                combo_ids = [f"C{i}" for i in range(1, 7)]
            items = []
            seen_ids: list[str] = []
            for cid in combo_ids:
                if cid in seen_ids:
                    continue
                seen_ids.append(cid)
                pattern = rf"['\"]combo_id['\"]\s*:\s*['\"]{re.escape(cid)}['\"]"
                match = re.search(pattern, content)
                if match:
                    rest = content[match.end() :]
                    nxt = re.search(r"['\"]combo_id['\"]", rest)
                    block = rest[: nxt.start()] if nxt else rest[:400]
                else:
                    block = ""
                has_h = "sg_h" in block or "dg_h" in block
                has_f = "sg_f" in block or "dg_f" in block or "sg_a" in block or "dg_a" in block
                if has_h and not has_f:
                    align = 0.55
                elif has_h and has_f:
                    align = 0.0
                elif has_f:
                    align = 0.85
                else:
                    align = 0.7
                items.append({"combo_id": cid, "activation_alignment": align})
            return {"items": items}

        if _is_alambana_grounding_schema(schema):
            custom = self.default_response.get("grounding_items")
            if isinstance(custom, list):
                return {"items": custom}
            content = user_message_content(messages)
            ids = re.findall(
                r"['\"]id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            if not ids:
                ids = re.findall(r"O_[A-Za-z0-9]+|O\d+", content)
            if not ids:
                ids = ["O_C1"]
            items = []
            for aid in ids:
                hallucinated = any(
                    tok in content
                    for tok in ("林可再次告诉", "林可说", "林可建议")
                )
                items.append(
                    {
                        "alambana_id": aid,
                        "grounded": not hallucinated,
                        "hallucinated_facts": hallucinated,
                        "preserves_critical_event": True,
                        "reason": "mock grounding",
                    }
                )
            return {"items": items}

        if _is_semantic_confirm_schema(schema):
            custom = self.default_response.get("semantic_confirm_items")
            if isinstance(custom, list):
                return {"items": custom}
            content = user_message_content(messages)
            # 默认：规则已过闸的配对视为同视角（兼容既有去重测）
            pair_ids = re.findall(r"['\"]pair_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
            if not pair_ids:
                pair_ids = ["p0"]
            return {
                "items": [
                    {
                        "pair_id": pid,
                        "same_perspective": True,
                    }
                    for pid in pair_ids
                ]
            }

        if _is_h_scope_schema(schema):
            custom = self.default_response.get("h_scope_items")
            if isinstance(custom, list):
                return {"items": custom}
            content = user_message_content(messages)
            frame_ids = re.findall(
                r"['\"]frame_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            if not frame_ids:
                frame_ids = ["bf_1"]
            return {
                "items": [
                    {"frame_id": fid, "triggered": True} for fid in frame_ids
                ]
            }

        if _is_h_bind_schema(schema):
            custom = self.default_response.get("h_bindings")
            if isinstance(custom, list):
                return {"bindings": custom}
            content = user_message_content(messages)
            frame_ids = re.findall(
                r"['\"]frame_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            grasping_ids = re.findall(
                r"['\"]id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            if not frame_ids:
                frame_ids = ["bf_1"]
            return {
                "bindings": [
                    {"frame_id": fid, "grasping_ids": list(grasping_ids)}
                    for fid in frame_ids
                ]
            }

        if _is_h_compress_schema(schema):
            custom = self.default_response.get("h_propositions")
            if isinstance(custom, list):
                return {"propositions": custom}
            content = user_message_content(messages)
            source_ids = re.findall(
                r"['\"]source_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            if not source_ids:
                source_ids = ["O_C1"]
            return {
                "propositions": [
                    {
                        "source_id": sid,
                        "text": "朋友骗了我，我对他不满，以及不信任",
                    }
                    for sid in source_ids
                ]
            }

        if _is_behavior_sanitize_schema(schema):
            custom = self.default_response.get("behavior_sanitize_items")
            if isinstance(custom, list):
                return {"items": custom}
            content = user_message_content(messages)
            texts = re.findall(
                r"['\"]text['\"]\s*:\s*['\"]([^'\"]*)['\"]", content
            )
            if not texts:
                texts = [content]
            return {
                "items": [
                    {"index": i, "cleaned_text": _mock_clean_behavior_text(t)}
                    for i, t in enumerate(texts)
                ]
            }

        if _is_observable_filter_schema(schema):
            custom = self.default_response.get("observable_filter")
            if isinstance(custom, dict):
                return custom
            content = user_message_content(messages)
            actor = "陈默"
            match = re.search(r"行动者：([^\n]+)", content)
            if match:
                actor = match.group(1).strip() or actor
            raw = content
            marker = "行动者的完整预测行动："
            if marker in content:
                raw = content.split(marker, 1)[1]
                raw = raw.split("返回 JSON", 1)[0].strip()
            return _mock_observable_from_action(actor, raw)

        if _is_branch_xiangs_schema(schema):
            custom = self.default_response.get("xiangs")
            if isinstance(custom, list):
                return {"xiangs": custom}
            content = user_message_content(messages)
            ids: list[str] = []
            for token in ("O1", "O2", "O3", "alambana_1", "alambana_2", "alambana_3"):
                if token in content and token not in ids:
                    ids.append(token)
            if not ids:
                ids = ["O1", "O2"]
            base = self.default_response.get(
                "xiang_text", "需要重新评估自己的位置与价值"
            )
            alt = self.default_response.get(
                "xiang_text_alt", "或许这是一次有价值的提醒"
            )
            xiangs = []
            for i, aid in enumerate(ids):
                xiangs.append(
                    {
                        "alambana_id": aid,
                        "text": base if i == 0 else alt,
                        "source": "推理",
                        "confidence": 0.7 if i == 0 else 0.65,
                        "reason_summary": "mock各支想",
                    }
                )
            return {"xiangs": xiangs}

        if _is_final_xiang_schema(schema):
            return {
                "text": self.default_response.get(
                    "xiang_text", "需要重新评估自己的位置与价值"
                ),
                "source": "推理",
                "confidence": 0.7,
                "reason_summary": "mock最终想",
            }

        if _is_self_representation_schema(schema):
            custom = self.default_response.get("self_representation")
            if isinstance(custom, dict):
                return custom
            return {
                "self_related_elements": ["该事件与自我位置直接相关"],
                "activated_self_identity": ["当事人角色"],
                "self_owned_elements": ["我的立场"],
                "self_continuity_change": "自我连续性被轻微触及",
            }

        if _is_world_representation_schema(schema):
            custom = self.default_response.get("world_representation")
            if isinstance(custom, dict):
                return custom
            return {
                "entity_understanding": ["相关对象被理解为互动方"],
                "relationship_understanding": ["关系被理解为协作中带张力"],
                "causal_understanding": ["对方发言导致局面变化"],
                "future_expectation": ["后续可能继续互动澄清"],
            }

        if _is_subjective_alambana_schema(schema):
            custom = self.default_response.get("candidates")
            if isinstance(custom, list):
                return {"candidates": custom}
            content = user_message_content(messages)
            combo_ids = re.findall(
                r"['\"]combo_id['\"]\s*:\s*['\"]([^'\"]+)['\"]", content
            )
            if not combo_ids:
                combo_ids = ["C1"]
            # 去重保持顺序
            seen: set[str] = set()
            ordered: list[str] = []
            for cid in combo_ids:
                if cid not in seen:
                    seen.add(cid)
                    ordered.append(cid)
            candidates = []
            for cid in ordered:
                block_match = re.search(
                    rf"['\"]combo_id['\"]\s*:\s*['\"]{re.escape(cid)}['\"]",
                    content,
                )
                block = ""
                if block_match:
                    rest = content[block_match.end() :]
                    nxt = re.search(r"['\"]combo_id['\"]", rest)
                    block = rest[: nxt.start()] if nxt else rest[:500]
                if "sg_h" in block or "dg_h" in block or "旧方法" in block:
                    text = (
                        "陈默把当前局面重新体验为旧方法可能已经失效、"
                        "自己的专业身份正在被重新衡量。"
                    )
                elif "共同提交" in content or "CEO" in content:
                    text = (
                        "陈默把CEO的安排理解为自己已经无法继续完全置身项目之外，"
                        "因为被重新纳入共同责任。"
                    )
                else:
                    text = self.default_response.get(
                        "alambana_text", "对方正在挑战我的资格"
                    )
                candidates.append(
                    {
                        "id": f"O_{cid}",
                        "combo_id": cid,
                        "text": text,
                    }
                )
            return {"candidates": candidates}

        if _is_alambana_schema(schema):
            custom = self.default_response.get("alambana_candidates")
            if isinstance(custom, list):
                return {"alambana_candidates": custom}
            return {
                "alambana_candidates": [
                    {
                        "id": "alambana_1",
                        "text": self.default_response.get(
                            "alambana_text", "对方正在挑战我的资格"
                        ),
                        "objective_basis": ["原文事实"],
                        "self_grasping_ids": ["self_grasping_1"],
                        "dharma_grasping_ids": ["dharma_grasping_1"],
                        "posterior_probability": 0.55,
                        "confidence": 0.6,
                        "added_meaning": ["挑战"],
                        "distortion_risk": 0.4,
                        "missing_conditions": [],
                        "source": "推理",
                    },
                    {
                        "id": "alambana_2",
                        "text": self.default_response.get(
                            "alambana_text_alt", "对方正在帮助发现问题"
                        ),
                        "objective_basis": ["原文事实"],
                        "self_grasping_ids": ["self_grasping_2"],
                        "dharma_grasping_ids": ["dharma_grasping_2"],
                        "posterior_probability": 0.45,
                        "confidence": 0.55,
                        "added_meaning": ["帮助"],
                        "distortion_risk": 0.3,
                        "missing_conditions": [],
                        "source": "推理",
                    },
                ]
            }

        if _is_subjective_from_base_schema(schema):
            content = user_message_content(messages)
            if (
                "self_grasping_2" in content
                or "协作成果" in content
                or "dharma_grasping_2" in content
            ):
                text = self.default_response.get(
                    "alambana_text_alt", "对方正在帮助发现问题"
                )
            else:
                text = self.default_response.get(
                    "alambana_text", "对方正在挑战我的资格"
                )
            return {
                "text": text,
                "confidence": 0.65,
                "objective_basis": ["原来的所缘"],
                "added_meaning": ["执取染色"],
                "distortion_risk": 0.35,
                "missing_conditions": [],
                "source": "推理",
            }

        if _is_branch_si_schema(schema):
            return {
                "text": self.default_response.get("si_text", "压制并恢复权威"),
                "source": self.default_response.get("si_source", "推理"),
                "confidence": self.default_response.get("si_confidence", 0.6),
                "alambana_id": "alambana_1",
                "xiang_text": self.default_response.get("xiang_text", "这是挑衅"),
                "supporting_grasping_ids": [],
                "reason_summary": "mock思",
            }

        if _is_branch_xiang_schema(schema):
            return {
                "text": self.default_response.get("xiang_text", "这是挑衅"),
                "source": self.default_response.get("xiang_source", "推理"),
                "confidence": self.default_response.get("xiang_confidence", 0.6),
                "alambana_id": "alambana_1",
                "supporting_grasping_ids": [],
                "reason_summary": "mock想",
            }

        if _is_behavior_canonicalize_schema(schema):
            custom = self.default_response.get("behavior_groups")
            if isinstance(custom, list):
                return {"groups": custom}
            content = user_message_content(messages)
            texts = re.findall(r"['\"]text['\"]\s*:\s*['\"]([^'\"]*)['\"]", content)
            silence = []
            other = []
            for t in texts:
                if any(m in t for m in ("沉默", "没有回应", "不回应", "未回应", "不说话")):
                    silence.append(t)
                else:
                    other.append(t)
            groups = []
            if silence:
                groups.append(
                    {"canonical_action": "保持沉默", "surface_forms": silence}
                )
            for t in other:
                groups.append({"canonical_action": t, "surface_forms": [t]})
            if not groups:
                groups = [{"canonical_action": "保持沉默", "surface_forms": ["保持沉默"]}]
            return {"groups": groups}

        if _is_behavior_candidates_schema(schema):
            custom = self.default_response.get("behavior_candidates")
            if isinstance(custom, list):
                return {"behavior_candidates": custom}
            return {
                "behavior_candidates": [
                    {"behavior_id": "silence", "text": "保持沉默，不主动参与"},
                    {"behavior_id": "join", "text": "开始参与共同方案讨论并提出建议"},
                    {
                        "behavior_id": "limited",
                        "text": "提供必要信息，但避免承担主导角色",
                    },
                ]
            }

        if _is_behavior_probability_schema(schema):
            custom = self.default_response.get("behavior_probabilities")
            if isinstance(custom, list):
                return {"behavior_probabilities": custom}
            content = user_message_content(messages)
            actions = re.findall(
                r"已去重行为：\[(.*?)\]", content
            )
            texts: list[str] = []
            marker = "已去重行为："
            if marker in content:
                blob = content.split(marker, 1)[1].split("\n", 1)[0].strip()
                texts = [
                    t.strip().strip("'\"")
                    for t in blob.strip("[]").split(",")
                    if t.strip()
                ]
            if not texts:
                texts = ["保持沉默，不主动参与", "开始参与共同方案讨论并提出建议"]
            n = max(len(texts), 1)
            items = []
            for i, t in enumerate(texts):
                items.append(
                    {
                        "canonical_action": t,
                        "conditional_probability": max(0.15, 0.55 - i * 0.2),
                        "confidence": 0.6,
                        "supporting_mental_factors": ["信"],
                    }
                )
            return {"behavior_probabilities": items}

        if _is_branch_behavior_schema(schema):
            custom = self.default_response.get("behavior_predictions")
            if isinstance(custom, list):
                return {"behavior_predictions": custom}
            return {
                "behavior_predictions": [
                    {
                        "behavior_id": "mock_behavior",
                        "text": self.default_response.get(
                            "predicted_behavior", "mock预测行为"
                        ),
                        "conditional_probability": 1.0,
                        "confidence": 0.6,
                        "supporting_mental_factors": ["信"],
                        "opposing_mental_factors": [],
                        "supporting_scores": {"信": 0.8},
                        "alambana_id": "alambana_1",
                        "xiang_text": "这是挑衅",
                        "si_text": "压制并恢复权威",
                        "reason_summary": "mock分支行为",
                    }
                ]
            }

        if _is_suoyuan_only_schema(schema):
            return {
                "suoyuan": {
                    "text": self.default_response.get("suoyuan_text", "mock所缘"),
                }
            }

        if _is_router_single_schema(schema):
            return {
                "relevance": self.default_response.get("router_relevance", 0.9),
            }

        if _is_xiang_evidence_schema(schema):
            has_explicit = bool(
                self.default_response.get("has_explicit_xiang", False)
            )
            return {
                "has_explicit_xiang": has_explicit,
                "xiang": {
                    "text": self.default_response.get("xiang_text", ""),
                    "source": "原文",
                    "confidence": 1.0,
                },
                "has_explicit_si": bool(
                    self.default_response.get("has_explicit_si", False)
                ),
                "si": {
                    "text": self.default_response.get("si_text", ""),
                    "source": "原文",
                    "confidence": 1.0,
                },
            }

        if _is_xiang_si_schema(schema):
            return {
                "xiang": {
                    "text": self.default_response.get("xiang_text", "mock想"),
                    "source": self.default_response.get("xiang_source", "推理"),
                    "confidence": self.default_response.get("xiang_confidence", 0.6),
                },
                "si": {
                    "text": self.default_response.get("si_text", "mock思"),
                    "source": self.default_response.get("si_source", "推理"),
                    "confidence": self.default_response.get("si_confidence", 0.6),
                },
            }

        if _is_activation_schema(schema):
            content = user_message_content(messages)
            names: list[str] = []
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and "（所缘相似度" in stripped:
                    names.append(stripped[2:].split("（", 1)[0].strip())
            default_a = float(self.default_response.get("activation", 1.0))
            custom = self.default_response.get("activations")
            if isinstance(custom, dict):
                if names:
                    return {
                        "activations": {
                            name: float(custom.get(name, default_a)) for name in names
                        }
                    }
                return {
                    "activations": {str(k): float(v) for k, v in custom.items()}
                }
            return {"activations": {name: default_a for name in names}}

        if schema and "winner" in _schema_properties(schema):
            return {
                "winner": self.default_response.get("winner", "none"),
            }

        if schema and "predicted_behavior" in _schema_properties(schema):
            return {
                "predicted_behavior": self.default_response.get(
                    "predicted_behavior", "mock预测行为"
                ),
                "prediction_reason": self.default_response.get(
                    "prediction_reason", "mock参考心所：信"
                ),
            }

        content = user_message_content(messages)
        if is_group_suoyuan_stage_content(content):
            return _group_suoyuan_mock_response(messages, self.default_response)
        if is_ye_stage_content(content):
            return {
                "score": self.default_response.get("ye_score", 1.0),
            }
        if is_xiandai_stage_content(content):
            return {
                "score": self.default_response.get("xiandai_xinli_score", 1.0),
            }
        return dict(self.default_response)
