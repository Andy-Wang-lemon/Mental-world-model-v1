"""主观世界建构层提示词。

L0 刺激/行动拆分 → L1 客观刺激 → L2 我执+法执 → L2.5 组合（仅激活）→ L3 所缘。
感知输入只用 stimulus_text；行动证据留给认知层（思 / L / 想）。
"""

from __future__ import annotations

from mental_world_model.engine.perception.config import PERCEPTION_PROMPT_VERSION
from mental_world_model.engine.perception.focus import (
    focus_person_instruction,
    focus_person_user_line,
)
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
    ObjectiveEventStructure,
)

STIMULUS_ACTION_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "stimulus_text": {"type": "string"},
        "action_span": {"type": "string"},
        "action_description": {"type": "string"},
    },
    "required": ["stimulus_text", "action_span", "action_description"],
}

OBJECTIVE_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "actors": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string"},
        "object": {"type": "string"},
        "scene": {"type": "string"},
        "outcome_changes": {"type": "array", "items": {"type": "string"}},
        "role_changes": {"type": "array", "items": {"type": "string"}},
        "relationship_changes": {"type": "array", "items": {"type": "string"}},
        "constraint_changes": {"type": "array", "items": {"type": "string"}},
        "resource_changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["actors", "action", "object", "scene", "outcome_changes"],
}

GRASPING_PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "self_grasping": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_bias_frame_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id", "text", "confidence"],
            },
        },
        "dharma_grasping": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_bias_frame_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id", "text", "confidence"],
            },
        },
    },
    "required": ["self_grasping", "dharma_grasping"],
}

COMBO_META_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "combo_id": {"type": "string"},
                    "activation_alignment": {"type": "number"},
                },
                "required": [
                    "combo_id",
                    "activation_alignment",
                ],
            },
        }
    },
    "required": ["items"],
}

# 二执路径：弱规则过闸后的语义二次确认（批量）
SEMANTIC_CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "string"},
                    "same_perspective": {"type": "boolean"},
                },
                "required": ["pair_id", "same_perspective"],
            },
        }
    },
    "required": ["items"],
}

SUBJECTIVE_ALAMBANA_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "combo_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": [
                    "id",
                    "combo_id",
                    "text",
                ],
            },
        }
    },
    "required": ["candidates"],
}

PSYCH_BAN_TERMS = (
    "情绪",
    "愤怒",
    "开心",
    "伤心",
    "恐惧",
    "焦虑",
    "嫉妒",
    "动机",
    "意图",
    "目的是",
    "心所",
    "觉得自己",
    "想要打压",
    "想打压",
    "让我愤怒",
    "使我愤怒",
    "故意为难",
)


_BG_INSTRUCTION = (
    "若提供了「背景信息」，仅将其视为无感情的客观背景参考，"
    "不要让背景信息主导或直接影响对当前事件的判断。\n"
)

# L2 专用：背景中仍可能激活的执取（任意类型）可写入；须能挂回当前事件
_GRASPING_BG_INSTRUCTION = (
    "若提供了「背景信息」：\n"
    "- 以当前刺激文本与客观刺激为主；背景不替代当前事件。\n"
    "- 背景中若有仍可能在当前情境激活的执取（任何类型），允许推理写入。\n"
    "- 禁止照抄背景原文；禁止与当前事件无关的空想。\n"
)


def _grasping_definition_block(who: str) -> str:
    """我执 / 法执定义 + 完整性 + confidence。L2 各路径共用。"""
    return (
        "定义（必须按此理解）：\n"
        f"- 我执：分析对象（{who}）把什么当成与自己分不开的。"
        "不限于所属和身份；人、事、关系、位置只要被放在「我这边」、"
        "不能当成与我无关，也是我执。\n"
        f"- 法执：分析对象（{who}）把人、事、规则、关系执成何种固定理解"
        "（理应如此、只能如此、本来如此）。\n"
        "完整性：有几种不能互相替代的我执或法执，就分别写几条；"
        "一条只表达一种黏着或一种理解。并进一条会丢掉另一种，就必须拆开。"
        "不要只写最显著的那条。刺激极空洞才可少写。\n"
        "confidence：表示当前刺激在多大程度上点燃了这一条，"
        "不是这条听起来有多强、有多惨。不同黏着可以同时高分；"
        "不要因为某一条更醒目就把其他条压低。\n"
    )


def _background_block(role_context: str, *, for_grasping: bool = False) -> str:
    if not role_context:
        return ""
    if for_grasping:
        return (
            "背景信息（若其中仍有可激活的执取可写入；不替代当前事件）：\n"
            f"{role_context}\n\n"
        )
    return f"背景信息（仅供参考，不直接影响当前分析）：\n{role_context}\n\n"


def build_stimulus_action_split_messages(
    text: str,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """L0：一次拆分刺激文本与行动证据骨架（不含 likelihood）。

    调用方保证本事件有主体行动；不由模型判断有无行动或置信度。
    """
    bg_instruction = _BG_INSTRUCTION if role_context else ""
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是刺激/行动拆分器。\n"
        f"{focus_block}"
        "任务：把原文拆成两块，彼此隔离。本事件一定含主体（分析对象）行动证据。\n\n"
        "1. stimulus_text：客观刺激窗。只保留分析对象面对的情境/他人言行/环境变化；"
        "删除或替换分析对象随后的造作、反应、成败结果。"
        "若删除后语句断裂，用「[行动已隔离]」占位，勿编造新情节。\n"
        "2. action_span / action_description：分析对象侧行动证据"
        "（内心造作意向、外在言行、明确无反应/未行动——原文写明的都算）。\n\n"
        "规则：\n"
        "- stimulus_text 不得含分析对象行动证据。\n"
        "- action_span 须能在原文核对；多处用分号连接。\n"
        "- action_description 只能是对 action_span 的忠实改写/通顺化："
        "可合并语序、补全指代，但不得增加动机、情绪、强度、"
        "原文没有的动作细节；禁止心理评价、情绪标签、心所名。\n"
        f"{bg_instruction}"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"{_background_block(role_context)}"
        f"原文：\n{text}\n\n"
        "返回 JSON：\n"
        '{"stimulus_text":"","action_span":"","action_description":""}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_objective_event_messages(
    text: str,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """L1：从刺激文本解析客观刺激结构（输入应为已隔离行动的 stimulus_text）。"""
    bg_instruction = _BG_INSTRUCTION if role_context else ""
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是客观刺激解析器。\n"
        f"{focus_block}"
        "任务：把刺激文本拆成可观察的刺激结构。\n"
        "只能写镜头可见、录音可闻的刺激侧事实。\n"
        "禁止写分析对象随后的反应、造作、成败结果。\n"
        "禁止写心理、情绪、动机、意图、评价。\n"
        "原文没有的信息填空，不要猜测。\n"
        f"{bg_instruction}"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"{_background_block(role_context)}"
        f"刺激文本：\n{text}\n\n"
        "请返回客观刺激 JSON：\n"
        "- actors：参与者\n"
        "- action：他人/环境施加的可观察刺激动作（不是分析对象反应）\n"
        "- object：作用对象（可空）\n"
        "- scene：场合/背景（可空）\n"
        "- outcome_changes：刺激侧局面变化（可空；禁止分析对象行动结果）\n"
        "- role_changes：客观角色/职责变化（可空）\n"
        "- relationship_changes：客观关系安排变化（可空）\n"
        "- constraint_changes：客观约束/期限/必须共同完成的要求（可空）\n"
        "- resource_changes：客观资源/场地安排变化（可空）\n\n"
        "以上变化字段只写事件本身造成的结构变化，禁止心理解释"
        "（如感到被迫、不愿意、觉得受威胁）。\n"
        "禁止：觉得、想要、故意、情绪词、心所、分析对象反应。\n"
        '返回：{"actors":[],"action":"","object":"","scene":"",'
        '"outcome_changes":[],"role_changes":[],"relationship_changes":[],'
        '"constraint_changes":[],"resource_changes":[]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_grasping_pair_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    role_context: str = "",
    cognitive_bias_frames: list[dict] | None = None,
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """Fresh L2：只根据当前 L1 生成我执/法执，不注入 H。"""
    del cognitive_bias_frames
    return build_fresh_grasping_messages(
        text,
        objective_event,
        role_context=role_context,
        focus_person=focus_person,
    )


def build_fresh_grasping_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """Path A：单看当前事件，它可能如何被主体执取。"""
    bg_instruction = _GRASPING_BG_INSTRUCTION if role_context else ""
    focus_block = focus_person_instruction(focus_person)
    who = str(focus_person or "").strip() or "主体"
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是「当前事件」我执 / 法执提取器。\n\n"
        f"{focus_block}"
        f"{_grasping_definition_block(who)}"
        f"任务：只根据当前刺激与客观刺激，回答「单看这件事，{who}可能如何取境」。\n"
        "不要依赖任何历史解释框架。不要否定过去，也不要求与过去不同。\n"
        "必须明确以客观刺激（含角色/约束/关系/资源变化）为事实底座。\n"
        "禁止根据分析对象后续行动倒推执取。禁止改写为其他人物的内心执取。\n"
        f"{bg_instruction}"
        "要求：\n"
        "1. 每条必须挂住当前客观刺激中的锚点；无锚点空想一律禁止。\n"
        "2. 不要输出行为预测、情绪标签、心所名。\n"
        "3. 每条给 id、text、confidence∈[0,1]。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"{_background_block(role_context, for_grasping=True)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        "请提取当前事件自身可能激活的 self_grasping 与 dharma_grasping。\n"
        "返回 JSON：\n"
        '{"self_grasping":[{"id":"sg_f1","text":"","confidence":0.0}],'
        '"dharma_grasping":[{"id":"dg_f1","text":"","confidence":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_accommodation_grasping_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    *,
    failed_reason: str = "",
    focus_person: str = "",
) -> list[dict[str, str]]:
    """现有候选无法解释当前事件时，仅凭 L1 重新寻找新执取。"""
    focus_block = focus_person_instruction(focus_person)
    who = str(focus_person or "").strip() or "主体"
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是认知调节（accommodation）提取器。\n"
        f"{focus_block}"
        f"{_grasping_definition_block(who)}"
        "现有认知候选无法解释当前事件。"
        f"请重新寻找由当前事件中新出现的角色变化、约束变化、关系变化、结果变化"
        f"所激活的、{who}的不同执取。\n"
        "只使用当前客观刺激。不要沿用已经失败的旧解释。"
        "禁止编造 L1 中不存在的事实。禁止情绪标签、心所名、行为预测。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"失败原因：{failed_reason or '已有组合无法贴合当前事件'}\n\n"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        "请重新提取 self_grasping 与 dharma_grasping。\n"
        "返回 JSON：\n"
        '{"self_grasping":[{"id":"sg_a1","text":"","confidence":0.0}],'
        '"dharma_grasping":[{"id":"dg_a1","text":"","confidence":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_grasping_pair_action_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    *,
    action_span: str = "",
    action_description: str = "",
    existing_alambana_summaries: list[dict] | None = None,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """行动扩支 L2'：客观所缘 + 行动 → 新我法二执（须与已有所缘视角不同）。"""
    focus_block = focus_person_instruction(focus_person)
    who = str(focus_person or "").strip() or "主体"
    action_block = (
        f"span={action_span}\n"
        f"description={action_description}"
        if (action_span or action_description)
        else "（无）"
    )
    existing_block = existing_alambana_summaries or []
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是「行动解释向」我执/法执提取器。\n"
        f"{focus_block}"
        f"{_grasping_definition_block(who)}"
        "任务：给定客观刺激与已观测行动，推断「何种内心执取」能更好解释该行动。\n"
        f"输出须能支撑一条与下方「已有主观所缘」**明显不同**的新视角。\n"
        "禁止复述行动本身当作执取；禁止输出心所名、行为预测清单。\n"
        "id 请用 sg_act1/dg_act1 等形式，避免与常规 sg1/dg1 冲突。\n"
        "每条给 id、text、confidence∈[0,1]。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"原文：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"行动证据：\n{action_block}\n\n"
        f"已有主观所缘（勿重复视角）：\n{existing_block}\n\n"
        f"请为「{who}」提取能解释上述行动的 self_grasping 与 dharma_grasping。\n"
        "返回 JSON：\n"
        '{"self_grasping":[{"id":"sg_act1","text":"","confidence":0.0}],'
        '"dharma_grasping":[{"id":"dg_act1","text":"","confidence":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_combo_meta_action_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    candidates: list[GraspingCombination],
    *,
    action_span: str = "",
    action_description: str = "",
    focus_person: str = "",
) -> list[dict[str, str]]:
    """行动扩支 L2.5'：组合在「刺激+行动」下的激活与行动解释度。"""
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    payload = []
    for c in candidates:
        payload.append(
            {
                "combo_id": c.combo_id,
                "used_self": [
                    {"id": i, "text": self_by_id[i].text}
                    for i in c.used_self_ids
                    if i in self_by_id
                ],
                "used_dharma": [
                    {"id": i, "text": dharma_by_id[i].text}
                    for i in c.used_dharma_ids
                    if i in dharma_by_id
                ],
            }
        )
    action_block = (
        f"span={action_span}\n"
        f"description={action_description}"
        if (action_span or action_description)
        else "（无）"
    )
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是行动解释向二执组合评估器。\n"
        f"{focus_block}"
        "输入：客观刺激、行动证据、候选我执×法执组合。\n"
        "activation_alignment∈[0,1]：该视角被刺激+行动共同激活/解释的程度。\n"
        "问：「若内心是这一执取组合，出现该行动有多自然？」\n"
        "禁止改 combo_id，禁止新增执取。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"行动证据：\n{action_block}\n\n"
        f"候选组合：\n{payload}\n\n"
        "返回：\n"
        '{"items":[{"combo_id":"C_act1","activation_alignment":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_combo_meta_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    candidates: list[GraspingCombination],
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """L2.5：为已枚举组合只打 activation 对齐分（禁看行动）。"""
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    payload = []
    for c in candidates:
        payload.append(
            {
                "combo_id": c.combo_id,
                "used_self": [
                    {"id": i, "text": self_by_id[i].text}
                    for i in c.used_self_ids
                    if i in self_by_id
                ],
                "used_dharma": [
                    {"id": i, "text": dharma_by_id[i].text}
                    for i in c.used_dharma_ids
                    if i in dharma_by_id
                ],
            }
        )
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是二执组合评估器。\n"
        f"{focus_block}"
        "输入是客观刺激与若干候选「我执×法执」组合。\n"
        "任务：把每个组合当作「分析对象潜在心理状态视角」，只评估是否被当前刺激激活。\n\n"
        "规则：\n"
        "1. activation_alignment∈[0,1]：当前刺激在多大程度上触及/激活该视角，"
        "使其成为一种可能的主观取境。"
        "问的是「是否被刺激点燃」，不是「是否解释了分析对象行动」。\n"
        "2. 禁止根据分析对象行动/成败打分；本阶段看不到也不应推断行动。\n"
        "3. 不要改组合 id，不要新增执取，不要输出所缘境正文或解释文案。\n"
        "4. 若多个组合描述相近侧面，仍各自打分；去重由本地完成。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"候选组合：\n{payload}\n\n"
        "返回：\n"
        '{"items":[{"combo_id":"C1","activation_alignment":0.0}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_semantic_confirm_messages(
    pairs: list[tuple[str, str, str]],
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """批量确认视角/执取文本是否同义（非反义、非主客对调）。

    pairs: [(pair_id, text_a, text_b), ...]
    """
    focus_block = focus_person_instruction(focus_person)
    payload = [
        {"pair_id": pid, "text_a": ta, "text_b": tb} for pid, ta, tb in pairs
    ]
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是心理视角等价判定器。\n"
        f"{focus_block}"
        "任务：判断每对文本是否描述**同一个**内心关注视角（可同义改写）。\n\n"
        "判定 same_perspective=true 仅当：关注点、极性、主客体关系实质相同。\n"
        "必须判 false 的情况：\n"
        "- 否定/极性相反（多/少一个「不」「没」等导致意思反转）\n"
        "- 主客体对调（A对B vs B对A）\n"
        "- 关注轴明显不同（能力 vs 关系 vs 公平 等）\n"
        "- 仅字面像但心理含义不同\n"
        "不要输出情绪标签、行为预测或解释文案；只返回 pair_id 与布尔判定。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"待判定配对：\n{payload}\n\n"
        "返回：\n"
        '{"items":[{"pair_id":"p1","same_perspective":false}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_subjective_alambana_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    role_context: str = "",
    max_candidates: int = 3,
    combinations: list[GraspingCombination] | None = None,
    *,
    focus_person: str = "",
    h_lookup: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """L3：按已选二执组合渲染主观所缘（不得改组合；禁看行动）。"""
    bg_instruction = _BG_INSTRUCTION if role_context else ""
    if combinations:
        return _build_render_alambana_messages(
            text,
            objective_event,
            self_grasping,
            dharma_grasping,
            combinations,
            role_context=role_context,
            bg_instruction=bg_instruction,
            focus_person=focus_person,
            h_lookup=h_lookup,
        )
    focus_block = focus_person_instruction(focus_person)
    who = str(focus_person or "").strip() or "主体"
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是主观所缘境渲染器。\n"
        f"{focus_block}"
        "输入是客观刺激、我执列表、法执列表。\n"
        "任务：写出主观所缘境，并声明所用 used_self_ids / used_dharma_ids。\n\n"
        f"主观所缘境 = 「{who}」此刻经验到的对象化世界。\n"
        "不是情绪标签，不是未来行为计划，不是动机宣判。\n"
        "不要写入分析对象后续行动或行动结果。\n"
        "禁止改写成其他人物的内心世界。\n\n"
        "规则：\n"
        "1. 每条必须声明 used_self_ids / used_dharma_ids，且只能引用已给 id。\n"
        "2. 不同候选应引用不同执取子集。\n"
        "3. 只输出 id / combo_id / text；不要输出一致性分数或其它诊断字段。\n"
        f"{bg_instruction}"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"{_background_block(role_context)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"我执：\n{_grasping_json(self_grasping)}\n\n"
        f"法执：\n{_grasping_json(dharma_grasping)}\n\n"
        f"请输出最多 {max_candidates} 条：\n"
        "- id / combo_id / text\n\n"
        "返回：\n"
        '{"candidates":[{"id":"O1","combo_id":"C1","text":""}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _grasping_json(items: list[GraspingHypothesis]) -> list[dict]:
    return [{"id": h.id, "text": h.text, "confidence": h.confidence} for h in items]


def _build_render_alambana_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    combinations: list[GraspingCombination],
    *,
    role_context: str = "",
    bg_instruction: str = "",
    focus_person: str = "",
    h_lookup: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """L3：按已选定的二执组合渲染主观所缘。"""
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    texts = dict(h_lookup or {})
    combo_payload = []
    for c in combinations:
        h_ids = list(c.reactivated_h_ids or c.source_bias_frame_ids)
        combo_payload.append(
            {
                "combo_id": c.combo_id,
                "used_self": [
                    {"id": i, "text": self_by_id[i].text}
                    for i in c.used_self_ids
                    if i in self_by_id
                ],
                "used_dharma": [
                    {"id": i, "text": dharma_by_id[i].text}
                    for i in c.used_dharma_ids
                    if i in dharma_by_id
                ],
                "reactivated_h": [
                    {"id": hid, "text": texts.get(hid, "")}
                    for hid in h_ids
                    if hid
                ],
            }
        )
    focus_block = focus_person_instruction(focus_person)
    who = str(focus_person or "").strip() or "主体"
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是主观所缘境渲染器。\n"
        f"{focus_block}"
        "输入是客观刺激与已选定的二执组合（心理视角）。\n"
        "任务：为每个组合各写一条主观所缘境。\n\n"
        f"主观所缘境 = 该组合视角下，「{who}」此刻经验到的对象化世界。\n"
        "不是情绪标签，不是未来行为计划，不是动机宣判。\n"
        "不要写入分析对象后续行动或行动结果。\n"
        "禁止改写成其他人物的内心世界。\n\n"
        "规则：\n"
        "1. 每个 combo_id 恰好一条；id 建议 O_<combo_id>。\n"
        "2. 所缘的核心解释必须由当前组合中的我执与法执产生。\n"
        "3. reactivated_h 不得产生新的执取维度，只允许作为历史证据或熟悉感，"
        "对当前组合已有解释进行强化；不要再判断这条历史是否相关。\n"
        "4. 不得添加当前客观刺激中不存在的他人言行或事实；主观解释可以，伪造事实不行。\n"
        "5. 解释必须针对当前事件的关键结构变化（角色/约束/关系/结果），不要完全忽略它们。\n"
        "6. 只输出 id / combo_id / text；不要输出一致性分数或其它诊断字段。\n"
        f"{bg_instruction}"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"{_background_block(role_context)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"已选组合：\n{combo_payload}\n\n"
        "返回：\n"
        '{"candidates":[{"id":"O_C1","combo_id":"C1","text":""}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_objective_fact_messages(
    text: str,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    return build_objective_event_messages(
        text, role_context=role_context, focus_person=focus_person
    )


def build_self_grasping_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    return build_grasping_pair_messages(
        text,
        objective_event,
        role_context=role_context,
        focus_person=focus_person,
    )


def build_dharma_grasping_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    role_context: str = "",
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    return build_grasping_pair_messages(
        text,
        objective_event,
        role_context=role_context,
        focus_person=focus_person,
    )


ALAMBANA_GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "alambana_id": {"type": "string"},
                    "grounded": {"type": "boolean"},
                    "hallucinated_facts": {"type": "boolean"},
                    "preserves_critical_event": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "alambana_id",
                    "grounded",
                    "hallucinated_facts",
                    "preserves_critical_event",
                ],
            },
        }
    },
    "required": ["items"],
}


def build_alambana_grounding_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    candidates: list,
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """语义 grounding：是否伪造事实、是否忽略当前关键变化。"""
    payload = [{"id": c.id, "text": c.text} for c in candidates]
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是当前事件 grounding 校验器。\n"
        f"{focus_block}"
        "不要判断心理解释是否正确，只判断两件事：\n"
        "1. hallucinated_facts：所缘是否添加了当前客观刺激中不存在的他人言行或事实。"
        "主观解释（理解为、当成）不是伪造。\n"
        "2. preserves_critical_event：所缘是否完全忽略了真正改变情境的核心信息"
        "（角色/约束/关系/结果变化）。不必复述全部细节。\n"
        "grounded = 未伪造事实 且 保留了关键变化。\n"
        "不要输出心理评价或行为预测。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"待校验所缘：\n{payload}\n\n"
        "返回：\n"
        '{"items":[{"alambana_id":"O_C1","grounded":true,'
        '"hallucinated_facts":false,"preserves_critical_event":true,"reason":""}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_alambana_grounding_retry_messages(
    text: str,
    objective_event: ObjectiveEventStructure,
    combo,
    self_grasping: list[GraspingHypothesis],
    dharma_grasping: list[GraspingHypothesis],
    previous_text: str,
    fail_reason: str,
    *,
    focus_person: str = "",
) -> list[dict[str, str]]:
    """保留该认知解释，针对当前事件重写所缘。"""
    self_by_id = {h.id: h for h in self_grasping}
    dharma_by_id = {h.id: h for h in dharma_grasping}
    used_self = [
        {"id": i, "text": self_by_id[i].text}
        for i in combo.used_self_ids
        if i in self_by_id
    ]
    used_dharma = [
        {"id": i, "text": dharma_by_id[i].text}
        for i in combo.used_dharma_ids
        if i in dharma_by_id
    ]
    focus_block = focus_person_instruction(focus_person)
    system = (
        f"[prompt_version={PERCEPTION_PROMPT_VERSION}] 你是主观所缘重写器。\n"
        f"{focus_block}"
        "保留给定的认知解释（我执/法执组合），重写所缘正文。\n"
        "不得创造客观刺激中不存在的事实；必须让该解释真正针对当前事件的关键变化。\n"
        "禁止情绪标签、行为预测。只返回一条候选。\n"
    )
    user = (
        f"{focus_person_user_line(focus_person)}"
        f"刺激文本：\n{text}\n\n"
        f"客观刺激：\n{objective_event.to_dict()}\n\n"
        f"组合 {combo.combo_id}：我执={used_self} 法执={used_dharma}\n"
        f"上次所缘：{previous_text}\n"
        f"失败原因：{fail_reason}\n\n"
        "返回：\n"
        f'{{"candidates":[{{"id":"O_{combo.combo_id}","combo_id":"{combo.combo_id}","text":""}}]}}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
