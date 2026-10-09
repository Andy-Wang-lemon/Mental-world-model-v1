"""所缘境合并评分分组：相同所缘定义的心所一次 LLM 批量打分。"""

from __future__ import annotations

from dataclasses import dataclass

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.scoring.results import DEPENDENT_ON_FEN
from mental_world_model.utils.confidence import as_unit_float

# 忿→恨→恼 链式依赖心所不在批量所缘阶段预评（恨/恼仍随链单独评）
SUOYUAN_BATCH_SKIP_NAMES = frozenset(DEPENDENT_ON_FEN)

_GROUP_SUOYUAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                },
                "required": ["score"],
            },
        },
    },
    "required": ["scores"],
}


@dataclass(frozen=True)
class SuoyuanScoringGroup:
    """一组共享所缘境定义、需一次 LLM 批量评分的心所。"""

    suoyuan_text: str
    xinsuo_names: tuple[str, ...]


# 13 个合并组：多心所共享所缘境定义，一次 LLM 返回多个分数。
MERGED_SUOYUAN_GROUPS: tuple[SuoyuanScoringGroup, ...] = (
    SuoyuanScoringGroup(
        "信的对象不是任何事物，而是正确、真实、善好的对象，也就是符合真理、值得学习和实践的对象，如正道、真理、善法等。",
        ("信", "不信"),
    ),
    SuoyuanScoringGroup(
        "所缘是具体应修的善法和应断的恶法，包括该培养的习惯、该完成的责任、该坚持的修行、该停止的烦恼、冲动和不善行为。重点是\"修断事务\"，不是一般忙碌。",
        ("精进", "懈怠"),
    ),
    SuoyuanScoringGroup(
        "所缘是客观正确、积极、健康的价值观、道德原则和行为规范。重点是：\"这种行为不符合正确的做人标准，我不应该这样做。\"",
        ("惭", "无惭"),
    ),
    SuoyuanScoringGroup(
        "所缘是他人的感受、公共规范、社会伦理、行为对他人和社会造成的影响，以及客观、公正的社会评价。重点是：\"这样做会伤害他人，也会受到合理、正当的批评。\"",
        ("愧", "无愧"),
    ),
    SuoyuanScoringGroup(
        "所缘是生命存在、身体、财物、地位、名声、关系、享受、成就、身份认同等可被占有或依附的对象。重点是面对\"有\"和\"有具\"时。",
        ("贪", "无贪"),
    ),
    SuoyuanScoringGroup(
        "所缘是痛苦、逆境、挫折、被冒犯、被伤害、令人不满的人事物，以及造成痛苦的条件。重点是在苦境中。",
        ("瞋", "无瞋"),
    ),
    SuoyuanScoringGroup(
        "所缘是一切事实、规律和正确的道理，包括事物发展的规律、伦理原则、心理动机、行为后果、现实条件以及解决问题的方法等。重点是客观、如实地理解事物，而不是受偏见、错觉、情绪或错误观念影响。",
        ("无痴", "无明", "善见", "恶见", "疑", "不正知"),
    ),
    SuoyuanScoringGroup(
        "所缘是身心的沉重、僵硬、紧绷、迟钝、疲惫、不适等状态，尤其是在专注、学习、思考或工作时，影响效率和发挥的各种身心障碍。",
        ("轻安", "昏沉"),
    ),
    SuoyuanScoringGroup(
        "所缘是所有应该坚持的正确行为和应该避免的错误行为，包括日常行为、想法、情绪、习惯、诱惑、过失、长期目标和正确的价值观。重点是主动做好该做的事，避免做不该做的事。",
        ("不放逸", "放逸"),
    ),
    SuoyuanScoringGroup(
        "所缘是自己与他人的各种关系，包括能力、身份、财富、地位、学识、道德、修行、外貌、成就、苦难经历等一切可比较的对象。面对这些差异时。",
        ("慢", "不慢"),
    ),
    SuoyuanScoringGroup(
        "所缘是自己已经犯下的错误、过失、不道德行为、违规行为、失败或缺点，以及这些事实可能被他人发现的情况。重点是自己的过失。",
        ("覆", "不覆"),
    ),
    SuoyuanScoringGroup(
        "所缘是他人的财富、地位、能力、成功、荣誉、人际关系、幸福、机会、赞美、资源等优越条件。重点是看到别人比自己好。",
        ("嫉", "不嫉"),
    ),
    SuoyuanScoringGroup(
        "所缘是他人，尤其是自己反感、敌视、认为可以欺负、报复或伤害的人。重点不是一时生气，而是将伤害对方视为解决问题、发泄情绪或达到目的的方式。",
        ("害", "不害"),
    ),
)

# 8 个单独心所：各用各的所缘境定义，一次 LLM 只评一个心所（不与其他心所合并）。
SINGLETON_SUOYUAN_GROUPS: tuple[SuoyuanScoringGroup, ...] = (
    SuoyuanScoringGroup(
        "所缘是当前、眼前直接发生的不饶益境，包括被辱骂、被否定、被阻碍、被冒犯、利益受损、遭受不公平待遇、受到攻击等立即刺激。重点是对象必须是现前正在发生。",
        ("忿",),
    ),
    SuoyuanScoringGroup(
        "所缘是他人的信任、尊敬、供养、利益、资源、职位、机会等可以通过他人相信自己而获得的利益。重点是获得利益。",
        ("诳",),
    ),
    SuoyuanScoringGroup(
        "所缘是可以被讨好、迎合、操控的人，包括上级、权威、客户、朋友、伴侣等。重点不是展示能力，而是迎合他人的喜好。",
        ("谄",),
    ),
    SuoyuanScoringGroup(
        "所缘是自己的财富、地位、名誉、外貌、健康、年轻、家庭、能力、知识、修行、成功经历等一切值得炫耀的优势。重点不是比较别人，而是沉醉于自己的优势。",
        ("憍",),
    ),
    SuoyuanScoringGroup(
        "所缘是自己已经拥有的财富、知识、经验、人脉、机会、资源、权力、时间、法义等可分享的东西。重点是已经属于自己的资源。",
        ("悭",),
    ),
    SuoyuanScoringGroup(
        "所缘是心的活动状态，尤其是偏斜、躁动、抓取、抗拒、用力过度、不平衡、不自然的心理运行。",
        ("行舍",),
    ),
    SuoyuanScoringGroup(
        "所缘是当前对象以及各种容易引起兴奋、后悔、担忧、幻想、欲望、期待或情绪波动的对象，包括过去经历、未来计划、未完成事件和各种联想。重点是心无法停留。",
        ("掉举",),
    ),
    SuoyuanScoringGroup(
        "所缘是各种不断变化的外境、念头、欲望、情绪、记忆、幻想、声音、图像以及任何能够吸引注意力的刺激。重点是没有稳定的注意中心。",
        ("散乱",),
    ),
)


def _merged_name_to_group() -> dict[str, SuoyuanScoringGroup]:
    mapping: dict[str, SuoyuanScoringGroup] = {}
    for group in MERGED_SUOYUAN_GROUPS:
        for name in group.xinsuo_names:
            if name in mapping:
                raise ValueError(f"心所 {name} 重复出现在多个所缘合并组")
            mapping[name] = group
    for group in SINGLETON_SUOYUAN_GROUPS:
        for name in group.xinsuo_names:
            if name in mapping:
                raise ValueError(f"心所 {name} 重复出现在合并组与单独组")
            mapping[name] = group
    return mapping


_MERGED_NAME_TO_GROUP = _merged_name_to_group()


def group_suoyuan_output_schema() -> dict:
    return dict(_GROUP_SUOYUAN_SCHEMA)


def is_group_suoyuan_schema(schema: dict | None) -> bool:
    properties = (schema or {}).get("properties", {})
    return isinstance(properties, dict) and "scores" in properties


def parse_group_suoyuan_response(
    response: dict,
    expected_names: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """解析批量所缘境评分 LLM 响应。"""
    raw_scores = response.get("scores")
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    parsed: dict[str, float] = {}
    for name in expected_names:
        item = raw_scores.get(name, {})
        if not isinstance(item, dict):
            item = {}
        parsed[name] = as_unit_float(item.get("score"), 0.0)
    return parsed


def plan_suoyuan_scoring_groups(
    xinsuo_by_name: dict[str, Xinsuo],
    *,
    names: list[str] | None = None,
) -> list[SuoyuanScoringGroup]:
    """为待评心所规划所缘境 LLM 批次（13 合并 + 8 单独 + 未映射者 fallback）。"""
    target_names = names or [
        name for name in xinsuo_by_name if name not in SUOYUAN_BATCH_SKIP_NAMES
    ]

    batches: list[SuoyuanScoringGroup] = []
    seen_merged: set[tuple[str, tuple[str, ...]]] = set()
    unmapped: list[str] = []

    for name in target_names:
        if name in _MERGED_NAME_TO_GROUP:
            group = _MERGED_NAME_TO_GROUP[name]
            if len(group.xinsuo_names) == 1:
                batches.append(SuoyuanScoringGroup(group.suoyuan_text, (name,)))
                continue
            key = (group.suoyuan_text, group.xinsuo_names)
            if key in seen_merged:
                continue
            present = tuple(
                candidate for candidate in group.xinsuo_names if candidate in target_names
            )
            if present:
                batches.append(SuoyuanScoringGroup(group.suoyuan_text, present))
                seen_merged.add(key)
        else:
            unmapped.append(name)

    for name in unmapped:
        xinsuo = xinsuo_by_name[name]
        batches.append(SuoyuanScoringGroup(xinsuo.suoyuan, (name,)))

    return batches
