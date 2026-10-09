"""评分阶段 prompt 上下文组装。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.scoring.context import ScoringContext

STAGE_YE_MARKER = "【第三轮：业】"
STAGE_XIANDAI_MARKER = "【第四轮：现代心理学】"
_SCORING_SYSTEM_PREFIX = (
    "你是心所评分助手。以下为本次评估的固定上下文，各轮评分请基于此。\n\n"
)


def is_ye_stage_content(content: str) -> bool:
    """判断 LLM 消息是否为业评分阶段。"""
    return STAGE_YE_MARKER in content


def is_xiandai_stage_content(content: str) -> bool:
    """判断 LLM 消息是否为现代心理学评分阶段。"""
    return STAGE_XIANDAI_MARKER in content


def _scoring_messages(context: ScoringContext, user_content: str) -> list[dict[str, str]]:
    """system 固定上下文 + user 阶段指令，便于 API 前缀缓存并减少重复拼接。"""
    return [
        {
            "role": "system",
            "content": f"{_SCORING_SYSTEM_PREFIX}{context.base_context_block}",
        },
        {"role": "user", "content": user_content},
    ]


def build_suoyuan_messages(xinsuo: Xinsuo, context: ScoringContext) -> list[dict[str, str]]:
    """组装所缘境阶段 LLM 消息（单心所，用于恨/恼链等未批量预取场景）。"""
    user_content = (
        f"心所名称：{xinsuo.name}\n\n"
        f"该心所的所缘境定义：\n{xinsuo.suoyuan}\n\n"
        "请评估原文与所缘境定义的匹配程度，只返回相似度分数 JSON："
        '{"score": 0.0-1.0}'
    )
    return _scoring_messages(context, user_content)


def build_group_suoyuan_messages(
    suoyuan_text: str,
    xinsuo_names: tuple[str, ...] | list[str],
    context: ScoringContext,
) -> list[dict[str, str]]:
    """组装所缘境批量评分 LLM 消息（合并组或单独心所）。"""
    names = list(xinsuo_names)
    names_block = "、".join(names)
    if len(names) == 1:
        intro = (
            f"请评估原文与下列心所「{names[0]}」的所缘境定义的匹配程度。\n\n"
            f"【所缘境定义】\n{suoyuan_text}\n\n"
        )
    else:
        intro = (
            "以下心所共享同一段所缘境定义。请分别评估原文对每个心所的匹配程度，"
            "每个心所独立打分（可一高一低）。\n\n"
            f"【共享所缘境定义】\n{suoyuan_text}\n\n"
            f"【心所列表】\n{names_block}\n\n"
        )
    user_content = (
        f"{intro}"
        "只返回各心所相似度分数 JSON（不要文字理由）："
        '{"scores": {"心所名": {"score": 0.0-1.0}, ...}}'
    )
    return _scoring_messages(context, user_content)


def is_group_suoyuan_stage_content(content: str) -> bool:
    """判断 LLM 消息是否为所缘境批量评分阶段。"""
    return "共享所缘境定义" in content or (
        "请评估原文与下列心所" in content and "所缘境定义" in content
    )


def build_shengqi_nian_messages(xinsuo: Xinsuo, context: ScoringContext) -> list[dict[str, str]]:
    """组装生起的念头阶段 LLM 消息。"""
    user_content = (
        f"心所名称：{xinsuo.name}\n\n"
        f"该心所的生起的念头定义：\n{xinsuo.shengqi_nian}\n\n"
        "请评估原文与生起的念头定义的匹配程度，只返回相似度分数 JSON："
        '{"score": 0.0-1.0}'
    )
    return _scoring_messages(context, user_content)


def build_ye_messages(xinsuo: Xinsuo, context: ScoringContext) -> list[dict[str, str]]:
    """组装业评分阶段 LLM 消息（第三轮）。"""
    user_content = (
        f"{STAGE_YE_MARKER}\n"
        f"心所名称：{xinsuo.name}\n\n"
        f"该心所的业定义：\n{xinsuo.ye}\n\n"
        "请评估原文中体现的行为倾向是否符合该心所的业定义（行为功能与影响）。\n"
        "只返回相似度分数 JSON："
        '{"score": 0.0-1.0}'
    )
    return _scoring_messages(context, user_content)


def build_xiandai_xinli_messages(
    xinsuo: Xinsuo,
    context: ScoringContext,
) -> list[dict[str, str]]:
    """组装现代心理学评分阶段 LLM 消息（第四轮）。"""
    user_content = (
        f"{STAGE_XIANDAI_MARKER}\n"
        f"心所名称：{xinsuo.name}\n\n"
        f"该心所的现代心理学解释：\n{xinsuo.xiandai_xinli}\n\n"
        "请从现代心理学角度评估，原文是否呈现该心所的心理特征。\n"
        "只返回相似度分数 JSON："
        '{"score": 0.0-1.0}'
    )
    return _scoring_messages(context, user_content)


def build_conflict_messages(
    name_a: str,
    name_b: str,
    xinsuo_a: Xinsuo,
    xinsuo_b: Xinsuo,
    context: ScoringContext,
    description: str,
) -> list[dict[str, str]]:
    """组装冲突消解阶段 LLM 消息。"""
    user_content = (
        f"以下两个心所不能同时成立（{description}）。\n\n"
        f"【{name_a}】\n"
        f"所缘境：{xinsuo_a.suoyuan}\n"
        f"生起的念头：{xinsuo_a.shengqi_nian}\n"
        f"业：{xinsuo_a.ye}\n"
        f"现代心理学解释：{xinsuo_a.xiandai_xinli}\n\n"
        f"【{name_b}】\n"
        f"所缘境：{xinsuo_b.suoyuan}\n"
        f"生起的念头：{xinsuo_b.shengqi_nian}\n"
        f"业：{xinsuo_b.ye}\n"
        f"现代心理学解释：{xinsuo_b.xiandai_xinli}\n\n"
        "请对照原文判断哪一个更符合，或两个都不符合。只返回 winner JSON（不要文字理由）："
        f'{{"winner": "{name_a}" | "{name_b}" | "none"}}'
    )
    return _scoring_messages(context, user_content)
