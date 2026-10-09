"""LLM 结构化 JSON 辅助。"""

import json
import re
from typing import Any


def build_schema_instruction(schema: dict[str, Any] | None) -> str:
    """生成要求模型返回 JSON 的指令。"""
    if not schema:
        return "请只返回一个 JSON 对象。"
    return (
        "请只返回一个 JSON 对象，不要包含 markdown 代码块或其它文字。"
        f"\nJSON Schema：\n{json.dumps(schema, ensure_ascii=False)}"
    )


def parse_json_response(content: str) -> dict[str, Any]:
    """从模型输出中解析 JSON 对象。

    空内容或非法 JSON 抛出 ``ValueError``（供上层重试）。
    """
    text = (content or "").strip()
    if not text:
        raise ValueError("LLM 返回空内容，无法解析 JSON")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    if not text:
        raise ValueError("LLM 返回空内容，无法解析 JSON")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM 返回非合法 JSON：{error}") from error
    if not isinstance(data, dict):
        raise ValueError("LLM 返回的不是 JSON 对象")
    return data


class LlmJsonParseError(ValueError):
    """LLM 响应 JSON 解析失败（可重试）。"""

    pass


def parse_json_response_or_raise(content: str) -> dict[str, Any]:
    """同 ``parse_json_response``，统一包装为 ``LlmJsonParseError``。"""
    try:
        return parse_json_response(content)
    except ValueError as error:
        raise LlmJsonParseError(str(error)) from error
