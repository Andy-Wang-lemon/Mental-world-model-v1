"""LLM 客户端协议。"""

from typing import Any, Protocol


class LlmClient(Protocol):
    """任意大语言模型适配器需实现的接口。

    心智世界模型通过此协议与具体 LLM 解耦：可注入自建客户端，
    或使用 ``HttpChatCompletionsClient`` 对接常见 HTTP Chat API。
    """

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发送对话消息，返回解析后的 JSON 对象（dict）。"""
        ...
