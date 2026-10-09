"""向后兼容：旧名 OpenAICompatibleClient 指向通用 HTTP 客户端。"""

from mental_world_model.llm.http_chat_client import HttpChatCompletionsClient

OpenAICompatibleClient = HttpChatCompletionsClient

__all__ = ["OpenAICompatibleClient", "HttpChatCompletionsClient"]
