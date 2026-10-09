"""LLM 客户端工厂。"""

from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.config import LlmConfig, LlmConfigurationError
from mental_world_model.llm.http_chat_client import HttpChatCompletionsClient
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.llm.tracing_client import ensure_tracing_client, is_tracing_enabled


def create_llm_client(
    *,
    use_mock: bool = False,
    config: LlmConfig | None = None,
    client: LlmClient | None = None,
    trace: bool | None = None,
) -> LlmClient:
    """创建或注入 LLM 客户端。

    优先级：显式 ``client`` > ``use_mock`` > ``config`` > 环境变量。
    任意实现 ``LlmClient.complete()`` 的适配器均可通过 ``client`` 注入。
    默认启用调用统计（``MWM_LLM_TRACE=1``，见 ``TracingLlmClient``）。
    """
    if client is not None:
        inner = client
    elif use_mock:
        inner = MockLlmClient()
    elif config is not None:
        inner = HttpChatCompletionsClient(config)
    else:
        return create_llm_client_from_env()

    if trace is False or not is_tracing_enabled():
        return inner
    return ensure_tracing_client(inner)


def create_llm_client_from_env() -> LlmClient:
    """从环境变量创建 HTTP Chat Completions 适配客户端。"""
    try:
        inner = HttpChatCompletionsClient(LlmConfig.from_env())
    except LlmConfigurationError as error:
        raise SystemExit(error.user_message()) from error
    if is_tracing_enabled():
        return ensure_tracing_client(inner)
    return inner
