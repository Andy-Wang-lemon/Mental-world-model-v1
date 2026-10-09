"""按人物/用途给 LLM 调用 stage 加前缀，便于统计与排障。"""

from __future__ import annotations

from typing import Any

from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import current_llm_stage, llm_call_context


class StagePrefixLlmClient:
    """包装已有客户端：complete 时 stage 变为 ``prefix.原stage``。"""

    def __init__(self, inner: LlmClient, prefix: str) -> None:
        self._inner = inner
        self.prefix = str(prefix or "").strip() or "llm"

    @property
    def inner(self) -> LlmClient:
        return self._inner

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = current_llm_stage() or "call"
        if base.startswith(f"{self.prefix}."):
            stage = base
        else:
            stage = f"{self.prefix}.{base}"
        with llm_call_context(stage=stage):
            return self._inner.complete(messages, schema=schema)
