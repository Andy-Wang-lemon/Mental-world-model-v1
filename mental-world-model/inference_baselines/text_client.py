"""要原始回答文本。temperature 固定为 0，失败按指数退避重试。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from mental_world_model.llm.config import LlmConfig

_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_MODEL = "deepseek-v4-flash"


class TextChatClient:
    def __init__(self, config: LlmConfig) -> None:
        if config.model != _MODEL:
            raise SystemExit(
                f"这三套基线只用 {_MODEL}，当前 MWM_LLM_MODEL={config.model or '(空)'}"
            )
        self._config = config

    def complete_text(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
        }
        if self._config.seed is not None:
            payload["seed"] = self._config.seed
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._config.api_key:
            headers[self._config.auth_header] = f"{self._config.auth_prefix}{self._config.api_key}"
        headers.update(self._config.extra_headers)
        url = _endpoint(self._config)
        attempts = 1 + max(0, self._config.max_retries)
        last_error = "未知错误"
        for attempt in range(1, attempts + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return _message_text(raw)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {error.code}: {detail[:300]}"
                if error.code not in _RETRYABLE_HTTP or attempt >= attempts:
                    raise RuntimeError(last_error) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt >= attempts:
                    raise RuntimeError(last_error) from error
            delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
            print(f"API 错误，{delay:.1f}s 后重试 ({attempt}/{attempts - 1})：{last_error}", flush=True)
            time.sleep(delay)
        raise RuntimeError(last_error)


def _endpoint(config: LlmConfig) -> str:
    path = config.chat_path if config.chat_path.startswith("/") else f"/{config.chat_path}"
    return f"{config.base_url.rstrip('/')}{path}"


def _message_text(raw: str) -> str:
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型返回空文本")
    return content
