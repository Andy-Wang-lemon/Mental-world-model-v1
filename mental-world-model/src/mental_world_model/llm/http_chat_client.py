"""通用 HTTP Chat Completions 适配客户端。

适配采用 ``POST {base_url}{chat_path}``、请求体含 ``model`` / ``messages``、
响应体含 ``choices[0].message.content`` 的各类大模型 HTTP API（不绑定特定厂商）。
"""

import http.client
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from mental_world_model.llm.config import LlmConfig, LlmConfigurationError
from mental_world_model.llm.json_helpers import (
    LlmJsonParseError,
    build_schema_instruction,
    parse_json_response_or_raise,
)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_HINT = (
    "Detected DeepSeek endpoint. You may need MWM_LLM_BASE_URL=https://api.deepseek.com/v1"
)
_RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _llm_debug_enabled() -> bool:
    """是否输出 LLM HTTP 请求诊断（MWM_LLM_DEBUG=1）。"""
    return os.environ.get("MWM_LLM_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _format_auth_header_preview(headers: dict[str, str], config: LlmConfig) -> str:
    """返回鉴权头预览，不泄露完整 API Key。"""
    auth_name = config.auth_header
    if auth_name not in headers or not headers[auth_name]:
        return f"{auth_name}: (not set)"
    prefix = config.auth_prefix or ""
    if prefix:
        return f"{auth_name}: {prefix}***"
    return f"{auth_name}: ***"


def _format_proxy_env() -> str:
    """汇总 urllib 可能使用的代理环境变量。"""
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    lines: list[str] = []
    for key in proxy_keys:
        value = os.environ.get(key)
        lines.append(f"  {key}={value if value else '(not set)'}")
    return "\n".join(lines)


def _truncate_body_preview(body: bytes, *, limit: int = 300) -> str:
    """请求体预览，超长时截断。"""
    text = body.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated, total {len(text)} chars)"


def _log_request_diagnostics(
    *,
    config: LlmConfig,
    url: str,
    method: str,
    headers: dict[str, str],
    body: bytes,
) -> None:
    """在发送 HTTP 请求前输出诊断信息（stderr）。"""
    chat_path = config.chat_path
    if not chat_path.startswith("/"):
        chat_path = f"/{chat_path}"

    lines = [
        "[MWM LLM DEBUG] 即将发送 HTTP 请求",
        f"  1. 最终请求 URL: {url}",
        f"  2. HTTP Method: {method}",
        f"  3. base_url: {config.base_url}",
        f"  4. chat_path: {config.chat_path}",
        f"  5. model: {config.model}",
        f"  6. Authorization Header: {_format_auth_header_preview(headers, config)}",
        "  7. urllib proxy 环境变量:",
        _format_proxy_env(),
        f"  8. 请求 Body 预览: {_truncate_body_preview(body)}",
        (
            "  URL 拼接: "
            f"{config.base_url.rstrip('/')}{chat_path} "
            f"(POST {{base_url}}{{chat_path}})"
        ),
    ]

    base_normalized = config.base_url.rstrip("/")
    path_normalized = config.chat_path if config.chat_path.startswith("/") else f"/{config.chat_path}"
    if base_normalized == _DEEPSEEK_BASE_URL and path_normalized == "/chat/completions":
        lines.append(f"  [MWM LLM HINT] {_DEEPSEEK_HINT}")
        lines.append(
            "  [MWM LLM HINT] 当前将请求 "
            "https://api.deepseek.com/chat/completions；"
            "DeepSeek OpenAI 兼容路径通常为 "
            "https://api.deepseek.com/v1/chat/completions"
        )

    print("\n".join(lines), file=sys.stderr)


def _log_retry(attempt: int, max_attempts: int, delay: float, reason: str) -> None:
    if not _llm_debug_enabled():
        return
    print(
        f"[MWM LLM DEBUG] 重试 {attempt}/{max_attempts - 1}，"
        f"{delay:.1f}s 后重试：{reason}",
        file=sys.stderr,
    )


def _pool_max_size() -> int:
    """连接池大小：优先 MWM_LLM_POOL_SIZE，否则与 MWM_LLM_MAX_CONCURRENCY 对齐。"""
    raw = os.environ.get("MWM_LLM_POOL_SIZE", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    from mental_world_model.engine.scoring.parallel_exec import max_concurrency

    return max_concurrency()


class _ConnectionPool:
    """线程安全的多连接 HTTP(S) 池（stdlib only）。

    借出连接后在锁外等待响应，真正支持并行 LLM 调用；
    旧实现单连接+全程锁会把所有并发请求串行化。
    """

    def __init__(
        self,
        parsed: urllib.parse.ParseResult,
        timeout: float,
        *,
        max_size: int | None = None,
    ) -> None:
        self._parsed = parsed
        self._timeout = timeout
        self._max_size = max(1, max_size if max_size is not None else _pool_max_size())
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._idle: list[http.client.HTTPConnection] = []
        self._created = 0
        self._closed = False

    def _new_conn(self) -> http.client.HTTPConnection:
        host = self._parsed.hostname or "localhost"
        port = self._parsed.port
        if self._parsed.scheme == "https":
            ctx = ssl.create_default_context()
            return http.client.HTTPSConnection(host, port, timeout=self._timeout, context=ctx)
        return http.client.HTTPConnection(host, port, timeout=self._timeout)

    def _acquire(self) -> http.client.HTTPConnection:
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("连接池已关闭")
                if self._idle:
                    return self._idle.pop()
                if self._created < self._max_size:
                    self._created += 1
                    break
                self._cond.wait(timeout=1.0)
        # 在锁外建连，避免 TLS 握手阻塞其他借还
        try:
            return self._new_conn()
        except Exception:
            with self._cond:
                self._created -= 1
                self._cond.notify()
            raise

    def _release(self, conn: http.client.HTTPConnection | None) -> None:
        if conn is None:
            return
        with self._cond:
            if self._closed:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._created = max(0, self._created - 1)
                self._cond.notify_all()
                return
            self._idle.append(conn)
            self._cond.notify()

    def _discard(self, conn: http.client.HTTPConnection | None) -> None:
        """损坏连接：关闭且不归还，腾出名额。"""
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        with self._cond:
            self._created = max(0, self._created - 1)
            self._cond.notify()

    def request(
        self, method: str, path: str, body: bytes, headers: dict[str, str],
    ) -> tuple[int, str]:
        """发送请求，返回 (status_code, response_body_text)。I/O 不占池锁。"""
        conn = self._acquire()
        try:
            for retry_new in (False, True):
                try:
                    conn.request(method, path, body=body, headers=headers)
                    resp = conn.getresponse()
                    data = resp.read().decode("utf-8", errors="replace")
                    self._release(conn)
                    conn = None  # type: ignore[assignment]
                    return resp.status, data
                except (
                    http.client.RemoteDisconnected,
                    http.client.IncompleteRead,
                    ConnectionError,
                    OSError,
                ):
                    self._discard(conn)
                    conn = None  # type: ignore[assignment]
                    if retry_new:
                        raise
                    conn = self._acquire()
                    continue
            raise RuntimeError("unreachable")
        except Exception:
            if conn is not None:
                self._discard(conn)
            raise

    def close(self) -> None:
        with self._cond:
            self._closed = True
            while self._idle:
                c = self._idle.pop()
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            self._created = 0
            self._cond.notify_all()


class HttpChatCompletionsClient:
    """通过 HTTP 调用 Chat Completions 风格端点的通用 LLM 客户端。

    内置多连接池，并行 ``complete()`` 可真正并发等待响应。
    """

    def __init__(self, config: LlmConfig) -> None:
        if not config.base_url:
            raise LlmConfigurationError("base_url 不能为空")
        if not config.model:
            raise LlmConfigurationError("model 不能为空")
        self._config = config
        parsed = urllib.parse.urlparse(config.base_url)
        self._pool = _ConnectionPool(parsed, timeout=config.timeout_seconds)
        self._request_path = self._build_request_path(parsed)

    def __del__(self) -> None:
        try:
            self._pool.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def config(self) -> LlmConfig:
        return self._config

    def _build_request_path(self, parsed: urllib.parse.ParseResult) -> str:
        base_path = parsed.path.rstrip("/")
        chat_path = self._config.chat_path
        if not chat_path.startswith("/"):
            chat_path = f"/{chat_path}"
        return f"{base_path}{chat_path}"

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            value = self._config.api_key
            if self._config.auth_prefix:
                value = f"{self._config.auth_prefix}{value}"
            headers[self._config.auth_header] = value
        headers.update(self._config.extra_headers)
        return headers

    def _endpoint_url(self) -> str:
        path = self._config.chat_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self._config.base_url.rstrip('/')}{path}"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发送消息并返回结构化 JSON 结果。

        使用持久连接池复用 TCP/TLS，对可重试状态码自动退避重试。
        """
        payload_messages = [dict(message) for message in messages]
        if schema is not None:
            payload_messages.append(
                {
                    "role": "user",
                    "content": build_schema_instruction(schema),
                }
            )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": payload_messages,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
        }
        if self._config.seed is not None:
            payload["seed"] = self._config.seed
        if self._config.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._build_headers()
        if _llm_debug_enabled():
            url = self._endpoint_url()
            _log_request_diagnostics(
                config=self._config,
                url=url,
                method="POST",
                headers=headers,
                body=body,
            )

        max_attempts = 1 + max(0, self._config.max_retries)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                status, resp_text = self._pool.request(
                    "POST", self._request_path, body, headers,
                )
            except (ConnectionError, OSError, http.client.IncompleteRead) as error:
                if attempt < max_attempts:
                    delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                    _log_retry(attempt, max_attempts, delay, f"{type(error).__name__}: {error}")
                    time.sleep(delay)
                    last_error = RuntimeError(f"LLM API 网络错误: {error}")
                    continue
                raise RuntimeError(f"LLM API 网络错误: {error}") from error

            if status in _RETRYABLE_HTTP_CODES and attempt < max_attempts:
                delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                _log_retry(attempt, max_attempts, delay, f"HTTP {status}: {resp_text[:200]}")
                time.sleep(delay)
                last_error = RuntimeError(f"LLM API 请求失败 ({status}): {resp_text}")
                continue

            if status >= 400:
                raise RuntimeError(f"LLM API 请求失败 ({status}): {resp_text}")

            try:
                response_data = json.loads(resp_text)
            except json.JSONDecodeError as error:
                if attempt < max_attempts:
                    delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                    _log_retry(
                        attempt,
                        max_attempts,
                        delay,
                        f"HTTP 响应体非 JSON: {error}",
                    )
                    time.sleep(delay)
                    last_error = RuntimeError(f"LLM API 响应体非 JSON: {error}")
                    continue
                raise RuntimeError(f"LLM API 响应体非 JSON: {error}") from error

            try:
                content = self._extract_message_content(response_data)
                return parse_json_response_or_raise(content)
            except (LlmJsonParseError, RuntimeError) as error:
                # 空 content / 非 JSON / 缺 choices 结构 → 可重试
                if attempt < max_attempts:
                    delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                    _log_retry(attempt, max_attempts, delay, str(error)[:200])
                    time.sleep(delay)
                    last_error = RuntimeError(f"LLM 响应解析失败: {error}")
                    continue
                raise RuntimeError(f"LLM 响应解析失败: {error}") from error

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM API 请求失败：未知错误")

    @staticmethod
    def _extract_message_content(response_data: Any) -> str:
        """从 Chat Completions 风格响应中取出 assistant 文本。"""
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"LLM API 响应格式异常（期望 choices[0].message.content）: {response_data}"
            ) from error
