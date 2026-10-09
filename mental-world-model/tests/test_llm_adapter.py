"""LLM 适配层测试。"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mental_world_model.llm.config import LlmConfig, LlmConfigurationError
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.llm.http_chat_client import HttpChatCompletionsClient
from mental_world_model.llm.json_helpers import parse_json_response
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.llm.openai_compatible_client import OpenAICompatibleClient


class TestLlmConfig:
    def test_from_env_missing_base_url(self, monkeypatch):
        monkeypatch.delenv("MWM_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("MWM_LLM_MODEL", "any-model")
        with pytest.raises(LlmConfigurationError, match="MWM_LLM_BASE_URL"):
            LlmConfig.from_env()

    def test_from_env_missing_model(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_BASE_URL", "https://example.com/v1")
        monkeypatch.delenv("MWM_LLM_MODEL", raising=False)
        with pytest.raises(LlmConfigurationError, match="MWM_LLM_MODEL"):
            LlmConfig.from_env()

    def test_from_env_optional_returns_none_without_base_url(self, monkeypatch):
        monkeypatch.delenv("MWM_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("MWM_LLM_MODEL", raising=False)
        assert LlmConfig.from_env_optional() is None

    def test_from_env_reads_values(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("MWM_LLM_MODEL", "test-model")
        monkeypatch.setenv("MWM_LLM_API_KEY", "test-key")
        monkeypatch.setenv("MWM_LLM_CHAT_PATH", "/v1/chat")
        monkeypatch.setenv("MWM_LLM_AUTH_HEADER", "X-Api-Key")
        monkeypatch.setenv("MWM_LLM_AUTH_PREFIX", "")
        monkeypatch.setenv("MWM_LLM_EXTRA_HEADERS", '{"X-Custom": "1"}')
        config = LlmConfig.from_env()
        assert config.base_url == "https://example.com/v1"
        assert config.model == "test-model"
        assert config.api_key == "test-key"
        assert config.chat_path == "/v1/chat"
        assert config.auth_header == "X-Api-Key"
        assert config.auth_prefix == ""
        assert config.extra_headers == {"X-Custom": "1"}

    def test_api_key_optional(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("MWM_LLM_MODEL", "local-model")
        monkeypatch.delenv("MWM_LLM_API_KEY", raising=False)
        config = LlmConfig.from_env()
        assert config.api_key == ""

    def test_from_env_reads_retry_settings(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("MWM_LLM_MODEL", "m")
        monkeypatch.setenv("MWM_LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("MWM_LLM_RETRY_BACKOFF", "0.5")
        config = LlmConfig.from_env()
        assert config.max_retries == 5
        assert config.retry_backoff_seconds == 0.5

    def test_from_env_decoding_defaults(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("MWM_LLM_MODEL", "m")
        monkeypatch.delenv("MWM_LLM_TEMPERATURE", raising=False)
        monkeypatch.delenv("MWM_LLM_TOP_P", raising=False)
        monkeypatch.delenv("MWM_LLM_SEED", raising=False)
        config = LlmConfig.from_env()
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.seed == 42

    def test_from_env_decoding_overrides_and_empty_seed(self, monkeypatch):
        monkeypatch.setenv("MWM_LLM_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("MWM_LLM_MODEL", "m")
        monkeypatch.setenv("MWM_LLM_TEMPERATURE", "0.2")
        monkeypatch.setenv("MWM_LLM_TOP_P", "0.9")
        monkeypatch.setenv("MWM_LLM_SEED", "")
        config = LlmConfig.from_env()
        assert config.temperature == 0.2
        assert config.top_p == 0.9
        assert config.seed is None


class TestHttpChatCompletionsClient:
    @staticmethod
    def _ok_body(payload: dict) -> str:
        return json.dumps(
            {"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    def test_complete_parses_json_response(self):
        config = LlmConfig(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test-model",
        )
        client = HttpChatCompletionsClient(config)
        with patch.object(
            client._pool,
            "request",
            return_value=(200, self._ok_body({"score": 0.8})),
        ):
            result = client.complete([{"role": "user", "content": "hi"}])
        assert result == {"score": 0.8}

    def test_custom_endpoint_and_headers(self):
        config = LlmConfig(
            base_url="https://provider.example/api",
            model="m",
            api_key="secret",
            chat_path="generate",
            auth_header="X-Api-Key",
            auth_prefix="",
            extra_headers={"X-App": "mwm"},
        )
        client = HttpChatCompletionsClient(config)
        captured: dict = {}

        def fake_request(method, path, body, headers):
            captured["method"] = method
            captured["path"] = path
            captured["headers"] = dict(headers)
            return 200, self._ok_body({})

        with patch.object(client._pool, "request", side_effect=fake_request):
            client.complete([{"role": "user", "content": "hi"}])

        assert captured["method"] == "POST"
        assert captured["path"].endswith("/generate")
        assert captured["headers"]["X-Api-Key"] == "secret"
        assert captured["headers"]["X-App"] == "mwm"

    def test_init_without_api_key(self):
        config = LlmConfig(
            base_url="http://localhost:11434/v1",
            model="local-model",
            api_key="",
        )
        client = HttpChatCompletionsClient(config)
        assert client.config.api_key == ""

    def test_init_with_none_api_key(self):
        config = LlmConfig(
            base_url="http://localhost:11434/v1",
            model="local-model",
            api_key=None,
        )
        client = HttpChatCompletionsClient(config)
        headers = client._build_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_without_api_key(self):
        config = LlmConfig(
            base_url="http://localhost:11434/v1",
            model="local-model",
            api_key="",
        )
        client = HttpChatCompletionsClient(config)
        headers = client._build_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_with_api_key(self):
        config = LlmConfig(
            base_url="https://example.com/v1",
            model="test-model",
            api_key="secret-key",
            auth_header="Authorization",
            auth_prefix="Bearer ",
        )
        client = HttpChatCompletionsClient(config)
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer secret-key"

    def test_rejects_empty_base_url_or_model(self):
        with pytest.raises(LlmConfigurationError):
            HttpChatCompletionsClient(LlmConfig(base_url="", model="m"))
        with pytest.raises(LlmConfigurationError):
            HttpChatCompletionsClient(LlmConfig(base_url="https://x", model=""))

    def test_openai_compatible_alias(self):
        assert OpenAICompatibleClient is HttpChatCompletionsClient

    def test_retries_on_connection_error_then_succeeds(self):
        config = LlmConfig(
            base_url="https://example.com/v1",
            model="test-model",
            max_retries=3,
            retry_backoff_seconds=0.01,
        )
        client = HttpChatCompletionsClient(config)
        calls = {"n": 0}

        def fake_request(method, path, body, headers):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("reset")
            return 200, self._ok_body({"ok": True})

        with (
            patch.object(client._pool, "request", side_effect=fake_request),
            patch("time.sleep") as sleep_mock,
        ):
            result = client.complete([{"role": "user", "content": "hi"}])

        assert result == {"ok": True}
        assert calls["n"] == 3
        assert sleep_mock.call_count == 2

    def test_retries_on_incomplete_read_then_succeeds(self):
        import http.client

        config = LlmConfig(
            base_url="https://example.com/v1",
            model="test-model",
            max_retries=2,
            retry_backoff_seconds=0.01,
        )
        client = HttpChatCompletionsClient(config)
        calls = {"n": 0}

        def fake_request(method, path, body, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                raise http.client.IncompleteRead(b"")
            return 200, self._ok_body({"ok": True})

        with (
            patch.object(client._pool, "request", side_effect=fake_request),
            patch("time.sleep") as sleep_mock,
        ):
            result = client.complete([{"role": "user", "content": "hi"}])

        assert result == {"ok": True}
        assert calls["n"] == 2
        sleep_mock.assert_called_once()

    def test_exhausts_retries_on_persistent_connection_error(self):
        config = LlmConfig(
            base_url="https://example.com/v1",
            model="test-model",
            max_retries=2,
            retry_backoff_seconds=0.01,
        )
        client = HttpChatCompletionsClient(config)

        with (
            patch.object(client._pool, "request", side_effect=OSError("timed out")),
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="网络错误"),
        ):
            client.complete([{"role": "user", "content": "hi"}])

    def test_does_not_retry_client_http_error(self):
        config = LlmConfig(
            base_url="https://example.com/v1",
            model="test-model",
            max_retries=3,
            retry_backoff_seconds=0.01,
        )
        client = HttpChatCompletionsClient(config)
        calls = {"n": 0}

        def fake_request(method, path, body, headers):
            calls["n"] += 1
            return 401, '{"error":"bad key"}'

        with (
            patch.object(client._pool, "request", side_effect=fake_request),
            patch("time.sleep") as sleep_mock,
            pytest.raises(RuntimeError, match="401"),
        ):
            client.complete([{"role": "user", "content": "hi"}])

        assert calls["n"] == 1
        sleep_mock.assert_not_called()

    def test_retries_on_503_then_succeeds(self):
        config = LlmConfig(
            base_url="https://example.com/v1",
            model="test-model",
            max_retries=2,
            retry_backoff_seconds=0.01,
        )
        client = HttpChatCompletionsClient(config)
        calls = {"n": 0}

        def fake_request(method, path, body, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                return 503, "busy"
            return 200, self._ok_body({"score": 1})

        with (
            patch.object(client._pool, "request", side_effect=fake_request),
            patch("time.sleep"),
        ):
            result = client.complete([{"role": "user", "content": "hi"}])

        assert result == {"score": 1}
        assert calls["n"] == 2

    def test_pool_allows_concurrent_in_flight_requests(self, monkeypatch):
        """多连接池：并行 request 不应被单锁串行化。"""
        import threading
        import time
        import urllib.parse

        from mental_world_model.llm.http_chat_client import _ConnectionPool

        monkeypatch.setenv("MWM_LLM_POOL_SIZE", "4")
        pool = _ConnectionPool(
            urllib.parse.urlparse("http://example.com"),
            timeout=5.0,
            max_size=4,
        )
        barrier = threading.Barrier(3)
        in_flight = {"n": 0, "peak": 0}
        lock = threading.Lock()

        class FakeConn:
            def request(self, *args, **kwargs):
                with lock:
                    in_flight["n"] += 1
                    in_flight["peak"] = max(in_flight["peak"], in_flight["n"])
                barrier.wait(timeout=2)
                time.sleep(0.05)
                with lock:
                    in_flight["n"] -= 1

            def getresponse(self):
                class Resp:
                    status = 200

                    def read(self):
                        return b"{}"

                return Resp()

            def close(self):
                return None

        pool._new_conn = FakeConn  # type: ignore[method-assign]
        errors: list = []

        def worker():
            try:
                pool.request("POST", "/x", b"{}", {"Content-Type": "application/json"})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        pool.close()
        assert not errors
        assert in_flight["peak"] >= 3


class TestJsonHelpers:
    def test_parse_json_response_strips_markdown_fence(self):
        content = '```json\n{"score": 1.0}\n```'
        assert parse_json_response(content) == {"score": 1.0}

    def test_parse_json_response_rejects_empty(self):
        with pytest.raises(ValueError, match="空内容"):
            parse_json_response("")
        with pytest.raises(ValueError, match="空内容"):
            parse_json_response("   ")

    def test_parse_json_response_rejects_invalid(self):
        with pytest.raises(ValueError, match="非合法 JSON"):
            parse_json_response("not-json")


class TestCreateLlmClient:
    def test_use_mock(self):
        client = create_llm_client(use_mock=True)
        assert client.complete([{"role": "user", "content": "x"}])["score"] == 1.0

    def test_inject_custom_client(self):
        custom = MockLlmClient(default_response={"score": 0.42})
        client = create_llm_client(client=custom)
        assert client.complete([{"role": "user", "content": "x"}])["score"] == 0.42

    def test_config_without_api_key_creates_client(self):
        config = LlmConfig(base_url="http://localhost:11434/v1", model="m")
        client = create_llm_client(config=config, trace=False)
        assert isinstance(client, HttpChatCompletionsClient)
        assert "Authorization" not in client._build_headers()


def _llm_integration_configured() -> bool:
    return bool(
        os.environ.get("MWM_LLM_BASE_URL", "").strip()
        and os.environ.get("MWM_LLM_MODEL", "").strip()
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _llm_integration_configured(),
    reason="未配置 MWM_LLM_BASE_URL / MWM_LLM_MODEL，跳过真实 LLM 集成测试",
)
class TestLlmIntegration:
    def test_real_llm_short_extraction_smoke(self):
        from mental_world_model.model import MentalWorldModel

        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        llm = create_llm_client()
        extracted = model.extract("我知道这样做是对的。", llm)
        assert extracted.suoyuan.text
        assert extracted.xiang.text
        assert extracted.si.text
