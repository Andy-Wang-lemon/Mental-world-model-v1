"""LLM 客户端配置（厂商无关）。"""

import json
import os
from dataclasses import dataclass, field

from mental_world_model.llm.env_file import load_project_dotenv


class LlmConfigurationError(ValueError):
    """LLM 配置缺失或无效。"""

    def user_message(self) -> str:
        """面向用户的简短说明。"""
        return (
            str(self.args[0])
            if self.args
            else (
                "LLM 配置无效。请设置 MWM_LLM_BASE_URL 与 MWM_LLM_MODEL，"
                "或使用 --mock / use_mock=True。"
            )
        )


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_extra_headers(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise LlmConfigurationError("MWM_LLM_EXTRA_HEADERS 应为 JSON 对象")
    return {str(key): str(value) for key, value in data.items()}


@dataclass
class LlmConfig:
    """通用 HTTP Chat Completions 适配配置。

    不预设任何厂商地址或模型名；由部署方指定 base_url、model 及鉴权方式。
    """

    base_url: str
    model: str
    api_key: str = ""
    chat_path: str = "/chat/completions"
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 42
    use_json_response_format: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "LlmConfig":
        """从环境变量加载配置。

        环境变量：
        - ``MWM_LLM_BASE_URL``（必填）API 根地址，如 ``https://your-host/v1``
        - ``MWM_LLM_MODEL``（必填）模型标识，由你所用服务定义
        - ``MWM_LLM_API_KEY``（可选）鉴权密钥；本地无鉴权服务可留空
        - ``MWM_LLM_CHAT_PATH``（可选，默认 ``/chat/completions``）
        - ``MWM_LLM_AUTH_HEADER``（可选，默认 ``Authorization``）
        - ``MWM_LLM_AUTH_PREFIX``（可选，默认 ``Bearer ``；无前缀时设为空字符串）
        - ``MWM_LLM_EXTRA_HEADERS``（可选）JSON 对象，附加 HTTP 头
        - ``MWM_LLM_TIMEOUT``（可选，秒）
        - ``MWM_LLM_TEMPERATURE``（可选，默认 0.0）采样温度
        - ``MWM_LLM_TOP_P``（可选，默认 1.0）nucleus 采样；与 temperature=0 搭配时保持 1
        - ``MWM_LLM_SEED``（可选，默认 42；设为空字符串则不传 seed）
        - ``MWM_LLM_JSON_RESPONSE``（可选，默认 true；不支持时设为 false）
        - ``MWM_LLM_MAX_RETRIES``（可选，默认 3）首次失败后的最大重试次数
        - ``MWM_LLM_RETRY_BACKOFF``（可选，默认 1.0）重试退避基数（秒，指数增长）

        另见运行时并发相关环境变量（非本 dataclass 字段）：
        ``MWM_LLM_PARALLEL`` / ``MWM_LLM_MAX_CONCURRENCY`` / ``MWM_LLM_POOL_SIZE`` /
        ``MWM_LLM_PROGRESS``（默认里程碑进度；``0`` 关闭；``all`` 逐次调用）。

        若环境变量未设置，会尝试从项目根目录的 ``.env`` 文件加载（见 ``load_project_dotenv``）。
        """
        load_project_dotenv()

        base_url = os.environ.get("MWM_LLM_BASE_URL", "").strip().rstrip("/")
        model = os.environ.get("MWM_LLM_MODEL", "").strip()

        if not base_url:
            raise LlmConfigurationError(
                "未设置 MWM_LLM_BASE_URL。请配置你所用 LLM 服务的 API 根地址，"
                "或使用 --mock / use_mock=True 进行本地测试。"
            )
        if not model:
            raise LlmConfigurationError(
                "未设置 MWM_LLM_MODEL。请配置模型标识（由你的 LLM 服务商提供），"
                "或使用 --mock / use_mock=True。"
            )

        timeout_raw = os.environ.get("MWM_LLM_TIMEOUT", "120").strip()
        chat_path = os.environ.get("MWM_LLM_CHAT_PATH", "/chat/completions").strip()
        auth_header = os.environ.get("MWM_LLM_AUTH_HEADER", "Authorization").strip()
        auth_prefix = os.environ.get("MWM_LLM_AUTH_PREFIX", "Bearer ")
        if "MWM_LLM_AUTH_PREFIX" in os.environ and not os.environ["MWM_LLM_AUTH_PREFIX"]:
            auth_prefix = ""

        retries_raw = os.environ.get("MWM_LLM_MAX_RETRIES", "3").strip()
        backoff_raw = os.environ.get("MWM_LLM_RETRY_BACKOFF", "1.0").strip()
        try:
            max_retries = max(0, int(retries_raw))
        except ValueError as error:
            raise LlmConfigurationError(
                f"MWM_LLM_MAX_RETRIES 应为非负整数：{retries_raw}"
            ) from error
        try:
            retry_backoff = max(0.0, float(backoff_raw))
        except ValueError as error:
            raise LlmConfigurationError(
                f"MWM_LLM_RETRY_BACKOFF 应为非负数字：{backoff_raw}"
            ) from error

        temperature_raw = os.environ.get("MWM_LLM_TEMPERATURE", "0").strip()
        top_p_raw = os.environ.get("MWM_LLM_TOP_P", "1").strip()
        try:
            temperature = float(temperature_raw)
        except ValueError as error:
            raise LlmConfigurationError(
                f"MWM_LLM_TEMPERATURE 应为数字：{temperature_raw}"
            ) from error
        try:
            top_p = float(top_p_raw)
        except ValueError as error:
            raise LlmConfigurationError(
                f"MWM_LLM_TOP_P 应为数字：{top_p_raw}"
            ) from error

        seed: int | None = 42
        if "MWM_LLM_SEED" in os.environ:
            seed_raw = os.environ["MWM_LLM_SEED"].strip()
            if not seed_raw:
                seed = None
            else:
                try:
                    seed = int(seed_raw)
                except ValueError as error:
                    raise LlmConfigurationError(
                        f"MWM_LLM_SEED 应为整数或空字符串：{seed_raw}"
                    ) from error

        extra_headers: dict[str, str] = {}
        extra_raw = os.environ.get("MWM_LLM_EXTRA_HEADERS", "").strip()
        if extra_raw:
            try:
                extra_headers = _parse_extra_headers(extra_raw)
            except json.JSONDecodeError as error:
                raise LlmConfigurationError(
                    f"MWM_LLM_EXTRA_HEADERS 不是合法 JSON：{error}"
                ) from error

        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("MWM_LLM_API_KEY", "").strip(),
            chat_path=chat_path or "/chat/completions",
            auth_header=auth_header or "Authorization",
            auth_prefix=auth_prefix,
            timeout_seconds=float(timeout_raw),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            use_json_response_format=_env_flag("MWM_LLM_JSON_RESPONSE", default=True),
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff,
        )

    @classmethod
    def from_env_optional(cls) -> "LlmConfig | None":
        """若未配置 BASE_URL 与 MODEL 则返回 None。"""
        base_url = os.environ.get("MWM_LLM_BASE_URL", "").strip()
        model = os.environ.get("MWM_LLM_MODEL", "").strip()
        if not base_url or not model:
            return None
        return cls.from_env()
