"""从项目根目录加载 .env（python-dotenv）。"""

from pathlib import Path


def get_project_root() -> Path:
    """mental-world-model 项目根目录（含 pyproject.toml 与 .env）。"""
    # .../src/mental_world_model/llm/env_file.py -> parents[3]
    return Path(__file__).resolve().parents[3]


def load_project_dotenv(*, override: bool = False) -> Path | None:
    """加载项目根目录 ``.env`` 到 ``os.environ``；文件不存在则返回 None。"""
    from dotenv import load_dotenv

    env_path = get_project_root() / ".env"
    if not env_path.is_file():
        return None
    load_dotenv(env_path, override=override)
    return env_path
