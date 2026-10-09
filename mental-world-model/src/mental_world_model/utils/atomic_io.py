"""同目录临时文件 + os.replace，避免半写入 JSON。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """先写 ``*.tmp``，再替换目标文件。"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, dest)


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """UTF-8 JSON 原子写盘。"""
    atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=indent),
    )
