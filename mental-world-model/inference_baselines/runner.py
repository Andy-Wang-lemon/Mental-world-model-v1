"""一题写一个文件。已完成的跳过，失败的记下来再继续。"""

from __future__ import annotations

import json
from pathlib import Path

from mental_world_model.utils.atomic_io import atomic_write_json

from inference_baselines.items import BenchmarkItem
from inference_baselines.methods import METHODS

_DONE = frozenset({"ok", "target_character_unresolved"})
_PROTECTED = frozenset(
    {
        "dyntom mwm",
        "dyntom_gemini",
        "dyntom_deepseek",
        "dyntom_grok",
        "dyntom_gpt",
        "mwm",
        "rectom_grok",
        "rectom_gemini",
        "rectom_deepseek",
        "rectom_gpt",
    }
)


def result_dir(root: Path, dataset: str, method: str) -> Path:
    resolved = root.expanduser().resolve()
    if dataset not in {"dyntom", "rectom"} or method not in METHODS:
        raise SystemExit(f"未知输出位置：{dataset}/{method}")
    out = resolved / dataset / method
    if any(part in _PROTECTED for part in out.parts):
        raise SystemExit("不能写进已有的心智世界模型或直接作答目录。")
    return out


def run_items(
    items: list[BenchmarkItem],
    *,
    method: str,
    client: object,
    out_dir: Path,
    resume: bool,
) -> int:
    fn = METHODS[method]
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for item in items:
        path = out_dir / f"{item.id}.json"
        existing = _read(path)
        if existing is not None and existing.get("status") in _DONE:
            print(f"跳过已完成 {item.id}", flush=True)
            continue
        if existing is not None and existing.get("status") == "error" and not resume:
            print(f"{item.id} 上次失败。续跑请加 --resume", flush=True)
            failed += 1
            continue
        if not item.target_character:
            row = _unresolved(item, method)
            atomic_write_json(path, row)
            print(f"{item.id} target_character_unresolved", flush=True)
            continue
        try:
            row = fn(client, item)
        except Exception as error:  # noqa: BLE001
            failed += 1
            row = _error(item, method, error)
            saved = getattr(error, "saved", None)
            if isinstance(saved, dict):
                row.update(saved)
                row["status"] = "error"
                row["correct"] = False
                row["error"] = f"{type(error).__name__}: {error}"
            atomic_write_json(path, row)
            print(f"{item.id} 失败，已记录，继续下一题：{error}", flush=True)
            continue
        row["status"] = "ok"
        row["dataset"] = item.dataset
        row["source"] = item.source
        atomic_write_json(path, row)
        mark = "对" if row.get("correct") else "错"
        print(f"{item.id} → {row.get('prediction') or '-'} {mark}", flush=True)
        print(path, flush=True)
    return failed


def _unresolved(item: BenchmarkItem, method: str) -> dict:
    return {
        "id": item.id,
        "method": method,
        "dataset": item.dataset,
        "source": item.source,
        "status": "target_character_unresolved",
        "target_character": "",
        "prediction": "",
        "gold": list(item.gold),
        "correct": False,
    }


def _error(item: BenchmarkItem, method: str, error: Exception) -> dict:
    return {
        "id": item.id,
        "method": method,
        "dataset": item.dataset,
        "source": item.source,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "prediction": "",
        "gold": list(item.gold) if len(item.gold) != 1 else item.gold[0],
        "correct": False,
    }


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
