"""按维度、操作、单时刻 / 跨时间汇总准确率。"""

from __future__ import annotations

import json
from pathlib import Path


def load_answer_rows(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("answers.jsonl"))
    rows: list[dict] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "total": _accuracy(rows, "correct"),
        "by_dimension": _group(rows, "target_dimension", "correct"),
        "by_operation": _group(rows, "operation", "correct"),
        "by_question_type": _group(rows, "question_type", "correct"),
        "by_scope": _group(rows, "scope", "correct"),
        "control": {
            "total": _accuracy(rows, "control_correct"),
            "by_dimension": _group(rows, "target_dimension", "control_correct"),
            "by_operation": _group(rows, "operation", "control_correct"),
            "by_scope": _group(rows, "scope", "control_correct"),
        },
    }


def _group(rows: list[dict], key: str, correct_field: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        if correct_field not in row:
            continue
        name = str(row.get(key) or "unknown")
        buckets.setdefault(name, []).append(row)
    return {name: _accuracy(items, correct_field) for name, items in sorted(buckets.items())}


def _accuracy(rows: list[dict], correct_field: str) -> dict:
    usable = [row for row in rows if correct_field in row]
    total = len(usable)
    correct = sum(1 for row in usable if row.get(correct_field) is True)
    failed = sum(1 for row in usable if row.get("error"))
    return {
        "total": total,
        "correct": correct,
        "failed": failed,
        "accuracy": round(correct / total, 4) if total else None,
    }
