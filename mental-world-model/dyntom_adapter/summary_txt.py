"""把 DynToM 的 MWM 结果写进 summary.txt。只改心智世界模型这一段。"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DIM_ORDER = ("belief", "emotion", "intention", "action")
_DIM_NAME = {"belief": "信念", "emotion": "情绪", "intention": "意图", "action": "行动"}
_KIND_ORDER = ("how", "what", "whether", "why")
_KIND = re.compile(r"_(how|what|whether|why)(?:_|$)")
_SCORE_LINE = re.compile(r"^(.+)：(\d+)/(\d+) = ([0-9.]+)$")


def update_summary(result_dir: Path, summary_path: Path | None = None) -> Path | None:
    """按结果目录里的 answers.jsonl 重写汇总。没有答题记录时不改文件。"""
    result_dir = result_dir.expanduser().resolve()
    trials = _trials(result_dir)
    if not trials:
        return None
    path = summary_path or _default_summary(result_dir)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = _block(result_dir, trials)
    total_correct = sum(item["correct"] for item in trials)
    total = sum(item["total"] for item in trials)
    updated = _apply(text, block, _score_line(total_correct, total), _range_line(trials))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return path


def mwm_root(path: Path) -> Path:
    """audit 可以指向 trial 目录或 answers.jsonl，汇总仍用上一层结果目录。"""
    path = path.expanduser().resolve()
    if path.is_file():
        path = path.parent
    if (path / "answers.jsonl").is_file():
        return path.parent
    return path


def _default_summary(result_dir: Path) -> Path:
    if result_dir.name == "dyntom mwm" and result_dir.parent.name == "dyntom":
        return result_dir.parent / "summary.txt"
    return result_dir / "summary.txt"


def _trials(result_dir: Path) -> list[dict]:
    found = []
    if not result_dir.is_dir():
        return found
    for trial_dir in sorted(path for path in result_dir.iterdir() if path.is_dir()):
        answer_path = trial_dir / "answers.jsonl"
        if not answer_path.is_file():
            continue
        rows = []
        for line in answer_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        if rows:
            found.append(_trial_stats(trial_dir.name, rows))
    return found


def _trial_stats(trial_id: str, rows: list[dict]) -> dict:
    manual = [
        _short_id(str(row.get("question_id") or ""))
        for row in rows
        if row.get("manual_correct") is True
    ]
    return {
        "trial_id": trial_id,
        "correct": sum(1 for row in rows if row.get("correct") is True),
        "total": len(rows),
        "manual": manual,
        "dimensions": _counts(rows, "target_dimension", _DIM_ORDER),
        "kinds": _counts(rows, "_kind", _KIND_ORDER),
    }


def _counts(rows: list[dict], key: str, order: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    buckets = {name: [0, 0] for name in order}
    for row in rows:
        name = _kind(row) if key == "_kind" else str(row.get(key) or "")
        if name not in buckets:
            continue
        buckets[name][1] += 1
        if row.get("correct") is True:
            buckets[name][0] += 1
    return {name: (correct, total) for name, (correct, total) in buckets.items() if total}


def _kind(row: dict) -> str:
    match = _KIND.search(str(row.get("question_id") or ""))
    return match.group(1) if match else ""


def _short_id(question_id: str) -> str:
    if question_id.startswith("type_d_"):
        return question_id[len("type_d_") :]
    if question_id.startswith("type_a_"):
        return question_id[len("type_a_") :]
    if question_id.startswith("type_"):
        return question_id[len("type_") :]
    return question_id


def _block(result_dir: Path, trials: list[dict]) -> str:
    lines = [f"心智世界模型（{_display(result_dir)}）"]
    for trial in trials:
        manual = ""
        if trial["manual"]:
            names = "、".join(trial["manual"])
            manual = f"（其中 {len(trial['manual'])} 题为人工改判：{names}）"
        lines.append(
            f"{trial['trial_id']}：{trial['correct']}/{trial['total']} = {_ratio(trial['correct'], trial['total'])}{manual}"
        )
    dimensions = _merge(trials, "dimensions", _DIM_ORDER)
    if dimensions:
        lines.append(
            "，".join(
                f"{_DIM_NAME[name]} {correct}/{total}"
                for name, (correct, total) in dimensions
            )
        )
    for name, (correct, total) in _merge(trials, "kinds", _KIND_ORDER):
        lines.append(f"{name} {correct}/{total} = {_ratio(correct, total)}")
    return "\n".join(lines)


def _merge(trials: list[dict], key: str, order: tuple[str, ...]) -> list[tuple[str, tuple[int, int]]]:
    totals = {name: [0, 0] for name in order}
    for trial in trials:
        for name, (correct, total) in trial[key].items():
            totals[name][0] += correct
            totals[name][1] += total
    return [(name, (correct, total)) for name, (correct, total) in totals.items() if total]


def _apply(text: str, block: str, score_line: str, range_line: str) -> str:
    if not text.strip():
        return f"DynToM 准确度总结\n\n{range_line}\n\n合计\n{score_line}\n\n{block}\n"
    parts = re.split(r"\n\s*\n", text.strip())
    replaced = False
    for index, part in enumerate(parts):
        if part.startswith("心智世界模型（"):
            parts[index] = block
            replaced = True
            break
    if not replaced:
        parts.append(block)
    for index, part in enumerate(parts):
        if part.startswith("范围："):
            lines = part.splitlines()
            lines[0] = range_line
            parts[index] = "\n".join(lines)
            break
    for index, part in enumerate(parts):
        if part.startswith("合计"):
            parts[index] = _ranking(part, score_line)
            break
    else:
        parts.insert(1, f"合计\n{score_line}")
    return "\n\n".join(parts).rstrip() + "\n"


def _ranking(part: str, score_line: str) -> str:
    lines = [line for line in part.splitlines() if not line.startswith("心智世界模型：")]
    if lines and lines[0] == "合计":
        body = lines[1:]
    else:
        body = lines
        lines = ["合计"]
    body.append(score_line)
    body.sort(key=_rank_key)
    return "\n".join(["合计", *body])


def _rank_key(line: str) -> tuple[float, int]:
    match = _SCORE_LINE.match(line.strip())
    if not match:
        return (1.0, 0)
    correct = int(match.group(2))
    total = int(match.group(3))
    accuracy = correct / total if total else 0.0
    return (-accuracy, -correct)


def _range_line(trials: list[dict]) -> str:
    names = "、".join(trial["trial_id"] for trial in trials)
    total = sum(trial["total"] for trial in trials)
    return (
        f"范围：心智世界模型目前有 {names}，合计 {total} 题。"
        "其余方法仍按各自原有范围，不随这次结果改动。"
    )


def _score_line(correct: int, total: int) -> str:
    return f"心智世界模型：{correct}/{total} = {_ratio(correct, total)}"


def _ratio(correct: int, total: int) -> str:
    return f"{(correct / total):.4f}" if total else "0.0000"


def _display(path: Path) -> str:
    parts = path.parts
    if "reports" in parts:
        start = parts.index("reports")
        return str(Path(*parts[start:]))
    return path.name
