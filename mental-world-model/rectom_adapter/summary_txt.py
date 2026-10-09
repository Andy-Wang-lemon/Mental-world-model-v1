"""把 RecToM 的 MWM 结果写进 summary.txt。只改心智世界模型这一段。"""

from __future__ import annotations

import json
import re
from pathlib import Path

_TITLES = {
    "3_pred_rec": "推荐者 3_pred_rec",
    "4_pred_seeker": "寻求者 4_pred_seeker",
}
_SCORE_LINE = re.compile(r"^(.+)：(\d+)/(\d+) = ([0-9.]+)$")
_ITEM_DIR = re.compile(r"^(\d{4})_")


def update_summary(result_dir: Path, summary_path: Path | None = None) -> Path | None:
    """按结果目录里的 answer.json 重写汇总。没有答题记录时不改文件。"""
    result_dir = result_dir.expanduser().resolve()
    groups = _groups(result_dir)
    if not any(group["rows"] or group["unfinished"] for group in groups):
        return None
    if not any(group["rows"] for group in groups):
        return None
    path = summary_path or _default_summary(result_dir)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    correct = sum(1 for group in groups for row in group["rows"] if row.get("correct") is True)
    total = sum(len(group["rows"]) for group in groups)
    updated = _apply(text, _block(result_dir, groups), _score_line(correct, total), _range_line(correct, total))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return path


def _default_summary(result_dir: Path) -> Path:
    if result_dir.name == "mwm" and result_dir.parent.name == "rectom":
        return result_dir.parent / "summary.txt"
    return result_dir / "summary.txt"


def _groups(result_dir: Path) -> list[dict]:
    if not result_dir.is_dir():
        return []
    found = []
    for dataset_dir in sorted(path for path in result_dir.iterdir() if path.is_dir()):
        rows = []
        unfinished = []
        for item_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            answer_path = item_dir / "answer.json"
            if answer_path.is_file():
                item = json.loads(answer_path.read_text(encoding="utf-8"))
                if isinstance(item, dict):
                    rows.append(item)
                continue
            match = _ITEM_DIR.match(item_dir.name)
            if match:
                unfinished.append(f"#{int(match.group(1))} {item_dir.name.split('_', 1)[1]}")
        if rows or unfinished:
            found.append({"name": dataset_dir.name, "rows": rows, "unfinished": unfinished})
    return found


def _block(result_dir: Path, groups: list[dict]) -> str:
    lines = [f"心智世界模型（{_display(result_dir)}）"]
    for group in groups:
        rows = group["rows"]
        correct = sum(1 for row in rows if row.get("correct") is True)
        title = _TITLES.get(group["name"], group["name"])
        lines.append(f"{title}：{correct}/{len(rows)} = {_ratio(correct, len(rows))}")
        direct = [row for row in rows if row.get("selector_correct") is True]
        manual_right = [
            row for row in rows if _manual(row) == "对" and row.get("selector_correct") is not True
        ]
        audited = [
            row
            for row in rows
            if row.get("correct") is True
            and row.get("selector_correct") is not True
            and row.get("mwm_matches_gold") is True
            and _manual(row) != "对"
        ]
        wrong = [row for row in rows if row.get("correct") is not True]
        lines.append(f"选题直接对：{len(direct)}/{len(rows)}")
        lines.append("最终仍错：" + ("无" if not wrong else "、".join(_wrong_label(row) for row in _by_index(wrong))))
        if manual_right:
            lines.append("人工改判为对：" + "、".join(_label(row) for row in _by_index(manual_right)))
        if audited:
            lines.append("核对后改对：" + "、".join(_label(row) for row in _by_index(audited)))
        if group["unfinished"]:
            lines.append("未完成、不计入：" + "、".join(group["unfinished"]))
    return "\n".join(lines)


def _manual(row: dict) -> str:
    return str(row.get("manual_judgment") or "").strip()


def _label(row: dict) -> str:
    return f"#{row.get('index')} {row.get('item_id') or ''}".strip()


def _wrong_label(row: dict) -> str:
    label = _label(row)
    if _manual(row) == "错":
        return f"{label}（人工判定错）"
    return label


def _by_index(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: int(row.get("index") or 0))


def _apply(text: str, block: str, score_line: str, range_line: str) -> str:
    if not text.strip():
        return f"RecToM 准确度总结\n\n{range_line}\n\n合计\n{score_line}\n\n{block}\n"
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
    body = lines[1:] if lines and lines[0] == "合计" else lines
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


def _range_line(correct: int, total: int) -> str:
    return (
        f"范围：心智世界模型按已有 answer.json 计，目前 {correct}/{total}。"
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
