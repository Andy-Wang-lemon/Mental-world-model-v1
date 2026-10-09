#!/usr/bin/env python3
"""将 OpenToM 官方 JSON 转为 mwm_test_adapter 可用的 JSONL。

支持：
1. HuggingFace / GitHub 上的 opentom.json、opentom_long.json（列表格式）
2. opentom_data/ 下按题型拆分的 JSON（如 attitude.json、location_cg_fo.json）

用法：
  python scripts/opentom/preprocess_opentom.py \\
    --input /path/to/opentom.json \\
    --output data/fixtures/benchmark_samples/opentom_300.jsonl \\
    --sample-size 300

  python scripts/opentom/preprocess_opentom.py \\
    --input /path/to/opentom_data \\
    --output data/fixtures/benchmark_samples/opentom.jsonl \\
    --question-types attitude location_cg_fo
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

from mwm_test_adapter.opentom_metrics import normalize_paper_question_type

# 题型 → 默认选项（无 per-row choices 时的回退）
DEFAULT_CHOICES: dict[str, list[str]] = {
    "attitude": ["positive", "neutral", "negative"],
    "location_cg_fo": ["yes", "no"],
    "location_cg_so": ["yes", "no"],
    "location-fo-coarse": ["yes", "no"],
    "location-so-coarse": ["yes", "no"],
    "multihop_fo": ["more full", "less full", "equally full", "more accessible", "less accessible", "equally accessible"],
    "multihop_so": ["more full", "less full", "equally full", "more accessible", "less accessible", "equally accessible"],
    "multihop-fo": ["more full", "less full", "equally full", "more accessible", "less accessible", "equally accessible"],
    "multihop-so": ["more full", "less full", "equally full", "more accessible", "less accessible", "equally accessible"],
}

QUESTION_TYPE_ALIASES = {
    "location-fo-coarse": "location_cg_fo",
    "location-so-coarse": "location_cg_so",
    "location-fo-fine": "location_fg_fo",
    "location-so-fine": "location_fg_so",
    "multihop-fo": "multihop_fo",
    "multihop-so": "multihop_so",
    "attitude": "attitude",
}


def _normalize_question_type(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return QUESTION_TYPE_ALIASES.get(key, key)


def _parse_plot_info(plot_info: Any) -> dict[str, Any]:
    if isinstance(plot_info, dict):
        return plot_info
    if isinstance(plot_info, str) and plot_info.strip():
        return json.loads(plot_info)
    return {}


def _infer_character(question: str, plot_info: dict[str, Any]) -> str:
    if plot_info.get("observer"):
        return str(plot_info["observer"])
    if plot_info.get("affected_char"):
        return str(plot_info["affected_char"])
    match = re.search(
        r"(?:does|is|what is|where does)\s+([A-Z][a-z]+)(?:'s|\s)",
        question,
    )
    if match:
        return match.group(1)
    mover = plot_info.get("mover")
    if mover:
        return str(mover)
    return ""


def _infer_choices(
    question_type: str,
    answer: str,
    plot_info: dict[str, Any],
    explicit: list[str] | None,
) -> list[str]:
    if explicit:
        return explicit
    normalized = _normalize_question_type(question_type)
    if normalized in ("location_fg_fo", "location_fg_so"):
        places: list[str] = []
        for key in ("original_place", "move_to_place", "new_location"):
            value = plot_info.get(key)
            if value and str(value).strip() and str(value) not in places:
                places.append(str(value).strip())
        if places:
            return places
    if normalized == "attitude":
        return DEFAULT_CHOICES["attitude"]
    if "location_cg" in normalized or "coarse" in question_type:
        return ["yes", "no"]
    if "multihop" in normalized:
        answer_lower = answer.lower()
        if "accessible" in answer_lower or "accessible" in question_type:
            return ["more accessible", "less accessible", "equally accessible"]
        return ["more full", "less full", "equally full"]
    return DEFAULT_CHOICES.get(normalized, [answer])


def _normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        cleaned = [str(item).strip() for item in answer if str(item).strip() and str(item) != "corrupted"]
        return cleaned[0] if cleaned else ""
    return str(answer).strip()


def _story_from_record(record: dict[str, Any]) -> str:
    for key in ("narrative", "story", "plot", "text"):
        value = record.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def record_to_row(record: dict[str, Any], *, fallback_type: str = "") -> dict[str, Any] | None:
    story = _story_from_record(record)
    question = str(record.get("question", "")).strip()
    answer = _normalize_answer(record.get("answer", record.get("label", "")))
    question_type = _normalize_question_type(
        str(record.get("question_type", record.get("type", fallback_type)))
    )
    if not story or not question or not answer:
        return None

    plot_info = _parse_plot_info(record.get("plot_info", {}))
    character = str(record.get("character", "")).strip() or _infer_character(question, plot_info)
    explicit_choices = record.get("choices")
    if explicit_choices is not None and not isinstance(explicit_choices, list):
        explicit_choices = None
    choices = _infer_choices(
        question_type,
        answer,
        plot_info,
        [str(c) for c in explicit_choices] if explicit_choices else None,
    )

    row_id = record.get("id")
    if row_id is None:
        row_id = record.get("idx", record.get("index"))
    if row_id is None:
        row_id = f"{question_type}-{hash((story[:40], question)) & 0xFFFFFFFF:08x}"

    return {
        "id": str(row_id),
        "story": story,
        "question": question,
        "character": character,
        "question_type": question_type,
        "choices": choices,
        "labels": [answer],
    }


def stratified_sample_rows(
    rows: list[dict[str, Any]],
    sample_size: int,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """按论文题型分层抽样（默认用于 ~300 条 benchmark）。"""
    if sample_size <= 0 or sample_size >= len(rows):
        return rows

    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        paper_type = normalize_paper_question_type(str(row.get("question_type", "")))
        by_type.setdefault(paper_type, []).append(row)

    types = sorted(by_type.keys())
    if not types:
        return rows[:sample_size]

    rng = random.Random(seed)
    base = sample_size // len(types)
    remainder = sample_size % len(types)
    sampled: list[dict[str, Any]] = []
    for index, paper_type in enumerate(types):
        pool = by_type[paper_type]
        take = base + (1 if index < remainder else 0)
        take = min(take, len(pool))
        sampled.extend(rng.sample(pool, take))

    rng.shuffle(sampled)
    return sampled


def iter_opentom_records(path: Path, question_types: set[str] | None) -> Iterator[dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(data, dict):
            for key, items in data.items():
                if not isinstance(items, list):
                    continue
                if question_types and key not in question_types:
                    continue
                for item in items:
                    if isinstance(item, dict):
                        item = dict(item)
                        item.setdefault("question_type", key)
                        yield item
            return
        raise ValueError(f"无法解析 OpenToM 文件：{path}")

    if path.is_dir():
        patterns = ["*.json"]
        if question_types:
            patterns = [f"{qt}.json" for qt in question_types]
        seen: set[Path] = set()
        for pattern in patterns:
            for file_path in sorted(path.glob(pattern)):
                if file_path.name in ("metadata.json", "metadata_long.json"):
                    continue
                if file_path in seen:
                    continue
                seen.add(file_path)
                fallback_type = file_path.stem
                data = json.loads(file_path.read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else data.get(fallback_type, [])
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        item = dict(item)
                        item.setdefault("question_type", fallback_type)
                        yield item
        return

    raise FileNotFoundError(path)


def convert_opentom(
    input_path: Path,
    output_path: Path,
    *,
    question_types: list[str] | None = None,
    limit: int | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> int:
    type_filter = set(question_types) if question_types else None
    rows: list[dict[str, Any]] = []
    for record in iter_opentom_records(input_path, type_filter):
        row = record_to_row(record)
        if row is None:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit and sample_size is None:
            break

    if sample_size is not None:
        rows = stratified_sample_rows(rows, sample_size, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenToM → mwm_test_adapter JSONL")
    parser.add_argument("--input", required=True, help="opentom.json 或 opentom_data 目录")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument(
        "--question-types",
        nargs="*",
        help="仅转换指定题型（如 attitude location_cg_fo）",
    )
    parser.add_argument("--limit", type=int, help="转换时最多读取条数（全量读取前截断）")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="按论文题型分层抽样条数（如 300；推荐用于 benchmark）",
    )
    parser.add_argument("--seed", type=int, default=42, help="分层抽样随机种子")
    args = parser.parse_args()

    count = convert_opentom(
        Path(args.input),
        Path(args.output),
        question_types=args.question_types,
        limit=args.limit,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"已写入 {count} 条 → {args.output}")
    if args.sample_size:
        print(f"（分层抽样 sample_size={args.sample_size}, seed={args.seed}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
