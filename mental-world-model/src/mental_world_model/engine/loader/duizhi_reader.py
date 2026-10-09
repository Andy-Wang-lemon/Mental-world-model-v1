"""从心所对治关系 Word 文档提取并解析结构化数据。"""

import re
from pathlib import Path

from mental_world_model.engine.loader.docx_reader import extract_paragraphs
from mental_world_model.utils.xinsuo_names import (
    format_relation_description,
    normalize_relation_type,
    normalize_xinsuo_name,
)

_SECTION_HEADER = "与其他心所的关系"

_RELATION_PATTERN = re.compile(r"^(?:(直接))?(对治|障碍)(.+?)[。]?$")


def _parse_relation_line(line: str) -> dict[str, str | bool]:
    """解析关系描述行，如「直接对治不信」。"""
    match = _RELATION_PATTERN.match(line.strip())
    if not match:
        raise ValueError(f"无法解析关系描述：{line!r}")

    is_direct, relation_type, target = match.groups()
    target = normalize_xinsuo_name(target.strip())
    is_direct_bool = is_direct is not None
    canonical_type = normalize_relation_type(relation_type)
    return {
        "target": target,
        "description": format_relation_description(relation_type, target, is_direct_bool),
        "relation_type": canonical_type,
        "is_direct": is_direct_bool,
    }


def parse_duizhi_paragraphs(paragraphs: list[str]) -> list[dict[str, str | bool]]:
    """将段落列表解析为心所对治关系字典列表。"""
    entries: list[dict[str, str | bool]] = []
    lines = [line.strip() for line in paragraphs if line.strip()]

    idx = 0
    while idx < len(lines):
        if lines[idx] != "原文":
            idx += 1
            continue

        idx += 1
        if idx < len(lines) and lines[idx] == _SECTION_HEADER:
            idx += 1

        while idx < len(lines):
            if lines[idx] == "原文":
                break
            if idx + 1 >= len(lines):
                break

            source = normalize_xinsuo_name(lines[idx])
            relation_line = lines[idx + 1]
            parsed = _parse_relation_line(relation_line)
            entries.append({
                "source": source,
                **parsed,
            })
            idx += 2

    return entries


def parse_duizhi_docx(docx_path: str | Path) -> list[dict[str, str | bool]]:
    """从 docx 文件解析心所对治关系列表。"""
    paragraphs = extract_paragraphs(docx_path)
    return parse_duizhi_paragraphs(paragraphs)
