"""从心所种子库 Word 文档（表格）解析默认权重。"""

import re
from pathlib import Path

from mental_world_model.engine.loader.docx_reader import extract_tables
from mental_world_model.utils.xinsuo_names import normalize_xinsuo_name

_WEIGHT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_weight_cell(raw: str) -> float:
    """解析单元格中的种子分数（容忍 Markdown 加粗等格式）。"""
    cleaned = raw.strip().replace("*", "").replace(" ", "")
    match = _WEIGHT_PATTERN.search(cleaned)
    if not match:
        raise ValueError(f"无法解析种子分数：{raw!r}")
    return float(match.group(0))


def _is_seed_header_row(row: list[str]) -> bool:
    joined = "".join(cell.replace(" ", "") for cell in row)
    return "心所" in joined and ("种子" in joined or "分数" in joined or "权重" in joined)


def _pick_seed_table(tables: list[list[list[str]]]) -> list[list[str]]:
    if not tables:
        raise ValueError("种子库 docx 中未找到表格")
    for table in tables:
        if table and _is_seed_header_row(table[0]):
            return table
    return tables[0]


def parse_seeds_table_rows(rows: list[list[str]]) -> dict:
    """将种子库表格解析为 xinsuo_seeds.json 结构。"""
    if not rows:
        raise ValueError("种子库表格为空")

    start_index = 1 if _is_seed_header_row(rows[0]) else 0
    categories: list[dict] = []
    category_index: dict[str, dict] = {}
    current_category: str | None = None

    for row in rows[start_index:]:
        cells = [cell.strip() for cell in row]
        while len(cells) < 3:
            cells.append("")

        category_cell, name_cell, weight_cell = cells[0], cells[1], cells[2]
        if category_cell:
            current_category = category_cell.strip("* ").strip()

        name = normalize_xinsuo_name(name_cell.strip("* ").strip())
        if not name:
            continue
        if current_category is None:
            current_category = "未分类"

        if current_category not in category_index:
            category = {"name": current_category, "entries": []}
            category_index[current_category] = category
            categories.append(category)

        category_index[current_category]["entries"].append(
            {"name": name, "weight": _parse_weight_cell(weight_cell)}
        )

    if not categories:
        raise ValueError("种子库表格未解析到任何心所条目")

    return {
        "version": 1,
        "description": "默认心所种子库权重（W_seed）",
        "categories": categories,
    }


def parse_seeds_docx(docx_path: str | Path) -> dict:
    """从 docx 文件解析心所种子库。"""
    tables = extract_tables(docx_path)
    table = _pick_seed_table(tables)
    return parse_seeds_table_rows(table)
