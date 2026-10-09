"""从 Word 文档提取心所定义段落并解析为结构化数据。"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_HEADER_LINES = frozenset({
    "所缘境",
    "生起的念头",
    "业",
    "用现代心理学语言",
    "用现代心理学语言，准确翻译",
})

_MAX_NAME_LENGTH = 4


def extract_paragraphs(docx_path: str | Path) -> list[str]:
    """从 docx 文件中提取非空段落文本。"""
    path = Path(docx_path)
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    return _extract_paragraphs_from_root(root)


def _extract_paragraphs_from_root(root: ET.Element) -> list[str]:
    paragraphs: list[str] = []
    for element in root.iter(f"{_W_NS}p"):
        texts = [
            node.text
            for node in element.iter(f"{_W_NS}t")
            if node.text
        ]
        if texts:
            paragraphs.append("".join(texts))
    return paragraphs


def extract_tables(docx_path: str | Path) -> list[list[list[str]]]:
    """从 docx 文件中提取表格（每个表格为行列表，每行为单元格文本列表）。"""
    path = Path(docx_path)
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    tables: list[list[list[str]]] = []
    for table_element in root.iter(f"{_W_NS}tbl"):
        rows: list[list[str]] = []
        for row_element in table_element.iter(f"{_W_NS}tr"):
            cells: list[str] = []
            for cell_element in row_element.iter(f"{_W_NS}tc"):
                texts = [
                    node.text
                    for node in cell_element.iter(f"{_W_NS}t")
                    if node.text
                ]
                cells.append("".join(texts).strip())
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def extract_document_root(docx_path: str | Path) -> ET.Element:
    """读取 docx 的 document.xml 根节点。"""
    path = Path(docx_path)
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    return ET.fromstring(xml_bytes)


def _is_entry_name(line: str) -> bool:
    """判断一行是否为心所名称。"""
    if not line or line in _HEADER_LINES or line == "原文":
        return False
    if line.startswith("所缘") or line.startswith("心中"):
        return False
    return len(line) <= _MAX_NAME_LENGTH


def parse_xinsuo_paragraphs(paragraphs: list[str]) -> list[dict[str, str]]:
    """将段落列表解析为心所定义字典列表。"""
    entries: list[dict[str, str]] = []
    lines = [line.strip() for line in paragraphs if line.strip()]

    idx = 0
    while idx < len(lines):
        if lines[idx] != "原文":
            idx += 1
            continue

        idx += 1
        while idx < len(lines) and lines[idx] in _HEADER_LINES:
            idx += 1

        while idx < len(lines):
            if lines[idx] == "原文":
                break
            if not _is_entry_name(lines[idx]):
                break
            if idx + 4 >= len(lines):
                break

            entries.append({
                "name": lines[idx],
                "suoyuan": lines[idx + 1],
                "shengqi_nian": lines[idx + 2],
                "ye": lines[idx + 3],
                "xiandai_xinli": lines[idx + 4],
            })
            idx += 5

    return entries


def parse_xinsuo_docx(docx_path: str | Path) -> list[dict[str, str]]:
    """从 docx 文件解析心所定义列表。"""
    paragraphs = extract_paragraphs(docx_path)
    return parse_xinsuo_paragraphs(paragraphs)
