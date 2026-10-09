"""从 Word 文档解析所缘境/想/思定义。"""

from pathlib import Path

from mental_world_model.engine.loader.docx_reader import extract_paragraphs


def parse_suoyuan_xiang_si_paragraphs(paragraphs: list[str]) -> dict[str, str]:
    """将段落解析为所缘境/想/思定义字典。"""
    lines = [line.strip() for line in paragraphs if line.strip()]
    if len(lines) < 4:
        raise ValueError("所缘境/想/思定义文档段落不足 4 段")

    suoyuan_jing = lines[0]
    xiang = lines[1]
    si = lines[2]
    summary = lines[3]

    if not suoyuan_jing.startswith("所缘境"):
        raise ValueError("第一段应为所缘境定义")
    if not xiang.startswith("想"):
        raise ValueError("第二段应为想定义")
    if not si.startswith("思"):
        raise ValueError("第三段应为思定义")

    return {
        "suoyuan_jing": suoyuan_jing,
        "xiang": xiang,
        "si": si,
        "summary": summary,
    }


def parse_suoyuan_xiang_si_docx(docx_path: str | Path) -> dict[str, str]:
    """从 docx 文件解析所缘境/想/思定义。"""
    paragraphs = extract_paragraphs(docx_path)
    return parse_suoyuan_xiang_si_paragraphs(paragraphs)
