#!/usr/bin/env python3
"""从 Markdown 生成排版整齐的 Word（无 python-docx 依赖）。"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _run(text: str, *, bold: bool = False, mono: bool = False, size: int = 22, color: str | None = None) -> str:
    rpr = ["<w:rPr>"]
    if bold:
        rpr.append("<w:b/>")
    if mono:
        rpr.append(
            '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="SimSun"/>'
        )
    else:
        rpr.append(
            '<w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/>'
        )
    rpr.append(f'<w:sz w:val="{size}"/>')
    if color:
        rpr.append(f'<w:color w:val="{color}"/>')
    rpr.append("</w:rPr>")
    return (
        f"<w:r>{''.join(rpr)}"
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    )


def _p(
    runs: str,
    *,
    style: str | None = None,
    align: str | None = None,
    before: int = 0,
    after: int = 120,
    indent_left: int = 0,
    border_bottom: bool = False,
) -> str:
    ppr = ["<w:pPr>"]
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if before or after:
        ppr.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="360" w:lineRule="auto"/>')
    if indent_left:
        ppr.append(f'<w:ind w:left="{indent_left}"/>')
    if border_bottom:
        ppr.append(
            "<w:pBdr><w:bottom w:val=\"single\" w:sz=\"6\" w:space=\"1\" w:color=\"CCCCCC\"/></w:pBdr>"
        )
    ppr.append("</w:pPr>")
    return f"<w:p>{''.join(ppr)}{runs}</w:p>"


def _rich_text(text: str, *, bold_prefix: bool = False, mono: bool = False, size: int = 22) -> str:
    runs: list[str] = []
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            runs.append(_run(text[pos : match.start()], bold=bold_prefix, mono=mono, size=size))
        token = match.group(0)
        if token.startswith("**"):
            runs.append(_run(token[2:-2], bold=True, mono=mono, size=size))
        else:
            runs.append(_run(token[1:-1], mono=True, size=20))
        pos = match.end()
    if pos < len(text):
        runs.append(_run(text[pos:], bold=bold_prefix, mono=mono, size=size))
    if not runs:
        runs.append(_run(""))
    return "".join(runs)


def _title(text: str) -> str:
    return _p(_run(text, bold=True, size=36), style="Title", align="center", before=240, after=160)


def _title_rule() -> str:
    return _p("", border_bottom=True, after=280)


def _subtitle(text: str) -> str:
    return _p(
        _rich_text(text, size=20),
        align="center",
        before=0,
        after=200,
    )


def _heading1(text: str) -> str:
    return _p(_run(text, bold=True, size=28), style="Heading1", before=360, after=160)


def _heading2(text: str) -> str:
    return _p(_run(text, bold=True, size=24), style="Heading2", before=240, after=120)


def _body(text: str, *, bold: bool = False) -> str:
    return _p(_rich_text(text, bold_prefix=bold), after=100)


def _bullet(text: str) -> str:
    ppr = (
        "<w:pPr>"
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
        '<w:spacing w:after="60"/>'
        "</w:pPr>"
    )
    return f"<w:p>{ppr}{_rich_text(text)}</w:p>"


def _code_block(text: str) -> str:
    parts: list[str] = []
    for line in text.splitlines() or [""]:
        if line.strip():
            parts.append(
                _p(_run(line, mono=True, size=20, color="333333"), indent_left=360, after=40)
            )
        else:
            parts.append(_p(_run(" "), after=40))
    return "".join(parts)


def _table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    col_w = 9000 // ncols
    xml_rows: list[str] = []
    for row_idx, row in enumerate(rows):
        while len(row) < ncols:
            row.append("")
        cells: list[str] = []
        for cell in row:
            inner = _run(cell, bold=(row_idx == 0), size=20)
            tcpr = ""
            if row_idx == 0:
                tcpr = (
                    "<w:tcPr>"
                    '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>'
                    f'<w:tcW w:w="{col_w}" w:type="dxa"/>'
                    "</w:tcPr>"
                )
            else:
                tcpr = f"<w:tcPr><w:tcW w:w=\"{col_w}\" w:type=\"dxa\"/></w:tcPr>"
            cells.append(f"<w:tc>{tcpr}<w:p>{inner}</w:p></w:tc>")
        xml_rows.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="5000" w:type="pct"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:color="BBBBBB"/>'
        '<w:left w:val="single" w:sz="4" w:color="BBBBBB"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="BBBBBB"/>'
        '<w:right w:val="single" w:sz="4" w:color="BBBBBB"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="DDDDDD"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="DDDDDD"/>'
        "</w:tblBorders>"
        "</w:tblPr>"
        f"{''.join(xml_rows)}"
        "</w:tbl>"
    )


def _parse_table_row(line: str) -> list[str] | None:
    line = line.strip()
    if not line.startswith("|") or line.count("|") < 2:
        return None
    return [c.strip() for c in line.strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) or c == "" for c in cells)


def md_to_docx_body(md_path: Path) -> str:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    i = 0
    in_code = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []
    title_done = False

    while i < len(lines):
        line = lines[i]

        if in_code:
            if line.strip().startswith("```"):
                body.append(_code_block("\n".join(code_lines)))
                body.append(_p(_run(""), after=80))
                in_code = False
                code_lines = []
            else:
                code_lines.append(line)
            i += 1
            continue

        if line.strip().startswith("```"):
            in_code = True
            i += 1
            continue

        if line.strip() == "---":
            i += 1
            continue

        row = _parse_table_row(line)
        if row is not None:
            if not _is_separator_row(row):
                table_rows.append(row)
            i += 1
            continue

        if table_rows:
            body.append(_table(table_rows))
            body.append(_p(_run(""), after=120))
            table_rows = []

        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            body.append(_title(line[2:].strip()))
            body.append(_title_rule())
            title_done = True
            i += 1
            continue

        if line.startswith("> "):
            body.append(_subtitle(line[2:].strip()))
            i += 1
            continue

        if line.startswith("## "):
            body.append(_heading1(line[3:].strip()))
            i += 1
            continue

        if line.startswith("### "):
            body.append(_heading2(line[4:].strip()))
            i += 1
            continue

        if line.strip().startswith("- "):
            body.append(_bullet(line.strip()[2:].strip()))
            i += 1
            continue

        if re.match(r"^\d+\.\s", line.strip()):
            text = re.sub(r"^\d+\.\s", "", line.strip())
            body.append(_bullet(text))
            i += 1
            continue

        if line.strip().startswith("**") and line.strip().endswith("**"):
            body.append(_body(line.strip()[2:-2], bold=True))
            i += 1
            continue

        body.append(_body(line.strip()))
        i += 1

    if table_rows:
        body.append(_table(table_rows))

    if not title_done:
        body.insert(0, _title(md_path.stem))

    return "".join(body)


def write_docx(md_path: Path, docx_path: Path) -> None:
    body = md_to_docx_body(md_path)
    numbering = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:numbering xmlns:w="{W_NS}">
  <w:abstractNum w:abstractNumId="0">
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="•"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
      <w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol" w:hint="default"/></w:rPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""
    styles = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/>
        <w:sz w:val="22"/>
        <w:lang w:val="zh-CN" w:eastAsia="zh-CN"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault><w:pPr/></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="Title" w:default="0">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:rPr><w:sz w:val="36"/><w:b/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1" w:default="0">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="360" w:after="160"/></w:pPr>
    <w:rPr><w:sz w:val="28"/><w:b/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2" w:default="0">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:sz w:val="24"/><w:b/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
</w:styles>"""
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/numbering.xml", numbering)
        archive.writestr("word/document.xml", document_xml)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "definitions/raw/心智世界模型原理.md"
    docx_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "definitions/raw/心智世界模型原理.docx"
    write_docx(md_path, docx_path)
    print(f"Wrote {docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
