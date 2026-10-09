"""raw → normalized 同步与种子库解析测试。"""

import json
import os
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from mental_world_model.engine.loader.definition_loader import (
    AmbiguousDocxError,
    DefinitionLoader,
)
from mental_world_model.engine.loader.seed_reader import (
    parse_seeds_docx,
    parse_seeds_table_rows,
)
from mental_world_model.engine.seeds.config import SEED_LIBRARY_ENABLED
from mental_world_model.model import MentalWorldModel


def _sample_seed_library() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "definitions" / "seeds" / "xinsuo_seeds.json").read_text(encoding="utf-8"))


def _make_docx_with_table(path: Path, rows: list[list[str]]) -> None:
    table_rows_xml = []
    for row in rows:
        cells_xml = "".join(
            f"<w:tc><w:p><w:r><w:t>{escape(cell)}</w:t></w:r></w:p></w:tc>"
            for cell in row
        )
        table_rows_xml.append(f"<w:tr>{cells_xml}</w:tr>")
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      {''.join(table_rows_xml)}
    </w:tbl>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            "</Types>",
        )
        archive.writestr("word/document.xml", document_xml)


class TestSeedReader:
    def test_parse_seeds_table_rows(self):
        rows = [
            ["类别", "心所", "默认种子分数"],
            ["善", "信", "0.60"],
            ["", "精进", "0.55"],
            ["根本烦恼", "无明（痴）", "0.55"],
        ]
        data = parse_seeds_table_rows(rows)
        assert data["categories"][0]["name"] == "善"
        assert data["categories"][0]["entries"][0] == {"name": "信", "weight": 0.6}
        assert data["categories"][1]["entries"][0]["name"] == "无明"
        assert data["categories"][1]["entries"][0]["weight"] == 0.55

    def test_parse_seeds_docx(self, tmp_path):
        docx_path = tmp_path / "心所种子库.docx"
        _make_docx_with_table(
            docx_path,
            [
                ["类别", "心所", "默认种子分数"],
                ["善", "信", "0.60"],
                ["", "精进", "0.55"],
            ],
        )
        data = parse_seeds_docx(docx_path)
        assert len(data["categories"]) == 1
        assert data["categories"][0]["entries"][1]["name"] == "精进"


class TestDefinitionRawSync:
    def test_sync_when_raw_is_newer_than_normalized(self, tmp_path):
        definitions = tmp_path
        raw_dir = definitions / "raw"
        seeds_dir = definitions / "seeds"
        raw_dir.mkdir()
        seeds_dir.mkdir()

        normalized_path = seeds_dir / "xinsuo_seeds.json"
        normalized_path.write_text(json.dumps({"weights": {"信": 0.1}}), encoding="utf-8")
        past = time.time() - 3600
        os.utime(normalized_path, (past, past))

        raw_payload = _sample_seed_library()
        raw_path = raw_dir / "心所种子库.json"
        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False), encoding="utf-8")

        loader = DefinitionLoader()
        loaded = loader.load_xinsuo_seed_definitions(definitions)
        assert loaded["categories"][0]["entries"][0]["name"] == "信"

        synced = json.loads(normalized_path.read_text(encoding="utf-8"))
        assert synced["categories"][0]["entries"][0]["weight"] == 0.6

    def test_use_normalized_when_newer_than_raw(self, tmp_path):
        definitions = tmp_path
        raw_dir = definitions / "raw"
        seeds_dir = definitions / "seeds"
        raw_dir.mkdir()
        seeds_dir.mkdir()

        normalized_path = seeds_dir / "xinsuo_seeds.json"
        normalized_payload = {
            "weights": {"信": 0.99},
        }
        normalized_path.write_text(json.dumps(normalized_payload), encoding="utf-8")

        raw_path = raw_dir / "心所种子库.json"
        raw_path.write_text(json.dumps(_sample_seed_library()), encoding="utf-8")
        past = time.time() - 3600
        os.utime(raw_path, (past, past))

        loader = DefinitionLoader()
        loaded = loader.load_xinsuo_seed_definitions(definitions)
        assert loaded["weights"]["信"] == 0.99

    def test_ambiguous_raw_files_raise(self, tmp_path):
        definitions = tmp_path
        raw_dir = definitions / "raw"
        raw_dir.mkdir()
        (raw_dir / "心所种子库.docx").write_bytes(b"")
        (raw_dir / "默认种子库.md").write_text("x", encoding="utf-8")

        loader = DefinitionLoader()
        with pytest.raises(AmbiguousDocxError, match="多个源文件"):
            loader.load_xinsuo_seed_definitions(definitions)

    @pytest.mark.skipif(not SEED_LIBRARY_ENABLED, reason="种子库已禁用")
    def test_model_loads_seed_weights_after_sync(self, tmp_path):
        definitions = tmp_path
        raw_dir = definitions / "raw"
        seeds_dir = definitions / "seeds"
        normalized_dir = definitions / "normalized"
        schemas_dir = definitions / "schemas"
        raw_dir.mkdir()
        seeds_dir.mkdir()
        normalized_dir.mkdir()
        schemas_dir.mkdir()

        project = Path(__file__).resolve().parents[1] / "definitions"
        for name in ("xinsuo.json", "xinsuo_duizhi.json", "suoyuan_xiang_si.json"):
            (normalized_dir / name).write_text(
                (project / "normalized" / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        for name in ("xinsuo.schema.json", "relation.schema.json", "suoyuan_xiang_si.schema.json"):
            (schemas_dir / name).write_text(
                (project / "schemas" / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        raw_path = raw_dir / "心所种子库.json"
        raw_path.write_text(
            json.dumps(_sample_seed_library(), ensure_ascii=False),
            encoding="utf-8",
        )

        model = MentalWorldModel.load(definitions, use_runtime_seeds=False)
        assert model.weights.get_seed_weight("信") == 0.6
