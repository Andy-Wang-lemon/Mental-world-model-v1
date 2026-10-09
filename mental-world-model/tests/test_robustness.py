"""健壮性与边界情况测试。"""

import json
from pathlib import Path

import pytest

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.loader.definition_loader import (
    AmbiguousDocxError,
    DefinitionLoader,
)
from mental_world_model.engine.reconciliation.reconciler import DuizhiReconciler
from mental_world_model.engine.scoring.context import ScoringContext
from mental_world_model.engine.scoring.pipeline import XinsuoScoringPipeline
from mental_world_model.engine.scoring.results import FEN_NAME, HEN_NAME, NAO_NAME
from mental_world_model.engine.validator.validation_result import DefinitionValidationError
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel
from tests.test_extraction import make_test_scoring_context


class TestLoadValidationBeforeParse:
    def test_missing_xinsuo_field_raises_definition_error(self, tmp_path):
        definitions = tmp_path / "definitions"
        (definitions / "normalized").mkdir(parents=True)
        (definitions / "schemas").mkdir(parents=True)
        for name in ("xinsuo.schema.json", "relation.schema.json", "suoyuan_xiang_si.schema.json"):
            src = Path(__file__).resolve().parents[1] / "definitions/schemas" / name
            (definitions / "schemas" / name).write_text(src.read_text(encoding="utf-8"))

        bad_xinsuo = [{"name": "信", "suoyuan": "x", "shengqi_nian": "x", "ye": "x"}]
        (definitions / "normalized/xinsuo.json").write_text(
            json.dumps(bad_xinsuo, ensure_ascii=False), encoding="utf-8"
        )
        root = Path(__file__).resolve().parents[1] / "definitions/normalized"
        for fname in ("xinsuo_duizhi.json", "suoyuan_xiang_si.json"):
            (definitions / "normalized" / fname).write_text(
                (root.parent / "normalized" / fname).read_text(encoding="utf-8")
            )

        with pytest.raises(DefinitionValidationError):
            MentalWorldModel.load(definitions)


class TestFenHenNaoMissingNames:
    def test_missing_hen_in_table_skips_chain(self):
        partial = [
            Xinsuo("信", "s", "s", "s", "s"),
            Xinsuo(FEN_NAME, "s", "s", "s", "s"),
            Xinsuo(NAO_NAME, "s", "s", "s", "s"),
        ]
        pipeline = XinsuoScoringPipeline(
            MockLlmClient(default_response={"score": 0.8})
        )
        results = pipeline.run(partial, make_test_scoring_context())
        by_name = {item.name: item for item in results}
        assert HEN_NAME not in by_name
        assert by_name[NAO_NAME].skipped

    def test_missing_fen_skips_chain(self):
        partial = [
            Xinsuo("信", "s", "s", "s", "s"),
            Xinsuo(HEN_NAME, "s", "s", "s", "s"),
            Xinsuo(NAO_NAME, "s", "s", "s", "s"),
        ]
        pipeline = XinsuoScoringPipeline(
            MockLlmClient(default_response={"score": 0.1})
        )
        results = pipeline.run(partial, make_test_scoring_context())
        by_name = {item.name: item for item in results}
        assert FEN_NAME not in by_name
        assert by_name[HEN_NAME].skipped
        assert by_name[NAO_NAME].skipped


class TestReconcilerWinnerMatch:
    def test_normalize_winner_name(self):
        winner, note = DuizhiReconciler._match_winner("无嗔", "贪", "无瞋")
        assert winner == "无瞋"
        assert note == ""

    def test_invalid_winner_note(self):
        winner, note = DuizhiReconciler._match_winner("乱写", "信", "不信")
        assert winner is None
        assert "无效 winner" in note


class TestAmbiguousDocx:
    def test_multiple_docx_raises(self, tmp_path):
        definitions = tmp_path / "definitions"
        raw = definitions / "raw"
        raw.mkdir(parents=True)
        (raw / "心所定义_a.docx").write_bytes(b"")
        (raw / "心所定义_b.docx").write_bytes(b"")
        loader = DefinitionLoader()
        with pytest.raises(AmbiguousDocxError):
            loader.load_xinsuo_definitions(definitions)


class TestEvaluationConflictsReport:
    def test_report_has_initial_and_final_conflicts(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        result = model.evaluate(
            make_test_scoring_context(),
            MockLlmClient(default_response={"score": 0.0}),
        )
        report = result.to_report_dict()
        assert "conflicts_initial" in report
        assert "conflicts_final" in report
        assert isinstance(result.conflicts_final, list)
