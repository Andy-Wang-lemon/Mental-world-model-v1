"""Golden test cases 回归测试。"""

import json
import re
from pathlib import Path

import pytest

from mental_world_model.engine.scoring.prompts import (
    is_group_suoyuan_stage_content,
    is_xiandai_stage_content,
    is_ye_stage_content,
)
from mental_world_model.engine.scoring.suoyuan_groups import is_group_suoyuan_schema
from mental_world_model.llm.mock_client import MockLlmClient, _group_xinsuo_names_from_content
from mental_world_model.llm.message_utils import user_message_content
from mental_world_model.model import MentalWorldModel


CASES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "test_cases"

_DEFAULT_EXTRACTION = {
    "suoyuan": {"text": "mock所缘境"},
    "xiang": {"text": "mock想", "has_direct_evidence": True},
    "si": {"text": "mock思", "has_direct_evidence": True},
}


def _load_cases() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CASES_DIR.glob("case_*.json"))
    ]


def _parse_xinsuo_name(content: str) -> str | None:
    marker = "心所名称："
    if marker not in content:
        return None
    return content.split(marker, 1)[1].split("\n", 1)[0].strip()


def _parse_conflict_pair(content: str) -> tuple[str, str] | None:
    if "以下两个心所不能同时成立" not in content:
        return None
    names = re.findall(r"【([^】]+)】", content)
    if len(names) >= 2:
        return names[0], names[1]
    return None


def _reconcile_winner(
    reconcile: dict[str, str],
    name_a: str,
    name_b: str,
) -> str | None:
    for key, winner in reconcile.items():
        parts = key.split("|")
        if len(parts) != 2:
            continue
        if {parts[0].strip(), parts[1].strip()} == {name_a, name_b}:
            return winner
    return None


def _field_from_extraction(field: dict) -> dict:
    if field.get("has_direct_evidence"):
        return {
            "text": field["text"],
            "source": "原文",
            "confidence": 1.0,
        }
    return {
        "text": field["text"],
        "source": "推理",
        "confidence": float(field.get("confidence", 0.6)),
    }


def _build_mock_handler(case: dict):
    mock = case.get("mock", {})
    extraction = mock.get("extraction", _DEFAULT_EXTRACTION)
    elevated_scores: dict[str, float] = mock.get("elevated_scores", {})
    stage_scores: dict[str, dict] = mock.get("stage_scores", {})
    reconcile: dict[str, str] = mock.get("reconcile", {})
    default_score = float(mock.get("default_score", 0.0))

    def score_for(name: str) -> float:
        return float(elevated_scores.get(name, default_score))

    def stage_score_for(name: str, column: str) -> float:
        stage = stage_scores.get(name, {})
        base = score_for(name)
        return float(stage.get(column, base))

    def handler(messages, schema):
        content = user_message_content(messages)
        props = schema.get("properties", {}) if schema else {}

        if "suoyuan" in props and "xiang" not in props:
            return {"suoyuan": extraction["suoyuan"]}

        if "relevance" in props:
            name = _parse_xinsuo_name(content)
            if name and name in elevated_scores:
                return {"relevance": 0.9}
            return {"relevance": 0.1}

        if "has_explicit_xiang" in props:
            xiang_field = extraction.get("xiang", {})
            si_field = extraction.get("si", {})
            has_x = bool(xiang_field.get("has_direct_evidence"))
            has_s = bool(si_field.get("has_direct_evidence"))
            return {
                "has_explicit_xiang": has_x,
                "xiang": (
                    _field_from_extraction(xiang_field)
                    if has_x
                    else {"text": "", "source": "原文", "confidence": 1.0}
                ),
                "has_explicit_si": has_s,
                "si": (
                    _field_from_extraction(si_field)
                    if has_s
                    else {"text": "", "source": "原文", "confidence": 1.0}
                ),
            }

        if "xiang" in props and "si" in props:
            return {
                "xiang": _field_from_extraction(extraction["xiang"]),
                "si": _field_from_extraction(extraction["si"]),
            }

        if schema and "activations" in props:
            names = []
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and "（所缘相似度" in stripped:
                    names.append(stripped[2:].split("（", 1)[0].strip())
            return {
                "activations": {
                    name: score_for(name) if name else default_score
                    for name in names
                }
            }

        if schema and "winner" in props:
            pair = _parse_conflict_pair(content)
            if pair is not None:
                winner = _reconcile_winner(reconcile, pair[0], pair[1])
                if winner is not None:
                    return {"winner": winner, "reason": "mock golden case"}
            return {"winner": "none", "reason": "mock default"}

        if schema and "predicted_behavior" in props:
            return {
                "predicted_behavior": mock.get(
                    "predicted_behavior", "mock预测行为"
                ),
                "prediction_reason": mock.get(
                    "prediction_reason", "mock参考心所：信"
                ),
            }

        if is_group_suoyuan_schema(schema) or is_group_suoyuan_stage_content(content):
            names = _group_xinsuo_names_from_content(content)
            return {
                "scores": {
                    name: {"score": score_for(name), "reason": "mock golden case"}
                    for name in names
                }
            }

        name = _parse_xinsuo_name(content)
        if is_ye_stage_content(content):
            score = stage_score_for(name, "ye") if name else default_score
            return {"score": score, "reason": "mock golden case"}
        if is_xiandai_stage_content(content):
            score = stage_score_for(name, "xiandai") if name else default_score
            return {"score": score, "reason": "mock golden case"}

        score = score_for(name) if name is not None else default_score
        return {"score": score, "reason": "mock golden case"}

    return handler


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_golden_case(case: dict, tmp_path, monkeypatch):
    import mental_world_model.engine.perception.config as perception_config

    monkeypatch.setattr(perception_config, "COGNITION_BRANCHING_ENABLED", False)
    root = Path(__file__).resolve().parents[1]
    model = MentalWorldModel.load(
        root / "definitions",
        seed_state_path=tmp_path / f"{case['id']}_seed.json",
    )
    llm = MockLlmClient(handler=_build_mock_handler(case))
    result = model.evaluate_from_yuanwen(case["yuanwen"], llm)
    assertions = case["assertions"]
    results_by_name = {item.name: item for item in result.stage_results}

    for name in assertions.get("nonzero_final", []):
        assert result.scores_final.get(name, 0.0) > 0, name

    for name in assertions.get("zero_final", []):
        assert result.scores_final.get(name, 0.0) == 0.0, name

    for name in assertions.get("skipped", []):
        assert name in results_by_name, name
        assert results_by_name[name].skipped, name

    if "extracted_suoyuan_confidence" in assertions:
        assert result.extracted is not None
        assert (
            result.extracted.suoyuan.confidence
            == assertions["extracted_suoyuan_confidence"]
        )

    report = result.to_report_dict(include_explanations=True)
    assert "score_breakdowns" in report
    assert "score_explanations" in report
    assert len(model.graph.node_ids()) == 40
