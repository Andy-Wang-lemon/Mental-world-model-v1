"""MWM 结果写回 summary.txt 时，只改心智世界模型，不动其它模型。"""

from __future__ import annotations

import json
from pathlib import Path

from dyntom_adapter.cli import build_parser as dyntom_parser
from dyntom_adapter.summary_txt import update_summary as update_dyntom
from rectom_adapter.cli import build_parser as rectom_parser
from rectom_adapter.summary_txt import update_summary as update_rectom

_ROOT = Path(__file__).resolve().parents[1] / "reports"


def test_defaults_point_at_existing_report_folders():
    dyntom = dyntom_parser().parse_args(["run", "--input", "x.json"])
    rectom = rectom_parser().parse_args(["run", "--input", "x.json"])
    assert str(dyntom.output).endswith("reports/dyntom/dyntom mwm")
    assert str(rectom.output).endswith("reports/rectom/mwm")
    baseline = dyntom_parser().parse_args(["baseline", "--input", "x.json", "--output", "elsewhere"])
    assert baseline.output == "elsewhere"


def test_dyntom_summary_replaces_only_mwm(tmp_path):
    result = tmp_path / "reports" / "dyntom" / "dyntom mwm"
    trial = result / "trial9"
    trial.mkdir(parents=True)
    row = {
        "question_id": "type_d_why_1",
        "target_dimension": "belief",
        "correct": True,
        "manual_correct": True,
    }
    (trial / "answers.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = tmp_path / "reports" / "dyntom" / "summary.txt"
    summary.write_text(
        "DynToM 准确度总结\n\n"
        "范围：trial497、trial235。每条 71 题，合计 142 题。\n"
        "其余说明保留。\n\n"
        "合计\n"
        "心智世界模型：1/2 = 0.5000\n"
        "Gemini gemini-3.8-flash：113/142 = 0.7958\n\n"
        "心智世界模型（reports/dyntom/dyntom mwm）\n"
        "旧内容\n\n"
        "Gemini gemini-3.8-flash（reports/dyntom/dyntom_gemini）\n"
        "trial235：55/71 = 0.7746\n",
        encoding="utf-8",
    )
    update_dyntom(result, summary)
    text = summary.read_text(encoding="utf-8")
    assert "trial235：55/71 = 0.7746" in text
    assert "其余说明保留。" in text
    assert "旧内容" not in text
    assert "心智世界模型：1/1 = 1.0000" in text
    assert "trial9：1/1 = 1.0000（其中 1 题为人工改判：why_1）" in text
    assert "信念 1/1" in text
    assert "why 1/1 = 1.0000" in text
    assert text.index("心智世界模型：") < text.index("Gemini gemini-3.8-flash：")


def test_rectom_summary_keeps_other_models_and_skips_unfinished(tmp_path):
    result = tmp_path / "reports" / "rectom" / "mwm"
    done = result / "3_pred_rec" / "0002_d474_u5"
    done.mkdir(parents=True)
    (done / "answer.json").write_text(
        json.dumps(
            {
                "index": 2,
                "item_id": "d474_u5",
                "selector_correct": False,
                "correct": False,
                "manual_judgment": "错",
            }
        ),
        encoding="utf-8",
    )
    pending = result / "4_pred_seeker" / "0013_d895_u1" / "runtime"
    pending.mkdir(parents=True)
    summary = tmp_path / "reports" / "rectom" / "summary.txt"
    summary.write_text(
        "RecToM 准确度总结\n\n"
        "范围：旧范围。\n"
        "推荐者数据集全量 2098 题，寻求者数据集全量 2149 题。\n\n"
        "合计\n"
        "心智世界模型：1/1 = 1.0000\n"
        "Grok grok-4.3：34/40 = 0.8500\n\n"
        "心智世界模型（reports/rectom/mwm）\n"
        "旧内容\n\n"
        "Grok grok-4.3（reports/rectom/rectom_grok）\n"
        "推荐者 3_pred_rec：16/20 = 0.8000\n",
        encoding="utf-8",
    )
    update_rectom(result, summary)
    text = summary.read_text(encoding="utf-8")
    assert "推荐者 3_pred_rec：16/20 = 0.8000" in text
    assert "推荐者数据集全量 2098 题" in text
    assert "旧内容" not in text
    assert "心智世界模型：0/1 = 0.0000" in text
    assert "#2 d474_u5（人工判定错）" in text
    assert "未完成、不计入：#13 d895_u1" in text
    assert text.index("Grok grok-4.3：") < text.index("心智世界模型：")


def test_saved_dyntom_still_totals_129():
    from dyntom_adapter.summary_txt import _trials

    trials = _trials(_ROOT / "dyntom" / "dyntom mwm")
    assert sum(item["total"] for item in trials) == 142
    assert sum(item["correct"] for item in trials) == 129
    by_id = {item["trial_id"]: item for item in trials}
    assert by_id["trial235"]["manual"] == ["why_4", "why_12", "why_14", "c_how_7", "what_8"]
    assert by_id["trial235"]["kinds"]["how"] == (17, 19)


def test_saved_rectom_counts_answer_files_only():
    from rectom_adapter.summary_txt import _groups

    groups = {item["name"]: item for item in _groups(_ROOT / "rectom" / "mwm")}
    rec = groups["3_pred_rec"]["rows"]
    seeker = groups["4_pred_seeker"]["rows"]
    assert len(rec) == 20
    assert sum(row.get("correct") is True for row in rec) == 18
    assert len(seeker) == 19
    assert sum(row.get("correct") is True for row in seeker) == 19
    assert groups["4_pred_seeker"]["unfinished"] == ["#13 d895_u1"]
