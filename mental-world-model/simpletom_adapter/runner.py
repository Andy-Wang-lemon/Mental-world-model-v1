"""一次只答 SimpleToM 行为预测的一道题。"""

from __future__ import annotations

import json
from pathlib import Path

from mental_world_model.utils.atomic_io import atomic_write_json, atomic_write_text

from simpletom_adapter.audit import review_miss
from simpletom_adapter.loader import SimpleTomItem
from simpletom_adapter.selector import choose_by_scores, option_text


class ItemExistsError(RuntimeError):
    pass


def run_mwm_item(
    item: SimpleTomItem,
    *,
    model,
    llm,
    out_dir: Path,
    dataset_name: str,
    force: bool,
) -> dict:
    folder = _item_dir(out_dir, dataset_name, item)
    answer_path = folder / "answer.json"
    if answer_path.is_file() and not force:
        raise ItemExistsError(f"{answer_path} 已存在。换 --index，或加 --force 覆盖这一题。")
    folder.mkdir(parents=True, exist_ok=True)
    from simpletom_adapter.rollout import run_item

    record = run_item(model, llm, item, folder / "runtime")
    atomic_write_json(folder / "state.json", record)
    return _finish(item, record, llm, folder, dataset_name)


def _finish(item: SimpleTomItem, record: dict, llm, folder: Path, dataset_name: str) -> dict:
    predicted, scores, reason = choose_by_scores(llm, record, item.question, list(item.choices))
    row = {
        "method": "mwm",
        "dataset": dataset_name,
        "index": item.index,
        "item_id": item.item_id,
        "scenario_name": item.scenario_name,
        "focus_person": item.focus_person,
        "story": item.story,
        "question": item.question,
        "gold": list(item.gold),
        "gold_output": "\n".join(option_text(letter, list(item.choices)) for letter in item.gold),
        "representation": record,
        "predicted_answer": predicted,
        "adapter_output": option_text(predicted, list(item.choices)),
        "option_scores": scores,
        "score_reason": reason,
    }
    row["selector_correct"] = predicted in set(item.gold)
    row["correct"] = row["selector_correct"]
    row["mwm_matches_gold"] = None
    row["audit_reason"] = ""
    if not row["selector_correct"]:
        same, level, audit_reason = review_miss(
            llm,
            question=item.question,
            record=record,
            gold_text=row["gold_output"],
        )
        row["mwm_matches_gold"] = same
        row["audit_level"] = level
        row["audit_reason"] = audit_reason
        row["correct"] = same
    atomic_write_json(folder / "answer.json", row)
    atomic_write_text(folder / "report.txt", render_report(row))
    return row


def render_report(row: dict) -> str:
    lines = [
        f"数据：{row.get('dataset') or ''}",
        f"题号：{row.get('index')}  {row.get('item_id') or ''}",
        f"焦点：{row.get('focus_person') or ''}",
        f"故事：{row.get('story') or ''}",
        f"原题：{row.get('question') or ''}",
        "MWM：",
        json.dumps(row.get("representation") or {}, ensure_ascii=False, indent=2),
        f"选择：{row.get('adapter_output') or row.get('predicted_answer') or ''}",
    ]
    if row.get("option_scores"):
        lines.append(f"分数：{json.dumps(row['option_scores'], ensure_ascii=False)}")
    if row.get("score_reason"):
        lines.append(f"选择说明：{row['score_reason']}")
    lines.append(f"标准：{row.get('gold_output') or ''}")
    lines.append("结果：对" if row.get("correct") else "结果：错")
    if row.get("manual_judgment"):
        lines.append(f"人工判定：{row['manual_judgment']}")
    if row.get("selector_correct") is False and row.get("mwm_matches_gold") is True:
        lines.append("核对：选题不在标准答案里，记录已支持标准行为，改为对")
    elif row.get("mwm_matches_gold") is False:
        lines.append("核对：记录未支持标准行为，保持错")
    if row.get("audit_reason"):
        lines.append(f"核对说明：{row['audit_reason']}")
    return "\n".join(lines).rstrip() + "\n"


def _item_dir(out_dir: Path, dataset_name: str, item: SimpleTomItem) -> Path:
    return out_dir / dataset_name / f"{item.index:04d}_{item.item_id}"
