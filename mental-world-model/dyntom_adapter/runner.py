"""一个 trial：五幕滚动 → 用所缘、想、行动给选项打分。题目不进入建模。"""

from __future__ import annotations

import json
from pathlib import Path

from mental_world_model.utils.atomic_io import atomic_write_json, atomic_write_text

from dyntom_adapter.audit import apply_audit, finalize_judgment
from dyntom_adapter.baseline import answer_baseline
from dyntom_adapter.loader import QuestionItem, TrialStory
from dyntom_adapter.rollout import (
    bind_fresh_continuity,
    bind_saved_continuity,
    run_scenario,
)
from dyntom_adapter.router import RouterError, bundle_for, bundle_lines, parse_route
from dyntom_adapter.selector import choose_by_scores, normalize_gold, split_option


class TrialExistsError(FileExistsError):
    """输出目录已有结果，需要 --resume 或换目录。"""


def run_trial(
    story: TrialStory,
    questions: list[QuestionItem],
    *,
    model,
    llm,
    out_dir: Path,
    resume: bool,
    question_limit: int | None,
    run_control: bool,
    placeholder_adapter: bool,
) -> dict:
    trial_dir = out_dir / _safe_name(story.trial_id)
    runtime_dir = trial_dir / "runtime"
    trial_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(trial_dir / "story_view.json", story.to_public_dict())

    if not resume and _has_progress(trial_dir):
        raise TrialExistsError(
            f"{trial_dir} 已有结果。续跑请加 --resume，或换一个 --output。"
        )

    snapshots = _ensure_states(
        story,
        model=model,
        llm=llm,
        trial_dir=trial_dir,
        runtime_dir=runtime_dir,
    )
    rows = _answer_questions(
        story.trial_id,
        questions,
        snapshots,
        llm=llm,
        trial_dir=trial_dir,
        resume=resume,
        question_limit=question_limit,
        run_control=run_control,
        placeholder_adapter=placeholder_adapter,
    )
    correct = sum(1 for row in rows if row.get("correct") is True)
    selector_correct = sum(1 for row in rows if row.get("selector_correct", row.get("letter_correct")) is True)
    rescued = sum(
        1
        for row in rows
        if row.get("selector_correct") is False and row.get("mwm_matches_gold") is True
    )
    summary = {
        "trial_id": story.trial_id,
        "scenarios": len(story.scenarios),
        "answered": len(rows),
        "selector_correct": selector_correct,
        "rescued": rescued,
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0,
    }
    atomic_write_json(trial_dir / "summary.json", summary)
    atomic_write_text(trial_dir / "report.txt", render_report(story.trial_id, rows))
    return summary


def audit_saved_answers(path: Path, llm, *, placeholder: bool = False) -> dict:
    """只核对已有结果里选题判错的题，不重跑 MWM，也不重答。"""
    path = path.expanduser().resolve()
    answer_path = path / "answers.jsonl" if path.is_dir() else path
    if not answer_path.is_file():
        raise FileNotFoundError(f"找不到答题记录：{answer_path}")
    rows = _read_jsonl(answer_path)
    checked = 0
    for row in rows:
        if "selector_correct" not in row:
            if row.get("letter_correct") is True or row.get("letter_correct") is False:
                row["selector_correct"] = row.get("letter_correct") is True
            else:
                row["selector_correct"] = row.get("correct") is True
        if row.get("selector_correct") is True:
            row["adapter_correct"] = True
            row["mwm_matches_gold"] = None
            row["audit_reason"] = ""
            row["correct"] = True
            continue
        if row.get("manual_correct") is True:
            row["adapter_correct"] = False
            row["mwm_matches_gold"] = True
            row["correct"] = True
            continue
        apply_audit(row, llm, placeholder=placeholder)
        checked += 1
        mark = "改为对" if row.get("mwm_matches_gold") else "保持错"
        print(f"  [{row.get('question_id')}] {mark} {row.get('audit_reason') or ''}", flush=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    atomic_write_text(answer_path, ("\n".join(lines) + "\n") if lines else "")
    trial_id = str(rows[0].get("trial_id") or answer_path.parent.name) if rows else answer_path.parent.name
    atomic_write_text(answer_path.parent / "report.txt", render_report(trial_id, rows))
    rescued = sum(
        1
        for row in rows
        if row.get("adapter_correct") is False and row.get("mwm_matches_gold") is True
    )
    adapter_correct = sum(1 for row in rows if row.get("adapter_correct") is True)
    final_correct = sum(1 for row in rows if row.get("correct") is True)
    summary_path = answer_path.parent / "summary.json"
    summary = {}
    if summary_path.is_file():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            summary = loaded
    summary.update(
        {
            "trial_id": trial_id,
            "answered": len(rows),
            "adapter_correct": adapter_correct,
            "rescued": rescued,
            "correct": final_correct,
            "accuracy": round(final_correct / len(rows), 4) if rows else 0,
        }
    )
    atomic_write_json(summary_path, summary)
    return {
        "trial_id": trial_id,
        "checked": checked,
        "rescued": rescued,
        "correct": final_correct,
        "total": len(rows),
    }


def run_baseline_trial(
    story: TrialStory,
    questions: list[QuestionItem],
    *,
    llm,
    out_dir: Path,
    resume: bool,
    question_limit: int | None,
) -> dict:
    trial_dir = out_dir / _safe_name(story.trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / "baseline.jsonl"
    if path.exists() and not resume:
        raise TrialExistsError(f"{path} 已存在。续跑请加 --resume，或换一个 --output。")
    done = _done_ids(path) if resume else set()
    written = 0
    correct = 0
    asked = 0
    for question in questions:
        if question.question_id in done:
            continue
        if question_limit is not None and asked >= question_limit:
            break
        asked += 1
        row = {
            "trial_id": story.trial_id,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "ground_truth": normalize_gold(question.true_answer, question.options),
        }
        try:
            route = parse_route(
                question.question_id,
                question.question,
                question_type=question.question_type,
            )
            row["target_dimension"] = route.target_dimension
            row["operation"] = route.operation
            row["scope"] = route.scope
            row["required_scenarios"] = list(route.required_scenarios)
        except RouterError as error:
            row["error"] = f"{type(error).__name__}: {error}"
        try:
            predicted = answer_baseline(llm, story, question)
            row["predicted_answer"] = predicted
            row["correct"] = predicted == row["ground_truth"]
        except Exception as error:  # noqa: BLE001
            row["predicted_answer"] = ""
            row["correct"] = False
            row["error"] = f"{type(error).__name__}: {error}"
        _append_jsonl(path, row)
        written += 1
        correct += int(row["correct"])
    return {"trial_id": story.trial_id, "answered": written, "correct": correct}


def _ensure_states(
    story: TrialStory,
    *,
    model,
    llm,
    trial_dir: Path,
    runtime_dir: Path,
) -> list[dict]:
    """只保证五幕 state 存在。答题不再做投影。"""
    total = len(story.scenarios)
    completed = _completed_scenarios(trial_dir, total)
    if completed < total:
        if model is None:
            raise RuntimeError(f"{story.trial_id} 还没有完整 state，需要加载 MWM。")
        if completed == 0:
            bind_fresh_continuity(model, runtime_dir, story.main_character)
            start = 1
        else:
            state = bind_saved_continuity(model, runtime_dir)
            actual = int(getattr(state, "event_count", 0) or 0)
            if actual != completed:
                raise RuntimeError(
                    f"{story.trial_id} 已有 {completed} 个 state，"
                    f"但连续性 event_count={actual}。请换一个输出目录。"
                )
            start = completed + 1
        for scenario in story.scenarios:
            if scenario.index < start:
                continue
            print(f"  [{story.trial_id}] scenario {scenario.index}/{total}", flush=True)
            frozen = run_scenario(model, llm, story, scenario)
            atomic_write_json(trial_dir / f"state_{scenario.index:02d}.json", frozen)
    return [
        _read_json(trial_dir / f"state_{scenario.index:02d}.json") for scenario in story.scenarios
    ]


def _answer_questions(
    trial_id: str,
    questions: list[QuestionItem],
    snapshots: list[dict],
    *,
    llm,
    trial_dir: Path,
    resume: bool,
    question_limit: int | None,
    run_control: bool,
    placeholder_adapter: bool,
) -> list[dict]:
    path = trial_dir / "answers.jsonl"
    done = _done_ids(path) if resume else set()
    rows = _read_jsonl(path) if resume else []
    asked = 0
    for question in questions:
        if question.question_id in done:
            continue
        if question_limit is not None and asked >= question_limit:
            break
        asked += 1
        row = _answer_one(
            trial_id,
            question,
            snapshots,
            llm=llm,
            run_control=run_control,
            placeholder_adapter=placeholder_adapter,
        )
        _append_jsonl(path, row)
        rows.append(row)
        if row.get("selector_correct") is False and row.get("mwm_matches_gold") is True:
            mark = "选题错，核对后改对"
        elif row.get("correct"):
            mark = "对"
        else:
            mark = "错"
        scores = row.get("option_scores") or {}
        chosen = str(row.get("predicted_answer") or "")
        score_text = f" {scores[chosen]}" if chosen in scores else ""
        print(
            f"  [{trial_id}] {question.question_id} → {chosen or '-'}{score_text} {mark}",
            flush=True,
        )
    return rows


def _answer_one(
    trial_id: str,
    question: QuestionItem,
    snapshots: list[dict],
    *,
    llm,
    run_control: bool,
    placeholder_adapter: bool,
) -> dict:
    gold = normalize_gold(question.true_answer, question.options)
    row = {
        "trial_id": trial_id,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "question": question.question,
        "ground_truth": gold,
    }
    try:
        route = parse_route(
            question.question_id,
            question.question,
            question_type=question.question_type,
        )
        bundle = bundle_for(route, snapshots)
        row.update(
            {
                "target_dimension": route.target_dimension,
                "operation": route.operation,
                "scope": route.scope,
                "required_scenarios": list(route.required_scenarios),
                "mwm_state_used": bundle_lines(bundle),
                "representation": bundle,
                "gold_output": _option_text(gold, question.options),
            }
        )
        if placeholder_adapter:
            predicted = "a" if "a" in _letters(question.options) else _first_letter(question.options)
            row["option_scores"] = {}
            row["score_reason"] = ""
        else:
            predicted, scores, reason = choose_by_scores(
                llm, bundle, question.question, question.options
            )
            row["option_scores"] = scores
            row["score_reason"] = reason
        row["predicted_answer"] = predicted
        row["adapter_output"] = _option_text(predicted, question.options)
        row["correct"] = predicted == gold
        row["letter_correct"] = row["correct"]
        if run_control:
            if placeholder_adapter:
                control = predicted
            else:
                control, _scores, _reason = choose_by_scores(
                    llm, {"scenarios": []}, question.question, question.options
                )
            row["control_answer"] = control
            row["control_correct"] = control == gold
    except (RouterError, ValueError, KeyError, RuntimeError) as error:
        row["predicted_answer"] = str(row.get("predicted_answer") or "")
        row["adapter_output"] = str(row.get("adapter_output") or "")
        row["correct"] = False
        row["error"] = f"{type(error).__name__}: {error}"
        if "gold_output" not in row:
            row["gold_output"] = _option_text(gold, question.options)
        row["letter_correct"] = False
    if row.get("selector_correct") is None and row.get("letter_correct") is None:
        predicted = str(row.get("predicted_answer") or "")
        gold_letter = str(row.get("ground_truth") or "")
        row["letter_correct"] = bool(predicted) and predicted == gold_letter
        row["selector_correct"] = row["letter_correct"]
    finalize_judgment(row, llm, placeholder=placeholder_adapter)
    return row


def render_report(trial_id: str, rows: list[dict]) -> str:
    """每次跑完留下原题、MWM 输出和 Adapter 选项，供之后对照升级。"""
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct") is True)
    lines = [
        f"trial：{trial_id}",
        f"题数：{total}",
        f"选题答对：{sum(1 for row in rows if row.get('selector_correct', row.get('letter_correct')) is True)}",
        f"核对改对：{sum(1 for row in rows if row.get('selector_correct') is False and row.get('mwm_matches_gold') is True)}",
        f"最终答对：{correct}",
        f"准确率：{round(correct / total, 4) if total else 0}",
        "",
        "唯一最高分等于标准答案时直接算对。",
        "选错的题只核对记录是否已经支持标准答案。支持则改为对，否则保持错。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"## {index}. {row.get('question_id') or ''}")
        lines.append(f"原题：{row.get('question') or ''}")
        lines.append("MWM：")
        lines.append(_format_mwm(row))
        adapter = str(row.get("adapter_output") or "").strip()
        if not adapter:
            adapter = "（没有选出选项）"
        lines.append(f"Adapter：{adapter}")
        if row.get("option_scores"):
            lines.append(f"分数：{json.dumps(row['option_scores'], ensure_ascii=False)}")
        if row.get("score_reason"):
            lines.append(f"选择说明：{row['score_reason']}")
        gold = str(row.get("gold_output") or row.get("ground_truth") or "").strip()
        lines.append(f"标准：{gold or '（无）'}")
        lines.append("结果：对" if row.get("correct") else "结果：错")
        if row.get("manual_correct") is True:
            lines.append("人工判定：对")
        if row.get("mwm_matches_gold") is True:
            lines.append(f"核对：记录已支持标准答案（{row.get('audit_level') or ''}）")
        elif row.get("mwm_matches_gold") is False:
            lines.append("核对：记录未支持标准答案，保持错")
        if row.get("audit_reason"):
            lines.append(f"核对说明：{row['audit_reason']}")
        if row.get("error"):
            lines.append(f"错误：{row['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_mwm(row: dict) -> str:
    representation = row.get("representation")
    if isinstance(representation, dict) and representation:
        return json.dumps(representation, ensure_ascii=False, indent=2)
    used = row.get("mwm_state_used")
    if isinstance(used, list) and used:
        return "\n".join(str(item) for item in used)
    return "（本题没有记下 MWM 输出）"


def _option_text(letter: str, options: list[str]) -> str:
    wanted = str(letter or "").strip().lower()
    if not wanted:
        return ""
    for option in options:
        option_letter, _body = split_option(option)
        if option_letter == wanted:
            return option.strip()
    return wanted


def _has_progress(trial_dir: Path) -> bool:
    if (trial_dir / "answers.jsonl").is_file():
        return True
    return any(trial_dir.glob("state_*.json"))


def _completed_scenarios(trial_dir: Path, total: int) -> int:
    done = 0
    for index in range(1, total + 1):
        if not (trial_dir / f"state_{index:02d}.json").is_file():
            break
        done = index
    return done


def needs_model(out_dir: Path, trial_id: str, scenario_count: int) -> bool:
    """已有完整 state 时，答题不再加载 MWM。"""
    trial_dir = out_dir / _safe_name(trial_id)
    return _completed_scenarios(trial_dir, scenario_count) < scenario_count


def _done_ids(path: Path) -> set[str]:
    return {str(row.get("question_id") or "") for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_name(trial_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in trial_id)
    return cleaned or "trial"


def _letters(options: list[str]) -> set[str]:
    from dyntom_adapter.selector import option_letters

    return option_letters(options)


def _first_letter(options: list[str]) -> str:
    letters = sorted(_letters(options))
    return letters[0] if letters else "a"
