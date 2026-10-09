"""把之前评过的 DynToM、RecToM 条目收成同一结构。不读取标准答案进提示。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dyntom_adapter.loader import build_scenario_text, parse_questions, parse_stage, render_story
from dyntom_adapter.selector import normalize_gold
from rectom_adapter.loader import load_one

EVAL_TRIALS = ("trial497", "trial235")
EVAL_RECTOM_END = 20
RECTOM_FILES = ("3_pred_rec.json", "4_pred_seeker.json")


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    dataset: str
    context: str
    events: tuple[str, ...]
    question: str
    choices: tuple[str, ...]
    target_character: str
    gold: tuple[str, ...]
    source: str


def data_root() -> Path:
    return Path(__file__).resolve().parents[3] / "公开数据集"


def load_split(
    dataset: str,
    *,
    data_dir: Path | None = None,
    trials: tuple[str, ...] | None = None,
    rectom_end: int | None = None,
) -> list[BenchmarkItem]:
    root = data_dir or data_root()
    if dataset == "dyntom":
        return load_dyntom(root / "DynToM.json", trials=trials or EVAL_TRIALS)
    if dataset == "rectom":
        return load_rectom(root / "RecTOM", end=EVAL_RECTOM_END if rectom_end is None else rectom_end)
    raise ValueError(f"未知数据集：{dataset}")


def load_dyntom(path: Path, trials: tuple[str, ...] = EVAL_TRIALS) -> list[BenchmarkItem]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 应为 trial 字典")
    items: list[BenchmarkItem] = []
    for trial_id in trials:
        raw = payload.get(trial_id)
        if not isinstance(raw, dict) or "stage" not in raw or "question" not in raw:
            raise FileNotFoundError(f"{path} 里没有 {trial_id}")
        stage = raw["stage"]
        if not isinstance(stage, dict):
            raise ValueError(f"{trial_id} 的 stage 不是对象")
        story = parse_stage(stage, trial_id=trial_id)
        questions = parse_questions(raw["question"])
        context = render_story(story).strip()
        events = tuple(build_scenario_text(scenario).strip() for scenario in story.scenarios)
        target = story.main_character.strip()
        for question in questions:
            gold = normalize_gold(question.true_answer, question.options)
            items.append(
                BenchmarkItem(
                    id=f"{trial_id}__{question.question_id}",
                    dataset="dyntom",
                    context=context,
                    events=events,
                    question=question.question.strip(),
                    choices=tuple(question.options),
                    target_character=target,
                    gold=(gold,) if gold else (),
                    source=trial_id,
                )
            )
    return items


def load_rectom(folder: Path, end: int = EVAL_RECTOM_END) -> list[BenchmarkItem]:
    if end < 1:
        raise ValueError("RecToM 结束下标至少为 1")
    items: list[BenchmarkItem] = []
    for name in RECTOM_FILES:
        path = folder / name
        stem = path.stem
        for index in range(end):
            raw = load_one(path, index)
            turns = tuple(line.strip() for line in raw.utterance_context.splitlines() if line.strip())
            items.append(
                BenchmarkItem(
                    id=f"{stem}__{index:04d}_{raw.item_id}",
                    dataset="rectom",
                    context=raw.utterance_context.strip(),
                    events=turns,
                    question=raw.question,
                    choices=raw.choices,
                    target_character=raw.focus_person,
                    gold=raw.gold,
                    source=stem,
                )
            )
    return items
