"""读取 DynToM，只放出建模允许看见的字段。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from dyntom_adapter.guard import assert_model_visible

_SCENARIO_RE = re.compile(r"(\d+)")


@dataclass
class Scenario:
    index: int
    background: str
    dialogue_text: str


@dataclass
class QuestionItem:
    question_id: str
    question: str
    options: list[str]
    true_answer: str
    question_type: str


@dataclass
class TrialStory:
    trial_id: str
    main_character: str
    characters_information: str
    relationships: str
    social_setting: str
    social_setting_type: str
    scenarios: list[Scenario] = field(default_factory=list)

    def to_public_dict(self) -> dict:
        """审计用。不含 sketch、心理标注、题目和答案。"""
        return {
            "trial_id": self.trial_id,
            "main_character": self.main_character,
            "social_setting": self.social_setting,
            "social_setting_type": self.social_setting_type,
            "characters_information": self.characters_information,
            "relationships": self.relationships,
            "scenarios": [
                {
                    "scenario": item.index,
                    "background": item.background,
                    "dialogue": item.dialogue_text,
                }
                for item in self.scenarios
            ],
        }


def build_role_context(story: TrialStory) -> str:
    parts = [f"分析对象：{story.main_character}"]
    setting = " ".join(
        part for part in (story.social_setting, story.social_setting_type) if part
    ).strip()
    if setting:
        parts.append(f"社会场景：{setting}")
    if story.characters_information.strip():
        parts.append("人物信息：\n" + story.characters_information.strip())
    if story.relationships.strip():
        parts.append("人物关系：\n" + story.relationships.strip())
    text = "\n\n".join(parts)
    assert_model_visible(text, where="role_context")
    return text


def build_scenario_text(scenario: Scenario) -> str:
    text = (
        f"Scenario {scenario.index}:\n"
        f"{scenario.background.strip()}\n\n"
        f"Dialogue:\n"
        f"{scenario.dialogue_text.strip()}"
    )
    assert_model_visible(text, where=f"scenario {scenario.index}")
    return text


def render_story(story: TrialStory) -> str:
    """LLM baseline 可见的完整故事。不含标注和选项。"""
    chunks = [build_role_context(story), ""]
    for scenario in story.scenarios:
        chunks.append(build_scenario_text(scenario))
        chunks.append("")
    return "\n".join(chunks).strip() + "\n"


def format_dialogue(dialogue: object) -> str:
    if dialogue is None:
        return ""
    if isinstance(dialogue, str):
        return dialogue.strip()
    lines: list[str] = []
    turns = [dialogue] if isinstance(dialogue, dict) else dialogue
    if not isinstance(turns, list):
        return str(dialogue).strip()
    for turn in turns:
        if isinstance(turn, str):
            if turn.strip():
                lines.append(turn.strip())
            continue
        if isinstance(turn, dict):
            for speaker, utterance in turn.items():
                lines.append(f"{speaker}: {utterance}")
            continue
        lines.append(str(turn))
    return "\n".join(lines)


def format_relationships(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        lines = []
        for key, value in raw.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines).strip()
    if isinstance(raw, list):
        return "\n".join(str(item) for item in raw).strip()
    return str(raw).strip()


def _scenario_index(key: object) -> int | None:
    match = _SCENARIO_RE.search(str(key))
    if not match:
        return None
    return int(match.group(1))


def _relationships_only(stage: dict) -> str:
    """关系在 sketch 里，只取这一项，其余标注丢弃。"""
    sketch = stage.get("sketch")
    if not isinstance(sketch, dict):
        return ""
    return format_relationships(sketch.get("relationships among characters"))


def parse_stage(stage: dict, *, trial_id: str) -> TrialStory:
    story_map = stage.get("story") if isinstance(stage.get("story"), dict) else stage
    scenarios: list[Scenario] = []
    if isinstance(story_map, dict):
        ranked: list[tuple[int, dict]] = []
        for key, value in story_map.items():
            if not isinstance(value, dict):
                continue
            if "background" not in value and "dialogue" not in value:
                continue
            index = _scenario_index(key)
            if index is None:
                continue
            ranked.append((index, value))
        for index, value in sorted(ranked, key=lambda item: item[0]):
            scenarios.append(
                Scenario(
                    index=index,
                    background=str(value.get("background") or "").strip(),
                    dialogue_text=format_dialogue(value.get("dialogue")),
                )
            )
    if not scenarios:
        raise ValueError(f"{trial_id} 没有可用的 scenario")
    return TrialStory(
        trial_id=trial_id,
        main_character=str(stage.get("main character") or "").strip(),
        characters_information=str(stage.get("characters information") or "").strip(),
        relationships=_relationships_only(stage),
        social_setting=str(stage.get("social setting") or "").strip(),
        social_setting_type=str(stage.get("social setting type") or "").strip(),
        scenarios=scenarios,
    )


def parse_questions(raw: dict) -> list[QuestionItem]:
    if not isinstance(raw, dict):
        raise ValueError("题目文件必须是 question_id → 题目对象 的字典")
    items: list[QuestionItem] = []
    for question_id, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        question = str(payload.get("question") or "").strip()
        options = payload.get("options")
        if not question or not isinstance(options, list) or not options:
            continue
        items.append(
            QuestionItem(
                question_id=str(question_id),
                question=question,
                options=[str(option) for option in options],
                true_answer=str(payload.get("true answer") or payload.get("true_answer") or "")
                .strip()
                .lower(),
                question_type=str(
                    payload.get("question type") or payload.get("question_type") or ""
                ).strip(),
            )
        )
    if not items:
        raise ValueError("题目文件里没有带 options 的题")
    return items


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _question_path(folder: Path) -> Path | None:
    preferred = folder / "question_new.json"
    if preferred.is_file():
        return preferred
    fallback = folder / "question.json"
    if fallback.is_file():
        return fallback
    return None


def _is_stage(payload: object) -> bool:
    return isinstance(payload, dict) and (
        "story" in payload or "main character" in payload or "characters information" in payload
    )


def iter_trials(path: Path) -> Iterator[tuple[TrialStory, list[QuestionItem]]]:
    """支持三种输入：trial 目录、含多个 trial 子目录的根目录、官方合并 JSON。"""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到 DynToM 输入：{path}")

    if path.is_dir() and (path / "story.json").is_file():
        yield _load_trial_dir(path, trial_id=path.name)
        return

    if path.is_dir():
        children = sorted(
            [item for item in path.iterdir() if item.is_dir() and (item / "story.json").is_file()],
            key=lambda item: _trial_sort_key(item.name),
        )
        if not children:
            raise FileNotFoundError(
                f"{path} 下没有 story.json。请指向 DynToM 的 data/script/data 目录。"
            )
        for child in children:
            yield _load_trial_dir(child, trial_id=child.name)
        return

    payload = _read_json(path)
    yield from _iter_combined(payload, default_id=path.stem)


def _load_trial_dir(folder: Path, *, trial_id: str) -> tuple[TrialStory, list[QuestionItem]]:
    stage = _read_json(folder / "story.json")
    if not isinstance(stage, dict):
        raise ValueError(f"{folder / 'story.json'} 不是对象")
    question_file = _question_path(folder)
    if question_file is None:
        raise FileNotFoundError(f"{folder} 缺少 question_new.json")
    questions_raw = _read_json(question_file)
    if not isinstance(questions_raw, dict):
        raise ValueError(f"{question_file} 不是对象")
    return parse_stage(stage, trial_id=trial_id), parse_questions(questions_raw)


def _iter_combined(payload: object, *, default_id: str) -> Iterator[tuple[TrialStory, list[QuestionItem]]]:
    if isinstance(payload, dict) and "stage" in payload and "question" in payload:
        stage = payload["stage"]
        if not isinstance(stage, dict):
            raise ValueError("stage 必须是对象")
        yield parse_stage(stage, trial_id=default_id), parse_questions(payload["question"])
        return
    if isinstance(payload, dict) and _is_stage(payload):
        raise ValueError(f"{default_id} 只有故事、没有 question。请使用 question_new.json 或合并 JSON。")
    if not isinstance(payload, dict):
        raise ValueError("合并 JSON 必须是 trial_id → {stage, question}")
    for trial_id in sorted(payload.keys(), key=_trial_sort_key):
        item = payload[trial_id]
        if not isinstance(item, dict) or "stage" not in item or "question" not in item:
            continue
        stage = item["stage"]
        if not isinstance(stage, dict):
            continue
        yield parse_stage(stage, trial_id=str(trial_id)), parse_questions(item["question"])


def _trial_sort_key(trial_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", trial_id)
    number = int(match.group(1)) if match else 10**12
    return number, trial_id
