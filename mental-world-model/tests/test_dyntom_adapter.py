"""DynToM 适配层：不触碰模型内部，只检查加载、冻结、路由和答题记录。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dyntom_adapter.guard import LeakError
from dyntom_adapter.loader import (
    build_role_context,
    build_scenario_text,
    iter_trials,
    parse_questions,
    parse_stage,
)
from dyntom_adapter.project import attach_deltas, project_state
from dyntom_adapter.router import bundle_for, parse_route
from dyntom_adapter.runner import TrialExistsError, run_trial
from dyntom_adapter.score import summarize
from dyntom_adapter.selector import extract_letter, normalize_gold
from dyntom_adapter.snapshot import freeze_raw_snapshot

QUESTIONS = [
    "What is the belief of Henry Lozano in scenario 1?",
    "What is the action of Henry Lozano in scenario 5?",
    "How does the belief of Henry Lozano change across the 5 scenarios?",
    "How does the emotion of Henry Lozano change across the 5 scenarios?",
    "Why does the belief of Henry Lozano change from scenario 1 to scenario 2?",
    "Why does the action of Henry Lozano change from scenario 4 to scenario 5?",
    "Whether the belief of Henry Lozano change from scenario 1 to scenario 2? if yes, from what to what?",
    "Whether the intention of Henry Lozano change from scenario 2 to scenario 3? if yes, from what to what?",
    "In scenario 1, how does the belief of Henry Lozano influence the emotion of Henry Lozano?",
    "In scenario 3, how does the belief&emotion of Henry Lozano influence the intention of Henry Lozano?",
    "In scenario 5, how does the intention of Henry Lozano influence the action of Henry Lozano?",
]


def _stage() -> dict:
    return {
        "social setting": "hotel lobby",
        "social setting type": "Accommodation and Residential",
        "main character": "Henry Lozano",
        "scenario numbers": 2,
        "characters information": "Henry Lozano is a business analyst. Charles Beyene is a guard.",
        "sketch": {
            "relationships among characters": "Henry and Charles are colleagues.",
            "mental states analysis in every scenario": "SECRET ANGER",
            "analysis of mental states across scenarios": "SECRET TRAJECTORY",
        },
        "story": {
            "scenario 2": {
                "background": "Charles reviews the footage again.",
                "dialogue": [{"Henry": "It was a false alarm.", "Charles": "Agreed."}],
            },
            "scenario 1": {
                "background": "Henry sees suspicious activity on the monitor.",
                "dialogue": [{"Henry": "That group looks suspicious.", "Charles": "Let us look."}],
            },
        },
    }


def _questions() -> list:
    from dyntom_adapter.loader import QuestionItem

    return [
        QuestionItem(
            question_id="type_a_what_1",
            question="What is the belief of Henry Lozano in scenario 1?",
            options=["a. nothing", "c. belief-1"],
            true_answer="c",
            question_type="type_a",
        ),
        QuestionItem(
            question_id="type_d_how_1",
            question="How does the belief of Henry Lozano change across the 2 scenarios?",
            options=["a. nothing", "c. belief-1 -> belief-2"],
            true_answer="c",
            question_type="type_d",
        ),
        QuestionItem(
            question_id="type_c_how_1",
            question="In scenario 1, how does the belief of Henry Lozano influence the emotion of Henry Lozano?",
            options=["a. nothing", "c. belief causes emotion 1"],
            true_answer="c",
            question_type="type_c",
        ),
    ]


class _Result:
    def to_event_snapshot(self, event_label=None):
        return {
            "event": event_label,
            "yuanwen": "SECRET STORY",
            "stimulus_text": "SECRET STORY",
            "objective_event": {"text": "SECRET STORY"},
            "focus_person": "Henry Lozano",
            "subjective_alambanas": [
                {
                    "text": "Charles underestimates the risk",
                    "posterior_probability": 0.61,
                },
                {"text": "Charles wants caution", "posterior_probability": 0.25},
            ],
            "branches": [
                {
                    "branch_id": "b1",
                    "activation_probability": 0.7,
                    "xiang": {"text": "He does not see the severity"},
                    "mental_factors_nonzero": {"忧": 0.8},
                }
            ],
            "aggregated_behaviors": [
                {"text": "present more evidence", "probability": 0.64}
            ],
            "action_evidence": {
                "action_span": "SECRET QUOTE",
                "action_description": "reviews the footage with Charles",
            },
            "mental_factors_event_nonzero": {"忧": 0.8},
            "continuity": {
                "cognitive_frames": [{"text": "threats get dismissed", "mass": 0.4}]
            },
            "option reasons": [{"reason": "true answer"}],
            "true answer": "g",
        }


class _Model:
    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []
        self.roles: list[str] = []
        self.flags: list[dict] = []
        self.continuity_state = None
        self.continuity_state_store = None

    def evaluate_text_with_cognition_branches(self, text, llm, **kwargs):
        self.calls += 1
        self.texts.append(text)
        self.roles.append(kwargs["role_context"])
        self.flags.append(
            {
                "update_seeds": kwargs["update_seeds"],
                "update_continuity": kwargs["update_continuity"],
                "use_action_feedback": kwargs["use_action_feedback"],
                "aggregate_behaviors": kwargs["aggregate_behaviors"],
                "predict_branch_behaviors": kwargs["predict_branch_behaviors"],
                "focus_person": kwargs["focus_person"],
            }
        )
        assert "SECRET" not in text
        assert "true answer" not in text.lower()
        assert "option reasons" not in text.lower()
        self.continuity_state.event_count += 1
        self.continuity_state_store.save(self.continuity_state)
        return SimpleNamespace(evaluation_result=_Result())


class _Llm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, schema=None):
        blob = "\n".join(message["content"] for message in messages)
        self.prompts.append(blob)
        assert "SECRET" not in blob
        assert "Dialogue:" not in blob
        assert "suspicious activity" not in blob.lower()
        empty = '"scenarios": []' in blob
        props = (schema or {}).get("properties") or {}
        if "level" in props or "same" in props:
            assert "Gold answer:" not in blob
            level = "explicit" if not empty else "inferred"
            return {"level": level, "reason": "stated" if not empty else "absent"}
        if "answer" in props:
            return {"answer": "a" if empty else "c", "reason": "ranked"}
        option = blob.split("Option:\n", 1)[-1].strip().splitlines()[0]
        letter = option.split(".", 1)[0].strip().lower()
        if empty:
            score = 1.0 if letter == "a" else 0.0
        else:
            score = 1.0 if letter == "c" else 0.0
        return {"score": score, "reason": "test"}


def test_loader_hides_annotations(tmp_path: Path):
    story = parse_stage(_stage(), trial_id="trial497")
    assert [item.index for item in story.scenarios] == [1, 2]
    visible = build_role_context(story) + "\n" + build_scenario_text(story.scenarios[0])
    assert "colleagues" in visible
    assert "SECRET" not in visible
    assert "mental states analysis" not in visible
    public = json.dumps(story.to_public_dict())
    assert "SECRET" not in public
    assert "sketch" not in public

    folder = tmp_path / "trial497"
    folder.mkdir()
    (folder / "story.json").write_text(json.dumps(_stage()), encoding="utf-8")
    (folder / "question_new.json").write_text(
        json.dumps(
            {
                "type_a_what_1": {
                    "question": "What is the belief of Henry Lozano in scenario 1?",
                    "true answer": "c",
                    "options": ["a. no", "c. yes"],
                    "question type": "type_a",
                    "option reasons": [{"reason": "true answer", "explain": "hidden", "short": "x"}],
                }
            }
        ),
        encoding="utf-8",
    )
    loaded, questions = next(iter_trials(folder))
    assert loaded.trial_id == "trial497"
    assert questions[0].true_answer == "c"
    assert not hasattr(questions[0], "option_reasons")


def test_banned_heading_in_background_is_refused():
    stage = _stage()
    stage["story"]["scenario 1"]["background"] = (
        "mental states analysis in every scenario: he is angry"
    )
    story = parse_stage(stage, trial_id="bad")
    with pytest.raises(LeakError):
        build_scenario_text(story.scenarios[0])


def test_combined_json_and_question_filter(tmp_path: Path):
    path = tmp_path / "all.json"
    path.write_text(
        json.dumps({"trial9": {"stage": _stage(), "question": {
            "type_a_what_1": {
                "question": "What is the belief of Henry Lozano in scenario 1?",
                "true answer": "a",
                "options": ["a. yes"],
                "question type": "type_a",
            },
            "skip_me": {"question": "no options here"},
        }}}),
        encoding="utf-8",
    )
    story, questions = next(iter_trials(path))
    assert story.trial_id == "trial9"
    assert [item.question_id for item in questions] == ["type_a_what_1"]
    with pytest.raises(ValueError):
        parse_questions({"only": {"question": "missing options"}})


def test_snapshot_keeps_distributions_and_drops_story():
    frozen = freeze_raw_snapshot(_Result().to_event_snapshot(), 2)
    dumped = json.dumps(frozen)
    assert "SECRET" not in dumped
    assert "true answer" not in dumped
    assert frozen["objects"][0]["probability"] == 0.61
    assert len(frozen["objects"]) == 2
    assert frozen["thoughts"][0]["content"] == "He does not see the severity"
    assert frozen["predicted_actions"][0]["probability"] == 0.64
    assert frozen["observed_action"] == "reviews the footage with Charles"
    assert frozen["history"][0]["mass"] == 0.4
    assert frozen["mental_state"]["event"]["忧"] == 0.8
    with pytest.raises(LeakError):
        project_state(None, {"scenario": 1, "yuanwen": "SECRET STORY"}, None)


def test_router_covers_official_templates():
    expected = {
        "what": ("belief", [1], "single_scenario"),
        "action-what": ("action", [5], "single_scenario"),
        "how": ("belief", [1, 2, 3, 4, 5], "cross_scenario"),
        "emotion-how": ("emotion", [1, 2, 3, 4, 5], "cross_scenario"),
        "why": ("belief", [1, 2], "cross_scenario"),
        "action-why": ("action", [4, 5], "cross_scenario"),
        "whether": ("belief", [1, 2], "cross_scenario"),
        "intention-whether": ("intention", [2, 3], "cross_scenario"),
        "influence-emotion": ("emotion", [1], "single_scenario"),
        "influence-intention": ("intention", [3], "single_scenario"),
        "influence-action": ("action", [5], "single_scenario"),
    }
    routes = [parse_route(f"q{i}", text) for i, text in enumerate(QUESTIONS)]
    assert routes[0].operation == "what"
    assert routes[2].operation == "how"
    assert routes[4].operation == "why"
    assert routes[6].operation == "whether"
    assert routes[8].operation == "influence"
    assert routes[9].source_dimensions == ["belief", "emotion"]
    assert routes[10].source_dimensions == ["intention"]
    for route, spec in zip(routes, expected.values()):
        dimension, scenarios, scope = spec
        assert route.target_dimension == dimension
        assert route.required_scenarios == scenarios
        assert route.scope == scope


def test_bundle_uses_objects_thoughts_and_actions():
    snapshots = [
        {
            "scenario": 1,
            "objects": [{"content": "Suspicious activity", "probability": 0.6}],
            "thoughts": [{"content": "He may be wrong", "activation": 0.5}],
            "observed_action": "Watch the monitor",
            "predicted_actions": [{"content": "Ask Charles", "probability": 0.4}],
            "mental_state": {"event": {"忧": 0.8}},
        },
        {
            "scenario": 2,
            "objects": [{"content": "Charles dismisses the footage", "probability": 0.7}],
            "thoughts": [{"content": "The risk is real", "activation": 0.6}],
            "observed_action": "Point at the screen",
            "predicted_actions": [{"content": "Argue", "probability": 0.5}],
        },
    ]
    route = parse_route(
        "type_d_why_1",
        "Why does the belief of Henry Lozano change from scenario 1 to scenario 2?",
        question_type="type_d",
    )
    view = bundle_for(route, snapshots)
    text = json.dumps(view)
    assert "Charles dismisses the footage" in text
    assert "The risk is real" in text
    assert "Point at the screen" in text
    assert "忧" not in text
    assert "change_reason" not in view


def test_run_trial_freezes_before_questions(tmp_path: Path):
    story = parse_stage(_stage(), trial_id="trial497")
    model = _Model()
    llm = _Llm()
    summary = run_trial(
        story,
        _questions(),
        model=model,
        llm=llm,
        out_dir=tmp_path,
        resume=False,
        question_limit=None,
        run_control=True,
        placeholder_adapter=False,
    )
    assert summary["correct"] == 3
    assert model.calls == 2
    assert model.continuity_state.event_count == 2
    assert model.flags[0]["update_seeds"] is False
    assert model.flags[0]["update_continuity"] is True
    assert model.flags[0]["use_action_feedback"] is True
    assert model.flags[0]["aggregate_behaviors"] is True
    assert model.texts[0].startswith("Scenario 1:")
    assert "Scenario 2:" not in model.texts[0]
    assert "Charles reviews" in model.texts[1]
    assert "suspicious activity" not in model.texts[1]

    state = json.loads((tmp_path / "trial497" / "state_01.json").read_text(encoding="utf-8"))
    assert "SECRET" not in json.dumps(state)
    assert len(state["objects"]) == 2
    answers = (tmp_path / "trial497" / "answers.jsonl").read_text(encoding="utf-8")
    assert "option reasons" not in answers
    assert "SECRET" not in answers
    rows = [json.loads(line) for line in answers.splitlines()]
    how = next(row for row in rows if row["question_id"] == "type_d_how_1")
    assert "Charles underestimates the risk" in json.dumps(how["representation"])
    assert "present more evidence" in json.dumps(how["representation"])
    assert how["required_scenarios"] == [1, 2]
    assert how["correct"] is True
    assert all(row["control_answer"] == "a" and row["control_correct"] is False for row in rows)
    assert "Dialogue:" not in "\n".join(llm.prompts)
    report = (tmp_path / "trial497" / "report.txt").read_text(encoding="utf-8")
    assert "原题：What is the belief of Henry Lozano in scenario 1?" in report
    assert "belief-1" in report
    assert "Adapter：c. belief-1" in report

    with pytest.raises(TrialExistsError):
        run_trial(
            story,
            _questions(),
            model=model,
            llm=llm,
            out_dir=tmp_path,
            resume=False,
            question_limit=None,
            run_control=True,
            placeholder_adapter=False,
        )
    run_trial(
        story,
        _questions(),
        model=model,
        llm=llm,
        out_dir=tmp_path,
        resume=True,
        question_limit=None,
        run_control=True,
        placeholder_adapter=False,
    )
    assert model.calls == 2


def test_score_groups_dimension_and_scope():
    summary = summarize(
        [
            {
                "target_dimension": "belief",
                "operation": "what",
                "question_type": "type_a",
                "scope": "single_scenario",
                "correct": True,
                "control_correct": False,
            },
            {
                "target_dimension": "belief",
                "operation": "why",
                "question_type": "type_d",
                "scope": "cross_scenario",
                "correct": False,
                "control_correct": False,
            },
        ]
    )
    assert summary["total"]["accuracy"] == 0.5
    assert summary["by_dimension"]["belief"]["total"] == 2
    assert summary["by_scope"]["cross_scenario"]["correct"] == 0
    assert summary["control"]["total"]["accuracy"] == 0.0


def test_selector_scores_each_option_alone():
    from dyntom_adapter.selector import choose_by_scores

    class _Scores:
        def __init__(self) -> None:
            self.calls = 0
            self.options: list[str] = []

        def complete(self, messages, schema=None):
            self.calls += 1
            props = (schema or {}).get("properties") or {}
            if "scores" in props:
                text = messages[-1]["content"]
                assert "a. nothing" in text and "c. a real threat" in text
                return {
                    "scores": [
                        {"letter": "a", "score": 0.4},
                        {"letter": "c", "score": 0.9},
                    ],
                    "reason": "c matches the record more directly",
                }
            option = messages[-1]["content"].split("Option:\n", 1)[1].strip()
            self.options.append(option)
            assert option.count("\n") == 0
            if option.startswith("b."):
                return {"score": "bad"}
            if option.startswith("a."):
                return {"score": 0.8, "reason": "also close"}
            if option.startswith("c."):
                return {"score": 0.8, "reason": "same claim"}
            return {"score": 0.1, "reason": "different"}

    llm = _Scores()
    letter, scores, reason = choose_by_scores(
        llm,
        {"scenarios": [{"scenario": 1, "objects": [{"content": "a real threat"}]}]},
        "What is the belief?",
        ["b. nearby event", "c. a real threat", "a. nothing"],
    )
    assert letter == "c"
    assert scores["c"] > scores["a"]
    assert scores["b"] == 0.0
    assert "more directly" in reason
    assert llm.calls == 5


def test_letters_and_deltas():
    options = ["a. no", "c. yes"]
    assert extract_letter({"answer": "C"}, options) == "c"
    assert normalize_gold("c", options) == "c"
    current = freeze_raw_snapshot(_Result().to_event_snapshot(), 2)
    previous = freeze_raw_snapshot(_Result().to_event_snapshot(), 1)
    previous["objects"][0]["probability"] = 0.2
    delta = attach_deltas(previous, current)["_deltas_from_prior_snapshot"]
    assert delta["objects"][0]["delta"] == 0.41


def test_audit_rescues_only_the_same_claim():
    from dyntom_adapter.audit import apply_audit

    class _Same:
        def complete(self, messages, schema=None):
            text = messages[-1]["content"]
            assert "Gold answer:" in text
            assert "apologizing necessary" in text
            return {"same": True, "reason": "Both say he wants to mend the relationship."}

    wrong = {
        "correct": False,
        "question": "Why does the intention change?",
        "representation": {"change_reason": "The misjudgment makes apologizing necessary."},
        "gold_output": "f. Desire to mend relationship after tension",
    }
    apply_audit(wrong, _Same())
    assert wrong["adapter_correct"] is False
    assert wrong["mwm_matches_gold"] is True
    assert wrong["correct"] is True

    class _Different:
        def complete(self, messages, schema=None):
            return {"same": False, "reason": "The statement names a reaction, not the frustration."}

    still = {
        "correct": False,
        "question": "Why does the emotion change?",
        "representation": {"change_reason": "Charles dismisses the footage."},
        "gold_output": "i. Frustration due to perceived underestimation",
    }
    apply_audit(still, _Different())
    assert still["correct"] is False
    assert still["mwm_matches_gold"] is False

    class _Unused:
        def complete(self, messages, schema=None):
            raise AssertionError("选题已对，不应再核对")

    right = {"correct": True, "representation": {"belief": "x"}, "gold_output": "a. yes"}
    apply_audit(right, _Unused())
    assert right["correct"] is True
    assert right["mwm_matches_gold"] is None


def test_letter_match_is_correct_without_review():
    from dyntom_adapter.audit import finalize_judgment

    class _Unused:
        def complete(self, messages, schema=None):
            raise AssertionError("最高分已是标准答案，不应再核对")

    row = {
        "predicted_answer": "g",
        "ground_truth": "g",
        "question": "How does the emotion change?",
        "adapter_output": "g. Relieved and embarrassed",
        "representation": {"scenarios": [{"thoughts": [{"content": "他向对方道歉"}]}]},
        "correct": True,
    }
    finalize_judgment(row, _Unused())
    assert row["selector_correct"] is True
    assert row["correct"] is True
    assert row["mwm_matches_gold"] is None
