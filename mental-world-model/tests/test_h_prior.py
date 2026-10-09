"""H 两级路由：作用域资格 → 调制哪条 Fresh 执。"""

from mental_world_model.engine.continuity.models import BiasFrame
from mental_world_model.engine.cognition.prompts import build_branch_xiang_messages
from mental_world_model.engine.continuity.prior import (
    apply_h_prior_to_fresh,
    bind_h_to_fresh,
    build_h_bind_messages,
    h_payload_for_candidate,
    trigger_h_scope,
    working_set_by_mass,
)
from mental_world_model.engine.perception.models import (
    GraspingCombination,
    GraspingHypothesis,
    ObjectiveEventStructure,
    SubjectiveAlambanaCandidate,
)
from mental_world_model.engine.perception.prompts import build_subjective_alambana_messages


class _RouterClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, messages, schema=None):
        props = (schema or {}).get("properties") or {}
        if "triggered" in str(schema):
            self.calls.append("scope")
            return {"items": [{"frame_id": "bf_1", "triggered": True}]}
        if "bindings" in props:
            self.calls.append("bind")
            return {"bindings": [{"frame_id": "bf_1", "grasping_ids": ["sg1"]}]}
        raise AssertionError(f"unexpected schema: {schema}")


def test_working_set_is_mass_only():
    frames = [
        BiasFrame(id="bf_1", text="低相关高mass", mass=0.9),
        BiasFrame(id="bf_2", text="高相关低mass", mass=0.2),
        BiasFrame(id="bf_3", text="中mass", mass=0.5),
    ]
    chosen = working_set_by_mass(frames, k=2)
    assert [f.id for f in chosen] == ["bf_1", "bf_3"]


def test_two_level_routing_does_not_retrigger_on_bind():
    frames = [
        BiasFrame(
            id="bf_1",
            text="朋友骗了我，我对他不满，以及不信任",
            mass=0.8,
        )
    ]
    selves = [
        GraspingHypothesis(
            id="sg1",
            type="self_grasping",
            text="把这次协作当成对自己责任的确认",
            confidence=0.5,
        )
    ]
    dharmas = [
        GraspingHypothesis(
            id="dg1",
            type="dharma_grasping",
            text="CEO的要求是必须完成的工作约束",
            confidence=0.5,
        )
    ]
    llm = _RouterClient()
    prior = apply_h_prior_to_fresh(
        "CEO要求两人48小时内共同提交方案",
        frames,
        selves,
        dharmas,
        llm,
    )
    assert llm.calls == ["scope", "bind"]
    assert prior["triggered_ids"] == ["bf_1"]
    assert prior["bindings"]["bf_1"] == ["sg1"]
    assert "bf_1" in selves[0].source_bias_frame_ids
    assert selves[0].confidence > 0.5
    assert dharmas[0].source_bias_frame_ids == []


def test_bind_does_not_ask_whether_h_should_trigger():
    frames = [
        BiasFrame(id="bf_1", text="已经确定进入本场的偏置", mass=0.7)
    ]
    graspings = [
        GraspingHypothesis(id="sg1", type="self_grasping", text="当场责任执")
    ]
    llm = _RouterClient()
    triggered = trigger_h_scope("当前刺激", frames, llm)
    assert [f.id for f in triggered] == ["bf_1"]
    bindings = bind_h_to_fresh(triggered, graspings, llm)
    assert bindings["bf_1"] == ["sg1"]
    text = build_h_bind_messages(triggered, graspings)[0]["content"]
    assert "不要再判断 H 该不该触发" in text
    assert "triggered" not in text


def test_l3_reads_reactivated_h_without_relevance_question():
    system = build_subjective_alambana_messages(
        "有人前来求助",
        ObjectiveEventStructure(scene="求助", action="前来"),
        [GraspingHypothesis(id="s1", type="self_grasping", text="经验即我")],
        [GraspingHypothesis(id="d1", type="dharma_grasping", text="求助即认可")],
        combinations=[
            GraspingCombination(
                combo_id="C1",
                used_self_ids=["s1"],
                used_dharma_ids=["d1"],
                reactivated_h_ids=["bf_1"],
                source_bias_frame_ids=["bf_1"],
            )
        ],
        h_lookup={"bf_1": "朋友骗了我，我对他不满，以及不信任"},
    )[0]["content"]
    assert "reactivated_h" in system
    assert "不得产生新的执取维度" in system
    assert "不要再判断这条历史是否相关" in system
    assert "禁止引入组合外的新心理因素" not in system
    user = build_subjective_alambana_messages(
        "有人前来求助",
        ObjectiveEventStructure(scene="求助", action="前来"),
        [GraspingHypothesis(id="s1", type="self_grasping", text="经验即我")],
        [GraspingHypothesis(id="d1", type="dharma_grasping", text="求助即认可")],
        combinations=[
            GraspingCombination(
                combo_id="C1",
                used_self_ids=["s1"],
                used_dharma_ids=["d1"],
                reactivated_h_ids=["bf_1"],
                source_bias_frame_ids=["bf_1"],
            )
        ],
        h_lookup={"bf_1": "朋友骗了我，我对他不满，以及不信任"},
    )[1]["content"]
    assert "朋友骗了我" in user
    assert "是否相关" not in user


class _FalseStringClient:
    def complete(self, messages, schema=None):
        del messages
        if "triggered" in str(schema):
            return {"items": [{"frame_id": "bf_1", "triggered": "false"}]}
        raise AssertionError("scope only")


class _EmptyItemsClient:
    def complete(self, messages, schema=None):
        del messages
        if "triggered" in str(schema):
            return {"items": []}
        raise AssertionError("scope only")


class _ReuseClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, messages, schema=None):
        del messages
        props = (schema or {}).get("properties") or {}
        if "triggered" in str(schema):
            self.calls.append("scope")
            return {"items": [{"frame_id": "bf_1", "triggered": True}]}
        if "bindings" in props:
            self.calls.append("bind")
            return {"bindings": [{"frame_id": "bf_1", "grasping_ids": ["sg1"]}]}
        raise AssertionError(f"unexpected schema: {schema}")


class _CompressClient:
    def complete(self, messages, schema=None):
        del messages, schema
        return {
            "propositions": [
                {"source_id": "O1", "text": "朋友骗了我，从此不太信任他的承诺"},
                {"source_id": "ghost", "text": "找不到来源却想进库的偏置"},
            ]
        }


def test_string_false_does_not_trigger():
    frames = [
        BiasFrame(id="bf_1", text="朋友曾经骗过我，我以后更不信任朋友", mass=0.8)
    ]
    hit = trigger_h_scope(
        "朋友这次又隐瞒了信息",
        frames,
        _FalseStringClient(),
    )
    assert hit == []


def test_empty_items_falls_back_to_lexical():
    frames = [
        BiasFrame(id="bf_1", text="朋友隐瞒重要信息我不信任", mass=0.8)
    ]
    hit = trigger_h_scope(
        "朋友隐瞒重要信息又不告诉我",
        frames,
        _EmptyItemsClient(),
    )
    assert [f.id for f in hit] == ["bf_1"]


def test_scope_runs_once_bind_can_repeat():
    from mental_world_model.engine.continuity.prior import (
        bind_h_scope_to_fresh,
        resolve_h_scope,
    )

    frames = [
        BiasFrame(id="bf_1", text="朋友骗了我，我对他不满，以及不信任", mass=0.8)
    ]
    llm = _ReuseClient()
    scope = resolve_h_scope("朋友这次又隐瞒了信息", frames, llm)
    g1 = [GraspingHypothesis(id="sg1", type="self_grasping", text="不能被排除")]
    g2 = [GraspingHypothesis(id="sg1", type="self_grasping", text="协作责任")]
    bind_h_scope_to_fresh(scope, g1, [], llm)
    bind_h_scope_to_fresh(scope, g2, [], llm)
    assert llm.calls == ["scope", "bind", "bind"]
    assert scope["triggered"][0].id == "bf_1"


def test_compress_skips_unknown_source():
    from mental_world_model.engine.continuity.cognitive import compress_belief_to_frames

    items = [
        {
            "id": "O1",
            "alambana": "朋友没有告诉我聚会时间",
            "self_grasping": ["我应当被告知"],
            "dharma_grasping": ["隐瞒就是排除"],
            "posterior": 0.18,
        }
    ]
    frames = compress_belief_to_frames("朋友没告诉我时间", items, _CompressClient())
    assert len(frames) == 1
    assert frames[0].text == "朋友骗了我，从此不太信任他的承诺"
    assert abs(frames[0].mass - 0.18) < 1e-9
    assert "找不到来源" not in frames[0].text


class _EmptyEvidence:
    has_observable_action = False
    action_span = ""
    action_description = ""


def test_h_payload_only_uses_bound_ids():
    cand = SubjectiveAlambanaCandidate(
        id="O1",
        text="隐瞒被揭穿",
        posterior_probability=0.4,
        confidence=0.6,
        reactivated_h_ids=["bf_1", "missing", "bf_empty"],
    )
    h_prior = {
        "working_set": [
            {"id": "bf_1", "text": "朋友骗了我，从此不太信任他的承诺", "mass": 0.82},
            {"id": "bf_2", "text": "不该出现的另一条", "mass": 0.9},
            {"id": "bf_empty", "text": "", "mass": 0.5},
        ]
    }
    payload = h_payload_for_candidate(cand, h_prior)
    assert [item["id"] for item in payload] == ["bf_1"]
    assert abs(payload[0]["mass"] - 0.82) < 1e-9


def test_xiang_prompt_reads_bound_h_without_relevance_judge():
    cand = SubjectiveAlambanaCandidate(
        id="O1",
        text="隐瞒被揭穿",
        posterior_probability=0.4,
        confidence=0.6,
        reactivated_h_ids=["bf_1"],
    )
    payload = [
        {
            "id": "bf_1",
            "text": "朋友骗了我，从此不太信任他的承诺",
            "mass": 0.82,
        }
    ]
    system, user = build_branch_xiang_messages(
        "朋友这次又没说",
        None,
        cand,
        [],
        [],
        _EmptyEvidence(),
        reactivated_h=payload,
    )
    system_text = system["content"]
    user_text = user["content"]
    assert "不要再判断是否相关" in system_text
    assert "mass 越大" in system_text
    assert "朋友骗了我，从此不太信任他的承诺" in user_text
    assert "mass=0.820" in user_text
    assert "不该出现的另一条" not in user_text
    assert "不要再做相关度判断" in user_text

    empty_user = build_branch_xiang_messages(
        "朋友这次又没说",
        None,
        cand,
        [],
        [],
        _EmptyEvidence(),
        reactivated_h=[],
    )[1]["content"]
    assert "本支未绑定历史总执" in empty_user
