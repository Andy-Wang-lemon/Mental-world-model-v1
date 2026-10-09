"""CoT 一次作答。SimToM 与 TimeToM-style 先重建，再答题。"""

from __future__ import annotations

import json
import re

from inference_baselines.agentic_tom import run_agentic
from inference_baselines.items import BenchmarkItem
from inference_baselines.parse_answer import is_correct, parse_answer
from inference_baselines.picture import run_picture
from inference_baselines.prompts import (
    cot_messages,
    simtom_stage1_messages,
    simtom_stage2_messages,
    timetom_stage1_messages,
    timetom_stage2_messages,
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def run_cot(client: object, item: BenchmarkItem) -> dict:
    raw = client.complete_text(cot_messages(item))
    prediction = parse_answer(raw, item.choices)
    return {
        "id": item.id,
        "method": "cot",
        "raw_response": raw,
        "prediction": prediction,
        "gold": _gold_value(item.gold),
        "correct": is_correct(prediction, item.gold),
    }


def run_simtom(client: object, item: BenchmarkItem) -> dict:
    perspective = client.complete_text(simtom_stage1_messages(item)).strip()
    final = client.complete_text(simtom_stage2_messages(perspective, item))
    prediction = parse_answer(final, item.choices)
    return {
        "id": item.id,
        "method": "simtom",
        "target_character": item.target_character,
        "perspective": perspective,
        "final_response": final,
        "prediction": prediction,
        "gold": _gold_value(item.gold),
        "correct": is_correct(prediction, item.gold),
    }


def run_timetom(client: object, item: BenchmarkItem) -> dict:
    raw_chain = client.complete_text(timetom_stage1_messages(item))
    chain = parse_state_chain(raw_chain)
    chain_text = json.dumps(chain, ensure_ascii=False, indent=2) if chain is not None else raw_chain
    final = client.complete_text(timetom_stage2_messages(chain_text, item))
    prediction = parse_answer(final, item.choices)
    return {
        "id": item.id,
        "method": "timetom",
        "implementation": "TimeToM-style",
        "target_character": item.target_character,
        "temporal_state_chain": chain if chain is not None else [],
        "temporal_state_raw": raw_chain,
        "final_response": final,
        "prediction": prediction,
        "gold": _gold_value(item.gold),
        "correct": is_correct(prediction, item.gold),
    }


def parse_state_chain(text: str) -> list | None:
    stripped = _FENCE_RE.sub("", str(text or "").strip()).strip()
    parsed = _load_array(stripped)
    if parsed is not None:
        return parsed
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end <= start:
        return None
    return _load_array(stripped[start : end + 1])


def _load_array(text: str) -> list | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return data
    return None


def _gold_value(gold: tuple[str, ...]) -> str | list[str]:
    if len(gold) == 1:
        return gold[0]
    return list(gold)


METHODS = {
    "cot": run_cot,
    "simtom": run_simtom,
    "timetom": run_timetom,
    "picture": run_picture,
    "agentictom": run_agentic,
}
