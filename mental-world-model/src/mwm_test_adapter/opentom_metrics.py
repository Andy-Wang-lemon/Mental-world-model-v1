"""OpenToM 论文级评测指标（macro-averaged F1 by question type）。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# 与 OpenToM 论文 / 官方 evaluator 一致的题型命名
PAPER_QUESTION_TYPES: tuple[str, ...] = (
    "location-fo-coarse",
    "location-fo-fine",
    "location-so-coarse",
    "location-so-fine",
    "multihop-fo",
    "multihop-so",
    "attitude",
)

_TYPE_ALIASES: dict[str, str] = {
    "location_cg_fo": "location-fo-coarse",
    "location_fg_fo": "location-fo-fine",
    "location_cg_so": "location-so-coarse",
    "location_fg_so": "location-so-fine",
    "location-fo-coarse": "location-fo-coarse",
    "location-fo-fine": "location-fo-fine",
    "location-so-coarse": "location-so-coarse",
    "location-so-fine": "location-so-fine",
    "multihop_fo": "multihop-fo",
    "multihop_so": "multihop-so",
    "multihop-fo": "multihop-fo",
    "multihop-so": "multihop-so",
    "attitude": "attitude",
}


def normalize_paper_question_type(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    if key in _TYPE_ALIASES:
        return _TYPE_ALIASES[key]
    # 回退：尽量匹配
    if "attitude" in key:
        return "attitude"
    if "multihop" in key and "so" in key:
        return "multihop-so"
    if "multihop" in key:
        return "multihop-fo"
    if "location" in key and "so" in key and ("fg" in key or "fine" in key):
        return "location-so-fine"
    if "location" in key and "so" in key:
        return "location-so-coarse"
    if "location" in key and ("fg" in key or "fine" in key):
        return "location-fo-fine"
    if "location" in key:
        return "location-fo-coarse"
    return key or "unknown"


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _class_f1(gold_labels: list[str], pred_labels: list[str], class_name: str) -> float:
    target = _normalize_label(class_name)
    tp = fp = fn = 0
    for gold, pred in zip(gold_labels, pred_labels, strict=True):
        g = _normalize_label(gold)
        p = _normalize_label(pred) if pred else ""
        if g == target and p == target:
            tp += 1
        elif g != target and p == target:
            fp += 1
        elif g == target and p != target:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1_single_label(gold_labels: list[str], pred_labels: list[str]) -> float:
    """单标签多分类的 macro-F1（对 gold 中出现过的类别取平均）。"""
    if not gold_labels:
        return 0.0
    classes = sorted({_normalize_label(g) for g in gold_labels if _normalize_label(g)})
    if not classes:
        return 0.0
    scores = [_class_f1(gold_labels, pred_labels, class_name) for class_name in classes]
    return sum(scores) / len(scores)


@dataclass
class OpenToMPaperMetrics:
    """OpenToM 论文报告格式：各题型 macro-F1 + 总体 macro-F1。"""

    by_question_type: dict[str, float] = field(default_factory=dict)
    counts_by_question_type: dict[str, int] = field(default_factory=dict)
    macro_avg_f1: float = 0.0
    accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_avg_f1": round(self.macro_avg_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "by_question_type": {
                key: round(value, 4)
                for key, value in sorted(self.by_question_type.items())
            },
            "counts_by_question_type": dict(sorted(self.counts_by_question_type.items())),
        }


def compute_opentom_paper_metrics(
    *,
    gold_labels: list[list[str]],
    pred_labels: list[list[str]],
    question_types: list[str],
) -> OpenToMPaperMetrics:
    if not gold_labels or len(gold_labels) != len(pred_labels) or len(gold_labels) != len(question_types):
        return OpenToMPaperMetrics()

    flat_gold = [g[0] if g else "" for g in gold_labels]
    flat_pred = [p[0] if p else "" for p in pred_labels]
    correct = sum(1 for g, p in zip(flat_gold, flat_pred, strict=True) if _normalize_label(g) == _normalize_label(p))
    accuracy = correct / len(flat_gold)

    grouped_gold: dict[str, list[str]] = defaultdict(list)
    grouped_pred: dict[str, list[str]] = defaultdict(list)
    for gold, pred, qtype in zip(flat_gold, flat_pred, question_types, strict=True):
        paper_type = normalize_paper_question_type(qtype)
        grouped_gold[paper_type].append(gold)
        grouped_pred[paper_type].append(pred)

    by_type: dict[str, float] = {}
    counts: dict[str, int] = {}
    for paper_type, g_list in grouped_gold.items():
        p_list = grouped_pred[paper_type]
        by_type[paper_type] = macro_f1_single_label(g_list, p_list)
        counts[paper_type] = len(g_list)

    type_scores = [by_type[t] for t in PAPER_QUESTION_TYPES if t in by_type]
    if not type_scores:
        type_scores = list(by_type.values())
    macro_avg = sum(type_scores) / len(type_scores) if type_scores else 0.0

    return OpenToMPaperMetrics(
        by_question_type=by_type,
        counts_by_question_type=counts,
        macro_avg_f1=macro_avg,
        accuracy=accuracy,
    )
