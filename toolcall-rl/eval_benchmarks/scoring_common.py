"""Small shared utilities used by the official benchmark adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import json_write, jsonl_read, jsonl_write, keyed_rows


def prediction_records(path: Path) -> dict[str, dict[str, Any]]:
    return keyed_rows(path)


def gold_records(path: Path) -> dict[str, dict[str, Any]]:
    return keyed_rows(path)


def join_gold_predictions(
    gold: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any] | None]], int, int]:
    """Join in gold order and retain missing predictions in the denominator."""

    rows: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    missing = 0
    extra = 0
    gold_ids = set(gold)
    for task_id, gold_row in gold.items():
        prediction = predictions.get(task_id)
        if prediction is None:
            missing += 1
        rows.append((task_id, gold_row, prediction))
    extra = len(set(predictions) - gold_ids)
    if extra:
        extra_ids = sorted(set(predictions) - gold_ids)
        raise ValueError(f"predictions contain {extra} task IDs absent from gold, first={extra_ids[:5]}")
    return rows, missing, extra


def prediction_text(prediction: dict[str, Any] | None) -> str:
    if not prediction:
        return ""
    return str(prediction.get("final_answer", "") or "")


def protocol_invalid(prediction: dict[str, Any] | None) -> bool:
    return bool(prediction is not None and not prediction.get("protocol_valid", False))


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _levenshtein_distance(left: str, right: str) -> int:
    try:
        import Levenshtein

        return int(Levenshtein.distance(left, right))
    except ImportError:
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, 1):
            current = [i]
            for j, right_char in enumerate(right, 1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[j] + 1,
                        previous[j - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1]


def nls(prediction: Any, reference: Any) -> float:
    pred = normalized_text(prediction)
    gold = normalized_text(reference)
    if pred == "" and gold == "":
        return 1.0
    if pred == "" or gold == "":
        return 0.0
    score = 1.0 - _levenshtein_distance(pred, gold) / max(len(pred), len(gold))
    return score if score >= 0.5 else 0.0


def anls(prediction: Any, references: Iterable[Any]) -> float:
    values = list(references)
    return max((nls(prediction, reference) for reference in values), default=0.0)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_scores(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    jsonl_write(path, rows)


def vendor_path(vendor_dir: Path) -> None:
    vendor_dir = vendor_dir.resolve()
    if not vendor_dir.is_dir():
        raise FileNotFoundError(f"official scorer vendor directory not found: {vendor_dir}")
    value = str(vendor_dir)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_numeric_metric(path: Path) -> float | None:
    """Find an ANLS/accuracy-like scalar in an official JSON result."""

    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    candidates = {
        "overall anls",
        "overall_anls",
        "anls",
        "accuracy",
        "score",
        "overall",
    }

    def walk(value: Any) -> float | None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).casefold().replace("-", "_").replace(" ", "_")
                if normalized_key in {c.replace(" ", "_") for c in candidates}:
                    if isinstance(item, (int, float)):
                        return float(item)
            for item in value.values():
                result = walk(item)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = walk(item)
                if result is not None:
                    return result
        return None

    return walk(payload)
