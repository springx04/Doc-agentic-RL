"""Rule-based rewards for document-understanding tool-call rollouts.

The scorer is intentionally dependency-free so it can run in every rollout
worker.  Labels may be plain strings/lists or JSON objects such as::

    {"answers": ["$1.2 million", "$1.2m"], "metric": "anls"}

Supported metrics are ``exact_match``, ``anls``, ``token_f1``, ``contains``,
``json`` and ``auto``.  ``auto`` takes the strongest of exact match, ANLS and
token F1, which works well for mixed short-answer document QA datasets.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any


FINAL_PATTERN = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)


def extract_final_answer(response: str) -> tuple[str, bool]:
    """Return the last strict ``<final>`` answer and whether it was present."""
    matches = FINAL_PATTERN.findall(response or "")
    if not matches:
        return "", False
    return matches[-1].strip(), True


def normalize_answer(value: Any) -> str:
    """Normalize a scalar answer without destroying non-Latin text."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = unicodedata.normalize("NFKC", value).casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _answer_tokens(value: Any) -> list[str]:
    normalized = normalize_answer(value)
    # Keep words/numbers together and score CJK text at character granularity.
    return re.findall(r"[a-z0-9]+(?:[.,:/+-][a-z0-9]+)*|[\u3400-\u9fff]|[^\W\s]", normalized)


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def exact_match(prediction: Any, reference: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def anls(prediction: Any, reference: Any, threshold: float = 0.5) -> float:
    """DocVQA-style average normalized Levenshtein similarity for one pair."""
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    similarity = 1.0 - _edit_distance(pred, ref) / max(len(pred), len(ref))
    return similarity if similarity >= threshold else 0.0


def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = _answer_tokens(prediction)
    ref_tokens = _answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_json(value: Any) -> Any | None:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def json_match(prediction: Any, reference: Any) -> float:
    pred = _parse_json(prediction)
    ref = _parse_json(reference)
    if pred is None or ref is None:
        return 0.0
    pred_text = json.dumps(pred, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ref_text = json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return exact_match(pred_text, ref_text)


def parse_label(label: Any, metadata: dict[str, Any] | None = None) -> tuple[list[Any], str]:
    """Extract acceptable answers and metric from a dataset label."""
    metadata = metadata or {}
    metric = str(metadata.get("metric") or metadata.get("answer_metric") or "auto").lower()
    parsed = _parse_json(label)

    if isinstance(parsed, dict):
        metric = str(parsed.get("metric") or metric).lower()
        answers = (
            parsed.get("answers")
            or parsed.get("acceptable_answers")
            or parsed.get("answer")
            or parsed.get("ground_truth")
            or parsed.get("label")
        )
    elif isinstance(parsed, list):
        answers = parsed
    else:
        answers = label

    if isinstance(answers, (list, tuple, set)):
        answer_list = list(answers)
    else:
        answer_list = [answers]
    return [answer for answer in answer_list if answer is not None], metric


def score_answer(prediction: str, answers: list[Any], metric: str = "auto") -> float:
    metric = (metric or "auto").lower().replace("-", "_")

    def score_pair(reference: Any) -> float:
        if metric in {"exact", "em", "exact_match", "accuracy", "acc"}:
            return exact_match(prediction, reference)
        if metric == "anls":
            return anls(prediction, reference)
        if metric in {"f1", "token_f1"}:
            return token_f1(prediction, reference)
        if metric == "contains":
            pred = normalize_answer(prediction)
            ref = normalize_answer(reference)
            return float(bool(ref) and ref in pred)
        if metric in {"json", "json_exact"}:
            return json_match(prediction, reference)
        if metric == "auto":
            return max(exact_match(prediction, reference), anls(prediction, reference), token_f1(prediction, reference))
        raise ValueError(f"Unsupported document reward metric: {metric}")

    return max((score_pair(reference) for reference in answers), default=0.0)


def compute_document_reward(
    response: str,
    label: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a GRPO-friendly reward in [-1, 1] and evaluation metrics."""
    prediction, format_ok = extract_final_answer(response)
    answers, metric = parse_label(label, metadata)
    quality = score_answer(prediction, answers, metric) if format_ok else 0.0
    quality = max(0.0, min(1.0, float(quality)))
    return {
        "score": 2.0 * quality - 1.0,
        "acc": max((exact_match(prediction, answer) for answer in answers), default=0.0) if format_ok else 0.0,
        "quality": quality,
        "format": float(format_ok),
        "pred": prediction,
        "metric": metric,
    }
