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

from tool_protocol import parse_assistant_action

ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:the\s+)?(?:final\s+)?answer|\u7b54\u6848|\u6700\u7ec8\u7b54\u6848)\s*(?:(?:is|\u662f)\s*)?[:\uff1a-]?\s*",
    re.IGNORECASE,
)
FIELD_LABEL_PATTERN = re.compile(
    # A document-QA answer is often returned as ``Field name: value``.  Keep
    # the raw candidate too, so this never turns into unconstrained substring
    # matching or prevents a reference that includes the field name from
    # matching exactly.
    r"^\s*[\w\u3400-\u9fff][\w\u3400-\u9fff #_./()'&-]{0,47}\s*:\s*(?=\S)",
    re.UNICODE,
)
_RELAXED_FINAL_RE = re.compile(
    r"<final>(?P<body>.*?)</final>",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_MARKER_RE = re.compile(r"</?\s*(?:tool_call|final)\b", re.IGNORECASE)


def extract_final_answer(response: str, metadata: dict[str, Any] | None = None) -> tuple[str, bool]:
    """Return ``(answer, protocol_validity)`` for reward evaluation.

    A generated rollout contains several assistant turns and tool
    observations.  The generation loop records the raw final assistant turn
    in ``metadata['final_action']``; using that field avoids searching the
    concatenated trajectory for a tag that may have appeared in an example or
    observation.  Standalone responses are parsed directly as one turn.

    The rollout parser remains strict: ``<final>answer</final>`` must occupy
    the whole assistant turn.  For scoring only, a single final span inside
    surrounding prose is recovered and returned with ``protocol_validity``
    false.  This keeps semantic correctness separate from protocol
    compliance instead of turning a correct, verbose answer into a false
    negative.
    """
    candidate = response or ""
    if isinstance(metadata, dict):
        final_action = metadata.get("final_action")
        raw_generation_text = metadata.get("raw_generation_text")
        if isinstance(final_action, str) and final_action.strip():
            candidate = final_action
        elif isinstance(raw_generation_text, str) and raw_generation_text.strip():
            candidate = raw_generation_text
    parsed = parse_assistant_action(candidate)
    if parsed.kind == "final":
        return str(parsed.value).strip(), True

    # A malformed tool turn must never be mistaken for a final answer.  One
    # and only one final span is recoverable for the semantic reward signal;
    # multiple spans are ambiguous and remain an invalid answer.
    matches = list(_RELAXED_FINAL_RE.finditer(candidate))
    if len(matches) != 1:
        return "", False
    if len(_ACTION_MARKER_RE.findall(candidate)) != 2:
        return "", False
    answer = matches[0].group("body").strip()
    if not answer:
        return "", False
    if answer.casefold() in {"and", "or"} and re.search(r"\b(?:return|format|tag|answer)\b", candidate, re.IGNORECASE):
        return "", False
    return answer, False


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}


def _replace_number_words(text: str) -> str:
    pattern = r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b"
    return re.sub(pattern, lambda match: _NUMBER_WORDS[match.group(0)], text)


def normalize_answer(value: Any) -> str:
    """Normalize answers while preserving decimals and numeric identity.

    Currency symbols, grouping commas, case and edge punctuation are
    presentation details.  Decimal points, signs and internal hyphens remain
    meaningful so values such as ``33.0`` and proposal identifiers are not
    damaged by normalization.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = "".join(
        ch if (ch.isalnum() or ch in ".+-/%") else " "
        for ch in text
    )
    # A period is retained only as a decimal point.  Other punctuation is a
    # separator, including leading/trailing periods.
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = _replace_number_words(" ".join(text.split()))
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


def answer_variants(prediction: Any) -> list[Any]:
    """Return raw and safely canonicalized answer candidates.

    We deliberately strip only conventional answer lead-ins (for example,
    ``The answer is:`` or ``绛旀鏄細``).  Free-form explanations are retained so
    exact-match metrics do not turn into unconstrained substring matching.
    """
    if not isinstance(prediction, str):
        return [prediction]
    stripped = ANSWER_PREFIX_PATTERN.sub("", prediction, count=1).strip()
    variants = [prediction]
    if stripped and stripped != prediction:
        variants.append(stripped)

    # Treat a short leading field label as presentation, not part of the
    # answer.  For example, ``Proposal #: 14-3006-14`` and the reference
    # ``14-3006-14`` are semantically the same short answer.  The raw form is
    # retained above, and only this anchored one-label form is stripped.
    field_value = FIELD_LABEL_PATTERN.sub("", prediction, count=1).strip()
    if field_value and field_value != prediction:
        variants.append(field_value)

    # VLMs often return a complete explanatory sentence such as
    # `The full form of "ILS" is "International Litigation Services".`.
    # The quoted spans are explicit candidate answers, not arbitrary
    # substrings, so retaining them avoids penalising a semantically exact
    # answer merely for a harmless explanatory wrapper.
    variants.extend(match.strip() for match in re.findall(r'"([^"]+)"', prediction) if match.strip())
    expansion = re.search(
        r"\b(?:stands\s+for|full\s+form\s+(?:of\s+.+?\s+)?is)\s+(.+?)[.!]?\s*$",
        prediction,
        re.IGNORECASE,
    )
    if expansion and expansion.group(1).strip():
        variants.append(expansion.group(1).strip().strip('"'))
    return list(dict.fromkeys(variants))


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


def _standalone_containment(prediction: Any, reference: Any) -> float:
    """Match a reference when it occurs as a complete normalized field."""
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return 0.0
    if pred == ref:
        return 1.0
    # Containment is evidence of correctness, but a long explanatory answer
    # should not become an exact match merely because it repeats the answer.
    # The bounded ratio keeps correctness positive while leaving
    # answer_conciseness to report the presentation penalty separately.
    if " " not in ref:
        contained = ref in pred.split()
        pred_length = len(pred.split())
    else:
        contained = f" {ref} " in f" {pred} "
        pred_length = len(pred.split())
    if not contained:
        return 0.0
    return min(0.95, max(0.25, len(ref.split()) / max(1, pred_length)))


def _numeric_values(value: Any) -> list[float]:
    normalized = normalize_answer(value)
    values: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", normalized):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def numeric_match(prediction: Any, reference: Any) -> float:
    """Match numeric answers despite currency, grouping or unit suffixes."""
    predicted = _numeric_values(prediction)
    expected = _numeric_values(reference)
    if not predicted or not expected:
        return 0.0
    return float(any(abs(candidate - target) <= max(1e-9, abs(target) * 1e-9) for candidate in predicted for target in expected))


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

    field_value = FIELD_LABEL_PATTERN.sub("", prediction, count=1).strip() if isinstance(prediction, str) else ""

    def is_truncated_field_reference(reference: Any) -> bool:
        if not field_value or field_value == prediction:
            return False
        candidate_tokens = _answer_tokens(field_value)
        reference_tokens = _answer_tokens(reference)
        return (
            bool(candidate_tokens)
            and len(candidate_tokens) < len(reference_tokens)
            and reference_tokens[: len(candidate_tokens)] == candidate_tokens
        )

    def score_pair(candidate: Any, reference: Any) -> float:
        candidate_tokens = _answer_tokens(candidate)
        reference_tokens = _answer_tokens(reference)
        incomplete_entity = (
            bool(candidate_tokens)
            and len(candidate_tokens) < len(reference_tokens)
            and reference_tokens[: len(candidate_tokens)] == candidate_tokens
        )
        robust_score = max(
            exact_match(candidate, reference),
            numeric_match(candidate, reference),
            _standalone_containment(candidate, reference),
            token_f1(candidate, reference),
            anls(candidate, reference),
        )
        if incomplete_entity:
            robust_score = 0.0
        if metric in {"exact", "em", "exact_match", "accuracy", "acc"}:
            # Keep the legacy exact-accuracy field strict (apart from the
            # explicit answer variants such as ``The answer is: ...``).  The
            # robust score is exported separately as answer_correctness.
            return max(exact_match(candidate, reference), 0.0)
        if metric == "anls":
            return max(anls(candidate, reference), robust_score)
        if metric in {"f1", "token_f1"}:
            return max(token_f1(candidate, reference), robust_score)
        if metric == "contains":
            return max(_standalone_containment(candidate, reference), robust_score)
        if metric in {"json", "json_exact"}:
            return json_match(candidate, reference)
        if metric == "auto":
            return robust_score
        raise ValueError(f"Unsupported document reward metric: {metric}")

    return max(
        (
            score_pair(candidate, reference)
            for candidate in answer_variants(prediction)
            for reference in answers
            if not is_truncated_field_reference(reference)
        ),
        default=0.0,
    )


def _answer_conciseness(prediction: str, answers: list[Any], correctness: float, format_ok: bool) -> float:
    if correctness <= 0.0:
        return 0.0
    prediction_tokens = _answer_tokens(prediction)
    reference_lengths = [len(_answer_tokens(answer)) for answer in answers if _answer_tokens(answer)]
    if not prediction_tokens or not reference_lengths:
        return 0.0
    if any(exact_match(prediction, answer) for answer in answers):
        return 1.0 if format_ok else 0.9
    # A correct explanatory sentence is still useful.  Apply only a mild,
    # bounded presentation penalty; correctness remains a separate signal.
    extra_tokens = max(0, len(prediction_tokens) - min(reference_lengths))
    floor = 0.70 if not format_ok else 0.75
    return max(floor, 1.0 - 0.05 * min(extra_tokens, 5))


def compute_document_reward(
    response: str,
    label: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a GRPO-friendly reward in [-1, 1] and evaluation metrics."""
    prediction, format_ok = extract_final_answer(response, metadata)
    answers, metric = parse_label(label, metadata)
    # ``quality`` is deliberately independent from protocol validity.  A
    # single final span wrapped in explanatory prose is still semantically
    # scoreable; the format component below records the protocol violation.
    quality = score_answer(prediction, answers, metric) if prediction else 0.0
    quality = max(0.0, min(1.0, float(quality)))
    answer_acc = score_answer(prediction, answers, "exact_match") if prediction else 0.0
    strict_acc = max((exact_match(prediction, answer) for answer in answers), default=0.0) if prediction else 0.0
    raw_anls = max((anls(prediction, answer) for answer in answers), default=0.0) if prediction else 0.0
    conciseness = _answer_conciseness(prediction, answers, quality, format_ok)
    score = 2.0 * quality - 1.0
    if quality > 0.0 and not format_ok:
        # Keep a correct but non-conforming answer positive while making the
        # protocol violation visible to both evaluation and training logs.
        score -= 0.25
    return {
        "score": max(-1.0, min(1.0, score)),
        # ``acc`` remains exact-answer accuracy, but tolerates a conventional
        # answer lead-in. ``exact_acc`` is retained for strict-format reporting.
        "acc": answer_acc,
        "exact_acc": strict_acc,
        "quality": quality,
        "format": float(format_ok),
        "answer_correctness": quality,
        "answer_conciseness": conciseness,
        "format_validity": float(format_ok),
        "raw_anls": raw_anls,
        "pred": prediction,
        "metric": metric,
    }
