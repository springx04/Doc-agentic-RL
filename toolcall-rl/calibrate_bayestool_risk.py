"""Fit the BayesTool AnswerRiskCalibrator on validation rollouts.

The input is JSONL exported from validation rollouts.  Each row may provide a
binary error/risk label directly, or one of the common correctness fields
(``answer_correct``, ``exact_acc``, ``acc``, or ``quality``).  The fitted
checkpoint is plain JSON so it can be loaded by rollout workers without
adding a runtime dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from bayestool.decision import AnswerRiskCalibrator
from bayestool.schema import TaskStateView


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [dict(json.loads(line)) for line in handle if line.strip()]


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return dict(value) if isinstance(value, Mapping) else dict(record)


def _dotted_value(value: Mapping[str, Any], key: str) -> Any:
    current: Any = value
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _binary_risk(value: Any) -> float | None:
    if isinstance(value, bool):
        return 0.0 if value else 1.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric:
            return None
        # Correctness/probability fields use [0, 1], where 1 means no risk.
        return 1.0 if numeric < 0.5 else 0.0
    text = str(value or "").strip().casefold()
    if text in {"1", "true", "yes", "correct", "supported", "success", "ok"}:
        return 0.0
    if text in {"0", "false", "no", "incorrect", "unsupported", "error", "failure", "failed"}:
        return 1.0
    return None


def _risk_value(value: Any) -> float | None:
    """Parse a field whose numeric meaning is explicitly risk probability."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        return None if numeric != numeric else max(0.0, min(1.0, numeric))
    text = str(value or "").strip().casefold()
    if text in {"1", "true", "yes", "risk", "error", "incorrect", "unsafe"}:
        return 1.0
    if text in {"0", "false", "no", "safe", "correct", "success", "ok"}:
        return 0.0
    return None


def _risk_label(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    label_key: str | None,
    *,
    label_is_risk: bool = False,
) -> float | None:
    if label_key:
        value = _dotted_value(record, label_key)
        if value is None:
            value = _dotted_value(metadata, label_key)
        return _risk_value(value) if label_is_risk else _binary_risk(value)
    value = metadata.get("risk_label", record.get("risk_label"))
    if value is not None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            text = str(value).strip().casefold()
            if text in {"1", "true", "yes", "risk", "error"}:
                return 1.0
            if text in {"0", "false", "no", "safe", "correct"}:
                return 0.0
    for key in ("answer_error", "answer_incorrect", "incorrect"):
        value = metadata.get(key, record.get(key))
        if value is not None:
            parsed = _binary_risk(value)
            if parsed is not None:
                return parsed
    for key in ("answer_correct", "correct", "exact_acc", "acc", "quality"):
        value = metadata.get(key, record.get(key))
        if value is not None:
            return _binary_risk(value)
    return None


def build_validation_rows(
    records: list[Mapping[str, Any]],
    *,
    label_key: str | None = None,
    label_is_risk: bool = False,
) -> tuple[list[dict[str, float]], list[float]]:
    seed_calibrator = AnswerRiskCalibrator()
    features: list[dict[str, float]] = []
    labels: list[float] = []
    for record in records:
        metadata = _metadata(record)
        label = _risk_label(record, metadata, label_key, label_is_risk=label_is_risk)
        if label is None:
            continue
        navigation = metadata.get("navigation_state")
        navigation = dict(navigation) if isinstance(navigation, Mapping) else metadata
        budget = navigation.get("remaining_tool_budget", metadata.get("tool_budget", 0))
        try:
            budget = int(budget or 0)
        except (TypeError, ValueError):
            budget = 0
        task = TaskStateView.from_navigation_state(navigation, tool_budget=budget)
        features.append(seed_calibrator.risk_features(task, metadata))
        labels.append(float(label))
    return features, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Validation rollout JSONL")
    parser.add_argument("--output", type=Path, required=True, help="JSON calibrator checkpoint")
    parser.add_argument("--label-key", default=None, help="Optional dotted risk/correctness label field")
    parser.add_argument(
        "--label-is-risk",
        action="store_true",
        help="Interpret --label-key as a risk probability/flag; otherwise it is treated as a correctness field.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    args = parser.parse_args(argv)

    records = _read_jsonl(args.input)
    features, labels = build_validation_rows(
        records,
        label_key=args.label_key,
        label_is_risk=args.label_is_risk,
    )
    if not features:
        raise ValueError("validation input contains no rows with a usable risk/correctness label")
    calibrator = AnswerRiskCalibrator()
    metrics = calibrator.fit(
        features,
        labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    payload = {
        "version": "bayestool-risk-calibrator-v1",
        "calibrator": calibrator.to_dict(),
        "metrics": metrics,
        "source_rows": len(records),
        "fitted_rows": len(features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
