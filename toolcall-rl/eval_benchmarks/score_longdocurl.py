"""Score LongDocURL with its official generalized-accuracy implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import bench_root_from, json_write, jsonl_write  # noqa: E402
from eval_benchmarks.scoring_common import (  # noqa: E402
    gold_records,
    join_gold_predictions,
    mean,
    prediction_records,
    prediction_text,
    protocol_invalid,
    write_scores,
    vendor_path,
)


def _official_eval_score(vendor_dir: Path) -> Callable[[Any, str, Any], Any]:
    vendor_path(vendor_dir)
    try:
        from utils.utils_score_v3 import eval_score
    except ImportError as exc:
        raise RuntimeError(
            f"LongDocURL official scorer is missing from {vendor_dir}; "
            "clone dengc2023/LongDocURL on the server"
        ) from exc
    return eval_score


def _scalar(value: Any) -> float:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else 0.0
    return float(value)


def _tag_group(value: Any) -> str | None:
    tag = str(value or "").casefold()
    for name in ("understanding", "reasoning", "locating"):
        if name in tag:
            return name
    return None


def score(
    predictions_path: Path,
    gold_path: Path,
    vendor_dir: Path,
    official_input_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = prediction_records(predictions_path)
    gold = gold_records(gold_path)
    evaluator = _official_eval_score(vendor_dir)
    joined, missing, _ = join_gold_predictions(gold, predictions)
    official_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    all_scores: list[float] = []
    grouped_scores: dict[str, list[float]] = {name: [] for name in ("understanding", "reasoning", "locating")}
    protocol_invalid_count = 0
    for task_id, gold_row, prediction in joined:
        final_answer = prediction_text(prediction)
        pred = final_answer if final_answer != "" else "Fail to extract"
        gold_answer = gold_row.get("answer", "")
        answer_format = gold_row.get("answer_format")
        score_value = 0.0 if pred == "Fail to extract" else _scalar(evaluator(gold_answer, pred, answer_format))
        score_value = max(0.0, min(1.0, score_value))
        tag_group = _tag_group(gold_row.get("task_tag"))
        if tag_group is not None:
            grouped_scores[tag_group].append(score_value)
        all_scores.append(score_value)
        if protocol_invalid(prediction):
            protocol_invalid_count += 1
        official_rows.append(
            {
                "question_id": gold_row.get("question_id"),
                "pred": pred,
                "answer": gold_answer,
                "answer_format": answer_format,
                "task_tag": gold_row.get("task_tag"),
                "question_type": gold_row.get("question_type"),
                "subTask": gold_row.get("subTask"),
            }
        )
        per_sample.append(
            {
                "task_id": task_id,
                "question_id": gold_row.get("question_id"),
                "final_answer": final_answer,
                "official_prediction": pred,
                "score": score_value,
                "task_tag": gold_row.get("task_tag"),
                "protocol_valid": bool(prediction and prediction.get("protocol_valid", False)),
            }
        )
    jsonl_write(official_input_path, official_rows)
    metrics = {
        "benchmark": "longdocurl",
        "split": "public",
        "num_questions": len(gold),
        "generalized_accuracy": mean(all_scores),
        "understanding": mean(grouped_scores["understanding"]),
        "reasoning": mean(grouped_scores["reasoning"]),
        "locating": mean(grouped_scores["locating"]),
        "missing_predictions": missing,
        "protocol_invalid": protocol_invalid_count,
    }
    return metrics, per_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--vendor-dir", type=Path, default=None)
    parser.add_argument("--official-input", type=Path, default=None)
    args = parser.parse_args()
    root = bench_root_from(args.bench_root)
    result_dir = root / "results" / "longdocurl"
    predictions = args.predictions or result_dir / "predictions.jsonl"
    gold = args.gold or root / "gold" / "longdocurl.jsonl"
    vendor = args.vendor_dir or Path(__file__).resolve().parent / "vendors" / "LongDocURL"
    official_input = args.official_input or result_dir / "official_input.jsonl"
    metrics, per_sample = score(predictions, gold, vendor, official_input)
    write_scores(result_dir / "per_sample_scores.jsonl", per_sample)
    json_write(result_dir / "official_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
