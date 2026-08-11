"""Score DocVQA 2026 predictions with the official evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import bench_root_from, json_write  # noqa: E402
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


def _official_evaluator(vendor_dir: Path) -> Callable[[str, list[str]], Any]:
    vendor_path(vendor_dir)
    try:
        from eval_utils import evaluate_docvqa_prediction
    except ImportError as exc:
        raise RuntimeError(
            f"DocVQA official evaluator is missing from {vendor_dir}; "
            "clone VLR-CVC/DocVQA2026 on the server"
        ) from exc
    return evaluate_docvqa_prediction


def score(
    predictions_path: Path,
    gold_path: Path,
    vendor_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = prediction_records(predictions_path)
    gold = gold_records(gold_path)
    evaluator = _official_evaluator(vendor_dir)
    joined, missing, _ = join_gold_predictions(gold, predictions)
    per_sample: list[dict[str, Any]] = []
    domain_scores: dict[str, list[float]] = defaultdict(list)
    protocol_invalid_count = 0
    for task_id, gold_row, prediction in joined:
        final_answer = prediction_text(prediction)
        answers = [str(value) for value in gold_row.get("answers", [])]
        raw_for_official = f"FINAL ANSWER: {final_answer}"
        result = evaluator(raw_for_official, answers)
        if isinstance(result, (tuple, list)):
            is_correct = bool(result[0]) if result else False
            extracted = result[1] if len(result) > 1 else final_answer
        else:
            is_correct = bool(result)
            extracted = final_answer
        score_value = 1.0 if is_correct else 0.0
        if protocol_invalid(prediction):
            protocol_invalid_count += 1
        domain = str(gold_row.get("doc_category") or "unknown")
        domain_scores[domain].append(score_value)
        per_sample.append(
            {
                "task_id": task_id,
                "question_id": gold_row.get("question_id"),
                "doc_id": gold_row.get("doc_id"),
                "doc_category": domain,
                "final_answer": final_answer,
                "official_extracted": extracted,
                "correct": is_correct,
                "protocol_valid": bool(prediction and prediction.get("protocol_valid", False)),
                "score": score_value,
            }
        )
    by_domain = {
        domain: {
            "num_questions": len(values),
            "accuracy": mean(values),
        }
        for domain, values in sorted(domain_scores.items())
    }
    metrics = {
        "benchmark": "docvqa2026",
        "split": "val",
        "num_questions": len(gold),
        "accuracy": mean(row["score"] for row in per_sample),
        "by_domain": by_domain,
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
    args = parser.parse_args()
    root = bench_root_from(args.bench_root)
    result_dir = root / "results" / "docvqa2026"
    predictions = args.predictions or result_dir / "predictions.jsonl"
    gold = args.gold or root / "gold" / "docvqa2026.jsonl"
    vendor = args.vendor_dir or Path(__file__).resolve().parent / "vendors" / "DocVQA2026"
    metrics, per_sample = score(predictions, gold, vendor)
    write_scores(result_dir / "per_sample_scores.jsonl", per_sample)
    json_write(result_dir / "official_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
