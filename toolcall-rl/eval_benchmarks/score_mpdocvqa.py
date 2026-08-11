"""Score MP-DocVQA validation with the standard threshold-0.5 ANLS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import bench_root_from, json_write  # noqa: E402
from eval_benchmarks.scoring_common import (  # noqa: E402
    anls,
    gold_records,
    join_gold_predictions,
    mean,
    prediction_records,
    prediction_text,
    protocol_invalid,
    write_scores,
)


def score(
    predictions_path: Path,
    gold_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = prediction_records(predictions_path)
    gold = gold_records(gold_path)
    joined, missing, _ = join_gold_predictions(gold, predictions)
    per_sample: list[dict[str, Any]] = []
    values: list[float] = []
    protocol_invalid_count = 0
    for task_id, gold_row, prediction in joined:
        final_answer = prediction_text(prediction)
        references = [str(value) for value in gold_row.get("answers", [])]
        score_value = anls(final_answer, references) if prediction is not None and final_answer != "" else 0.0
        values.append(score_value)
        if protocol_invalid(prediction):
            protocol_invalid_count += 1
        per_sample.append(
            {
                "task_id": task_id,
                "question_id": gold_row.get("question_id"),
                "doc_id": gold_row.get("doc_id"),
                "final_answer": final_answer,
                "references": references,
                "anls": score_value,
                "protocol_valid": bool(prediction and prediction.get("protocol_valid", False)),
            }
        )
    metrics = {
        "benchmark": "mpdocvqa",
        "split": "val",
        "num_questions": len(gold),
        "anls": mean(values),
        "missing_predictions": missing,
        "protocol_invalid": protocol_invalid_count,
    }
    return metrics, per_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    args = parser.parse_args()
    root = bench_root_from(args.bench_root)
    result_dir = root / "results" / "mpdocvqa"
    predictions = args.predictions or result_dir / "predictions.jsonl"
    gold = args.gold or root / "gold" / "mpdocvqa.jsonl"
    metrics, per_sample = score(predictions, gold)
    write_scores(result_dir / "per_sample_scores.jsonl", per_sample)
    json_write(result_dir / "official_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
