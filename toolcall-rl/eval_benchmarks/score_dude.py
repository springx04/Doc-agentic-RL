"""Generate and score the DUDE validation submission with DUDEeval."""

from __future__ import annotations

import argparse
import json
import subprocess
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
    load_numeric_metric,
    mean,
    prediction_records,
    prediction_text,
    protocol_invalid,
    write_scores,
)


EMPTY_SENTINELS = {
    "",
    "unknown",
    "unanswerable",
    "not answerable",
    "not-answerable",
    "n/a",
}


def _load_root(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open(encoding="utf-8") as handle:
        root = json.load(handle)
    if not isinstance(root, dict) or not isinstance(root.get("data"), list):
        raise ValueError(f"DUDE annotation must contain a data list: {annotation_path}")
    return root


def build_official_gt(gold_path: Path, annotation_path: Path, output_path: Path) -> None:
    """Preserve the official root schema while restricting data to validation."""

    root = _load_root(annotation_path)
    validation_rows = [row for row in root["data"] if isinstance(row, dict) and row.get("data_split") == "val"]
    expected_ids = [str(row.get("questionId")) for row in validation_rows]
    gold = gold_records(gold_path)
    gold_ids = [str(row.get("question_id")) for row in gold.values()]
    if set(expected_ids) != set(gold_ids):
        raise RuntimeError(
            "DUDE annotation validation IDs do not match prepared gold: "
            f"annotation={len(expected_ids)} gold={len(gold_ids)}"
        )
    root["data"] = validation_rows
    json_write(output_path, root)


def build_submission(gold_path: Path, predictions_path: Path, output_path: Path) -> tuple[int, int]:
    predictions = prediction_records(predictions_path)
    gold = gold_records(gold_path)
    joined, missing, _ = join_gold_predictions(gold, predictions)
    submission: list[dict[str, Any]] = []
    for _, gold_row, prediction in joined:
        question_id = gold_row.get("questionId", gold_row.get("question_id"))
        prediction_text_value = prediction_text(prediction)
        if prediction_text_value.strip().casefold() in EMPTY_SENTINELS:
            answer = ""
        else:
            answer = prediction_text_value
        submission.append(
            {
                "questionId": str(question_id),
                "answers": [answer],
                "answers_confidence": [1],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_write(output_path, submission)
    return len(submission), missing


def run_official(
    vendor_dir: Path,
    gt_path: Path,
    submission_path: Path,
    output_dir: Path,
) -> Path:
    evaluator = vendor_dir / "evaluate_submission.py"
    if not evaluator.is_file():
        raise RuntimeError(
            f"DUDE official evaluator is missing from {vendor_dir}; "
            "clone Jordy-VL/DUDEeval on the server"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(evaluator),
        f"-g={gt_path}",
        f"-s={submission_path}",
        f"-o={output_dir}",
    ]
    subprocess.run(command, cwd=vendor_dir, check=True)
    result_path = output_dir / "results.json"
    if result_path.is_file():
        return result_path
    candidates = sorted(output_dir.rglob("*.json"))
    if not candidates:
        raise RuntimeError(f"DUDE evaluator completed without JSON output in {output_dir}")
    return candidates[0]


def score(
    predictions_path: Path,
    gold_path: Path,
    annotation_path: Path,
    vendor_dir: Path,
    result_dir: Path,
    *,
    skip_official: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = prediction_records(predictions_path)
    gold = gold_records(gold_path)
    joined, missing, _ = join_gold_predictions(gold, predictions)
    gt_path = result_dir / "dude_val_gt.json"
    submission_path = result_dir / "dude_val_submission.json"
    build_official_gt(gold_path, annotation_path, gt_path)
    _, submission_missing = build_submission(gold_path, predictions_path, submission_path)
    if submission_missing != missing:
        raise RuntimeError("DUDE submission/gold join disagrees on missing predictions")

    per_sample: list[dict[str, Any]] = []
    local_scores: list[float] = []
    protocol_invalid_count = 0
    for task_id, gold_row, prediction in joined:
        final_answer = prediction_text(prediction)
        references = [str(value) for value in gold_row.get("answers", [])]
        local_score = anls(final_answer, references) if prediction is not None and final_answer != "" else 0.0
        local_scores.append(local_score)
        if protocol_invalid(prediction):
            protocol_invalid_count += 1
        per_sample.append(
            {
                "task_id": task_id,
                "question_id": gold_row.get("question_id", gold_row.get("questionId")),
                "final_answer": final_answer,
                "local_anls": local_score,
                "protocol_valid": bool(prediction and prediction.get("protocol_valid", False)),
            }
        )

    official_result: Path | None = None
    official_score: float | None = None
    if not skip_official:
        official_result = run_official(vendor_dir, gt_path, submission_path, result_dir / "official")
        official_score = load_numeric_metric(official_result)
        if official_score is None:
            raise RuntimeError(f"could not find Overall ANLS in DUDE result: {official_result}")
    metrics: dict[str, Any] = {
        "benchmark": "dude",
        "split": "val",
        "num_questions": len(gold),
        "overall_anls": official_score,
        "local_mean_anls": mean(local_scores),
        "missing_predictions": missing,
        "protocol_invalid": protocol_invalid_count,
        "official_result": str(official_result) if official_result else None,
        "official_pending": bool(skip_official),
    }
    return metrics, per_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--vendor-dir", type=Path, default=None)
    parser.add_argument("--skip-official", action="store_true", help="prepare artifacts without running DUDEeval")
    args = parser.parse_args()
    root = bench_root_from(args.bench_root)
    result_dir = root / "results" / "dude"
    predictions = args.predictions or result_dir / "predictions.jsonl"
    gold = args.gold or root / "gold" / "dude.jsonl"
    annotation = args.annotation or root / "raw" / "dude" / "2023-03-23_DUDE_gt_test_PUBLIC.json"
    vendor = args.vendor_dir or Path(__file__).resolve().parent / "vendors" / "DUDEeval"
    metrics, per_sample = score(
        predictions,
        gold,
        annotation,
        vendor,
        result_dir,
        skip_official=args.skip_official,
    )
    write_scores(result_dir / "per_sample_scores.jsonl", per_sample)
    json_write(result_dir / "official_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
