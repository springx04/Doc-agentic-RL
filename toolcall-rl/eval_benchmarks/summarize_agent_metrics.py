"""Summarize Agent-only rollout and post-hoc navigation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks import BENCHMARKS  # noqa: E402
from eval_benchmarks.common import bench_root_from, integer_list, json_write  # noqa: E402
from eval_benchmarks.scoring_common import (  # noqa: E402
    gold_records,
    join_gold_predictions,
    mean,
    prediction_records,
    prediction_text,
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _call_pages(call: Any) -> set[int]:
    call = _mapping(call)
    arguments = call.get("arguments", call.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    arguments = _mapping(arguments)
    pages: list[int] = []
    for key in ("page_number", "page", "answer_page"):
        if key in arguments:
            pages.extend(integer_list(arguments[key]))
    for key in ("page_numbers", "pages"):
        if key in arguments:
            pages.extend(integer_list(arguments[key]))
    return set(pages)


def _tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("tool_calls", [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _official_score(benchmark: str, root: Path) -> float | None:
    path = root / "results" / benchmark / "official_metrics.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("accuracy", "generalized_accuracy", "anls", "overall_anls"):
        if isinstance(metrics.get(key), (int, float)):
            return float(metrics[key])
    return None


def summarize(
    benchmark: str,
    bench_root: Path,
    *,
    predictions_path: Path | None = None,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    result_dir = bench_root / "results" / benchmark
    predictions = prediction_records(predictions_path or result_dir / "predictions.jsonl")
    gold = gold_records(gold_path or bench_root / "gold" / f"{benchmark}.jsonl")
    joined, missing, _ = join_gold_predictions(gold, predictions)
    n = len(gold)
    protocol_valid_values: list[float] = []
    completion_values: list[float] = []
    tool_calls: list[float] = []
    valid_tool_calls: list[float] = []
    tool_errors: list[float] = []
    unique_pages: list[float] = []
    rendered_pages: list[float] = []
    ocr_pages: list[float] = []
    duplicate_calls = 0
    no_information_gain_calls = 0
    has_duplicate_metric = False
    has_no_gain_metric = False
    page_visited_values: list[float] = []
    page_first_hits: list[float] = []
    page_normalized_hits: list[float] = []
    evidence_recalls: list[float] = []
    evidence_precisions: list[float] = []
    any_evidence_hits: list[float] = []
    all_evidence_hits: list[float] = []
    hit_conditioned_first_hits: list[float] = []
    for _, gold_row, prediction in joined:
        prediction = prediction or {}
        visited = set(integer_list(prediction.get("visited_pages", [])))
        rendered = set(integer_list(prediction.get("rendered_pages", [])))
        ocr = set(integer_list(prediction.get("ocr_pages", [])))
        calls = _tool_calls(prediction)
        call_count = int(prediction.get("tool_call_count", len(calls)) or 0)
        error_count = int(prediction.get("tool_error_count", 0) or 0)
        protocol_valid_values.append(float(bool(prediction.get("protocol_valid", False))))
        completion_values.append(float(bool(prediction.get("final_answer", ""))))
        tool_calls.append(float(call_count))
        valid_tool_calls.append(float(prediction.get("valid_tool_call_count", 0) or 0))
        tool_errors.append(float(error_count))
        unique_pages.append(float(len(visited)))
        rendered_pages.append(float(len(rendered)))
        ocr_pages.append(float(len(ocr)))
        if "duplicate_page_calls" in prediction:
            has_duplicate_metric = True
            duplicate_calls += int(prediction.get("duplicate_page_calls", 0) or 0)
        if "no_information_gain_calls" in prediction:
            has_no_gain_metric = True
            no_information_gain_calls += int(prediction.get("no_information_gain_calls", 0) or 0)

        if benchmark == "mpdocvqa":
            try:
                gold_page = int(gold_row["gold_answer_page_1based"])
            except (KeyError, TypeError, ValueError):
                gold_page = None
            if gold_page is not None:
                hit = gold_page in visited
                page_visited_values.append(float(hit))
                first_hit: int | None = None
                for step, call in enumerate(calls, 1):
                    if gold_page in _call_pages(call):
                        first_hit = step
                        break
                if first_hit is not None:
                    hit_conditioned_first_hits.append(float(first_hit))
                    page_first_hits.append(float(first_hit))
                    page_normalized_hits.append(1.0 - (first_hit - 1) / max(call_count, 1))
                else:
                    page_first_hits.append(0.0)
                    page_normalized_hits.append(0.0)
        elif benchmark == "longdocurl":
            gold_pages = set(integer_list(gold_row.get("evidence_pages", [])))
            intersection = gold_pages & visited
            evidence_recalls.append(len(intersection) / len(gold_pages) if gold_pages else 0.0)
            evidence_precisions.append(len(intersection) / len(visited) if visited else 0.0)
            any_evidence_hits.append(float(bool(intersection)))
            all_evidence_hits.append(float(bool(gold_pages) and intersection == gold_pages))

    total_tool_calls = sum(tool_calls)
    metrics: dict[str, Any] = {
        "benchmark": benchmark,
        "num_questions": n,
        "missing_predictions": missing,
        "protocol_valid_rate": mean(protocol_valid_values),
        "completion_rate": mean(completion_values),
        "mean_tool_calls": mean(tool_calls),
        "mean_valid_tool_calls": mean(valid_tool_calls),
        "tool_error_rate": sum(tool_errors) / total_tool_calls if total_tool_calls else 0.0,
        "mean_unique_pages_visited": mean(unique_pages),
        "mean_rendered_pages": mean(rendered_pages),
        "mean_ocr_pages": mean(ocr_pages),
        "duplicate_page_call_rate": (
            duplicate_calls / total_tool_calls if has_duplicate_metric and total_tool_calls else (0.0 if has_duplicate_metric else None)
        ),
        "no_information_gain_call_rate": (
            no_information_gain_calls / total_tool_calls if has_no_gain_metric and total_tool_calls else (0.0 if has_no_gain_metric else None)
        ),
        "official_score": _official_score(benchmark, bench_root),
    }
    if benchmark == "mpdocvqa":
        metrics.update(
            {
                "gold_page_visited_rate": mean(page_visited_values),
                "mean_gold_page_first_hit_tool_step": mean(page_first_hits),
                "hit_conditioned_mean_first_hit_step": mean(hit_conditioned_first_hits),
                "unconditional_first_hit_normalized_score": mean(page_normalized_hits),
                "gold_page_first_hit_missing_count": len(page_first_hits) - len(hit_conditioned_first_hits),
            }
        )
    elif benchmark == "longdocurl":
        metrics.update(
            {
                "evidence_page_recall": mean(evidence_recalls),
                "evidence_page_precision": mean(evidence_precisions),
                "any_evidence_hit_rate": mean(any_evidence_hits),
                "all_evidence_hit_rate": mean(all_evidence_hits),
            }
        )
    output = result_dir / "agent_metrics.json"
    json_write(output, metrics)
    return metrics


def write_summary(bench_root: Path, metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        metric = metrics[benchmark]
        split = {"docvqa2026": "val", "longdocurl": "public", "mpdocvqa": "val", "dude": "val"}[benchmark]
        score = metric.get("official_score")
        rows.append(
            {
                "benchmark": benchmark,
                "split": split,
                "num_questions": metric.get("num_questions", 0),
                "official_metric": {
                    "docvqa2026": "Accuracy",
                    "longdocurl": "Generalized Accuracy",
                    "mpdocvqa": "ANLS",
                    "dude": "ANLS",
                }[benchmark],
                "score": score,
                "protocol_valid": metric.get("protocol_valid_rate"),
                "mean_tool_calls": metric.get("mean_tool_calls"),
                "mean_pages_visited": metric.get("mean_unique_pages_visited"),
            }
        )
    navigation = []
    if "longdocurl" in metrics:
        navigation.extend(
            {"benchmark": "LongDocURL", "evidence_metric": name, "value": metrics["longdocurl"].get(name)}
            for name in ("evidence_page_recall", "any_evidence_hit_rate")
        )
    if "mpdocvqa" in metrics:
        navigation.extend(
            {"benchmark": "MP-DocVQA", "evidence_metric": name, "value": metrics["mpdocvqa"].get(name)}
            for name in ("gold_page_visited_rate", "mean_gold_page_first_hit_tool_step")
        )
    summary = {"benchmarks": rows, "agent_navigation": navigation}
    json_write(bench_root / "results" / "summary.json", summary)
    lines = [
        "# Doc-Agentic-RL benchmark summary",
        "",
        "| Benchmark | Split | #Q | Official metric | Score | Protocol valid | Mean tool calls | Mean pages visited |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {split} | {num_questions} | {official_metric} | {score} | {protocol_valid} | {mean_tool_calls} | {mean_pages_visited} |".format(**row)
        )
    lines.extend(["", "## Agent navigation", "", "| Benchmark | Evidence metric | Value |", "|---|---|---:|"])
    for row in navigation:
        lines.append(f"| {row['benchmark']} | {row['evidence_metric']} | {row['value']} |")
    (bench_root / "results" / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (bench_root / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], required=True)
    parser.add_argument("--bench-root", default=None)
    args = parser.parse_args()
    root = bench_root_from(args.bench_root)
    if args.benchmark == "all":
        metrics = {benchmark: summarize(benchmark, root) for benchmark in BENCHMARKS}
        result = write_summary(root, metrics)
    else:
        result = summarize(args.benchmark, root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
