"""Validate prepared eval/gold sidecars before any model rollout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks import BENCHMARKS  # noqa: E402
from eval_benchmarks.common import (  # noqa: E402
    bench_root_from,
    forbidden_prompt_markers,
    jsonl_read,
    keyed_rows,
    pdf_page_count,
)
from eval_benchmarks.common import FORBIDDEN_PROMPT_MARKERS  # noqa: E402


EXPECTED_COUNTS = {
    "longdocurl": (2325, None),
    "mpdocvqa": (5187, 927),
    "dude": (6315, None),
}


def _contains_direct_label_concat(prompt: str, answers: list[str]) -> bool:
    """Catch adapter mistakes without rejecting normal question wording."""

    explicit_label = re.search(
        r"(?:^|\n)\s*(?:gold|label|reference|answer(?:s)?)\s*(?:answers?)?\s*[:=]",
        prompt,
        re.IGNORECASE,
    )
    if explicit_label is None:
        return False
    folded_prompt = prompt.casefold()
    return any(answer and str(answer).casefold() in folded_prompt for answer in answers)


def validate(
    benchmark: str,
    bench_root: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    bench_root = bench_root_from(bench_root)
    eval_path = bench_root / "eval" / f"{benchmark}.jsonl"
    gold_path = bench_root / "gold" / f"{benchmark}.jsonl"
    if not eval_path.is_file() or not gold_path.is_file():
        raise FileNotFoundError(f"missing prepared files for {benchmark}: {eval_path}, {gold_path}")
    eval_rows = list(jsonl_read(eval_path))
    gold_rows = list(jsonl_read(gold_path))
    eval_by_id: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(eval_rows, 1):
        metadata = row.get("metadata")
        task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
        if not task_id:
            raise ValueError(f"{eval_path}:{row_number}: metadata.task_id missing")
        task_id = str(task_id)
        if task_id in eval_by_id:
            raise ValueError(f"{eval_path}:{row_number}: duplicate task_id={task_id}")
        eval_by_id[task_id] = row
    gold_by_id = keyed_rows(gold_path)
    if set(eval_by_id) != set(gold_by_id):
        missing_in_gold = sorted(set(eval_by_id) - set(gold_by_id))
        missing_in_eval = sorted(set(gold_by_id) - set(eval_by_id))
        raise ValueError(
            f"{benchmark}: eval/gold task IDs differ; "
            f"missing_in_gold={missing_in_gold[:5]}, missing_in_eval={missing_in_eval[:5]}"
        )

    document_ids: set[str] = set()
    question_ids: set[str] = set()
    for row_number, row in enumerate(eval_rows, 1):
        prompt = str(row.get("prompt", ""))
        label = row.get("label")
        metadata = row.get("metadata")
        if not isinstance(label, dict) or not isinstance(label.get("answers"), list):
            raise ValueError(f"{eval_path}:{row_number}: label.answers must be a list")
        if not isinstance(metadata, dict):
            raise ValueError(f"{eval_path}:{row_number}: metadata must be an object")
        forbidden_metadata = [
            field for field in FORBIDDEN_PROMPT_MARKERS if field in metadata
        ]
        if forbidden_metadata:
            raise ValueError(
                f"{eval_path}:{row_number}: GT/navigation fields in metadata: "
                f"{forbidden_metadata}"
            )
        task_id = str(metadata.get("task_id", ""))
        expected_prefix = f"{benchmark}:"
        if not task_id.startswith(expected_prefix):
            raise ValueError(f"{eval_path}:{row_number}: invalid task_id {task_id!r}")
        question_id = str(metadata.get("question_id", ""))
        if question_id in question_ids:
            raise ValueError(f"{eval_path}:{row_number}: duplicate question_id={question_id}")
        question_ids.add(question_id)
        document_ids.add(str(metadata.get("doc_id", "")))
        document_path_value = metadata.get("document_path")
        if not isinstance(document_path_value, str):
            raise ValueError(f"{eval_path}:{row_number}: metadata.document_path missing")
        document_path = Path(document_path_value)
        if not document_path.is_absolute():
            raise ValueError(f"{eval_path}:{row_number}: document_path is not absolute")
        if not document_path.is_file():
            raise FileNotFoundError(f"{eval_path}:{row_number}: document_path missing: {document_path}")
        actual_pages = pdf_page_count(document_path)
        expected_pages = int(metadata.get("page_count", 0) or 0)
        if expected_pages < 1 or actual_pages != expected_pages:
            raise ValueError(
                f"{eval_path}:{row_number}: page_count metadata={expected_pages}, actual={actual_pages}"
            )
        markers = forbidden_prompt_markers(prompt)
        if markers:
            raise ValueError(f"{eval_path}:{row_number}: GT/navigation markers in prompt: {markers}")
        if _contains_direct_label_concat(prompt, [str(value) for value in label["answers"]]):
            raise ValueError(f"{eval_path}:{row_number}: prompt appears to contain concatenated gold label")

    if not eval_rows:
        raise ValueError(f"{eval_path}: no rows")
    if not allow_partial:
        expected = EXPECTED_COUNTS.get(benchmark)
        if expected is not None:
            expected_questions, expected_docs = expected
            if len(eval_rows) != expected_questions:
                raise ValueError(f"{benchmark}: expected {expected_questions} rows, found {len(eval_rows)}")
            if expected_docs is not None and len(document_ids) != expected_docs:
                raise ValueError(f"{benchmark}: expected {expected_docs} documents, found {len(document_ids)}")
        elif benchmark == "docvqa2026" and len(document_ids) != 25:
            raise ValueError(f"docvqa2026: expected 25 documents, found {len(document_ids)}")
    return {
        "benchmark": benchmark,
        "eval_path": str(eval_path),
        "gold_path": str(gold_path),
        "num_questions": len(eval_rows),
        "num_documents": len(document_ids),
        "task_ids_unique": True,
        "allow_partial": allow_partial,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], required=True)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--allow-partial", action="store_true", help="permit an explicit smoke subset")
    args = parser.parse_args()
    benchmarks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    reports = [validate(name, bench_root_from(args.bench_root), allow_partial=args.allow_partial) for name in benchmarks]
    print(json.dumps(reports[0] if len(reports) == 1 else reports, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
