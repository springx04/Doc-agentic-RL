"""Prepare the public DocVQA 2026 validation parquet on the server.

Only ``val.parquet`` is consumed.  Each document row is materialized as one
multi-page PDF and each question becomes one canonical eval row.  The gold
page/evidence fields remain in a separate sidecar.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import (  # noqa: E402
    as_string_list,
    bench_root_from,
    images_to_pdf,
    jsonl_write,
    make_eval_row,
    pdf_page_count,
    safe_component,
)


def _iter_rows(parquet_path: Path) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on minimal servers
        raise RuntimeError("DocVQA preparation requires the datasets package") from exc
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    for row in dataset:
        yield dict(row)


def _field(mapping: Any, name: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(name)
    return getattr(mapping, name, None)


def _parallel_question_fields(row: Mapping[str, Any]) -> tuple[list[str], list[str], list[Any]]:
    questions = row.get("questions")
    answers = row.get("answers")
    question_ids = list(_field(questions, "question_id") or [])
    question_texts = list(_field(questions, "question") or [])
    answer_qids = list(_field(answers, "question_id") or [])
    answer_values = list(_field(answers, "answer") or [])
    if question_ids != answer_qids:
        raise ValueError(
            f"DocVQA question/answer IDs differ for doc_id={row.get('doc_id')!r}: "
            f"{question_ids!r} != {answer_qids!r}"
        )
    if not (len(question_ids) == len(question_texts) == len(answer_values)):
        raise ValueError(f"DocVQA fields are not aligned for doc_id={row.get('doc_id')!r}")
    return [str(value) for value in question_ids], [str(value) for value in question_texts], answer_values


DOCVQA_EXTRA_INSTRUCTION = """Follow the DocVQA 2026 answer formatting rules for the content inside <final>:
- If the question is unanswerable, return exactly Unknown.
- For multiple answers, preserve document order and separate items with ", ".
- Use standardized abbreviated units with one space between number and unit.
- Attach % directly to the number.
- Format dates as YYYY-MM-DD.
- Do not use thousands separators.
- Do not add explanatory prose."""


def prepare(
    bench_root: Path,
    input_path: Path | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    source = input_path or bench_root / "raw" / "docvqa2026" / "val.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"DocVQA validation parquet not found: {source}")

    documents_root = bench_root / "documents" / "docvqa2026"
    eval_path = bench_root / "eval" / "docvqa2026.jsonl"
    gold_path = bench_root / "gold" / "docvqa2026.jsonl"
    eval_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    document_count = 0
    seen_documents: set[str] = set()

    for row_index, row in enumerate(_iter_rows(source), 1):
        if limit is not None and len(eval_rows) >= limit:
            break
        doc_id = str(row["doc_id"])
        category = str(row.get("doc_category") or "unknown")
        images = row.get("document")
        if not isinstance(images, (list, tuple)):
            raise ValueError(f"DocVQA row {row_index} document is not a page list")
        if not images:
            raise ValueError(f"DocVQA row {row_index} has no document pages")
        question_ids, questions, answer_values = _parallel_question_fields(row)
        pdf_path = documents_root / f"{safe_component(doc_id)}.pdf"
        images_to_pdf(images, pdf_path, expected_page_count=len(images))
        page_count = pdf_page_count(pdf_path)
        if doc_id not in seen_documents:
            seen_documents.add(doc_id)
            document_count += 1

        for question_id, question, answer_value in zip(question_ids, questions, answer_values):
            if limit is not None and len(eval_rows) >= limit:
                break
            answers = as_string_list(answer_value)
            if not answers:
                raise ValueError(f"DocVQA {doc_id}:{question_id} has no answer reference")
            eval_rows.append(
                make_eval_row(
                    benchmark="docvqa2026",
                    question_id=question_id,
                    doc_id=doc_id,
                    document_path=pdf_path,
                    page_count=page_count,
                    question=question,
                    answers=answers,
                    extra_instruction=DOCVQA_EXTRA_INSTRUCTION,
                )
            )
            gold_rows.append(
                {
                    "task_id": f"docvqa2026:{question_id}",
                    "question_id": question_id,
                    "doc_id": doc_id,
                    "doc_category": category,
                    "answers": answers,
                }
            )

    if not eval_rows:
        raise RuntimeError("DocVQA preparation produced zero questions")
    if limit is None:
        if document_count != 25:
            raise RuntimeError(f"expected 25 DocVQA validation documents, found {document_count}")
    if len({row["metadata"]["task_id"] for row in eval_rows}) != len(eval_rows):
        raise RuntimeError("DocVQA preparation produced duplicate task IDs")
    jsonl_write(eval_path, eval_rows)
    jsonl_write(gold_path, gold_rows)
    return {
        "benchmark": "docvqa2026",
        "split": "val",
        "num_questions": len(eval_rows),
        "num_documents": document_count,
        "eval_path": str(eval_path),
        "gold_path": str(gold_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--input", type=Path, default=None, help="override raw val.parquet")
    parser.add_argument("--limit", type=int, default=None, help="bounded local smoke preparation")
    args = parser.parse_args()
    print(prepare(bench_root_from(args.bench_root), args.input, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
