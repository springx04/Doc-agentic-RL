"""Prepare LongDocURL's public PDF+QA validation-style benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import (  # noqa: E402
    as_string_list,
    bench_root_from,
    jsonl_read,
    jsonl_write,
    make_eval_row,
    pdf_page_count,
)


def _pdf_index(documents_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in documents_root.rglob("*.pdf"):
        key = path.stem
        if key in index and index[key] != path:
            raise RuntimeError(f"duplicate LongDocURL PDF stem {key!r}: {index[key]} and {path}")
        index[key] = path
    return index


def prepare(
    bench_root: Path,
    input_path: Path | None = None,
    documents_root: Path | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    source = input_path or bench_root / "raw" / "longdocurl" / "LongDocURL_public.jsonl"
    documents_root = documents_root or bench_root / "documents" / "longdocurl"
    if not source.is_file():
        raise FileNotFoundError(f"LongDocURL QA JSONL not found: {source}")
    if not documents_root.is_dir():
        raise FileNotFoundError(f"LongDocURL document directory not found: {documents_root}")

    pdf_index = _pdf_index(documents_root)
    eval_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    page_counts: dict[Path, int] = {}
    for line_number, row in enumerate(jsonl_read(source), 1):
        if limit is not None and len(eval_rows) >= limit:
            break
        if "question_id" not in row or "doc_no" not in row:
            raise ValueError(f"{source}:{line_number}: missing question_id or doc_no")
        question_id = str(row["question_id"])
        doc_id = str(row["doc_no"])
        document_path = pdf_index.get(doc_id)
        if document_path is None:
            raise FileNotFoundError(f"LongDocURL doc_no={doc_id!r} has no extracted PDF")
        if document_path not in page_counts:
            page_counts[document_path] = pdf_page_count(document_path)
        answer = row.get("answer")
        answers = as_string_list(answer)
        if not answers:
            answers = [""]
        eval_rows.append(
            make_eval_row(
                benchmark="longdocurl",
                question_id=question_id,
                doc_id=doc_id,
                document_path=document_path,
                page_count=page_counts[document_path],
                question=str(row.get("question", "")),
                answers=answers,
            )
        )
        gold_rows.append(
            {
                "task_id": f"longdocurl:{question_id}",
                "question_id": question_id,
                "doc_id": doc_id,
                "answer": answer,
                "answer_format": row.get("answer_format"),
                "task_tag": row.get("task_tag"),
                "question_type": row.get("question_type"),
                "subTask": row.get("subTask"),
                "evidence_pages": row.get("evidence_pages", []),
                "evidence_sources": row.get("evidence_sources", []),
                "detailed_evidences": row.get("detailed_evidences", []),
            }
        )

    if not eval_rows:
        raise RuntimeError("LongDocURL preparation produced zero questions")
    if limit is None and len(eval_rows) != 2325:
        raise RuntimeError(f"expected 2325 LongDocURL rows, found {len(eval_rows)}")
    task_ids = [row["metadata"]["task_id"] for row in eval_rows]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("LongDocURL preparation produced duplicate task IDs")
    eval_path = bench_root / "eval" / "longdocurl.jsonl"
    gold_path = bench_root / "gold" / "longdocurl.jsonl"
    jsonl_write(eval_path, eval_rows)
    jsonl_write(gold_path, gold_rows)
    return {
        "benchmark": "longdocurl",
        "split": "public",
        "num_questions": len(eval_rows),
        "num_documents": len({row["metadata"]["doc_id"] for row in eval_rows}),
        "eval_path": str(eval_path),
        "gold_path": str(gold_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--documents-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="bounded local smoke preparation")
    args = parser.parse_args()
    report = prepare(
        bench_root_from(args.bench_root),
        args.input,
        args.documents_root,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
