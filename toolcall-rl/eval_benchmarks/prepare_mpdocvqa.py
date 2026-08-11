"""Prepare MP-DocVQA validation shards as one PDF per document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import (  # noqa: E402
    as_string_list,
    bench_root_from,
    images_to_pdf,
    jsonl_write,
    make_eval_row,
    parse_maybe_literal,
    pdf_page_count,
    safe_component,
)


def _iter_rows(shards: list[Path]) -> Iterator[dict[str, Any]]:
    if not shards:
        raise FileNotFoundError("no MP-DocVQA validation shards matching val-*.parquet")
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "parquet",
            data_files={"validation": [str(path) for path in shards]},
            split="validation",
            streaming=True,
        )
        for row in dataset:
            yield dict(row)
        return
    except ImportError:
        pass

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - minimal server fallback
        raise RuntimeError("MP-DocVQA preparation requires datasets or pyarrow") from exc
    for shard in shards:
        parquet_file = parquet.ParquetFile(shard)
        for batch in parquet_file.iter_batches():
            yield from batch.to_pylist()


def _page_ids(row: dict[str, Any]) -> list[Any]:
    value = parse_maybe_literal(row.get("page_ids"))
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"MP-DocVQA {row.get('questionId')!r}: page_ids is not a list")
    result = list(value)
    if not result or len(result) > 20:
        raise ValueError(f"MP-DocVQA {row.get('questionId')!r}: invalid page count {len(result)}")
    return result


def _answers_for_eval(value: Any) -> tuple[list[str], list[str]]:
    """Return source answers and the non-empty project label representation."""

    source_answers = as_string_list(value)
    return source_answers, source_answers if source_answers else [""]


def prepare(
    bench_root: Path,
    raw_root: Path | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    raw_root = raw_root or bench_root / "raw" / "mpdocvqa"
    shards = sorted(raw_root.rglob("val-*.parquet"))
    documents_root = bench_root / "documents" / "mpdocvqa"
    eval_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    materialized_docs: dict[str, tuple[str, ...]] = {}
    page_counts: dict[str, int] = {}

    for row_number, row in enumerate(_iter_rows(shards), 1):
        if limit is not None and len(eval_rows) >= limit:
            break
        question_id = str(row["questionId"])
        doc_id = str(row["doc_id"])
        page_ids = _page_ids(row)
        answers, eval_answers = _answers_for_eval(row.get("answers"))
        page_key = tuple(str(value) for value in page_ids)
        previous = materialized_docs.get(doc_id)
        if previous is None:
            images = [row.get(f"image_{index}") for index in range(1, len(page_ids) + 1)]
            if any(image is None for image in images):
                raise ValueError(f"MP-DocVQA {question_id}: page image list is incomplete")
            pdf_path = documents_root / f"{safe_component(doc_id)}.pdf"
            images_to_pdf(images, pdf_path, expected_page_count=len(page_ids))
            materialized_docs[doc_id] = page_key
            page_counts[doc_id] = pdf_page_count(pdf_path)
        elif previous != page_key:
            raise ValueError(
                f"MP-DocVQA doc_id={doc_id!r} has inconsistent page_ids: {previous!r} != {page_key!r}"
            )

        document_path = documents_root / f"{safe_component(doc_id)}.pdf"
        eval_rows.append(
            make_eval_row(
                benchmark="mpdocvqa",
                question_id=question_id,
                doc_id=doc_id,
                document_path=document_path,
                page_count=page_counts[doc_id],
                question=str(row.get("question", "")),
                answers=eval_answers,
            )
        )
        gold_index = int(row["answer_page_idx"])
        if not 0 <= gold_index < len(page_ids):
            raise ValueError(f"MP-DocVQA {question_id}: answer_page_idx={gold_index} out of range")
        gold_rows.append(
            {
                "task_id": f"mpdocvqa:{question_id}",
                "question_id": question_id,
                "doc_id": doc_id,
                "answers": answers,
                "page_ids": list(page_key),
                "gold_answer_page_0based": gold_index,
                "gold_answer_page_1based": gold_index + 1,
            }
        )

    if not eval_rows:
        raise RuntimeError("MP-DocVQA preparation produced zero questions")
    if limit is None:
        if len(eval_rows) != 5187:
            raise RuntimeError(f"expected 5187 MP-DocVQA rows, found {len(eval_rows)}")
        if len(materialized_docs) != 927:
            raise RuntimeError(f"expected 927 MP-DocVQA documents, found {len(materialized_docs)}")
    task_ids = [row["metadata"]["task_id"] for row in eval_rows]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("MP-DocVQA preparation produced duplicate task IDs")
    eval_path = bench_root / "eval" / "mpdocvqa.jsonl"
    gold_path = bench_root / "gold" / "mpdocvqa.jsonl"
    jsonl_write(eval_path, eval_rows)
    jsonl_write(gold_path, gold_rows)
    return {
        "benchmark": "mpdocvqa",
        "split": "val",
        "num_questions": len(eval_rows),
        "num_documents": len(materialized_docs),
        "eval_path": str(eval_path),
        "gold_path": str(gold_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="bounded local smoke preparation")
    args = parser.parse_args()
    report = prepare(bench_root_from(args.bench_root), args.raw_root, limit=args.limit)
    import json

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
