"""Prepare only DUDE validation PDFs and annotations from the public archive."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import (  # noqa: E402
    as_string_list,
    bench_root_from,
    jsonl_write,
    make_eval_row,
    pdf_page_count,
    safe_component,
)


def _read_annotation(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        root = json.load(handle)
    if not isinstance(root, dict) or not isinstance(root.get("data"), list):
        raise ValueError(f"DUDE annotation must be a JSON object with a data list: {path}")
    rows = [row for row in root["data"] if isinstance(row, dict) and row.get("data_split") == "val"]
    return root, rows


def _extract_validation_pdfs(archive: Path, documents_root: Path, val_doc_ids: set[str]) -> set[str]:
    documents_root.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    for doc_id in val_doc_ids:
        destination = documents_root / f"{safe_component(doc_id)}.pdf"
        if destination.is_file():
            try:
                if pdf_page_count(destination) >= 1:
                    existing.add(doc_id)
            except Exception:
                destination.unlink(missing_ok=True)
    if existing == val_doc_ids:
        return existing
    if not archive.is_file():
        raise FileNotFoundError(f"DUDE binary archive not found: {archive}")
    extracted: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            path = PurePosixPath(member.name)
            matches = set(path.parts)
            matches.add(path.stem)
            matches.intersection_update(val_doc_ids)
            if not matches:
                continue
            if len(matches) > 1:
                raise RuntimeError(f"ambiguous DUDE PDF member {member.name!r}: {sorted(matches)}")
            doc_id = next(iter(matches))
            destination = documents_root / f"{safe_component(doc_id)}.pdf"
            if destination.is_file():
                try:
                    if pdf_page_count(destination) >= 1:
                        extracted.add(doc_id)
                        continue
                except Exception:
                    destination.unlink()
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read DUDE archive member {member.name!r}")
            with tempfile.NamedTemporaryFile(
                dir=documents_root,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                try:
                    shutil.copyfileobj(source, handle)
                    handle.flush()
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(destination)
            if pdf_page_count(destination) < 1:
                raise RuntimeError(f"extracted DUDE PDF is empty: {destination}")
            extracted.add(doc_id)
    return extracted


def prepare(
    bench_root: Path,
    annotation_path: Path | None = None,
    archive_path: Path | None = None,
    *,
    limit: int | None = None,
    delete_archive: bool = False,
) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    annotation_path = annotation_path or bench_root / "raw" / "dude" / "2023-03-23_DUDE_gt_test_PUBLIC.json"
    archive_path = archive_path or bench_root / "raw" / "dude" / "data" / "DUDE_train-val-test_binaries.tar.gz"
    root, all_val_rows = _read_annotation(annotation_path)
    if limit is None and len(all_val_rows) != 6315:
        raise RuntimeError(f"expected 6315 DUDE validation rows, found {len(all_val_rows)}")
    val_rows = all_val_rows if limit is None else all_val_rows[:limit]
    val_doc_ids = {str(row["docId"]) for row in val_rows}
    documents_root = bench_root / "documents" / "dude"
    extracted = _extract_validation_pdfs(archive_path, documents_root, val_doc_ids)
    missing = val_doc_ids - extracted
    if missing:
        raise RuntimeError(f"DUDE validation PDFs missing from archive: {sorted(missing)[:20]}")
    if len(extracted) != len(val_doc_ids):
        raise RuntimeError(f"DUDE extracted {len(extracted)} documents for {len(val_doc_ids)} IDs")

    eval_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for row in val_rows:
        question_id = str(row["questionId"])
        doc_id = str(row["docId"])
        document_path = documents_root / f"{safe_component(doc_id)}.pdf"
        answers = as_string_list(row.get("answers"))
        eval_answers = answers if answers else [""]
        eval_rows.append(
            make_eval_row(
                benchmark="dude",
                question_id=question_id,
                doc_id=doc_id,
                document_path=document_path,
                page_count=pdf_page_count(document_path),
                question=str(row.get("question", "")),
                answers=eval_answers,
                extra_instruction="If the document does not contain an answer, return exactly Unknown inside <final>.",
            )
        )
        gold = dict(row)
        gold.update(
            {
                "task_id": f"dude:{question_id}",
                "question_id": question_id,
                "doc_id": doc_id,
                "answers": answers,
            }
        )
        gold_rows.append(gold)

    task_ids = [row["metadata"]["task_id"] for row in eval_rows]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("DUDE preparation produced duplicate task IDs")
    eval_path = bench_root / "eval" / "dude.jsonl"
    gold_path = bench_root / "gold" / "dude.jsonl"
    jsonl_write(eval_path, eval_rows)
    jsonl_write(gold_path, gold_rows)
    if delete_archive:
        archive_path.unlink(missing_ok=True)
    return {
        "benchmark": "dude",
        "split": "val",
        "num_questions": len(eval_rows),
        "num_documents": len(val_doc_ids),
        "eval_path": str(eval_path),
        "gold_path": str(gold_path),
        "deleted_archive": bool(delete_archive),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="bounded local smoke preparation")
    parser.add_argument("--delete-archive", action="store_true")
    args = parser.parse_args()
    report = prepare(
        bench_root_from(args.bench_root),
        args.annotation,
        args.archive,
        limit=args.limit,
        delete_archive=args.delete_archive,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
