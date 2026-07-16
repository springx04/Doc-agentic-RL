"""Build a slime JSONL dataset for document-understanding tool-call RL.

Input is a JSONL manifest.  Common field names are detected automatically::

    {"document_path": "/data/report.pdf", "question": "What was revenue?",
     "answers": ["$1.2 million"], "metric": "anls"}

The output contains ``prompt``, a JSON-encoded ``label`` (answer aliases and
metric), and ``metadata``.  Document files must be mounted at the same paths in
the rollout workers because the tools open them locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


QUESTION_KEYS = ("question", "query", "instruction", "prompt")
DOCUMENT_KEYS = ("document_path", "file_path", "document", "image_path")
ANSWER_KEYS = ("answers", "acceptable_answers", "answer", "ground_truth", "label")


def _first(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                records.append(value)
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("data") or value.get("records") or value.get("examples")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("JSON input must be an object list or contain data/records/examples")
    return value


def _resolve_document_path(raw_path: Any, document_root: Path | None) -> str:
    path = Path(str(raw_path)).expanduser()
    if document_root is not None and not path.is_absolute():
        path = document_root / path
    return str(path)


def transform_record(
    record: dict[str, Any],
    *,
    document_root: Path | None = None,
    default_metric: str = "auto",
    check_files: bool = False,
) -> dict[str, Any]:
    question = _first(record, QUESTION_KEYS)
    document = _first(record, DOCUMENT_KEYS)
    answers = _first(record, ANSWER_KEYS)
    if question is None:
        raise ValueError("record has no question/query/instruction/prompt field")
    if document is None:
        raise ValueError("record has no document_path/file_path/document/image_path field")
    if answers is None:
        raise ValueError("record has no answers/answer/ground_truth/label field")

    document_path = _resolve_document_path(document, document_root)
    if check_files and not Path(document_path).is_file():
        raise FileNotFoundError(f"document does not exist: {document_path}")
    answer_list = list(answers) if isinstance(answers, (list, tuple)) else [answers]
    metric = str(record.get("metric") or default_metric)
    task_id = record.get("id") or record.get("task_id")
    metadata = dict(record.get("metadata") or {})
    metadata.update({"document_path": document_path, "metric": metric})
    if task_id is not None:
        metadata["task_id"] = task_id

    prompt = (
        f"Document path: {document_path}\n"
        f"Question: {question}\n\n"
        "Inspect the document with the available document tools. Return only the supported answer "
        "inside <final>...</final> when you are done."
    )
    label = json.dumps({"answers": answer_list, "metric": metric}, ensure_ascii=False)
    return {"prompt": prompt, "label": label, "metadata": metadata}


def build_dataset(args: argparse.Namespace) -> int:
    records = _load_json_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            try:
                transformed = transform_record(
                    record,
                    document_root=args.document_root,
                    default_metric=args.default_metric,
                    check_files=args.check_files,
                )
            except Exception as exc:
                raise ValueError(f"failed to transform record {index}: {exc}") from exc
            handle.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source JSON or JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="Output slime JSONL path")
    parser.add_argument("--document-root", type=Path, help="Prefix for relative document paths")
    parser.add_argument(
        "--default-metric",
        default="auto",
        choices=("auto", "exact_match", "anls", "token_f1", "contains", "json"),
    )
    parser.add_argument("--check-files", action="store_true", help="Fail if a referenced document is missing")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    count = build_dataset(parsed_args)
    print(f"Wrote {count} document RL examples to {parsed_args.output}")
