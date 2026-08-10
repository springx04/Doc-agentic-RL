"""Shared filesystem, schema, and document helpers for benchmark adapters.

The benchmark data itself is intentionally not part of the repository.  All
preparation scripts receive one ``BENCH_ROOT`` and materialize only the
documents and sidecars needed by that run.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import BENCHMARKS


TOOL_BUDGET = {
    "max_turns": 10,
    "max_tool_calls": 8,
    "max_obs_chars": 8192,
    "tool_concurrency": 32,
}

FORBIDDEN_PROMPT_MARKERS = (
    "answer_page",
    "answer_type",
    "target_page",
    "answer_bbox",
    "evidence_pages",
    "detailed_evidences",
    "answers_page_bounding_boxes",
    "gold_answer_page",
    "gold_bbox",
)


def bench_root_from(value: str | Path | None = None) -> Path:
    """Resolve the one shared benchmark root used by all adapters."""

    root = value or os.environ.get("BENCH_ROOT") or "/data/doc_agentic_benchmarks"
    return Path(root).expanduser().resolve()


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write UTF-8 JSONL so interrupted preparation is recoverable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def json_write(path: Path, payload: Any) -> None:
    """Atomically write a JSON artifact with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def jsonl_read(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file with useful line-number errors."""

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(jsonl_read(path))


def pdf_page_count(path: Path) -> int:
    """Return the page count after asking PyMuPDF to open the PDF."""

    import fitz

    with fitz.open(str(path)) as document:
        return len(document)


def make_task_id(benchmark: str, question_id: str) -> str:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return f"{benchmark}:{question_id}"


def make_prompt(document_path: Path, question: str, extra_instruction: str = "") -> str:
    """Build the only prompt shape allowed by the benchmark specification."""

    base = (
        f"Document path: {Path(document_path).resolve()}\n"
        f"Question: {str(question).strip()}\n\n"
        "Inspect the document with the available tools before answering. "
        "Use only evidence from the document. "
        "Return only the concise answer inside <final>...</final>."
    )
    if extra_instruction.strip():
        base += "\n" + extra_instruction.strip()
    return base


def make_eval_row(
    *,
    benchmark: str,
    question_id: str,
    doc_id: str,
    document_path: Path,
    page_count: int,
    question: str,
    answers: list[str],
    extra_instruction: str = "",
) -> dict[str, Any]:
    """Create the canonical agent-visible row without gold navigation fields."""

    if page_count < 1:
        raise ValueError(f"page_count must be positive, got {page_count}")
    answer_list = [str(answer) for answer in answers]
    return {
        "prompt": make_prompt(document_path, question, extra_instruction),
        "label": {"answers": answer_list, "metric": "anls"},
        "metadata": {
            "task_id": make_task_id(benchmark, str(question_id)),
            "benchmark": benchmark,
            "question_id": str(question_id),
            "doc_id": str(doc_id),
            "document_path": str(Path(document_path).resolve()),
            "page_count": int(page_count),
        },
    }


def parse_maybe_literal(value: Any) -> Any:
    """Parse the stringified lists used by MP-DocVQA parquet shards."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def as_string_list(value: Any, *, empty_fallback: list[str] | None = None) -> list[str]:
    value = parse_maybe_literal(value)
    if value is None:
        return list(empty_fallback or [])
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def safe_component(value: Any) -> str:
    """Make a source ID safe as one local filename component."""

    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or "document"


def images_to_pdf(images: Iterable[Any], output_pdf: Path, expected_page_count: int | None = None) -> None:
    """Materialize one image sequence as one PDF, using only a temporary page set.

    Existing PDFs are reused only after PyMuPDF validates their page count.  A
    malformed or stale output is replaced by the newly materialized document.
    """

    from io import BytesIO

    import img2pdf
    from PIL import Image

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf.is_file():
        try:
            existing_count = pdf_page_count(output_pdf)
        except Exception:
            existing_count = -1
        if (expected_page_count is None and existing_count >= 1) or existing_count == expected_page_count:
            return
        output_pdf.unlink()

    image_list = list(images)
    if not image_list:
        raise ValueError("cannot materialize an empty document")

    def to_pil(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.copy()
        if isinstance(value, dict):
            raw = value.get("bytes")
            if raw is not None:
                with Image.open(BytesIO(raw)) as image:
                    return image.copy()
            image_path = value.get("path")
            if image_path:
                with Image.open(str(image_path)) as image:
                    return image.copy()
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(value))) as image:
                return image.copy()
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return Image.fromarray(value)
        raise TypeError(f"unsupported page image type: {type(value)!r}")

    with tempfile.TemporaryDirectory(prefix="docbench_pages_") as temporary_dir:
        page_paths: list[str] = []
        for index, raw_image in enumerate(image_list, 1):
            if raw_image is None:
                raise ValueError(f"missing page image {index}")
            image = to_pil(raw_image)
            page_path = Path(temporary_dir) / f"{index:04d}.png"
            image.save(page_path, format="PNG")
            page_paths.append(str(page_path))
            image.close()
        output_pdf.write_bytes(img2pdf.convert(page_paths))

    actual_count = pdf_page_count(output_pdf)
    if expected_page_count is not None and actual_count != expected_page_count:
        raise RuntimeError(
            f"materialized {output_pdf} with {actual_count} pages, expected {expected_page_count}"
        )


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy one file through a temporary sibling and preserve binary bytes."""

    import shutil

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def keyed_rows(path: Path, key: str = "task_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in jsonl_read(path):
        value = row.get(key)
        if not value:
            raise ValueError(f"{path}: row missing {key}")
        value = str(value)
        if value in result:
            raise ValueError(f"{path}: duplicate {key}={value}")
        result[value] = row
    return result


def nested_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def integer_list(value: Any) -> list[int]:
    value = parse_maybe_literal(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def forbidden_prompt_markers(prompt: str) -> list[str]:
    lowered = str(prompt).casefold()
    return [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker.casefold() in lowered]
