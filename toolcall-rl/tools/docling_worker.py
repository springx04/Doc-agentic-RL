"""Isolated Docling conversion worker for Windows Tool Studio runs.

The worker is launched with ``python -S`` so automatic user-site .pth files
cannot select an unrelated Python/Torch runtime for Docling model workers.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import site
import sys
import tempfile
from pathlib import Path
from typing import Any


def _bootstrap_paths() -> None:
    runtime = os.environ.get("OPENCLAW_DOC_TOOL_RUNTIME_PATH")
    if runtime and Path(runtime).is_dir():
        sys.path.insert(0, runtime)
    store_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Packages"
    store_sites = list(store_root.glob("PythonSoftwareFoundation.Python.3.11_*/LocalCache/local-packages/Python311/site-packages")) if store_root.is_dir() else []
    for path in (Path(site.getusersitepackages()), *store_sites, Path(tempfile.gettempdir()) / "openclaw_pydeps"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))


def _page_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str) and "-" in value:
        start, end = value.split("-", 1)
        return int(start), int(end)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    page = int(value)
    return page, page


def main() -> None:
    _bootstrap_paths()
    os.environ.setdefault("CONDA_AUTO_ACTIVATE_BASE", "false")
    os.environ.setdefault("CONDA_CHANGEPS1", "false")
    multiprocessing.set_executable(sys.executable)
    os.environ["PYTHONEXECUTABLE"] = sys.executable
    request = json.loads(sys.stdin.read())
    artifacts = request.get("artifacts_path")
    if artifacts:
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(artifacts)

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    options = None
    if artifacts:
        pipeline_options = PdfPipelineOptions(artifacts_path=str(artifacts))
        options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    converter = DocumentConverter(format_options=options)
    kwargs: dict[str, Any] = {"raises_on_error": True}
    page_range = _page_range(request.get("page_range"))
    if page_range:
        kwargs["page_range"] = page_range
    # Backend progress must not corrupt the JSON protocol on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        result = converter.convert(str(request["document_path"]), **kwargs)
    status = str(getattr(result, "status", "success")).split(".")[-1].lower()
    if status not in {"success", "partial_success"}:
        raise RuntimeError(f"Docling conversion failed with status={status}: {getattr(result, 'errors', None)}")
    document = result.document
    payload = {
        "status": status,
        "markdown": str(document.export_to_markdown()),
        "document_json": document.export_to_dict(),
    }
    output_path = request.get("output_path")
    if output_path:
        Path(str(output_path)).write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({"status": status, "output_path": str(output_path)}, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stdout)
        raise
