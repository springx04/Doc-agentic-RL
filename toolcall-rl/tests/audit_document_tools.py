"""Run every registered document tool against a real PDF and report JSON results.

This is an integration audit, not a unit test: it intentionally loads the
actual OCR, Docling, table, and DePlot backends through ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
import os
import site
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLCALL_DIR = ROOT / "toolcall-rl"
STUDIO_DIR = TOOLCALL_DIR / "tool_studio"
RUNTIME_DIR = STUDIO_DIR / "runtime"
PDF_PATH = Path(r"C:\Users\30738\Desktop\project\agent\paper\ToolRL.pdf")
REPORT_PATH = STUDIO_DIR / "outputs" / "full_tool_audit_report.json"


def prepare_runtime() -> None:
    if RUNTIME_DIR.is_dir():
        sys.path.insert(0, str(RUNTIME_DIR))
    sys.path.insert(1, str(TOOLCALL_DIR))
    store_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Packages"
    store_sites = list(store_root.glob("PythonSoftwareFoundation.Python.3.11_*/LocalCache/local-packages/Python311/site-packages")) if store_root.is_dir() else []
    for path in (Path(site.getusersitepackages()), *store_sites, Path(os.environ.get("TEMP", "")) / "openclaw_pydeps"):
        if path.is_dir():
            sys.path.append(str(path))

    os.environ["OPENCLAW_TOOL_OUTPUT_DIR"] = str(STUDIO_DIR / "outputs" / "full_tool_audit")
    os.environ["OPENCLAW_TOOL_CACHE_DIR"] = str(STUDIO_DIR / "cache")
    temp_dir = Path(os.environ.get("TEMP", ""))
    model_dir = temp_dir / "openclaw_docling_models"
    deplot_dir = temp_dir / "openclaw_deplot_model"
    modelscope_dir = temp_dir / "openclaw_modelscope_runtime"
    paddle_dir = temp_dir / "openclaw_paddle_runtime"
    if (model_dir / "docling-project--docling-layout-heron").is_dir():
        os.environ["OPENCLAW_DOCLING_ARTIFACTS_PATH"] = str(model_dir)
    if deplot_dir.is_dir():
        os.environ["OPENCLAW_DEPLOT_MODEL"] = str(deplot_dir)
    if modelscope_dir.is_dir():
        os.environ["OPENCLAW_MODELSCOPE_RUNTIME"] = str(modelscope_dir)
    if paddle_dir.is_dir():
        os.environ["OPENCLAW_PADDLE_RUNTIME"] = str(paddle_dir)


async def main() -> int:
    prepare_runtime()
    from tool_sandbox import tool_registry

    if not PDF_PATH.is_file():
        raise FileNotFoundError(PDF_PATH)

    report: list[dict[str, Any]] = []

    async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        raw = await tool_registry.execute_tool(name, arguments)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"status": "error", "error": f"non-JSON tool result: {raw[:1000]}"}
        item = {
            "tool": name,
            "arguments": arguments,
            "seconds": round(time.perf_counter() - started, 2),
            "status": result.get("status"),
            "error": result.get("error"),
            "engine": result.get("engine"),
        }
        report.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    page_one = await call("render_page", {"document_path": str(PDF_PATH), "page_number": 1, "dpi": 160})
    image_path = page_one.get("image_path")
    await call("render_page", {"document_path": str(PDF_PATH), "page_number": 10, "dpi": 100})
    await call("crop_region", {"image_path": image_path, "bbox": [0.05, 0.05, 0.45, 0.30], "unit": "relative", "padding": 8})
    await call("crop_region", {"document_path": str(PDF_PATH), "page_number": 1, "bbox": [100, 100, 650, 500], "unit": "pixel", "dpi": 120})
    await call("zoom_region", {"image_path": image_path, "bbox": [0.45, 0.10, 0.95, 0.55], "unit": "relative", "scale": 2.5})
    await call("zoom_region", {"document_path": str(PDF_PATH), "page_number": 10, "bbox": [80, 80, 520, 430], "unit": "pixel", "scale": 1.5, "dpi": 120})
    await call("parse_document", {"document_path": str(PDF_PATH), "output_format": "both", "page_range": "1-2", "max_chars": 1500})
    await call("parse_document", {"document_path": str(PDF_PATH), "output_format": "markdown", "page_range": [10, 10], "max_chars": 1000})
    await call("detect_layout", {"document_path": str(PDF_PATH), "page_number": 1, "max_items": 30})
    await call("detect_layout", {"document_path": str(PDF_PATH), "page_number": 10, "element_types": ["table", "picture", "title", "paragraph"], "max_items": 30})
    await call("ocr_region", {"document_path": str(PDF_PATH), "page_number": 1, "engine": "rapidocr", "lang": "en", "dpi": 160, "max_lines": 25, "max_chars": 2000})
    await call("ocr_region", {"image_path": image_path, "bbox": [0.45, 0.10, 0.95, 0.55], "unit": "relative", "engine": "paddleocr", "lang": "en", "max_lines": 25, "max_chars": 2000})
    await call("extract_table", {"document_path": str(PDF_PATH), "page_number": 9, "engine": "docling", "table_index": 0, "output_format": "all"})
    await call("extract_table", {"document_path": str(PDF_PATH), "page_number": 10, "engine": "pdfplumber", "table_index": 0, "output_format": "markdown"})
    await call("extract_table", {"document_path": str(PDF_PATH), "page_number": 10, "engine": "camelot", "flavor": "stream", "table_index": 0, "output_format": "csv"})
    await call("chart_to_table", {"document_path": str(PDF_PATH), "page_number": 1, "bbox": [0.05, 0.25, 0.95, 0.80], "unit": "relative", "device": "cpu", "local_files_only": True, "max_new_tokens": 128, "output_format": "markdown"})
    errors = [item for item in report if item["status"] != "ok"]
    print(json.dumps({"report": str(REPORT_PATH), "calls": len(report), "errors": len(errors)}, ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
