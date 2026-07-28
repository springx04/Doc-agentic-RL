import json
from pathlib import Path
import sys


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from tool_sandbox import tool_registry
from tools import document_tools
from tools.document_tools import DOC_TOOL_SPECS, _normalize_bbox, _parse_deplot_table, _rows_to_markdown


def test_document_tools_are_registered():
    expected = {
        "render_page",
        "crop_region",
        "zoom_region",
        "parse_document",
        "detect_layout",
        "ocr_region",
        "extract_table",
        "chart_to_table",
    }
    assert expected.issubset(DOC_TOOL_SPECS)
    assert expected.issubset(tool_registry.tools)
    assert "code_interpreter" not in tool_registry.tools


def test_relative_bbox_normalization():
    assert _normalize_bbox([0.1, 0.2, 0.4, 0.6], width=1000, height=500) == (100, 100, 400, 300)


def test_pixel_bbox_with_padding_and_clamp():
    assert _normalize_bbox([10, 20, 30, 40], width=35, height=35, unit="pixel", padding=10) == (0, 10, 35, 35)


def test_parse_deplot_table_text():
    text = "Year | Sales<0x0A>2024 | 10<0x0A>2025 | 12"
    assert _parse_deplot_table(text) == [["Year", "Sales"], ["2024", "10"], ["2025", "12"]]


def test_rows_to_markdown():
    assert _rows_to_markdown([["A", "B"], ["1", "2"]]) == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_docling_rejects_torch_development_build(monkeypatch, tmp_path):
    package = tmp_path / "torch"
    package.mkdir()
    init_file = package / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    (package / "version.py").write_text("__version__ = '2.11.0.dev20260206'", encoding="utf-8")

    class Spec:
        origin = str(init_file)

    monkeypatch.setattr(document_tools.importlib.util, "find_spec", lambda name: Spec())
    try:
        document_tools._assert_docling_torch_runtime()
    except RuntimeError as exc:
        assert "development build" in str(exc)
    else:
        raise AssertionError("development Torch build should be rejected")


def test_parse_document_returns_page_boundaries_and_supports_follow_up_page(monkeypatch, tmp_path):
    fitz = __import__("fitz")
    pdf_path = tmp_path / "five-pages.pdf"
    with fitz.open() as document:
        for page_number in range(1, 6):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {page_number} unique evidence")
        document.save(str(pdf_path))

    monkeypatch.setattr(document_tools, "_docling_convert", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no docling")))

    first_read = json.loads(
        document_tools.parse_document(
            {"document_path": str(pdf_path), "output_format": "markdown", "max_chars": 55}
        )
    )
    assert first_read["status"] == "ok"
    assert first_read["page_count"] == 5
    assert first_read["returned_pages"] == [1]
    assert first_read["truncated"] is True
    assert first_read["has_more_pages"] is True
    assert "Page 1" in first_read["pages"][0]["markdown"]

    later_read = json.loads(
        document_tools.parse_document(
            {"document_path": str(pdf_path), "page_numbers": [5], "output_format": "markdown"}
        )
    )
    assert later_read["status"] == "ok"
    assert later_read["page_count"] == 5
    assert later_read["returned_pages"] == [5]
    assert later_read["pages"][0]["page_number"] == 5
    assert "Page 5" in later_read["pages"][0]["markdown"]
