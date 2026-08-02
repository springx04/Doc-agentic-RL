import json
from pathlib import Path
import sys

from PIL import Image

TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from tool_sandbox import tool_registry
from tools import document_tools
from tools.document_tools import (
    DOC_TOOL_SPECS,
    _normalize_bbox,
    _page_observation,
    _parse_deplot_table,
    _rows_to_markdown,
)


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


def test_ocr_region_separates_pdf_region_and_image_inputs(monkeypatch, tmp_path):
    fitz = __import__("fitz")
    pdf_path = tmp_path / "ocr-input.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=120)
        page.insert_text((20, 60), "PDF-CANARY", fontsize=20)
        document.save(str(pdf_path))
    image_path = tmp_path / "ocr-input.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    monkeypatch.setattr(
        document_tools,
        "_run_ocr",
        lambda path, lang, engine: ("fake", [{"bbox": None, "text": "OCR-CANARY", "confidence": 1.0}]),
    )
    pdf_result = json.loads(
        document_tools.ocr_region(
            {
                "document_path": str(pdf_path),
                "page_number": 1,
                "bbox": [0, 0, 200, 120],
                "output_path": str(tmp_path / "pdf-ocr.json"),
            }
        )
    )
    image_result = json.loads(
        document_tools.ocr_region(
            {"image_path": str(image_path), "output_path": str(tmp_path / "image-ocr.json")}
        )
    )
    assert pdf_result["status"] == "ok"
    assert pdf_result["tool"] == "ocr_region"
    assert Path(pdf_result["image_path"]).suffix.lower() == ".png"
    assert Path(pdf_result["json_path"]).suffix.lower() == ".json"
    assert image_result["status"] == "ok"
    assert image_result["tool"] == "ocr_region"
    assert image_result["image_path"] == str(image_path)

    invalid_path = tmp_path / "wrong-input.json"
    invalid_path.write_text("{}", encoding="utf-8")
    invalid_result = json.loads(document_tools.ocr_region({"image_path": str(invalid_path)}))
    assert invalid_result["status"] == "error"
    assert invalid_result["tool"] == "ocr_region"
    assert "image_path" in invalid_result["error"]


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


def test_parse_document_page_contract_has_one_consistent_representation(monkeypatch, tmp_path):
    fitz = __import__("fitz")
    pdf_path = tmp_path / "four-pages.pdf"
    with fitz.open() as document:
        for page_number in range(1, 5):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {page_number} declared content")
        document.save(str(pdf_path))

    monkeypatch.setattr(document_tools, "_docling_convert", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no docling")))

    all_pages = json.loads(document_tools.parse_document({"document_path": str(pdf_path), "output_format": "both"}))
    assert "markdown" not in all_pages
    assert all_pages["returned_pages"] == [1, 2, 3, 4]
    assert all_pages["truncated"] is False
    assert all_pages["content_truncated"] is False
    assert all_pages["has_more_pages"] is False
    assert all_pages["document_has_unreturned_pages"] is False
    assert {page["page_number"] for page in all_pages["pages"]} == set(all_pages["returned_pages"])
    assert json.loads(Path(all_pages["json_path"]).read_text(encoding="utf-8"))["returned_pages"] == [1, 2, 3, 4]
    assert "Page 4 declared content" in Path(all_pages["markdown_path"]).read_text(encoding="utf-8")

    one_page = json.loads(
        document_tools.parse_document(
            {"document_path": str(pdf_path), "page_numbers": [3], "output_format": "json"}
        )
    )
    assert "markdown" not in one_page
    assert one_page["returned_pages"] == [3]
    assert one_page["truncated"] is False
    assert one_page["content_truncated"] is False
    assert one_page["has_more_pages"] is True
    assert one_page["document_has_unreturned_pages"] is True
    assert {page["page_number"] for page in one_page["pages"]} == {3}
    json_artifact = json.loads(Path(one_page["json_path"]).read_text(encoding="utf-8"))
    assert json_artifact["returned_pages"] == [3]
    assert {page["page_number"] for page in json_artifact["pages"]} == {3}
    assert "Page 1 declared content" not in json.dumps(json_artifact, ensure_ascii=False)
    assert "Page 4 declared content" not in json.dumps(json_artifact, ensure_ascii=False)


def test_page_observation_exposes_table_and_visual_follow_up_routes():
    observation = _page_observation(
        {
            "page_number": 3,
            "markdown": "## Page 3\n\n| Name | Value |\n| --- | --- |\n| A | 1 |",
            "source": {
                "page_number": 3,
                "tables": [{"bbox": [10, 20, 100, 200], "rows": [["Name", "Value"], ["A", "1"]]}],
                "blocks": [{"type": "image", "bbox": [200, 300, 400, 500]}],
            },
        }
    )
    assert observation["has_tables"] is True
    assert observation["table_count"] == 1
    assert observation["table_extraction_recommended"] is True
    assert observation["table_regions"] == [{"page_number": 3, "bbox": [10, 20, 100, 200]}]
    assert observation["has_images"] is True
    assert observation["visual_content_omitted"] is True
    assert observation["image_regions"] == [{"page_number": 3, "bbox": [200, 300, 400, 500]}]


def test_extract_table_auto_falls_back_from_headless_docling(monkeypatch, tmp_path):
    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    def fail_docling(*args, **kwargs):
        raise RuntimeError("libGL.so.1: cannot open shared object file")

    def fake_pdfplumber(arguments, table_index, output_format):
        return {
            "rows": [["Item", "Amount"], ["Total", "7,265,516"]],
            "markdown": "| Item | Amount |\n| --- | --- |\n| Total | 7,265,516 |",
            "row_count": 2,
            "column_count": 2,
        }, "pdfplumber"

    monkeypatch.setattr(document_tools, "_extract_table_docling", fail_docling)
    monkeypatch.setattr(document_tools, "_extract_table_pdfplumber", fake_pdfplumber)
    result = json.loads(
        document_tools.extract_table(
            {"document_path": str(pdf_path), "page_number": 1, "table_index": 0, "engine": "auto"}
        )
    )

    assert result["status"] == "ok"
    assert result["engine"] == "pdfplumber"
    assert result["requested_engine"] == "auto"
    assert result["fallback_from"] == "docling"
    assert "libGL.so.1" in result["fallback_error"]
    assert "| Total | 7,265,516 |" in result["markdown"]


def test_detect_layout_auto_falls_back_to_local_pdf_boxes(monkeypatch, tmp_path):
    fitz = __import__("fitz")
    pdf_path = tmp_path / "layout.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "VISUAL-CANARY-123")
        document.save(str(pdf_path))

    monkeypatch.setattr(
        document_tools,
        "_docling_convert",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("libGL.so.1 unavailable")),
    )
    result = json.loads(document_tools.detect_layout({"document_path": str(pdf_path), "page_number": 1}))

    assert result["status"] == "ok"
    assert result["fallback_from"] == "docling"
    assert result["engine"] == "openclaw_pymupdf_pdfplumber"
    assert result["element_count"] >= 1
    assert any("VISUAL-CANARY-123" in str(item.get("text")) for item in result["elements"])


def test_ocr_backend_failure_keeps_visual_input_as_partial_result(monkeypatch, tmp_path):
    image_path = tmp_path / "visual-canary.png"
    from PIL import Image

    Image.new("RGB", (32, 32), "white").save(image_path)

    def fail_ocr(*args, **kwargs):
        raise RuntimeError("libGL.so.1 unavailable")

    monkeypatch.setattr(document_tools, "_run_ocr", fail_ocr)
    result = json.loads(document_tools.ocr_region({"image_path": str(image_path)}))

    assert result["status"] == "partial"
    assert result["ocr_unavailable"] is True
    assert result["text"] == ""
    assert result["image_path"] == str(image_path)
    assert "libGL.so.1" in result["error"]


def test_region_tools_pad_extreme_aspect_for_vlm(tmp_path):
    from PIL import Image

    image_path = tmp_path / "wide-line.png"
    Image.new("RGB", (1000, 100), "white").save(image_path)

    crop_path = tmp_path / "crop.png"
    crop = json.loads(
        document_tools.crop_region(
            {
                "image_path": str(image_path),
                "bbox": [0, 0, 900, 1],
                "unit": "pixel",
                "output_path": str(crop_path),
            }
        )
    )
    assert crop["status"] == "ok"
    assert crop["aspect_padding_applied"] is True
    assert max(crop["width"] / crop["height"], crop["height"] / crop["width"]) < 200
    with Image.open(crop_path) as rendered:
        assert rendered.size == (crop["width"], crop["height"])

    zoom_path = tmp_path / "zoom.png"
    zoom = json.loads(
        document_tools.zoom_region(
            {
                "image_path": str(image_path),
                "bbox": [0, 0, 900, 1],
                "unit": "pixel",
                "scale": 2,
                "output_path": str(zoom_path),
            }
        )
    )
    assert zoom["status"] == "ok"
    assert zoom["aspect_padding_applied"] is True
    assert max(zoom["width"] / zoom["height"], zoom["height"] / zoom["width"]) < 200
