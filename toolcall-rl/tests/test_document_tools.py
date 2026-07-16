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
