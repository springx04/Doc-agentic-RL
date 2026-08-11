"""Document understanding tools for tool-call RL rollouts.

The tools in this module keep their runtime integration code inside
toolcall-rl. Heavy backends are imported lazily, and missing/failed backends
return explicit tool errors instead of producing substitute results.
"""

from __future__ import annotations

import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import multiprocessing
import subprocess
import sys
import tempfile
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

try:
    from .ocr_runtime import prepare_headless_ocr_runtime
except ImportError:  # pragma: no cover - compatibility with direct module loading
    from ocr_runtime import prepare_headless_ocr_runtime


DOC_TOOL_SPECS: dict[str, dict[str, Any]] = {}

_OCR_ENGINES: dict[str, Any] = {}
_DEPLOT_CACHE: dict[tuple[str, str, bool], tuple[Any, Any]] = {}
_DOCLING_CONVERTER: Any | None = None
_DOCLING_CACHE: dict[tuple[str, float, int | None, int | None], tuple[Any, Any]] = {}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
PDF_SUFFIXES = {".pdf"}
DOC_SUFFIXES = {".docx"}
PPT_SUFFIXES = {".pptx"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    value = os.environ.get(name, default)
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _docling_ocr_languages() -> list[str]:
    return _env_list("OPENCLAW_DOCLING_OCR_LANG", "en")


def _docling_ocr_options(RapidOcrOptions: Any) -> Any:
    engine = os.environ.get("OPENCLAW_DOCLING_OCR_ENGINE", "rapidocr").strip().lower()
    if engine != "rapidocr":
        raise RuntimeError(
            "OpenClaw Docling OCR is restricted to the offline RapidOCR backend; "
            f"got {engine!r}"
        )
    return RapidOcrOptions(
        lang=_docling_ocr_languages(),
        backend=os.environ.get("OPENCLAW_DOCLING_RAPIDOCR_BACKEND", "onnxruntime").strip().lower(),
        print_verbose=False,
    )


def _rapidocr_artifact_params(lang: str = "en", backend: str = "onnxruntime") -> dict[str, Any] | None:
    artifacts = os.environ.get("OPENCLAW_DOCLING_ARTIFACTS_PATH") or os.environ.get("DOCLING_ARTIFACTS_PATH")
    if not artifacts:
        if _env_flag("OPENCLAW_OCR_OFFLINE"):
            raise RuntimeError(
                "offline OCR requires OPENCLAW_DOCLING_ARTIFACTS_PATH with the bundled RapidOCR models"
            )
        return None
    backend = backend.strip().lower()
    if backend == "onnxruntime":
        root = Path(artifacts).expanduser() / "RapidOcr" / "onnx"
        paths = {
            "det": root / "PP-OCRv6" / "det" / "PP-OCRv6_det_small.onnx",
            "cls": root / "PP-OCRv4" / "cls" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "rec": root / "PP-OCRv6" / "rec" / "PP-OCRv6_rec_small.onnx",
            "keys": None,
        }
        engine_type_name = "ONNXRUNTIME"
    elif backend == "torch":
        root = Path(artifacts).expanduser() / "RapidOcr"
        chinese = lang.strip().lower() in {"zh", "zho", "chi", "chinese", "ch"}
        paths = {
            "det": root / "torch" / "PP-OCRv4" / "det" / ("ch_PP-OCRv4_det_mobile.pth" if chinese else "en_PP-OCRv3_det_mobile.pth"),
            "cls": root / "torch" / "PP-OCRv4" / "cls" / "ch_ptocr_mobile_v2.0_cls_mobile.pth",
            "rec": root / "torch" / "PP-OCRv4" / "rec" / ("ch_PP-OCRv4_rec_mobile.pth" if chinese else "en_PP-OCRv4_rec_mobile.pth"),
            "keys": root / "paddle" / "PP-OCRv4" / "rec" / ("ch_PP-OCRv4_rec_mobile" if chinese else "en_PP-OCRv4_rec_mobile") / ("ppocr_keys_v1.txt" if chinese else "en_dict.txt"),
        }
        engine_type_name = "TORCH"
    else:
        raise RuntimeError(f"unsupported local RapidOCR backend: {backend}")
    missing = [str(path) for path in paths.values() if path is not None and not path.is_file()]
    if missing:
        raise RuntimeError("bundled RapidOCR model files are missing: " + ", ".join(missing))
    from rapidocr import EngineType
    engine_type = getattr(EngineType, engine_type_name)

    params: dict[str, Any] = {
        "Global.text_score": 0.5,
        "Det.model_path": str(paths["det"]),
        "Cls.model_path": str(paths["cls"]),
        "Rec.model_path": str(paths["rec"]),
        "Det.engine_type": engine_type,
        "Cls.engine_type": engine_type,
        "Rec.engine_type": engine_type,
        "Det.use_cuda": False,
        "Cls.use_cuda": False,
        "Rec.use_cuda": False,
        "Det.use_dml": False,
        "Cls.use_dml": False,
        "Rec.use_dml": False,
        "EngineConfig.onnxruntime.intra_op_num_threads": int(os.environ.get("OPENCLAW_OCR_THREADS", "4")),
    }
    if paths["keys"] is not None:
        params["Rec.rec_keys_path"] = str(paths["keys"])
    return params


def _register_tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> None:
    DOC_TOOL_SPECS[name] = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


def _json_result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _error(tool_name: str, message: str, **extra: Any) -> str:
    payload = {"status": "error", "tool": tool_name, "error": message}
    payload.update(extra)
    return _json_result(**payload)


def _plain_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _path_arg(arguments: dict[str, Any], key: str) -> Path | None:
    value = arguments.get(key)
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def _ensure_file(path: Path, tool_name: str) -> str | None:
    if not path.exists():
        return f"{tool_name}: file does not exist: {path}"
    if not path.is_file():
        return f"{tool_name}: path is not a file: {path}"
    return None


def _output_dir() -> Path:
    root = os.environ.get("OPENCLAW_TOOL_OUTPUT_DIR")
    out_dir = Path(root).expanduser() if root else Path(tempfile.gettempdir()) / "openclaw_doc_tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _backend_cache_dir() -> Path:
    root = os.environ.get("OPENCLAW_TOOL_CACHE_DIR")
    cache_dir = Path(root).expanduser() if root else Path(tempfile.gettempdir()) / "openclaw_doc_tool_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _prepare_ml_backend_environment() -> Path:
    cache_dir = _backend_cache_dir()
    dependency_paths = [Path(tempfile.gettempdir()) / "openclaw_pydeps"]
    configured_paddle = os.environ.get("OPENCLAW_PADDLE_RUNTIME")
    if configured_paddle:
        dependency_paths.insert(0, Path(configured_paddle).expanduser())
    configured_modelscope = os.environ.get("OPENCLAW_MODELSCOPE_RUNTIME")
    if configured_modelscope:
        dependency_paths.insert(0, Path(configured_modelscope).expanduser())
    for extra_packages in reversed(dependency_paths):
        if extra_packages.is_dir():
            # Place optional PaddleOCR dependencies before the Store user
            # site, which may contain an incomplete modelscope installation.
            while str(extra_packages) in sys.path:
                sys.path.remove(str(extra_packages))
            insert_at = 1 if sys.path and "tool_studio" in sys.path[0] else 0
            sys.path.insert(insert_at, str(extra_packages))
    model_cache_dir = Path(os.environ.get("OPENCLAW_PADDLEOCR_CACHE_DIR", str(cache_dir))).expanduser()
    home_dir = model_cache_dir / "home"
    paddle_home = home_dir / ".cache" / "paddle"
    paddlex_home = model_cache_dir / "paddlex"
    paddleocr_home = model_cache_dir / "paddleocr"
    xdg_home = home_dir / ".cache"
    for path in (home_dir, paddle_home, paddlex_home, paddleocr_home, xdg_home):
        path.mkdir(parents=True, exist_ok=True)

    # Paddle hardcodes expanduser("~") for some dataset/cache paths on import.
    os.environ["HOME"] = str(home_dir)
    os.environ["USERPROFILE"] = str(home_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_home))
    os.environ.setdefault("PADDLE_HOME", str(paddle_home))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paddlex_home))
    os.environ.setdefault("PADDLEOCR_HOME", str(paddleocr_home))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    return cache_dir


def _preload_torch_for_windows() -> None:
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"PyTorch preload failed on Windows: {exc}") from exc


def _stable_stem(prefix: str, source: Path | None = None, payload: Any | None = None) -> str:
    raw = f"{source or ''}|{json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False) if payload is not None else ''}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}_{digest}_{uuid.uuid4().hex[:8]}"


def _resolve_output_path(value: Any, suffix: str, prefix: str, source: Path | None = None, payload: Any | None = None) -> Path:
    if value:
        path = Path(str(value)).expanduser()
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{_stable_stem(prefix, source, payload)}{suffix}"
    return _output_dir() / f"{_stable_stem(prefix, source, payload)}{suffix}"


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]", True


def _import_pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for this tool. Install with: pip install pillow") from exc
    return Image


def _import_fitz():
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF rendering. Install with: pip install pymupdf") from exc
    return fitz


def _page_number(arguments: dict[str, Any], default: int = 1) -> int:
    page_number = int(arguments.get("page_number", default))
    if page_number < 1:
        raise ValueError("page_number is 1-based and must be >= 1")
    return page_number


def _parse_bbox(value: Any) -> list[float]:
    if value is None:
        raise ValueError("bbox is required")
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = [part.strip() for part in stripped.split(",")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be [x0, y0, x1, y1]")
    return [float(v) for v in value]


def _normalize_bbox(
    bbox: Any,
    width: int | float,
    height: int | float,
    unit: str = "auto",
    padding: int | float = 0,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _parse_bbox(bbox)
    unit = (unit or "auto").lower()
    if unit == "relative" or (unit == "auto" and all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1))):
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    elif unit not in {"auto", "pixel", "pixels"}:
        raise ValueError("unit must be one of: auto, pixel, relative")

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    pad = float(padding or 0)
    x0 = max(0, int(round(x0 - pad)))
    y0 = max(0, int(round(y0 - pad)))
    x1 = min(int(width), int(round(x1 + pad)))
    y1 = min(int(height), int(round(y1 + pad)))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox is empty after normalization: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


_VLM_MAX_ASPECT_RATIO = 100.0


def _pad_extreme_aspect(image: Any, max_aspect_ratio: float = _VLM_MAX_ASPECT_RATIO) -> tuple[Any, bool]:
    """Pad ultra-thin crops so multimodal processors can encode them.

    Layout/OCR boxes can legitimately be a one-pixel-high text line. Qwen-VL's
    image processor rejects images whose absolute aspect ratio is >= 200, so
    preserve the pixels and add a white margin rather than stretching or
    discarding the visual evidence.
    """
    width, height = int(image.width), int(image.height)
    if width <= 0 or height <= 0:
        return image, False
    limit = max(1.0, float(max_aspect_ratio))
    ratio = max(width / height, height / width)
    if ratio < limit:
        return image, False

    Image = _import_pil()
    if width >= height:
        target_height = max(height, int(math.ceil(width / limit)))
        canvas = Image.new("RGB", (width, target_height), (255, 255, 255))
        canvas.paste(image, (0, (target_height - height) // 2))
    else:
        target_width = max(width, int(math.ceil(height / limit)))
        canvas = Image.new("RGB", (target_width, height), (255, 255, 255))
        canvas.paste(image, ((target_width - width) // 2, 0))
    return canvas, True


def _render_pdf_page(document_path: Path, page_number: int, dpi: int, output_path: Path) -> dict[str, Any]:
    fitz = _import_fitz()
    with fitz.open(str(document_path)) as doc:
        if page_number > len(doc):
            raise ValueError(f"page_number {page_number} is out of range; document has {len(doc)} pages")
        page = doc[page_number - 1]
        zoom = float(dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(output_path))
        return {
            "image_path": str(output_path),
            "document_path": str(document_path),
            "page_number": page_number,
            "page_count": len(doc),
            "width": pix.width,
            "height": pix.height,
            "dpi": dpi,
            "coordinate_space": "pixel_top_left",
        }


def _render_image_page(image_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    Image = _import_pil()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if output_path is None:
            output_path = image_path
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
        return {
            "image_path": str(output_path),
            "page_number": 1,
            "page_count": 1,
            "width": image.width,
            "height": image.height,
            "dpi": None,
            "coordinate_space": "pixel_top_left",
        }


def _render_page_to_image(
    *,
    document_path: Path,
    page_number: int = 1,
    dpi: int = 144,
    output_path: Path | None = None,
) -> dict[str, Any]:
    suffix = document_path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        out_path = output_path or _resolve_output_path(None, ".png", "render_page", document_path, {"page": page_number, "dpi": dpi})
        return _render_pdf_page(document_path, page_number, dpi, out_path)
    if suffix in IMAGE_SUFFIXES:
        return _render_image_page(document_path, output_path=output_path)
    raise ValueError(f"rendering is supported for PDF and image files, got: {document_path.suffix}")


def render_page(arguments: dict[str, Any]) -> str:
    tool = "render_page"
    document_path = _path_arg(arguments, "document_path")
    if document_path is None:
        return _error(tool, "document_path is required")
    problem = _ensure_file(document_path, tool)
    if problem:
        return _error(tool, problem)
    try:
        page_number = _page_number(arguments)
        dpi = int(arguments.get("dpi", 144))
        output_path = _resolve_output_path(
            arguments.get("output_path"), ".png", tool, document_path, {"page_number": page_number, "dpi": dpi}
        )
        result = _render_page_to_image(
            document_path=document_path,
            page_number=page_number,
            dpi=dpi,
            output_path=output_path,
        )
        result = {"document_path": str(document_path), **result}
        return _json_result(status="ok", tool=tool, **result)
    except Exception as exc:
        return _error(tool, str(exc), document_path=str(document_path))


def _image_for_region(arguments: dict[str, Any], tool: str) -> tuple[Path, dict[str, Any]]:
    image_path = _path_arg(arguments, "image_path")
    document_path = _path_arg(arguments, "document_path")
    if image_path is not None and document_path is not None:
        raise ValueError("provide exactly one of image_path or document_path")
    if image_path is not None:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"{tool}: image_path must point to an image ({sorted(IMAGE_SUFFIXES)}); "
                "use document_path for a PDF"
            )
        problem = _ensure_file(image_path, tool)
        if problem:
            raise ValueError(problem)
        Image = _import_pil()
        with Image.open(image_path) as image:
            return image_path, {
                "image_path": str(image_path),
                "width": image.width,
                "height": image.height,
                "page_number": 1,
                "page_count": 1,
                "dpi": None,
                "coordinate_space": "pixel_top_left",
            }

    if document_path is None:
        raise ValueError("Either image_path or document_path is required")
    if document_path.suffix.lower() not in PDF_SUFFIXES | IMAGE_SUFFIXES:
        raise ValueError(
            f"{tool}: document_path must point to a PDF or image; "
            "pass a rendered PNG/JPG through image_path"
        )
    problem = _ensure_file(document_path, tool)
    if problem:
        raise ValueError(problem)
    page_number = _page_number(arguments)
    dpi = int(arguments.get("dpi", 144))
    rendered = _render_page_to_image(document_path=document_path, page_number=page_number, dpi=dpi)
    return Path(rendered["image_path"]), rendered


def crop_region(arguments: dict[str, Any]) -> str:
    tool = "crop_region"
    try:
        source_path, source_meta = _image_for_region(arguments, tool)
        Image = _import_pil()
        padding = float(arguments.get("padding", 0))
        unit = str(arguments.get("unit", "auto"))
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            box = _normalize_bbox(arguments.get("bbox"), image.width, image.height, unit=unit, padding=padding)
            crop = image.crop(box)
            original_size = crop.size
            crop, aspect_padding_applied = _pad_extreme_aspect(crop)
            out_path = _resolve_output_path(arguments.get("output_path"), ".png", tool, source_path, {"bbox": box})
            crop.save(out_path)
        return _json_result(
            status="ok",
            tool=tool,
            image_path=str(out_path),
            source_image_path=str(source_path),
            source=source_meta,
            bbox_pixels=list(box),
            width=crop.width,
            height=crop.height,
            original_width=original_size[0],
            original_height=original_size[1],
            aspect_padding_applied=aspect_padding_applied,
            coordinate_space="pixel_top_left",
        )
    except Exception as exc:
        return _error(tool, str(exc))


def zoom_region(arguments: dict[str, Any]) -> str:
    tool = "zoom_region"
    try:
        source_path, source_meta = _image_for_region(arguments, tool)
        Image = _import_pil()
        padding = float(arguments.get("padding", 0))
        unit = str(arguments.get("unit", "auto"))
        scale = float(arguments.get("scale", arguments.get("zoom", 2.0)))
        if scale <= 0:
            raise ValueError("scale must be > 0")
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            box = _normalize_bbox(arguments.get("bbox"), image.width, image.height, unit=unit, padding=padding)
            crop = image.crop(box)
            target_size = (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale))))
            resized = crop.resize(target_size, Image.Resampling.LANCZOS)
            original_size = resized.size
            resized, aspect_padding_applied = _pad_extreme_aspect(resized)
            out_path = _resolve_output_path(arguments.get("output_path"), ".png", tool, source_path, {"bbox": box, "scale": scale})
            resized.save(out_path)
        return _json_result(
            status="ok",
            tool=tool,
            image_path=str(out_path),
            source_image_path=str(source_path),
            source=source_meta,
            bbox_pixels=list(box),
            scale=scale,
            width=resized.width,
            height=resized.height,
            original_width=original_size[0],
            original_height=original_size[1],
            aspect_padding_applied=aspect_padding_applied,
            coordinate_space="pixel_top_left",
        )
    except Exception as exc:
        return _error(tool, str(exc))


def _as_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _as_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_as_jsonable(item) for item in value]
        return str(value)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _parse_page_range(value: Any) -> tuple[int, int] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        text = value.strip()
        if "-" in text:
            left, right = text.split("-", 1)
            return int(left), int(right)
        page = int(text)
        return page, page
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    page = int(value)
    return page, page


def _assert_docling_torch_runtime() -> None:
    """Reject known-unstable Torch builds before Docling imports model code.

    The check reads package metadata only.  Importing a broken Windows Torch
    development build can spawn child processes and terminate the caller before
    a tool error can be returned.
    """
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        raise RuntimeError("Docling requires PyTorch, but no torch package is available.")
    version_file = Path(spec.origin).with_name("version.py")
    try:
        version_text = version_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if ".dev" in version_text.lower():
        raise RuntimeError(
            "Docling cannot run with the detected PyTorch development build. "
            "Install a stable CPU or CUDA PyTorch release, then restart Tool Studio. "
            "The development build is rejected before import because it is unstable on this Windows runtime."
        )


def _prepare_docling_runtime() -> None:
    """Keep Docling model workers in the same Python runtime on Windows.

    Windows process spawning otherwise resolved ``python`` to an unrelated
    Anaconda 3.13 installation, which cannot load this Python 3.11 tool stack.
    """
    if os.name == "nt":
        executable = sys.executable
        multiprocessing.set_executable(executable)
        os.environ["PYTHONEXECUTABLE"] = executable


def _get_docling_converter() -> Any:
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        prepare_headless_ocr_runtime()
        _assert_docling_torch_runtime()
        _prepare_docling_runtime()
        artifacts_path = os.environ.get("OPENCLAW_DOCLING_ARTIFACTS_PATH") or os.environ.get("DOCLING_ARTIFACTS_PATH")
        if artifacts_path:
            os.environ["DOCLING_ARTIFACTS_PATH"] = str(Path(artifacts_path).expanduser())
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
        except ImportError as exc:
            raise ImportError("Docling is required. Install with: pip install docling") from exc
        format_options = None
        if artifacts_path:
            pipeline_options = PdfPipelineOptions(
                artifacts_path=str(Path(artifacts_path).expanduser()),
                ocr_options=_docling_ocr_options(RapidOcrOptions),
            )
            if _env_flag("OPENCLAW_DOCLING_DISABLE_OCR_TABLE"):
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = False
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            }
        _DOCLING_CONVERTER = DocumentConverter(format_options=format_options)
    return _DOCLING_CONVERTER


def _docling_worker_runtime_path() -> Path | None:
    configured = os.environ.get("OPENCLAW_DOC_TOOL_RUNTIME_PATH")
    candidate = Path(configured).expanduser() if configured else Path(__file__).resolve().parents[1] / "tool_studio" / "runtime"
    return candidate if (candidate / "torch").is_dir() else None


def _use_docling_worker() -> bool:
    return os.name == "nt" and os.environ.get("OPENCLAW_DOCLING_WORKER", "1").lower() not in {"0", "false", "no"}


def _docling_convert_worker(path: Path, page_range: tuple[int, int] | None) -> tuple[Any, Any]:
    runtime = _docling_worker_runtime_path()
    if runtime is None:
        raise RuntimeError("Docling worker requires the isolated stable runtime at tool_studio/runtime.")
    handle, payload_path_text = tempfile.mkstemp(prefix="openclaw_docling_worker_", suffix=".json")
    os.close(handle)
    payload_path = Path(payload_path_text)
    request = {
        "document_path": str(path),
        "page_range": list(page_range) if page_range else None,
        "artifacts_path": os.environ.get("OPENCLAW_DOCLING_ARTIFACTS_PATH") or os.environ.get("DOCLING_ARTIFACTS_PATH"),
        "output_path": str(payload_path),
    }
    environment = os.environ.copy()
    environment["OPENCLAW_DOC_TOOL_RUNTIME_PATH"] = str(runtime)
    environment.setdefault("CONDA_AUTO_ACTIVATE_BASE", "false")
    environment.setdefault("CONDA_CHANGEPS1", "false")
    worker_path = Path(__file__).with_name("docling_worker.py")
    try:
        completed = subprocess.run(
            [sys.executable, "-S", str(worker_path)],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=int(os.environ.get("OPENCLAW_DOCLING_TIMEOUT", "300")),
            check=False,
        )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = completed.stderr[-2000:]
            raise RuntimeError(f"Docling worker returned invalid JSON (exit {completed.returncode}): {detail}") from exc
        if completed.returncode != 0 or response.get("status") == "error":
            detail = response.get("error") or completed.stderr[-2000:]
            raise RuntimeError(f"Docling worker failed: {detail}")
        try:
            worker_payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Docling worker did not write a readable payload: {exc}") from exc
    finally:
        payload_path.unlink(missing_ok=True)
    document_json = worker_payload.get("document_json")
    if not isinstance(document_json, dict):
        raise RuntimeError("Docling worker did not return a document JSON object")
    document = {"_worker_markdown": str(worker_payload.get("markdown", "")), "_worker_document_json": document_json}
    return {"status": worker_payload.get("status")}, document


def _docling_convert(path: Path, page_range: tuple[int, int] | None = None) -> tuple[Any, Any]:
    stat = path.stat()
    cache_key = (
        str(path.resolve()),
        stat.st_mtime,
        page_range[0] if page_range else None,
        page_range[1] if page_range else None,
    )
    if cache_key in _DOCLING_CACHE:
        return _DOCLING_CACHE[cache_key]

    if _use_docling_worker():
        result, document = _docling_convert_worker(path, page_range)
    else:
        converter = _get_docling_converter()
        kwargs: dict[str, Any] = {"raises_on_error": True}
        if page_range is not None:
            kwargs["page_range"] = page_range
        result = converter.convert(str(path), **kwargs)
        status = str(getattr(result, "status", "success")).split(".")[-1].lower()
        if status not in {"success", "partial_success"}:
            errors = getattr(result, "errors", None)
            raise RuntimeError(f"Docling conversion failed with status={status}: {errors}")
        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError("Docling conversion did not return a document")
    _DOCLING_CACHE[cache_key] = (result, document)
    return result, document


def _docling_export_dict(document: Any) -> dict[str, Any]:
    if isinstance(document, dict) and "_worker_document_json" in document:
        return document["_worker_document_json"]
    if not hasattr(document, "export_to_dict"):
        raise RuntimeError("Docling document does not expose export_to_dict()")
    exported = document.export_to_dict()
    exported = _as_jsonable(exported)
    if not isinstance(exported, dict):
        return {"document": exported}
    return exported


def _docling_export_markdown(document: Any) -> str:
    if isinstance(document, dict) and "_worker_markdown" in document:
        return document["_worker_markdown"]
    if not hasattr(document, "export_to_markdown"):
        raise RuntimeError("Docling document does not expose export_to_markdown()")
    return str(document.export_to_markdown())


def _docling_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if {"l", "t", "r", "b"}.issubset(value):
            return [float(value["l"]), float(value["t"]), float(value["r"]), float(value["b"])]
        keys = ("x0", "y0", "x1", "y1")
        if all(key in value for key in keys):
            return [float(value[key]) for key in keys]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(item) for item in value]
    for attr_set in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
        if all(hasattr(value, attr) for attr in attr_set):
            return [float(getattr(value, attr)) for attr in attr_set]
    return None


def _docling_first_prov(item: Any) -> Any | None:
    prov = item.get("prov") if isinstance(item, dict) else getattr(item, "prov", None)
    if isinstance(prov, list) and prov:
        return prov[0]
    return prov


def _docling_page_no(item: Any) -> int | None:
    prov = _docling_first_prov(item)
    if prov is None:
        return None
    if isinstance(prov, dict):
        page = prov.get("page_no") or prov.get("page_number")
    else:
        page = getattr(prov, "page_no", None) or getattr(prov, "page_number", None)
    return int(page) if page not in (None, "") else None


def _docling_item_bbox(item: Any) -> list[float] | None:
    prov = _docling_first_prov(item)
    bbox = None
    if isinstance(prov, dict):
        bbox = prov.get("bbox")
    elif prov is not None:
        bbox = getattr(prov, "bbox", None)
    if bbox is None and isinstance(item, dict):
        bbox = item.get("bbox")
    elif bbox is None:
        bbox = getattr(item, "bbox", None)
    return _docling_bbox(bbox)


def _docling_label(value: Any, container: str | None = None) -> str:
    raw = None
    if isinstance(value, dict):
        raw = value.get("label") or value.get("type")
    else:
        raw = getattr(value, "label", None) or getattr(value, "type", None)
    raw = getattr(raw, "name", raw)
    label = str(raw or container or "unknown").lower()
    mapping = {
        "section_header": "title",
        "title": "title",
        "heading": "title",
        "text": "paragraph",
        "paragraph": "paragraph",
        "list_item": "paragraph",
        "code": "paragraph",
        "caption": "paragraph",
        "document_index": "paragraph",
        "table": "table",
        "picture": "image",
        "figure": "image",
        "formula": "formula",
        "equation": "formula",
        "chart": "chart",
    }
    if container == "tables":
        return "table"
    if container == "pictures":
        return "image"
    return mapping.get(label, label)


def _docling_item_text(item: Any, document: Any | None = None) -> str | None:
    if isinstance(item, dict):
        for key in ("text", "orig", "caption", "name"):
            text = item.get(key)
            if text:
                return str(text)
        return None
    text = getattr(item, "text", None)
    if text:
        return str(text)
    if hasattr(item, "caption_text") and document is not None:
        try:
            caption = item.caption_text(document)
            if caption:
                return str(caption)
        except Exception:
            return None
    return None


def _docling_table_rows_from_data(data: Any) -> list[list[Any]]:
    if not isinstance(data, dict):
        return []
    cells = data.get("table_cells") or data.get("cells") or []
    row_count = int(data.get("num_rows") or data.get("row_count") or 0)
    col_count = int(data.get("num_cols") or data.get("column_count") or 0)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row_count = max(row_count, int(cell.get("end_row_offset_idx", cell.get("row", 0) + 1) or 0))
        col_count = max(col_count, int(cell.get("end_col_offset_idx", cell.get("col", 0) + 1) or 0))
    if not row_count or not col_count:
        return []
    rows = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = int(cell.get("start_row_offset_idx", cell.get("row", 0)) or 0)
        col = int(cell.get("start_col_offset_idx", cell.get("col", 0)) or 0)
        if row < row_count and col < col_count:
            rows[row][col] = cell.get("text", "")
    return rows


def _docling_layout_elements(document: Any, document_json: dict[str, Any]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    if hasattr(document, "iterate_items"):
        try:
            for index, item_level in enumerate(document.iterate_items()):
                item = item_level[0] if isinstance(item_level, (tuple, list)) and item_level else item_level
                level = item_level[1] if isinstance(item_level, (tuple, list)) and len(item_level) > 1 else None
                item_type = _docling_label(item)
                element: dict[str, Any] = {
                    "id": f"docling:item:{index}",
                    "type": item_type,
                    "source_container": "docling_iterate_items",
                    "page_number": _docling_page_no(item),
                    "bbox": _docling_item_bbox(item),
                    "text": _short_text(_docling_item_text(item, document)),
                }
                if level is not None:
                    element["level"] = level
                if item_type == "table" and hasattr(item, "export_to_dataframe"):
                    try:
                        rows = _df_to_rows(item.export_to_dataframe(doc=document))
                        element["row_count"] = len(rows)
                        element["column_count"] = max((len(row) for row in rows), default=0)
                        element["text"] = _short_text(_rows_to_markdown(rows), 320)
                    except Exception:
                        pass
                elements.append(element)
            if elements:
                return elements
        except Exception:
            elements = []

    for container in ("texts", "tables", "pictures", "form_items", "key_value_items", "groups"):
        items = document_json.get(container)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            element_type = _docling_label(item, container)
            element = {
                "id": str(item.get("self_ref") or f"docling:{container}:{index}") if isinstance(item, dict) else f"docling:{container}:{index}",
                "type": element_type,
                "source_container": container,
                "page_number": _docling_page_no(item),
                "bbox": _docling_item_bbox(item),
                "text": _short_text(_docling_item_text(item)),
            }
            if element_type == "table" and isinstance(item, dict):
                rows = _docling_table_rows_from_data(item.get("data"))
                element["row_count"] = len(rows)
                element["column_count"] = max((len(row) for row in rows), default=0)
                element["text"] = _short_text(_rows_to_markdown(rows), 320) or element["text"]
            elements.append(element)
    return elements


def _pdf_table_records(path: Path, page_number: int) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        return []
    records: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return []
            page = pdf.pages[page_number - 1]
            for table_index, table in enumerate(page.find_tables() or []):
                rows = table.extract() or []
                records.append(
                    {
                        "table_index": table_index,
                        "page_number": page_number,
                        "bbox": list(getattr(table, "bbox", []) or []),
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": max((len(row) for row in rows), default=0),
                    }
                )
    except Exception:
        return []
    return records


def _bbox_overlaps(left: Any, right: Any) -> bool:
    """Return whether two xyxy boxes have a non-empty intersection."""
    try:
        lx1, ly1, lx2, ly2 = (float(value) for value in left[:4])
        rx1, ry1, rx2, ry2 = (float(value) for value in right[:4])
    except (TypeError, ValueError, IndexError):
        return False
    return min(lx2, rx2) > max(lx1, rx1) and min(ly2, ry2) > max(ly1, ry1)


def _parse_pdf_document(path: Path, page_range: tuple[int, int] | None = None) -> tuple[str, dict[str, Any], str]:
    fitz = _import_fitz()
    markdown_chunks: list[str] = []
    pages: list[dict[str, Any]] = []
    with fitz.open(str(path)) as doc:
        start, end = page_range or (1, len(doc))
        start = max(1, start)
        end = min(len(doc), end)
        for page_number in range(start, end + 1):
            page = doc[page_number - 1]
            raw_text = page.get_text("text").strip()
            blocks: list[dict[str, Any]] = []
            for block_index, block in enumerate((page.get_text("dict") or {}).get("blocks", [])):
                block_type = block.get("type")
                bbox = [float(value) for value in block.get("bbox", [])]
                if block_type == 1:
                    blocks.append(
                        {
                            "id": f"page:{page_number}:image:{block_index}",
                            "type": "image",
                            "bbox": bbox,
                            "width": block.get("width"),
                            "height": block.get("height"),
                        }
                    )
                    continue
                line_texts: list[str] = []
                max_font = 0.0
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if line_text.strip():
                        line_texts.append(line_text.strip())
                    for span in line.get("spans", []):
                        max_font = max(max_font, float(span.get("size", 0.0) or 0.0))
                block_text = "\n".join(line_texts).strip()
                if not block_text:
                    continue
                block_label = "formula" if _looks_like_formula(block_text) else ("title" if max_font >= 18 else "paragraph")
                blocks.append(
                    {
                        "id": f"page:{page_number}:text:{block_index}",
                        "type": block_label,
                        "bbox": bbox,
                        "text": block_text,
                        "max_font": max_font,
                    }
                )
            tables = _pdf_table_records(path, page_number)
            # pdfplumber gives us the table cell matrix.  Do not append the
            # same cells once more as PyMuPDF's linear text stream: that was
            # the source of the old label/value split and duplicate tables.
            body_blocks = [
                block
                for block in blocks
                if block.get("type") != "image"
                and not any(_bbox_overlaps(block.get("bbox"), table.get("bbox")) for table in tables)
            ]
            text = "\n".join(str(block.get("text", "")).strip() for block in body_blocks if str(block.get("text", "")).strip()).strip()
            if not text and not tables:
                text = raw_text
            page_markdown = [f"## Page {page_number}"]
            if text:
                page_markdown.append(text)
            for table in tables:
                table_md = _rows_to_markdown(table["rows"])
                if table_md:
                    page_markdown.append(f"### Table {table['table_index'] + 1}\n\n{table_md}")
            markdown_chunks.append("\n\n".join(page_markdown))
            pages.append(
                {
                    "page_number": page_number,
                    "markdown": "\n\n".join(page_markdown),
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "text": text,
                    "blocks": blocks,
                    "tables": tables,
                    "image_count": len([block for block in blocks if block["type"] == "image"]),
                }
            )
    data = {
        "document_path": str(path),
        "format": "pdf",
        "engine": "openclaw_pymupdf_pdfplumber",
        "page_count": len(pages),
        "pages": pages,
        "tables": [table for page in pages for table in page["tables"]],
    }
    return "\n\n".join(markdown_chunks).strip(), data, "openclaw_pymupdf_pdfplumber"


def _parse_docx_document(path: Path) -> tuple[str, dict[str, Any], str]:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx") from exc
    doc = Document(str(path))
    markdown_parts: list[str] = []
    paragraphs: list[dict[str, Any]] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append({"text": text, "style": paragraph.style.name if paragraph.style else None})
            markdown_parts.append(text)
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        tables.append(
            {
                "table_index": table_index,
                "rows": rows,
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
            }
        )
        markdown_parts.append(f"### Table {table_index + 1}\n\n{_rows_to_markdown(rows)}")
    data = {
        "document_path": str(path),
        "format": "docx",
        "engine": "openclaw_python_docx",
        "paragraphs": paragraphs,
        "tables": tables,
    }
    return "\n\n".join(markdown_parts).strip(), data, "openclaw_python_docx"


def _parse_pptx_document(path: Path) -> tuple[str, dict[str, Any], str]:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise ImportError("python-pptx is required for PPTX parsing. Install with: pip install python-pptx") from exc
    prs = Presentation(str(path))
    slide_markdown: list[str] = []
    slides: list[dict[str, Any]] = []
    for index, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        shapes: list[dict[str, Any]] = []
        for shape in slide.shapes:
            shape_info = {
                "type": str(getattr(shape, "shape_type", "")),
                "left": int(getattr(shape, "left", 0) or 0),
                "top": int(getattr(shape, "top", 0) or 0),
                "width": int(getattr(shape, "width", 0) or 0),
                "height": int(getattr(shape, "height", 0) or 0),
            }
            if hasattr(shape, "text"):
                text = str(shape.text).strip()
                if text:
                    texts.append(text)
                    shape_info["text"] = text
            shapes.append(shape_info)
        slide_markdown.append(f"## Slide {index}\n\n" + "\n\n".join(texts))
        slides.append({"slide_number": index, "texts": texts, "shapes": shapes})
    data = {
        "document_path": str(path),
        "format": "pptx",
        "engine": "openclaw_python_pptx",
        "slides": slides,
    }
    return "\n\n".join(slide_markdown).strip(), data, "openclaw_python_pptx"


def _parse_text_document(path: Path) -> tuple[str, dict[str, Any], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    data = {
        "document_path": str(path),
        "format": path.suffix.lower().lstrip(".") or "text",
        "engine": "openclaw_plain_text",
        "text": text,
    }
    return text, data, "openclaw_plain_text"


def _parse_image_document(path: Path) -> tuple[str, dict[str, Any], str]:
    result = json.loads(ocr_region({"image_path": str(path)}))
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "OCR failed"))
    text = str(result.get("text", ""))
    data = {
        "document_path": str(path),
        "format": "image",
        "engine": result.get("engine", "ocr"),
        "ocr": result,
    }
    return text, data, str(result.get("engine", "ocr"))


def _parse_document_local(path: Path, page_range: tuple[int, int] | None = None) -> tuple[str, dict[str, Any], str]:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _parse_pdf_document(path, page_range)
    if suffix in DOC_SUFFIXES:
        return _parse_docx_document(path)
    if suffix in PPT_SUFFIXES:
        return _parse_pptx_document(path)
    if suffix in IMAGE_SUFFIXES:
        return _parse_image_document(path)
    return _parse_text_document(path)


def _requested_page_numbers(arguments: dict[str, Any]) -> list[int] | None:
    value = arguments.get("page_numbers")
    if value not in (None, "", []):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("page_numbers must be a list of 1-based page numbers")
        pages = sorted({int(item) for item in value})
        if any(page < 1 for page in pages):
            raise ValueError("page_numbers must contain values >= 1")
        return pages
    page_range = _parse_page_range(arguments.get("page_range"))
    if page_range is None:
        return None
    start, end = page_range
    if start < 1 or end < start:
        raise ValueError("page_range must be a valid 1-based inclusive range")
    return list(range(start, end + 1))


def _page_records_from_document(path: Path, document: dict[str, Any], markdown: str) -> list[dict[str, Any]]:
    """Convert local parser output into stable page-boundary records."""
    raw_pages = document.get("pages") if isinstance(document, dict) else None
    if isinstance(raw_pages, list) and raw_pages:
        records: list[dict[str, Any]] = []
        for index, page in enumerate(raw_pages, start=1):
            if not isinstance(page, dict):
                continue
            page_number = int(page.get("page_number", index))
            page_markdown = str(page.get("markdown") or page.get("text") or "").strip()
            if not page_markdown.startswith("#"):
                page_markdown = f"## Page {page_number}\n\n{page_markdown}".strip()
            records.append({"page_number": page_number, "markdown": page_markdown, "source": page})
        if records:
            return records

    slides = document.get("slides") if isinstance(document, dict) else None
    if isinstance(slides, list) and slides:
        records = []
        for index, slide in enumerate(slides, start=1):
            slide_number = int(slide.get("slide_number", index)) if isinstance(slide, dict) else index
            texts = slide.get("texts", []) if isinstance(slide, dict) else []
            body = "\n\n".join(str(item) for item in texts if str(item).strip())
            records.append({"page_number": slide_number, "markdown": f"## Slide {slide_number}\n\n{body}".strip(), "source": slide})
        return records

    # DOCX, image and plain-text inputs are single logical pages for the
    # purpose of follow-up reads.  The explicit boundary is still valuable to
    # the agent and keeps the return schema uniform across formats.
    body = markdown.strip()
    return [{"page_number": 1, "markdown": f"## Page 1\n\n{body}".strip(), "source": document}]


def _page_observation(page: dict[str, Any]) -> dict[str, Any]:
    """Build the only page representation exposed by parse_document.

    Parser internals such as raw blocks and source-page JSON stay in the
    artifact, never beside a second full-document markdown string in the
    model observation.  Table/image metadata is deliberately small and gives
    the agent an explicit route to visual follow-up tools.
    """
    source = page.get("source") if isinstance(page.get("source"), dict) else page
    tables = source.get("tables", []) if isinstance(source, dict) else []
    tables = tables if isinstance(tables, list) else []
    table_regions = [
        {"page_number": int(source.get("page_number", page.get("page_number", 1))), "bbox": table.get("bbox")}
        for table in tables
        if isinstance(table, dict) and table.get("bbox")
    ]
    blocks = source.get("blocks", []) if isinstance(source, dict) else []
    blocks = blocks if isinstance(blocks, list) else []
    image_regions = [
        {"page_number": int(source.get("page_number", page.get("page_number", 1))), "bbox": block.get("bbox")}
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "image" and block.get("bbox")
    ]
    has_images = bool(image_regions) or int(source.get("image_count", 0) or 0) > 0
    return {
        "page_number": int(page.get("page_number", source.get("page_number", 1))),
        "markdown": str(page.get("markdown", "")),
        "has_tables": bool(tables),
        "table_count": len(tables),
        "table_extraction_recommended": bool(tables),
        "table_regions": table_regions,
        "has_images": has_images,
        "visual_content_omitted": has_images,
        "image_regions": image_regions,
    }


def _select_page_records(
    records: list[dict[str, Any]],
    requested_pages: list[int] | None,
    max_chars: int,
) -> tuple[list[dict[str, Any]], bool, bool]:
    page_count = len(records)
    available = {int(page["page_number"]) for page in records}
    selected_numbers = requested_pages if requested_pages is not None else [int(page["page_number"]) for page in records]
    missing = [page for page in selected_numbers if page not in available]
    if missing:
        raise ValueError(f"requested page does not exist: {missing[0]}")

    selected = [page for page in records if int(page["page_number"]) in set(selected_numbers)]
    returned: list[dict[str, Any]] = []
    used_chars = 0
    truncated = False
    for page in selected:
        page_copy = _page_observation(page)
        page_markdown = str(page_copy.get("markdown", ""))
        separator_chars = 2 if returned else 0
        if max_chars > 0 and used_chars + separator_chars + len(page_markdown) > max_chars:
            if not returned:
                remaining = max(1, max_chars - separator_chars)
                page_copy["markdown"] = page_markdown[:remaining] + f"\n... [truncated {len(page_markdown) - remaining} chars]"
                returned.append(page_copy)
            truncated = True
            break
        returned.append(page_copy)
        used_chars += separator_chars + len(page_markdown)

    returned_numbers = {int(page["page_number"]) for page in returned}
    has_more_pages = len(returned_numbers) < page_count
    if len(returned) < len(selected):
        truncated = True
    return returned, truncated, has_more_pages


def _project_document_json(
    document: dict[str, Any],
    returned_pages: list[int],
    page_count: int,
    content_truncated: bool,
    has_more_pages: bool,
    selected_page_observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write a page-scoped JSON artifact without leaking unreturned pages."""
    returned = set(returned_pages)
    projected: dict[str, Any] = {}
    for key, value in document.items():
        if key in {"pages", "markdown"}:
            continue
        if key == "tables" and isinstance(value, list):
            projected[key] = [
                item
                for item in value
                if not isinstance(item, dict) or int(item.get("page_number", 0) or 0) in returned
            ]
        else:
            projected[key] = value
    projected.update(
        {
            "page_count": page_count,
            "returned_pages": returned_pages,
            "truncated": bool(content_truncated),
            "content_truncated": bool(content_truncated),
            "has_more_pages": bool(has_more_pages),
            "document_has_unreturned_pages": bool(has_more_pages),
            "pages": [
                {
                    **(
                        dict(page.get("source"))
                        if isinstance(page.get("source"), dict)
                        else {"page_number": page.get("page_number")}
                    ),
                    "page_number": int(page.get("page_number", 0)),
                    "markdown": str(page.get("markdown", "")),
                }
                for page in selected_page_observations
            ],
        }
    )
    return projected


def parse_document(arguments: dict[str, Any]) -> str:
    tool = "parse_document"
    document_path = _path_arg(arguments, "document_path")
    if document_path is None:
        return _error(tool, "document_path is required")
    problem = _ensure_file(document_path, tool)
    if problem:
        return _error(tool, problem)

    output_format = str(arguments.get("output_format", "markdown")).lower()
    if output_format not in {"markdown", "json", "both"}:
        return _error(tool, "output_format must be one of: markdown, json, both")
    max_chars = int(arguments.get("max_chars", 6000))

    try:
        requested_pages = _requested_page_numbers(arguments)
        # The deterministic local parser is the fast/default path.  Docling
        # conversion may initialize a CPU OCR pipeline and take minutes even
        # for a text PDF, while its preview is not used to build the returned
        # page records.  Keep it available for an explicit request (or an
        # application-wide opt-in) so scan-heavy workflows can still use the
        # bundled OCR artifacts without slowing every rollout.
        use_docling_value = arguments.get("use_docling", os.environ.get("OPENCLAW_PARSE_USE_DOCLING", "0"))
        use_docling = str(use_docling_value).strip().lower() in {"1", "true", "yes", "on"}
        docling_markdown = ""
        docling_document: dict[str, Any] | None = None
        docling_engine = ""
        local_markdown, local_document, local_engine = _parse_document_local(document_path)
        if use_docling:
            try:
                page_range = None
                if requested_pages:
                    page_range = (min(requested_pages), max(requested_pages))
                _, converted = _docling_convert(document_path, page_range=page_range)
                docling_markdown = _docling_export_markdown(converted)
                docling_document = _docling_export_dict(converted)
                docling_engine = "docling"
            except Exception:
                # The local parser remains the documented fallback when an
                # explicit Docling/OCR request cannot be initialized.
                docling_document = None
        page_records = _page_records_from_document(document_path, local_document, local_markdown)
        page_count = len(page_records)
        selected_pages, truncated, has_more_pages = _select_page_records(page_records, requested_pages, max_chars)
        returned_pages = [int(page["page_number"]) for page in selected_pages]
        selected_markdown = "\n\n".join(str(page["markdown"]) for page in selected_pages)
        document_json = _project_document_json(
            local_document,
            returned_pages,
            page_count,
            truncated,
            has_more_pages,
            selected_pages,
        )
        engine = f"{docling_engine}+{local_engine}" if docling_engine else local_engine

        payload: dict[str, Any] = {
            "status": "ok",
            "tool": tool,
            "engine": engine,
            "document_path": str(document_path),
            "output_format": output_format,
            "docling_used": bool(docling_engine),
            "page_count": page_count,
            "returned_pages": returned_pages,
            "truncated": truncated,
            "content_truncated": truncated,
            "has_more_pages": has_more_pages,
            "document_has_unreturned_pages": has_more_pages,
            "pages": selected_pages,
        }

        if output_format in {"markdown", "both"}:
            md_path = _resolve_output_path(arguments.get("markdown_path") or arguments.get("output_path"), ".md", tool, document_path)
            _write_text(md_path, selected_markdown)
            payload.update({"markdown_path": str(md_path)})

        if output_format in {"json", "both"}:
            json_path = _resolve_output_path(arguments.get("json_path") or arguments.get("output_path"), ".json", tool, document_path)
            _write_json(json_path, document_json)
            json_preview = json.dumps(document_json, ensure_ascii=False, indent=2, default=str)
            payload.update({"json_preview": json_preview, "json_path": str(json_path)})

        return _json_result(**payload)
    except Exception as exc:
        return _error(tool, str(exc), document_path=str(document_path))


def _short_text(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\n", " ").strip()
    if not text:
        return None
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _looks_like_formula(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    math_markers = (
        "=",
        "\u2264",
        "\u2265",
        "\u2211",
        "\u222b",
        "\u221a",
        "\u03b1",
        "\u03b2",
        "\u03b3",
        "\u03bb",
        "\u03b8",
        "\\frac",
        "\\sum",
        "\\int",
    )
    if any(marker in stripped for marker in math_markers):
        letters = sum(ch.isalpha() for ch in stripped)
        digits_symbols = sum(ch.isdigit() or ch in "+-*/=()[]{}^_<>|.,:;\\/" for ch in stripped)
        return digits_symbols >= max(3, letters // 2)
    return False


def _pymupdf_pdf_table_elements(path: Path, page_no: int) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        return []
    elements: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            if page_no < 1 or page_no > len(pdf.pages):
                return []
            page = pdf.pages[page_no - 1]
            for table_index, table in enumerate(page.find_tables() or []):
                bbox = getattr(table, "bbox", None)
                rows = table.extract() or []
                elements.append(
                    {
                        "id": f"page:{page_no}:table:{table_index}",
                        "type": "table",
                        "source_container": "pdfplumber_tables",
                        "page_number": page_no,
                        "bbox": list(bbox) if bbox else None,
                        "row_count": len(rows),
                        "column_count": max((len(row) for row in rows), default=0),
                        "text": _short_text(_rows_to_markdown(rows), 320),
                    }
                )
    except Exception:
        return []
    return elements


def _pymupdf_pdf_layout(path: Path, page_number: int | None = None) -> list[dict[str, Any]]:
    fitz = _import_fitz()
    elements: list[dict[str, Any]] = []
    with fitz.open(str(path)) as doc:
        pages = [page_number] if page_number else list(range(1, len(doc) + 1))
        for page_no in pages:
            if page_no < 1 or page_no > len(doc):
                continue
            page = doc[page_no - 1]
            elements.extend(_pymupdf_pdf_table_elements(path, page_no))
            info = page.get_text("dict")
            for block_index, block in enumerate(info.get("blocks", [])):
                bbox = block.get("bbox")
                block_type = block.get("type")
                text_parts: list[str] = []
                max_font = 0.0
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    text_parts.append(line_text)
                    for span in line.get("spans", []):
                        max_font = max(max_font, float(span.get("size", 0.0) or 0.0))
                text = "\n".join(part for part in text_parts if part.strip()).strip()
                if block_type == 1:
                    label = "image"
                elif _looks_like_formula(text):
                    label = "formula"
                elif text and max_font >= 18:
                    label = "title"
                else:
                    label = "paragraph"
                elements.append(
                    {
                        "id": f"page:{page_no}:block:{block_index}",
                        "type": label,
                        "source_container": "pymupdf_blocks",
                        "page_number": page_no,
                        "bbox": bbox,
                        "text": _short_text(text),
                    }
                )
    return elements


def _filter_elements(
    elements: list[dict[str, Any]],
    page_number: int | None,
    element_types: list[str] | None,
    max_items: int,
) -> list[dict[str, Any]]:
    normalized_types = {item.lower() for item in element_types} if element_types else None
    filtered: list[dict[str, Any]] = []
    for element in elements:
        if page_number is not None and element.get("page_number") not in {page_number, str(page_number)}:
            continue
        if normalized_types:
            element_type = str(element.get("type", "")).lower()
            container = str(element.get("source_container", "")).lower()
            if element_type not in normalized_types and container not in normalized_types:
                continue
        filtered.append(element)
        if len(filtered) >= max_items:
            break
    return filtered


def _local_layout_elements(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a lightweight layout view without Docling or GUI dependencies.

    The headless server may not have ``libGL.so.1``.  PyMuPDF already gives
    us stable text/image boxes and pdfplumber gives us table boxes, so this
    fallback is sufficient for navigation and region selection even when the
    richer Docling layout backend cannot be imported.
    """
    elements: list[dict[str, Any]] = []
    pages = document.get("pages", []) if isinstance(document, dict) else []
    if not isinstance(pages, list):
        return elements
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            continue
        blocks = page.get("blocks", [])
        if isinstance(blocks, list):
            for index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                element_type = str(block.get("type") or "paragraph")
                elements.append(
                    {
                        "id": str(block.get("id") or f"page:{page_number}:block:{index}"),
                        "type": element_type,
                        "source_container": "pymupdf_blocks",
                        "page_number": page_number,
                        "bbox": block.get("bbox"),
                        "text": _short_text(block.get("text")),
                    }
                )
        tables = page.get("tables", [])
        if isinstance(tables, list):
            for index, table in enumerate(tables):
                if not isinstance(table, dict):
                    continue
                rows = table.get("rows") if isinstance(table.get("rows"), list) else []
                elements.append(
                    {
                        "id": f"page:{page_number}:table:{index}",
                        "type": "table",
                        "source_container": "pdfplumber_tables",
                        "page_number": page_number,
                        "bbox": table.get("bbox"),
                        "row_count": int(table.get("row_count", len(rows)) or 0),
                        "column_count": int(table.get("column_count", max((len(row) for row in rows), default=0)) or 0),
                        "text": _short_text(_rows_to_markdown(rows), 320),
                    }
                )
    return elements


def detect_layout(arguments: dict[str, Any]) -> str:
    tool = "detect_layout"
    document_path = _path_arg(arguments, "document_path")
    if document_path is None:
        return _error(tool, "document_path is required")
    problem = _ensure_file(document_path, tool)
    if problem:
        return _error(tool, problem)
    page_number = arguments.get("page_number")
    page_number = int(page_number) if page_number not in (None, "") else None
    raw_types = arguments.get("element_types")
    if isinstance(raw_types, str):
        element_types = [part.strip() for part in raw_types.split(",") if part.strip()]
    else:
        element_types = raw_types if isinstance(raw_types, list) else None
    max_items = int(arguments.get("max_items", 200))

    try:
        page_range = (page_number, page_number) if page_number else None
        _, document = _docling_convert(document_path, page_range=page_range)
        document_json = _docling_export_dict(document)
        elements = _docling_layout_elements(document, document_json)
        filtered = _filter_elements(elements, page_number, element_types, max_items)
        out_path = _resolve_output_path(arguments.get("output_path"), ".json", tool, document_path)
        _write_json(out_path, {"engine": "docling", "elements": filtered})
        payload = {
            "status": "ok",
            "tool": tool,
            "engine": "docling",
            "document_path": str(document_path),
            "page_number": page_number,
            "element_count": len(filtered),
            "elements": filtered,
            "json_path": str(out_path),
            "coordinate_space": "docling_backend_native",
        }
        return _json_result(**payload)
    except Exception as docling_exc:
        # Docling is useful when available, but it is not a prerequisite for
        # visual navigation.  In particular, headless workers commonly lack
        # libGL.so.1.  Fall back to the deterministic local PDF parser and
        # preserve the original backend error for diagnostics.
        try:
            _, local_document, local_engine = _parse_document_local(document_path, page_range=page_range)
            elements = _local_layout_elements(local_document)
            filtered = _filter_elements(elements, page_number, element_types, max_items)
            out_path = _resolve_output_path(arguments.get("output_path"), ".json", tool, document_path)
            _write_json(out_path, {"engine": local_engine, "elements": filtered})
            payload = {
                "status": "ok",
                "tool": tool,
                "engine": local_engine,
                "document_path": str(document_path),
                "page_number": page_number,
                "element_count": len(filtered),
                "elements": filtered,
                "json_path": str(out_path),
                "coordinate_space": "pdf_points",
                "fallback_from": "docling",
                "fallback_error": str(docling_exc),
            }
            return _json_result(**payload)
        except Exception as fallback_exc:
            return _error(
                tool,
                str(fallback_exc),
                document_path=str(document_path),
                fallback_from="docling",
                fallback_error=str(docling_exc),
            )


def _rapidocr_lines(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "txts"):
        boxes = _plain_value(getattr(result, "boxes", None)) or []
        txts = _plain_value(getattr(result, "txts", None)) or []
        scores = _plain_value(getattr(result, "scores", None)) or []
        lines: list[dict[str, Any]] = []
        for i, txt in enumerate(txts):
            if txt is None:
                continue
            lines.append(
                {
                    "bbox": boxes[i] if i < len(boxes) else None,
                    "text": str(txt),
                    "confidence": scores[i] if i < len(scores) else None,
                }
            )
        return lines
    if isinstance(result, tuple) and result:
        result = result[0]
    lines: list[dict[str, Any]] = []
    if isinstance(result, list):
        for item in result:
            item = _plain_value(item)
            if item is None or item == []:
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("txt") or item.get("rec_text")
                if text:
                    bbox = item.get("bbox")
                    if bbox is None:
                        bbox = item.get("box")
                    if bbox is None:
                        bbox = item.get("points")
                    confidence = item.get("score")
                    if confidence is None:
                        confidence = item.get("confidence")
                    lines.append(
                        {
                            "bbox": _plain_value(bbox),
                            "text": str(text),
                            "confidence": _plain_value(confidence),
                        }
                    )
                continue
            if len(item) >= 3 and isinstance(item[1], str):
                lines.append({"bbox": item[0], "text": item[1], "confidence": item[2]})
            elif len(item) >= 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                lines.append({"bbox": item[0], "text": item[1][0], "confidence": item[1][1]})
    return lines


def _paddleocr_lines(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        page = _plain_value(page)
        if page is None or page == []:
            continue
        if isinstance(page, dict):
            texts = page.get("rec_texts") or page.get("texts")
            scores = page.get("rec_scores") or page.get("scores")
            boxes = page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes") or page.get("boxes")
            if isinstance(texts, list):
                scores = scores if isinstance(scores, list) else []
                boxes = boxes if isinstance(boxes, list) else []
                for index, text in enumerate(texts):
                    if text:
                        lines.append(
                            {
                                "bbox": boxes[index] if index < len(boxes) else None,
                                "text": str(text),
                                "confidence": scores[index] if index < len(scores) else None,
                            }
                        )
                continue
            text = page.get("text") or page.get("rec_text")
            if text:
                bbox = page.get("bbox")
                if bbox is None:
                    bbox = page.get("points")
                confidence = page.get("score")
                if confidence is None:
                    confidence = page.get("confidence")
                lines.append({"bbox": _plain_value(bbox), "text": str(text), "confidence": _plain_value(confidence)})
            continue
        if not isinstance(page, list):
            continue
        for item in page:
            item = _plain_value(item)
            if isinstance(item, dict):
                text = item.get("text") or item.get("rec_text")
                if text:
                    bbox = item.get("bbox")
                    if bbox is None:
                        bbox = item.get("points")
                    lines.append({"bbox": _plain_value(bbox), "text": str(text), "confidence": _plain_value(item.get("score"))})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                rec = item[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    lines.append({"bbox": item[0], "text": rec[0], "confidence": rec[1]})
    return lines


def _easyocr_lines(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for item in result or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            lines.append({"bbox": item[0], "text": item[1], "confidence": item[2]})
    return lines


def _paddle_local_model_dir(kind: str, model_name: str) -> Path:
    override = os.environ.get(f"OPENCLAW_PADDLEOCR_{kind.upper()}_DIR")
    candidates = [Path(override).expanduser()] if override else []
    cache = os.environ.get("OPENCLAW_PADDLEOCR_CACHE_DIR")
    if cache:
        cache_path = Path(cache).expanduser()
        candidates.extend(
            [
                cache_path / model_name,
                cache_path / "models" / model_name,
                cache_path / "official_models" / model_name,
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        f"local PaddleOCR {kind} model is not cached ({model_name}); refusing to download it"
    )


def _easyocr_local_model_dir(lang: str) -> Path:
    override = os.environ.get("OPENCLAW_EASYOCR_MODEL_DIR")
    if override:
        model_dir = Path(override).expanduser()
    else:
        cache = os.environ.get("OPENCLAW_PADDLEOCR_CACHE_DIR")
        model_dir = Path(cache).expanduser() / "home" / ".EasyOCR" / "model" if cache else Path.home() / ".EasyOCR" / "model"
    required = [model_dir / "craft_mlt_25k.pth"]
    if lang.strip().lower() in {"en", "eng", "english"}:
        required.append(model_dir / "english_g2.pth")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "local EasyOCR model files are missing; refusing to download them: " + ", ".join(missing)
        )
    return model_dir


def _pdf_text_fallback_lines(region_meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Read selectable PDF text when all image OCR engines are unavailable.

    This is intentionally local and deterministic.  It is not presented as
    OCR: the returned engine name tells the policy that the observation came
    from the PDF text layer, which is often the correct fallback for a crop
    rendered from a digitally-created document.
    """

    if not isinstance(region_meta, dict):
        return []
    source = region_meta.get("source") if isinstance(region_meta.get("source"), dict) else region_meta
    document_value = source.get("document_path") or region_meta.get("document_path")
    if not document_value:
        return []
    document_path = Path(str(document_value)).expanduser()
    if document_path.suffix.lower() not in PDF_SUFFIXES or not document_path.is_file():
        return []
    try:
        fitz = _import_fitz()
        page_number = int(source.get("page_number", region_meta.get("page_number", 1)) or 1)
        with fitz.open(str(document_path)) as document:
            if page_number < 1 or page_number > len(document):
                return []
            page = document[page_number - 1]
            blocks = page.get_text("blocks") or []
            selected_bbox = region_meta.get("bbox_pixels")
            dpi = float(source.get("dpi") or 72.0)
            scale = 72.0 / max(1.0, dpi)
            target = None
            if isinstance(selected_bbox, (list, tuple)) and len(selected_bbox) == 4:
                try:
                    target = tuple(float(value) * scale for value in selected_bbox)
                except (TypeError, ValueError):
                    target = None

            def intersects(block: tuple[float, float, float, float], wanted: tuple[float, float, float, float]) -> bool:
                left, top, right, bottom = block
                w_left, w_top, w_right, w_bottom = wanted
                return right > w_left and left < w_right and bottom > w_top and top < w_bottom

            rows: list[dict[str, Any]] = []
            for block in blocks:
                if len(block) < 5:
                    continue
                text = str(block[4] or "").strip()
                if not text:
                    continue
                block_bbox = tuple(float(value) for value in block[:4])
                if target is not None and not intersects(block_bbox, target):
                    continue
                rows.append({"bbox": list(block_bbox), "text": text, "confidence": 1.0})
            # If a crop did not intersect the PDF text boxes due to differing
            # coordinate conventions, a page-scoped text observation is still
            # more useful than an empty OCR result and remains explicitly
            # labelled as such by the caller.
            if not rows and target is not None:
                for block in blocks:
                    if len(block) < 5:
                        continue
                    text = str(block[4] or "").strip()
                    if text:
                        rows.append({"bbox": list(tuple(float(value) for value in block[:4])), "text": text, "confidence": 1.0})
            return rows
    except Exception:
        return []


def _run_ocr(
    image_path: Path,
    lang: str,
    engine: str,
    region_meta: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    prepare_headless_ocr_runtime()
    requested = (engine or "auto").lower()
    errors: list[str] = []
    # Production runs are offline.  PaddleOCR/EasyOCR are optional only when
    # their model directories are explicitly provisioned; they must never be
    # reached by an automatic path that could trigger a download.  The two
    # bundled RapidOCR engines provide a deterministic local fallback chain.
    auto_backends = _env_list(
        # RapidOCR is preferred, but an empty/failed RapidOCR result is not
        # a reason to abandon the observation.  PaddleOCR and EasyOCR are
        # local-only fallbacks; their artifact checks below reject missing
        # models before import and neither backend is allowed to download.
        "OPENCLAW_OCR_AUTO_BACKENDS", "rapidocr,rapidocr_torch,paddleocr,easyocr"
    )
    backends = auto_backends if requested == "auto" else [requested]

    if "rapidocr" in backends:
        try:
            if "rapidocr" not in _OCR_ENGINES:
                from rapidocr import RapidOCR

                _OCR_ENGINES["rapidocr"] = RapidOCR(
                    params=_rapidocr_artifact_params(lang=lang, backend="onnxruntime")
                )
            result = _OCR_ENGINES["rapidocr"](str(image_path))
            lines = _rapidocr_lines(result)
            if not lines:
                raise RuntimeError("rapidocr returned no text")
            return "rapidocr", lines
        except Exception as exc:
            errors.append(f"rapidocr: {exc}")
            if requested == "rapidocr":
                raise RuntimeError("; ".join(errors)) from exc

    if "rapidocr_torch" in backends:
        try:
            if "rapidocr_torch" not in _OCR_ENGINES:
                from rapidocr import RapidOCR

                _OCR_ENGINES["rapidocr_torch"] = RapidOCR(
                    params=_rapidocr_artifact_params(lang=lang, backend="torch")
                )
            result = _OCR_ENGINES["rapidocr_torch"](str(image_path))
            lines = _rapidocr_lines(result)
            if not lines:
                raise RuntimeError("RapidOCR torch backend returned no text")
            return "rapidocr_torch", lines
        except Exception as exc:
            errors.append(f"rapidocr_torch: {exc}")
            if requested == "rapidocr_torch":
                raise RuntimeError("; ".join(errors)) from exc

    if "paddleocr" in backends:
        try:
            det_model = os.environ.get("OPENCLAW_PADDLEOCR_DET_MODEL", "PP-OCRv6_small_det")
            rec_model = os.environ.get("OPENCLAW_PADDLEOCR_REC_MODEL", "PP-OCRv6_small_rec")
            # Resolve local artifacts before importing optional ML stacks.  A
            # missing model must report the offline/no-download contract,
            # even on hosts with a broken torchvision preload.
            det_dir = _paddle_local_model_dir("det", det_model)
            rec_dir = _paddle_local_model_dir("rec", rec_model)
            _prepare_ml_backend_environment()
            _preload_torch_for_windows()
            if "paddleocr" not in _OCR_ENGINES:
                from paddleocr import PaddleOCR
                try:
                    _OCR_ENGINES["paddleocr"] = PaddleOCR(
                        lang=lang,
                        text_detection_model_name=det_model,
                        text_detection_model_dir=str(det_dir),
                        text_recognition_model_name=rec_model,
                        text_recognition_model_dir=str(rec_dir),
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        # Paddle 3.3's automatic Windows CPU path can select
                        # oneDNN/PIR, which currently fails for OCR models.
                        engine="paddle_static",
                        enable_hpi=False,
                        enable_mkldnn=False,
                        enable_cinn=False,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "installed PaddleOCR does not support local model directories"
                    ) from exc
            ocr = _OCR_ENGINES["paddleocr"]
            if hasattr(ocr, "predict"):
                try:
                    result = ocr.predict(
                        str(image_path),
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                except TypeError:
                    result = ocr.predict(str(image_path))
            elif hasattr(ocr, "ocr"):
                try:
                    result = ocr.ocr(str(image_path), cls=True)
                except TypeError as exc:
                    if "unexpected keyword argument 'cls'" not in str(exc):
                        raise
                    result = ocr.ocr(str(image_path))
            else:
                raise RuntimeError("PaddleOCR instance has neither predict nor ocr method")
            lines = _paddleocr_lines(result)
            if not lines:
                raise RuntimeError("PaddleOCR returned no text")
            return "paddleocr", lines
        except Exception as exc:
            errors.append(f"paddleocr: {exc}")
            if requested == "paddleocr":
                raise RuntimeError("; ".join(errors)) from exc

    if "easyocr" in backends:
        try:
            langs = [lang] if lang else ["en"]
            # Check the explicitly provisioned local model first.  This keeps
            # an offline run from failing in an unrelated torch/torchvision
            # import before it can state that a download is forbidden.
            model_dir = _easyocr_local_model_dir(langs[0])
            _prepare_ml_backend_environment()
            _preload_torch_for_windows()
            if "easyocr" not in _OCR_ENGINES:
                import easyocr

                _OCR_ENGINES["easyocr"] = easyocr.Reader(
                    langs,
                    model_storage_directory=str(model_dir),
                    download_enabled=False,
                )
            result = _OCR_ENGINES["easyocr"].readtext(str(image_path))
            lines = _easyocr_lines(result)
            if not lines:
                raise RuntimeError("EasyOCR returned no text")
            return "easyocr", lines
        except Exception as exc:
            errors.append(f"easyocr: {exc}")
            if requested == "easyocr":
                raise RuntimeError("; ".join(errors)) from exc

    if requested == "auto":
        pdf_lines = _pdf_text_fallback_lines(region_meta)
        if pdf_lines:
            return "pdf_text_fallback", pdf_lines

    raise RuntimeError("No OCR backend succeeded. " + "; ".join(errors))


def ocr_region(arguments: dict[str, Any]) -> str:
    tool = "ocr_region"
    image_path: Path | None = None
    region_meta: dict[str, Any] | None = None
    try:
        if _path_arg(arguments, "image_path") is not None and _path_arg(arguments, "document_path") is not None:
            raise ValueError("provide exactly one of image_path or document_path")
        bbox = arguments.get("bbox")
        if bbox is not None:
            crop_arguments = dict(arguments)
            # ``output_path`` belongs to the OCR text artifact.  Passing a
            # .json path through to crop_region makes PIL try to encode the
            # image as JSON (the historical ``unknown file extension: .json``
            # failure).  A separate image output may be requested explicitly.
            region_output_path = crop_arguments.pop("region_output_path", None)
            if region_output_path:
                crop_arguments["output_path"] = region_output_path
            else:
                crop_arguments.pop("output_path", None)
            crop_result = json.loads(crop_region(crop_arguments))
            if crop_result.get("status") != "ok":
                return _json_result(
                    status="error",
                    tool=tool,
                    error=str(crop_result.get("error", "crop_region failed")),
                    source_tool="crop_region",
                    source_result=crop_result,
                )
            image_path = Path(crop_result["image_path"])
            region_meta = crop_result
        else:
            image_path, source_meta = _image_for_region(arguments, tool)
            region_meta = source_meta
        lang = str(arguments.get("lang", "en"))
        engine_name = str(arguments.get("engine", "auto"))
        try:
            engine, lines = _run_ocr(
                image_path,
                lang=lang,
                engine=engine_name,
                region_meta=region_meta,
            )
        except TypeError as exc:
            # Keep compatibility with small test/integration adapters that
            # implement the historical three-argument OCR hook.
            if "unexpected keyword argument" not in str(exc):
                raise
            engine, lines = _run_ocr(image_path, lang=lang, engine=engine_name)
        max_lines = int(arguments.get("max_lines", 200))
        lines = lines[:max_lines]
        text = "\n".join(str(line.get("text", "")).strip() for line in lines if str(line.get("text", "")).strip())
        max_chars = int(arguments.get("max_chars", 12000))
        text, text_truncated = _truncate_text(text, max_chars)
        out_path = _resolve_output_path(arguments.get("output_path"), ".json", tool, image_path)
        payload = {
            "status": "ok",
            "tool": tool,
            "engine": engine,
            "image_path": str(image_path),
            "text": text,
            "text_truncated": text_truncated,
            "lines": lines,
            "line_count": len(lines),
            "region": region_meta,
            "json_path": str(out_path),
        }
        _write_json(out_path, {"engine": engine, "text": text, "lines": lines, "region": region_meta, **payload})
        return _json_result(**payload)
    except Exception as exc:
        # OCR is an optional accelerator.  Keep the rendered/cropped image
        # usable by the vision model when every OCR backend is unavailable
        # (for example because libGL.so.1 is missing), instead of turning a
        # recoverable visual observation into a terminal tool_error.
        input_error = isinstance(exc, ValueError) and any(
            marker in str(exc).casefold()
            for marker in ("image_path", "document_path", "provide exactly one", "either image_path")
        )
        payload: dict[str, Any] = {
            "status": "error" if input_error else "partial",
            "tool": tool,
            "engine": "unavailable",
            "text": "",
            "lines": [],
            "line_count": 0,
            "ocr_unavailable": not input_error,
            "error": str(exc),
        }
        if image_path is not None and image_path.is_file():
            payload["image_path"] = str(image_path)
        if region_meta is not None:
            payload["region"] = region_meta
        try:
            if image_path is not None:
                out_path = _resolve_output_path(arguments.get("output_path"), ".json", tool, image_path)
                payload["json_path"] = str(out_path)
                _write_json(out_path, payload)
        except Exception as write_exc:
            payload["diagnostic_write_error"] = str(write_exc)
        return _json_result(**payload)


def _rows_to_csv(rows: list[list[Any]]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue().strip()


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [[str(cell) for cell in row] + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    separator = ["---"] * width
    body = padded[1:]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |"

    return "\n".join([fmt(header), fmt(separator), *[fmt(row) for row in body]])


def _rows_to_html(rows: list[list[Any]]) -> str:
    lines = ["<table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        cells = "".join(f"<{tag}>{html.escape(str(cell))}</{tag}>" for cell in row)
        lines.append(f"  <tr>{cells}</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _df_to_rows(df: Any) -> list[list[Any]]:
    if hasattr(df, "columns") and hasattr(df, "values"):
        rows = [list(df.columns)]
        rows.extend(df.astype(str).values.tolist())
        return rows
    if isinstance(df, list):
        return df
    return []


def _table_payload(rows: list[list[Any]], output_format: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"rows": rows, "row_count": len(rows), "column_count": max((len(r) for r in rows), default=0)}
    if output_format in {"all", "csv"}:
        payload["csv"] = _rows_to_csv(rows)
    if output_format in {"all", "markdown"}:
        payload["markdown"] = _rows_to_markdown(rows)
    if output_format in {"all", "html"}:
        payload["html"] = _rows_to_html(rows)
    return payload


def _extract_table_docling(arguments: dict[str, Any], table_index: int, output_format: str) -> tuple[dict[str, Any], str]:
    path = _path_arg(arguments, "document_path")
    if path is None:
        raise ValueError("document_path is required")
    page_arg = arguments.get("page_number")
    page_range = None
    if path.suffix.lower() in PDF_SUFFIXES and page_arg not in (None, ""):
        page_number = int(page_arg)
        if page_number < 1:
            raise ValueError("page_number is 1-based and must be >= 1")
        page_range = (page_number, page_number)
    _, document = _docling_convert(path, page_range=page_range)
    document_json = _docling_export_dict(document)
    json_tables = document_json.get("tables") if isinstance(document_json, dict) else None
    worker_document = isinstance(document, dict) and "_worker_document_json" in document
    if worker_document:
        tables = json_tables if isinstance(json_tables, list) else []
    else:
        tables = list(getattr(document, "tables", []) or [])
    if table_index >= len(tables):
        location = f"page has {len(tables)} tables" if page_range else f"document has {len(tables)} tables"
        raise IndexError(f"table_index {table_index} is out of range; {location}")
    table = tables[table_index]
    rows: list[list[Any]] = []
    if not worker_document and hasattr(table, "export_to_dataframe"):
        rows = _df_to_rows(table.export_to_dataframe(doc=document))
    if not rows:
        if isinstance(json_tables, list) and table_index < len(json_tables):
            table_data = json_tables[table_index].get("data") if isinstance(json_tables[table_index], dict) else None
            rows = _docling_table_rows_from_data(table_data)
    if not rows:
        raise RuntimeError("Docling table export returned no cells")
    payload = _table_payload(rows, output_format)
    if not worker_document and output_format in {"all", "markdown"} and hasattr(table, "export_to_markdown"):
        try:
            payload["markdown"] = str(table.export_to_markdown(doc=document))
        except TypeError:
            payload["markdown"] = str(table.export_to_markdown())
    if not worker_document and output_format in {"all", "html"} and hasattr(table, "export_to_html"):
        try:
            payload["html"] = str(table.export_to_html(doc=document))
        except TypeError:
            payload["html"] = str(table.export_to_html())
    payload["table_index"] = table_index
    payload["page_range"] = list(page_range) if page_range else None
    return payload, "docling"


def _extract_table_pdfplumber(arguments: dict[str, Any], table_index: int, output_format: str) -> tuple[dict[str, Any], str]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required for this backend. Install with: pip install pdfplumber") from exc

    path = _path_arg(arguments, "document_path")
    if path is None:
        raise ValueError("document_path is required")
    page_number = _page_number(arguments)
    with pdfplumber.open(str(path)) as pdf:
        if page_number > len(pdf.pages):
            raise ValueError(f"page_number {page_number} is out of range; document has {len(pdf.pages)} pages")
        page = pdf.pages[page_number - 1]
        if arguments.get("bbox") is not None:
            box = _normalize_bbox(arguments.get("bbox"), page.width, page.height, unit=str(arguments.get("unit", "auto")))
            page = page.crop(box)
        tables = page.extract_tables() or []
    if table_index >= len(tables):
        raise IndexError(f"table_index {table_index} is out of range; page has {len(tables)} tables")
    return _table_payload(tables[table_index], output_format), "pdfplumber"


def _extract_table_docx(arguments: dict[str, Any], table_index: int, output_format: str) -> tuple[dict[str, Any], str]:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required for DOCX table extraction. Install with: pip install python-docx") from exc

    path = _path_arg(arguments, "document_path")
    if path is None:
        raise ValueError("document_path is required")
    doc = Document(str(path))
    if table_index >= len(doc.tables):
        raise IndexError(f"table_index {table_index} is out of range; document has {len(doc.tables)} tables")
    table = doc.tables[table_index]
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    return _table_payload(rows, output_format), "openclaw_python_docx"


def _extract_table_camelot(arguments: dict[str, Any], table_index: int, output_format: str) -> tuple[dict[str, Any], str]:
    try:
        import camelot
    except ImportError as exc:
        raise ImportError("Camelot is required for this backend. Install with: pip install camelot-py[cv]") from exc

    path = _path_arg(arguments, "document_path")
    if path is None:
        raise ValueError("document_path is required")
    page_number = _page_number(arguments)
    flavor = str(arguments.get("flavor", "lattice"))
    tables = camelot.read_pdf(str(path), pages=str(page_number), flavor=flavor)
    if table_index >= len(tables):
        raise IndexError(f"table_index {table_index} is out of range; page has {len(tables)} tables")
    rows = _df_to_rows(tables[table_index].df)
    return _table_payload(rows, output_format), "camelot"


def extract_table(arguments: dict[str, Any]) -> str:
    tool = "extract_table"
    document_path = _path_arg(arguments, "document_path")
    if document_path is None:
        return _error(tool, "document_path is required")
    problem = _ensure_file(document_path, tool)
    if problem:
        return _error(tool, problem)
    output_format = str(arguments.get("output_format", "all")).lower()
    if output_format not in {"all", "csv", "markdown", "html", "json"}:
        return _error(tool, "output_format must be one of: all, csv, markdown, html, json")
    table_index = int(arguments.get("table_index", 0))
    requested_engine = str(arguments.get("engine", "auto")).lower()
    engine = requested_engine

    def finish(payload: dict[str, Any], used_engine: str, **extra: Any) -> str:
        out_path = _resolve_output_path(arguments.get("output_path"), ".json", tool, document_path)
        _write_json(out_path, payload)
        return _json_result(
            status="ok",
            tool=tool,
            engine=used_engine,
            document_path=str(document_path),
            json_path=str(out_path),
            **extra,
            **payload,
        )

    try:
        if engine == "auto" and document_path.suffix.lower() in PDF_SUFFIXES:
            # Docling can import a GPU/GUI dependency that is unavailable in
            # headless workers (notably libGL.so.1).  A table request must not
            # become an infrastructure failure merely because the preferred
            # extractor is unavailable; fall back to the deterministic PDF
            # backend and expose both engines in the observation.
            try:
                payload, used_engine = _extract_table_docling(arguments, table_index, output_format)
                return finish(payload, used_engine)
            except Exception as docling_exc:
                payload, used_engine = _extract_table_pdfplumber(arguments, table_index, output_format)
                return finish(
                    payload,
                    used_engine,
                    requested_engine="auto",
                    fallback_from="docling",
                    fallback_error=str(docling_exc),
                )

        if engine == "auto":
            engine = "docling"

        if engine == "docling":
            payload, used_engine = _extract_table_docling(arguments, table_index, output_format)
            return finish(payload, used_engine)

        if document_path.suffix.lower() in DOC_SUFFIXES and engine == "docx":
            payload, used_engine = _extract_table_docx(arguments, table_index, output_format)
            return finish(payload, used_engine)

        if engine == "pdfplumber" and document_path.suffix.lower() in PDF_SUFFIXES:
            payload, used_engine = _extract_table_pdfplumber(arguments, table_index, output_format)
            return finish(payload, used_engine)

        if engine == "camelot" and document_path.suffix.lower() in PDF_SUFFIXES:
            payload, used_engine = _extract_table_camelot(arguments, table_index, output_format)
            return finish(payload, used_engine)

        return _error(tool, f"engine={engine!r} does not support file type {document_path.suffix!r}", document_path=str(document_path))
    except Exception as exc:
        return _error(tool, str(exc), document_path=str(document_path), engine=requested_engine)


def _get_deplot(model_name: str, device: str, local_files_only: bool) -> tuple[Any, Any]:
    key = (model_name, device, local_files_only)
    if key in _DEPLOT_CACHE:
        return _DEPLOT_CACHE[key]
    _prepare_ml_backend_environment()
    _preload_torch_for_windows()
    try:
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for DePlot. Install with: pip install transformers torch") from exc
    processor = Pix2StructProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name, local_files_only=local_files_only)
    if device and device != "auto":
        model = model.to(device)
    model.eval()
    _DEPLOT_CACHE[key] = (processor, model)
    return processor, model


def _default_deplot_font_path() -> str | None:
    configured = os.environ.get("OPENCLAW_DEPLOT_FONT_PATH")
    if configured and Path(configured).exists():
        return configured
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "Arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _patch_pix2struct_font(font_path: str) -> None:
    try:
        from transformers.models.pix2struct import image_processing_pix2struct as pix2struct_image_processing
    except Exception as exc:
        raise RuntimeError(f"Unable to patch Pix2Struct font handling: {exc}") from exc

    original = getattr(
        pix2struct_image_processing,
        "_openclaw_original_render_text",
        pix2struct_image_processing.render_text,
    )
    pix2struct_image_processing._openclaw_original_render_text = original

    def render_text_with_local_font(text: str, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("font_bytes") is None and kwargs.get("font_path") is None:
            kwargs["font_path"] = font_path
        return original(text, *args, **kwargs)

    pix2struct_image_processing.render_text = render_text_with_local_font


def _parse_deplot_table(text: str) -> list[list[str]]:
    normalized = text.replace("<0x0A>", "\n").replace("\\n", "\n")
    rows: list[list[str]] = []
    for line in normalized.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def chart_to_table(arguments: dict[str, Any]) -> str:
    tool = "chart_to_table"
    try:
        bbox = arguments.get("bbox")
        if bbox is not None:
            crop_result = json.loads(crop_region(arguments))
            if crop_result.get("status") != "ok":
                return _json_result(**crop_result)
            image_path = Path(crop_result["image_path"])
            region_meta = crop_result
        else:
            image_path, source_meta = _image_for_region(arguments, tool)
            region_meta = source_meta

        Image = _import_pil()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            model_name = str(arguments.get("model_name") or os.environ.get("OPENCLAW_DEPLOT_MODEL", "google/deplot"))
            device = str(arguments.get("device", "cpu"))
            local_files_only = bool(arguments.get("local_files_only", False))
            processor, model = _get_deplot(model_name, device, local_files_only)
            prompt = str(arguments.get("prompt", "Generate underlying data table of the figure below:"))
            processor_kwargs: dict[str, Any] = {"images": image, "text": prompt, "return_tensors": "pt"}
            font_path = str(arguments.get("font_path") or _default_deplot_font_path() or "")
            if font_path:
                _patch_pix2struct_font(font_path)
            inputs = processor(**processor_kwargs)
            if device and device not in {"cpu", "auto"}:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            output_ids = model.generate(**inputs, max_new_tokens=int(arguments.get("max_new_tokens", 512)))
            raw_text = processor.decode(output_ids[0], skip_special_tokens=True)

        rows = _parse_deplot_table(raw_text)
        table = _table_payload(rows, str(arguments.get("output_format", "all")).lower())
        out_path = _resolve_output_path(arguments.get("output_path"), ".json", tool, image_path)
        _write_json(out_path, {"raw_text": raw_text, "rows": rows, "region": region_meta})
        return _json_result(
            status="ok",
            tool=tool,
            engine="deplot",
            image_path=str(image_path),
            raw_text=raw_text,
            region=region_meta,
            json_path=str(out_path),
            **table,
        )
    except Exception as exc:
        return _error(tool, str(exc))


def execute_document_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    dispatch = {
        "render_page": render_page,
        "crop_region": crop_region,
        "zoom_region": zoom_region,
        "parse_document": parse_document,
        "detect_layout": detect_layout,
        "ocr_region": ocr_region,
        "extract_table": extract_table,
        "chart_to_table": chart_to_table,
    }
    func = dispatch.get(tool_name)
    if func is None:
        return _error(tool_name, f"Unknown document tool: {tool_name}")
    return func(arguments or {})


_register_tool(
    "render_page",
    "Render one PDF page or image file to a PNG image. Page numbers are 1-based. Output coordinates use pixel_top_left.",
    {
        "document_path": {"type": "string", "description": "Path to a PDF or image file."},
        "page_number": {"type": "integer", "description": "1-based page number for PDF files.", "default": 1},
        "dpi": {"type": "integer", "description": "Rendering resolution for PDF pages.", "default": 144},
        "output_path": {"type": "string", "description": "Optional output PNG path or directory."},
    },
    ["document_path"],
)

_register_tool(
    "crop_region",
    "Crop a rectangular region from an image or rendered document page.",
    {
        "image_path": {"type": "string", "description": "Path to an existing image. Use either image_path or document_path."},
        "document_path": {"type": "string", "description": "Path to a PDF or image file. Used when image_path is not provided."},
        "page_number": {"type": "integer", "description": "1-based page number when document_path is a PDF.", "default": 1},
        "bbox": {"type": "array", "items": {"type": "number"}, "description": "[x0, y0, x1, y1] in pixels or relative coordinates."},
        "unit": {"type": "string", "enum": ["auto", "pixel", "relative"], "default": "auto"},
        "padding": {"type": "number", "description": "Extra pixels to include around the region.", "default": 0},
        "dpi": {"type": "integer", "description": "Rendering resolution when a PDF page must be rendered.", "default": 144},
        "output_path": {"type": "string", "description": "Optional output PNG path or directory."},
    },
    ["bbox"],
)

_register_tool(
    "zoom_region",
    "Crop a region and enlarge it for visual inspection.",
    {
        "image_path": {"type": "string", "description": "Path to an existing image. Use either image_path or document_path."},
        "document_path": {"type": "string", "description": "Path to a PDF or image file. Used when image_path is not provided."},
        "page_number": {"type": "integer", "description": "1-based page number when document_path is a PDF.", "default": 1},
        "bbox": {"type": "array", "items": {"type": "number"}, "description": "[x0, y0, x1, y1] in pixels or relative coordinates."},
        "unit": {"type": "string", "enum": ["auto", "pixel", "relative"], "default": "auto"},
        "scale": {"type": "number", "description": "Zoom scale for the cropped region.", "default": 2.0},
        "padding": {"type": "number", "description": "Extra pixels to include around the region.", "default": 0},
        "dpi": {"type": "integer", "description": "Rendering resolution when a PDF page must be rendered.", "default": 144},
        "output_path": {"type": "string", "description": "Optional output PNG path or directory."},
    },
    ["bbox"],
)

_register_tool(
    "parse_document",
    "Read a document with a consistent page-scoped schema. The observation contains only pages listed in returned_pages; use page_numbers for targeted reads. Table/image metadata explicitly points to extract_table, render_page, crop_region, zoom_region, or ocr_region when text parsing is insufficient.",
    {
        "document_path": {"type": "string", "description": "Path to PDF, DOCX, PPTX, image, Markdown, or text file."},
        "output_format": {"type": "string", "enum": ["markdown", "json", "both"], "default": "markdown"},
        "page_range": {"description": "Optional 1-based page range for Docling conversion, e.g. '1-3' or [1, 3]."},
        "page_numbers": {"type": "array", "items": {"type": "integer", "minimum": 1}, "description": "Optional explicit 1-based pages to return, e.g. [3, 5]."},
        "max_chars": {"type": "integer", "description": "Maximum preview characters returned in the tool observation.", "default": 6000},
        "use_docling": {"type": "boolean", "description": "Opt into the slower Docling/OCR conversion for scan-heavy documents; the default uses the fast local parser."},
        "output_path": {"type": "string", "description": "Optional output file path or directory."},
        "markdown_path": {"type": "string", "description": "Optional Markdown output path."},
        "json_path": {"type": "string", "description": "Optional JSON output path."},
    },
    ["document_path"],
)

_register_tool(
    "detect_layout",
    "Detect page layout elements such as titles, paragraphs, tables, pictures, formulas, and charts using Docling.",
    {
        "document_path": {"type": "string", "description": "Path to a PDF/DOCX/PPTX/image document."},
        "page_number": {"type": "integer", "description": "Optional 1-based page number filter."},
        "element_types": {"type": "array", "items": {"type": "string"}, "description": "Optional element type filters."},
        "max_items": {"type": "integer", "description": "Maximum number of layout elements to return.", "default": 200},
        "output_path": {"type": "string", "description": "Optional JSON output path or directory."},
    },
    ["document_path"],
)

_register_tool(
    "ocr_region",
    "OCR an image or a selected region of a rendered document page using RapidOCR, PaddleOCR, or EasyOCR.",
    {
        "image_path": {"type": "string", "description": "Path to an existing image. Use either image_path or document_path."},
        "document_path": {"type": "string", "description": "Path to a PDF or image file. Used when image_path is not provided."},
        "page_number": {"type": "integer", "description": "1-based page number when document_path is a PDF.", "default": 1},
        "bbox": {"type": "array", "items": {"type": "number"}, "description": "Optional [x0, y0, x1, y1] region in pixels or relative coordinates."},
        "unit": {"type": "string", "enum": ["auto", "pixel", "relative"], "default": "auto"},
        "lang": {"type": "string", "description": "OCR language code.", "default": "en"},
        "engine": {"type": "string", "enum": ["auto", "rapidocr", "paddleocr", "easyocr"], "default": "auto"},
        "dpi": {"type": "integer", "description": "Rendering resolution when a PDF page must be rendered.", "default": 144},
        "max_lines": {"type": "integer", "description": "Maximum OCR lines returned in the tool observation.", "default": 200},
        "max_chars": {"type": "integer", "description": "Maximum OCR text characters returned in the tool observation.", "default": 12000},
        "output_path": {"type": "string", "description": "Optional JSON output path or directory."},
        "region_output_path": {"type": "string", "description": "Optional PNG/JPG path for the intermediate crop; separate from the OCR JSON output_path."},
    },
)

_register_tool(
    "extract_table",
    "Extract a table as HTML/CSV/Markdown/JSON using Docling, pdfplumber, Camelot, or python-docx.",
    {
        "document_path": {"type": "string", "description": "Path to a PDF/DOCX/PPTX/image document."},
        "page_number": {"type": "integer", "description": "1-based page number for pdfplumber/Camelot extraction.", "default": 1},
        "bbox": {"type": "array", "items": {"type": "number"}, "description": "Optional table bbox for pdfplumber, in pixels/points or relative coordinates."},
        "unit": {"type": "string", "enum": ["auto", "pixel", "relative"], "default": "auto"},
        "table_index": {"type": "integer", "description": "0-based table index among detected tables.", "default": 0},
        "engine": {"type": "string", "enum": ["auto", "docling", "pdfplumber", "camelot", "docx"], "default": "auto"},
        "flavor": {"type": "string", "enum": ["lattice", "stream"], "description": "Camelot flavor.", "default": "lattice"},
        "output_format": {"type": "string", "enum": ["all", "csv", "markdown", "html", "json"], "default": "all"},
        "output_path": {"type": "string", "description": "Optional JSON output path or directory."},
    },
    ["document_path"],
)

_register_tool(
    "chart_to_table",
    "Convert a chart image or selected chart region into an underlying data table using DePlot.",
    {
        "image_path": {"type": "string", "description": "Path to an existing chart image. Use either image_path or document_path."},
        "document_path": {"type": "string", "description": "Path to a PDF or image file. Used when image_path is not provided."},
        "page_number": {"type": "integer", "description": "1-based page number when document_path is a PDF.", "default": 1},
        "bbox": {"type": "array", "items": {"type": "number"}, "description": "Optional chart bbox in pixels or relative coordinates."},
        "unit": {"type": "string", "enum": ["auto", "pixel", "relative"], "default": "auto"},
        "model_name": {"type": "string", "description": "Hugging Face model id or local path.", "default": "google/deplot"},
        "device": {"type": "string", "description": "Torch device such as cpu or cuda.", "default": "cpu"},
        "local_files_only": {"type": "boolean", "description": "Load model only from local Hugging Face cache.", "default": False},
        "font_path": {"type": "string", "description": "Optional TrueType font path for Pix2Struct prompt rendering."},
        "max_new_tokens": {"type": "integer", "description": "Maximum generated tokens.", "default": 512},
        "output_format": {"type": "string", "enum": ["all", "csv", "markdown", "html", "json"], "default": "all"},
        "output_path": {"type": "string", "description": "Optional JSON output path or directory."},
    },
)
