"""Run the full 200-question DocVQA eval with the repaired multi-turn agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _enable_headless_ocr_runtime() -> None:
    """Expose the already-installed GL dependencies to Ray OCR workers.

    The server image contains libGL/libglib under /opt/conda, but the launch
    environment normally keeps only the NVIDIA library paths.  cv2 (used by
    rapidocr/PaddleOCR/easyocr) is therefore importable only when these
    existing system paths are inherited by the driver and its Ray children.
    Do not download or install anything here; only append directories that
    actually contain the shared libraries.
    """
    if os.name == "nt":
        return
    candidates = (
        os.environ.get("OPENCLAW_OCR_LIBRARY_DIR"),
        "/opt/conda/lib",
        "/opt/conda/pkgs/libgl-1.7.0-ha4b6fd6_2/lib",
    )
    current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    additions: list[str] = []
    for item in candidates:
        if not item or item in current or item in additions:
            continue
        directory = Path(item)
        if (directory / "libGL.so.1").exists() or (directory / "libgthread-2.0.so.0").exists():
            additions.append(str(directory))
    if additions:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(current + additions)


_enable_headless_ocr_runtime()

import run_qwen3_vl_4b_full_docvqa_baseline_08 as profile


DATA_DIR = profile.DATA_DIR
RAW_TEST_DATA = DATA_DIR / "test.jsonl"
FULL_EVAL_DATA = DATA_DIR / "full_docvqa_eval_20260730_09.jsonl"
OUTPUT_DIR = (
    profile.profile.base.PROJECT
    / "outputs"
    / "qwen3-vl-4b-docvqa-baseline-20260730-09-multiturn200"
).resolve()
RAY_TEMP_DIR = (profile.profile.base.WORKSPACE / ".ray" / "r09").resolve()
RUN_PROFILE = "full_docvqa_baseline_09_multiturn200"


def ensure_eval_data() -> None:
    if FULL_EVAL_DATA.exists():
        return
    rows: list[dict[str, Any]] = []
    with RAW_TEST_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            pdf_path = raw.get("pdf_path")
            question = raw.get("question")
            answers = raw.get("answers")
            if not isinstance(pdf_path, str) or not isinstance(question, str) or not isinstance(answers, list) or not answers:
                raise RuntimeError(f"invalid raw row {line_number}")
            document_path = (DATA_DIR / pdf_path).resolve()
            if DATA_DIR not in document_path.parents or not document_path.is_file():
                raise RuntimeError(f"invalid PDF at raw row {line_number}")
            rows.append(
                {
                    "prompt": f"Document path: {document_path}\nQuestion: {question}\n\nInspect the document with the available document tools. Return only the supported answer inside <final>...</final> when you are done.",
                    "label": json.dumps({"answers": answers, "metric": "anls"}, ensure_ascii=False),
                    "metadata": {
                        "document_path": str(document_path),
                        "metric": "anls",
                        "sample_id": raw.get("sample_id"),
                        "source_pdf_path": pdf_path,
                        "target_page": raw.get("target_page"),
                        "answer_page": raw.get("answer_page", raw.get("target_page")),
                        "answer_bbox": raw.get("answer_bbox"),
                        "num_pages": raw.get("num_pages"),
                        "page_count": raw.get("page_count", raw.get("num_pages")),
                    },
                }
            )
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 test rows, found {len(rows)}")
    with FULL_EVAL_DATA.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure() -> None:
    base = profile.profile.base
    base.TRAIN_DATA = profile.profile.REAL_TRAIN_DATA
    base.EVAL_DATA = FULL_EVAL_DATA
    base.OUTPUT_DIR = OUTPUT_DIR
    base.CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    base.RAY_TEMP_DIR = RAY_TEMP_DIR
    base.DOCUMENT_ROOT = DATA_DIR
    base.DOCUMENT_PROBE = None


def validate() -> dict[str, Any]:
    required = (
        profile.profile.base.DOCLING_ARTIFACTS_DIR / "docling-project--docling-layout-heron",
        profile.profile.base.DOCLING_ARTIFACTS_DIR / "docling-project--docling-models",
        profile.profile.base.TOOL_MODELS_DIR / "deplot" / "config.json",
        profile.profile.base.TOOL_MODELS_DIR / "deplot" / "model.safetensors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(json.dumps(missing))
    report = profile.profile._base_validate()
    if report["train_rows"] != 4 or report["eval_rows"] != 200:
        raise RuntimeError(f"expected 4 train rows and 200 eval rows, got {report}")
    report.update(
        {
            "profile": RUN_PROFILE,
            "eval_only": True,
            "agent_detail_log_dir": str(OUTPUT_DIR / "dump_details"),
            "tool_output_log_dir": str(OUTPUT_DIR / "tool_outputs"),
        }
    )
    return report


ensure_eval_data()
configure()
profile.profile.base.training_argv = profile.profile._training_argv
profile.profile.base.validate = validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", default=None, help="override the rollout output directory")
    parser.add_argument("--ray-temp-dir", default=None, help="override the Ray temporary directory")
    args = parser.parse_args()
    global OUTPUT_DIR, RAY_TEMP_DIR, RUN_PROFILE
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
        RUN_PROFILE = f"{RUN_PROFILE}_override"
    if args.ray_temp_dir:
        RAY_TEMP_DIR = Path(args.ray_temp_dir).expanduser().resolve()
    if args.output_dir or args.ray_temp_dir:
        configure()
        profile.profile.base.training_argv = profile.profile._training_argv
        profile.profile.base.validate = validate
    if args.check_only:
        print(json.dumps(validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    profile.profile.base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
