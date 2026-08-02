"""Evaluate the complete 200-question DocVQA test split without RL updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_qwen3_vl_4b_real_docvqa_test_06 as two_sample


base = two_sample.base
DATA_DIR = two_sample.DATA_DIR
RAW_TEST_DATA = DATA_DIR / "test.jsonl"
FULL_EVAL_DATA = DATA_DIR / "full_docvqa_eval_20260723_07.jsonl"
FULL_OUTPUT_DIR = (base.PROJECT / "outputs" / "qwen3-vl-4b-docvqa-baseline-20260728-01-full200").resolve()
FULL_RAY_TEMP_DIR = (base.WORKSPACE / ".ray" / "b20260728-01-full200").resolve()


def _configure() -> None:
    # Keep the already validated four-row train fixture because eval-only mode
    # never performs a rollout update; use the authoritative 200-QA test split
    # only for evaluation.
    base.TRAIN_DATA = two_sample.REAL_TRAIN_DATA
    base.EVAL_DATA = FULL_EVAL_DATA
    base.OUTPUT_DIR = FULL_OUTPUT_DIR
    base.CHECKPOINT_DIR = FULL_OUTPUT_DIR / "checkpoints"
    base.RAY_TEMP_DIR = FULL_RAY_TEMP_DIR
    base.DOCUMENT_ROOT = DATA_DIR
    base.DOCUMENT_PROBE = None


def _ensure_eval_data() -> None:
    """Create the rollout-format 200-QA manifest once without altering raw data."""
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
                raise RuntimeError(f"{RAW_TEST_DATA}:{line_number} has invalid DocVQA fields")
            document_path = (DATA_DIR / pdf_path).resolve()
            if DATA_DIR not in document_path.parents or not document_path.is_file():
                raise RuntimeError(f"{RAW_TEST_DATA}:{line_number} has invalid PDF path")
            rows.append(
                {
                    "prompt": f"Document path: {document_path}\nQuestion: {question}\n\nInspect the document with the available document tools. Return only the supported answer inside <final>...</final> when you are done.",
                    "label": json.dumps({"answers": answers, "metric": "anls"}, ensure_ascii=False),
                    "metadata": {"document_path": str(document_path), "metric": "anls", "sample_id": raw.get("sample_id"), "source_pdf_path": pdf_path, "target_page": raw.get("target_page"), "answer_page": raw.get("answer_page", raw.get("target_page")), "answer_bbox": raw.get("answer_bbox"), "num_pages": raw.get("num_pages"), "page_count": raw.get("page_count", raw.get("num_pages"))},
                }
            )
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 full-test rows, found {len(rows)}")
    with FULL_EVAL_DATA.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate() -> dict[str, Any]:
    report = two_sample._base_validate()
    if report["train_rows"] != 4 or report["eval_rows"] != 200:
        raise RuntimeError("full baseline requires four fixture train rows and 200 test evaluation rows")
    report.update(
        {
            "profile": "full_docvqa_baseline_07",
            "source_dataset": "nielsr/docvqa_1200_examples",
            "dynamic_tool_images": True,
            "agent_detail_log_dir": str(FULL_OUTPUT_DIR / "dump_details"),
            "tool_output_log_dir": str(FULL_OUTPUT_DIR / "tool_outputs"),
        }
    )
    return report


_ensure_eval_data()
_configure()
base.training_argv = two_sample._training_argv
base.validate = _validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(base.validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
