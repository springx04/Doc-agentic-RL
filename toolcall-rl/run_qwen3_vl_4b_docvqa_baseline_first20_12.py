"""Run the fixed first-20 DocVQA eval subset with the multi-turn hook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_qwen3_vl_4b_full_docvqa_baseline_08 as profile


FULL_EVAL_DATA = profile.profile.base.PROJECT / "data" / "full_docvqa_eval_20260723_07.jsonl"
SUBSET_EVAL_DATA = profile.profile.base.PROJECT / "data" / "full_docvqa_eval_20260723_07_first20.jsonl"
OUTPUT_DIR = (
    profile.profile.base.PROJECT
    / "outputs"
    / "qwen3-vl-4b-docvqa-baseline-20260730-08-multiturn20"
).resolve()
RAY_TEMP_DIR = (profile.profile.base.WORKSPACE / ".ray" / "r08").resolve()


def ensure_subset() -> None:
    if SUBSET_EVAL_DATA.exists():
        return
    rows = FULL_EVAL_DATA.read_text(encoding="utf-8").splitlines()
    if len(rows) < 20:
        raise RuntimeError(f"expected at least 20 eval rows, found {len(rows)}")
    with SUBSET_EVAL_DATA.open("x", encoding="utf-8") as handle:
        for row in rows[:20]:
            json.loads(row)
            handle.write(row + "\n")


def configure() -> None:
    ensure_subset()
    base = profile.profile.base
    base.TRAIN_DATA = profile.profile.REAL_TRAIN_DATA
    base.EVAL_DATA = SUBSET_EVAL_DATA
    base.OUTPUT_DIR = OUTPUT_DIR
    base.CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    base.RAY_TEMP_DIR = RAY_TEMP_DIR
    base.DOCUMENT_ROOT = profile.DATA_DIR
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
    if report["train_rows"] != 4 or report["eval_rows"] != 20:
        raise RuntimeError(f"expected 4 train rows and 20 eval rows, got {report}")
    report.update(
        {
            "profile": "full_docvqa_baseline_first20_multiturn",
            "eval_only": True,
            "dynamic_tool_images": True,
            "agent_detail_log_dir": str(OUTPUT_DIR / "dump_details"),
            "tool_output_log_dir": str(OUTPUT_DIR / "tool_outputs"),
        }
    )
    return report


configure()
profile.profile.base.training_argv = profile.profile._training_argv
profile.profile.base.validate = validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    profile.profile.base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
