"""Evaluate two repaired real document-QA PDFs without RL updates."""

from __future__ import annotations

import argparse
import json
from typing import Any

from baseline_eval import make_eval_only_argv
import run_qwen3_vl_4b_smoke as base


DATA_DIR = (base.PROJECT / "data").resolve()
SUBSET_DIR = DATA_DIR / "real_docvqa_rl_test_20260717-01"
REAL_TRAIN_DATA = SUBSET_DIR / "train.jsonl"
REAL_EVAL_DATA = SUBSET_DIR / "eval.jsonl"
REAL_OUTPUT_DIR = (base.PROJECT / "outputs" / "qwen3-vl-4b-docvqa-baseline-20260723-06").resolve()
REAL_RAY_TEMP_DIR = (base.WORKSPACE / ".ray" / "b06").resolve()
_BASE_TRAINING_ARGV = base.training_argv


def _replace_value(argv: list[str], option: str, value: str) -> None:
    index = argv.index(option)
    argv[index + 1] = value


def _configure() -> None:
    base.TRAIN_DATA = REAL_TRAIN_DATA
    base.EVAL_DATA = REAL_EVAL_DATA
    base.OUTPUT_DIR = REAL_OUTPUT_DIR
    base.CHECKPOINT_DIR = REAL_OUTPUT_DIR / "checkpoints"
    base.RAY_TEMP_DIR = REAL_RAY_TEMP_DIR
    base.DOCUMENT_ROOT = DATA_DIR
    base.DOCUMENT_PROBE = None


def _training_argv() -> list[str]:
    argv = _BASE_TRAINING_ARGV()
    argv.remove("--disable-rewards-normalization")
    _replace_value(argv, "--rollout-batch-size", "2")
    _replace_value(argv, "--n-samples-per-prompt", "4")
    _replace_value(argv, "--global-batch-size", "8")
    argv.append("--gradient-checkpointing")
    argv = make_eval_only_argv(argv)
    eval_index = argv.index("--eval-prompt-data")
    argv[eval_index + 1] = "real_docvqa_test_with_dynamic_tool_images"
    return argv


def _validate() -> dict[str, Any]:
    required_model_paths = (
        base.DOCLING_ARTIFACTS_DIR / "docling-project--docling-layout-heron",
        base.DOCLING_ARTIFACTS_DIR / "docling-project--docling-models",
        base.TOOL_MODELS_DIR / "deplot" / "config.json",
        base.TOOL_MODELS_DIR / "deplot" / "model.safetensors",
    )
    missing = [str(path) for path in required_model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("uploaded tool model bundle is incomplete: " + json.dumps(missing))
    report = _base_validate()
    if report["train_rows"] != 4 or report["eval_rows"] != 2:
        raise RuntimeError("real test requires exactly four train rows and two evaluation rows")
    report.update(
        {
            "profile": "real_docvqa_repaired_pdf_baseline_06",
            "source_dataset": "nielsr/docvqa_1200_examples",
            "selection_manifest": str(SUBSET_DIR / "selection_manifest.json"),
            "dynamic_tool_images": True,
            "agent_detail_log_dir": str(REAL_OUTPUT_DIR / "dump_details"),
            "tool_output_log_dir": str(REAL_OUTPUT_DIR / "tool_outputs"),
        }
    )
    return report


_configure()
_base_validate = base.validate
base.training_argv = _training_argv
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
