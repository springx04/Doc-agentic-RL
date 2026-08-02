"""Run a fresh full DocVQA baseline after the multi-turn document-tool fix."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_qwen3_vl_4b_full_docvqa_baseline_08 as profile


OUTPUT_DIR = (
    profile.profile.base.PROJECT
    / "outputs"
    / "qwen3-vl-4b-docvqa-baseline-20260730-04-multiturn200"
).resolve()
RAY_TEMP_DIR = (profile.profile.base.WORKSPACE / ".ray" / "r04").resolve()

profile.OUTPUT_DIR = OUTPUT_DIR
profile.RAY_TEMP_DIR = RAY_TEMP_DIR
profile.profile.base.OUTPUT_DIR = OUTPUT_DIR
profile.profile.base.CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
profile.profile.base.RAY_TEMP_DIR = RAY_TEMP_DIR
profile.profile.base.DOCUMENT_ROOT = profile.DATA_DIR
profile.profile.base.DOCUMENT_PROBE = None
profile.profile.base.validate = profile.validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        import json

        print(json.dumps(profile.profile.base.validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    profile.profile.base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
