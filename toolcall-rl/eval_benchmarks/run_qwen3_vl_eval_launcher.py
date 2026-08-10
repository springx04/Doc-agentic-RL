"""Bridge the validated Qwen3-VL launcher into strict eval-only mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_eval import make_eval_only_argv
import run_qwen3_vl_4b_real_docvqa_test_05 as configured


_CONFIGURED_TRAINING_ARGV = configured._training_argv


def _training_argv() -> list[str]:
    """Reuse the real launcher and force the project's no-update contract."""

    return make_eval_only_argv(_CONFIGURED_TRAINING_ARGV())


configured.base.training_argv = _training_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(configured.base.validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    configured.base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
