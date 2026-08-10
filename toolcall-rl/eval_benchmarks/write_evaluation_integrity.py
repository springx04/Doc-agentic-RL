"""Write the reproducibility manifest required before formal benchmark results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks import BENCHMARKS  # noqa: E402
from eval_benchmarks.common import TOOL_BUDGET, bench_root_from, json_write, load_jsonl  # noqa: E402


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_integrity(
    bench_root: Path,
    *,
    model: str,
    checkpoint: str,
    repo_root: Path,
    git_commit: str | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    bench_root = bench_root_from(bench_root)
    counts: dict[str, int] = {}
    for benchmark in BENCHMARKS:
        path = bench_root / "eval" / f"{benchmark}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"prepared eval manifest not found: {path}")
        counts[benchmark] = len(load_jsonl(path))
    manifest = {
        "model": model,
        "model_parameter_class": "<=8B",
        "checkpoint": checkpoint,
        "git_commit": git_commit or _git_commit(repo_root),
        "tool_budget": dict(TOOL_BUDGET),
        "samples_per_question": 1,
        "benchmarks": {
            "docvqa2026": {"split": "val", "num_questions": counts["docvqa2026"]},
            "longdocurl": {"split": "public", "num_questions": counts["longdocurl"]},
            "mpdocvqa": {"split": "val", "num_questions": counts["mpdocvqa"]},
            "dude": {"split": "val", "num_questions": counts["dude"]},
        },
    }
    destination = output or bench_root / "results" / "evaluation_integrity.json"
    json_write(destination, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = write_integrity(
        bench_root_from(args.bench_root),
        model=args.model,
        checkpoint=args.checkpoint,
        repo_root=args.repo_root,
        git_commit=args.git_commit,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
