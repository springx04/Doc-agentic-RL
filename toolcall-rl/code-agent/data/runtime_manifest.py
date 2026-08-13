"""Create policy-safe manifests from SWE manifests with private evaluators.

The generic Slime dataset loader copies a row's ``metadata`` into ``Sample``.
Consequently the original SWE manifest, which retains evaluator-private test
patches for the clean evaluator, must never be passed directly to Slime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocess_swe import read_rows
from .schema import SWEInstance


def materialize_public_manifest(input_path: str | Path, output_path: str | Path) -> int:
    """Write a manifest suitable for policy/rollout loading, with no secrets."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [SWEInstance.from_raw(row).to_runtime_row() for row in read_rows(input_path)]
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a public-only Code SWE runtime manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"rows": materialize_public_manifest(args.input, args.output), "output": args.output}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["materialize_public_manifest"]
