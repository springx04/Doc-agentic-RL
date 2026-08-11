"""Preprocess local SWE-Bench/SWE-Gym JSON or JSONL into Code manifests.

The CLI is intentionally offline.  Dataset downloads belong to an explicit
data-preparation step and are not performed by Code rollout or tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .schema import SWEInstance
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from data.schema import SWEInstance


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, Mapping):
        return [dict(value)]
    raise ValueError("input must contain a JSON object/list or JSONL objects")


def preprocess(rows: Iterable[Mapping[str, Any]], *, data_source: str = "") -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        instance = SWEInstance.from_raw(raw, data_source=data_source)
        output.append(instance.to_manifest_row())
    return output


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline Code SWE JSONL manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-source", default="")
    args = parser.parse_args(argv)
    write_jsonl(preprocess(read_rows(args.input), data_source=args.data_source), args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "preprocess", "read_rows", "write_jsonl"]
