"""Convert document-tool conversation data to slime SFT parquet.

Each input JSON/JSONL row must contain a ``messages`` list using standard
``system``/``user``/``assistant`` roles. Assistant tool calls should use the
same ``<tool_call>`` JSON or XML format as RL, and final answers should use
``<final>...</final>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


VALID_ROLES = {"system", "user", "assistant", "tool"}


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("data") or value.get("records") or value.get("examples")
    if not isinstance(value, list):
        raise ValueError("input must be a JSON object list or JSONL")
    return value


def convert(row: dict) -> dict:
    conversations = row.get("messages")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("each row must contain a non-empty messages list")
    messages = []
    for turn in conversations:
        role = str(turn.get("role", ""))
        if role not in VALID_ROLES:
            raise ValueError(f"unknown role: {role}")
        messages.append({"role": role, "content": str(turn.get("content", ""))})
    return {"messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converted = [convert(row) for row in load_rows(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(converted).to_parquet(str(args.output))
    print(f"Wrote {len(converted)} document-tool SFT examples to {args.output}")


if __name__ == "__main__":
    main()
