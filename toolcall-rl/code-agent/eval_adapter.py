"""Final checkpoint evaluation logging for the isolated Code environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def log_eval_rollout_data(rollout_id: int, args: Any, data: Mapping[str, Any], extra_metrics: Mapping[str, Any] | None = None) -> bool:
    """Write a public, per-instance 50-task audit without evaluator secrets."""

    output_root = Path(os.environ["CODE_OUTPUT_DIR"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset_name, value in data.items():
        samples = value.get("samples") if isinstance(value, Mapping) else None
        if not isinstance(samples, list):
            continue
        for sample in samples:
            metadata = getattr(sample, "metadata", {})
            metadata = metadata if isinstance(metadata, Mapping) else {}
            trajectory = metadata.get("code_trajectory")
            trajectory = trajectory if isinstance(trajectory, Mapping) else {}
            public = metadata.get("public_instance")
            public = public if isinstance(public, Mapping) else {}
            rows.append(
                {
                    "environment": "code",
                    "dataset": str(dataset_name),
                    "instance_id": str(public.get("instance_id") or ""),
                    "reward": float(getattr(sample, "reward", 0.0) or 0.0),
                    "resolved": bool(trajectory.get("resolved", False)),
                    "valid_for_rl": bool(trajectory.get("valid_for_rl", False)),
                    "termination_reason": str(trajectory.get("termination_reason") or ""),
                    "tool_calls_used": int(trajectory.get("tool_calls_used", 0) or 0),
                }
            )
    if len(rows) != 50:
        raise RuntimeError(f"Code final evaluation requires exactly 50 rows, got {len(rows)}")
    blocked = ("patch", "test", "fail_to_pass", "pass_to_pass", "evaluator", "gold")
    if any(any(token in json.dumps(row, sort_keys=True).lower() for token in blocked) for row in rows):
        raise RuntimeError("refusing to export evaluator-private evaluation content")
    path = output_root / "eval" / f"final_checkpoint_eval_{rollout_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return False


__all__ = ["log_eval_rollout_data"]
