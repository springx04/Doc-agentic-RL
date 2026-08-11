"""Export one compact prediction record per benchmark question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_reward import extract_final_answer  # noqa: E402
from export_rollout_workflows import _find_samples  # noqa: E402
from eval_benchmarks.common import jsonl_write  # noqa: E402


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, dict) else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def _load_payload(path: Path) -> Any:
    if path.suffix.casefold() == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - rollout servers have torch
        raise RuntimeError("exporting .pt rollouts requires PyTorch") from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def export_predictions(eval_pt: Path, output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if not eval_pt.is_file():
        raise FileNotFoundError(f"rollout artifact not found: {eval_pt}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"prediction output already exists; pass --overwrite: {output}")
    samples = _find_samples(_load_payload(eval_pt))
    if not samples:
        raise RuntimeError(f"no rollout samples found in {eval_pt}")

    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for position, sample in enumerate(samples, 1):
        sample = _mapping(sample)
        metadata = _mapping(sample.get("metadata"))
        task_id = metadata.get("task_id")
        if not task_id:
            raise RuntimeError(f"rollout sample {position} missing metadata.task_id")
        task_id = str(task_id)
        if task_id in seen_task_ids:
            raise RuntimeError(
                f"duplicate prediction for {task_id}; benchmark eval requires "
                "n_samples_per_eval_prompt=1"
            )
        seen_task_ids.add(task_id)
        response = str(sample.get("response", ""))
        final_answer, protocol_valid = extract_final_answer(response, metadata)
        tool_execution = _mapping(metadata.get("tool_execution"))
        navigation = _mapping(metadata.get("navigation_state"))
        rollout_status = str(
            sample.get("rollout_status")
            or metadata.get("rollout_status")
            or ""
        )
        rows.append(
            {
                "task_id": task_id,
                "benchmark": metadata.get("benchmark"),
                "question_id": metadata.get("question_id"),
                "doc_id": metadata.get("doc_id"),
                "final_answer": final_answer,
                "protocol_valid": bool(protocol_valid),
                "rollout_status": rollout_status,
                "rollout_status_reason": metadata.get("rollout_status_reason"),
                "tool_call_count": int(
                    metadata.get("tool_call_count", tool_execution.get("call_count", 0)) or 0
                ),
                "valid_tool_call_count": int(
                    metadata.get("valid_tool_call_count", tool_execution.get("valid_call_count", 0)) or 0
                ),
                "tool_error_count": int(
                    metadata.get("tool_error_count", tool_execution.get("error_count", 0)) or 0
                ),
                "visited_pages": _json_value(
                    metadata.get("visited_pages", navigation.get("visited_pages", []))
                ),
                "rendered_pages": _json_value(
                    metadata.get("rendered_pages", navigation.get("rendered_pages", []))
                ),
                "ocr_pages": _json_value(
                    metadata.get("ocr_pages", navigation.get("ocr_pages", []))
                ),
                "supporting_pages": _json_value(
                    metadata.get("supporting_pages", navigation.get("supporting_pages", []))
                ),
                "duplicate_page_calls": int(metadata.get("duplicate_page_calls", 0) or 0),
                "no_information_gain_calls": int(metadata.get("no_information_gain_calls", 0) or 0),
                "tool_calls": _json_value(tool_execution.get("calls", [])),
            }
        )
    jsonl_write(output, rows)
    return {"output": str(output), "num_predictions": len(rows), "eval_pt": str(eval_pt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-pt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_predictions(args.eval_pt, args.output, overwrite=args.overwrite), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
