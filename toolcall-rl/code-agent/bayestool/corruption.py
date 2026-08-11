"""Policy-observation corruption operators for Code tools."""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any

try:
    from ..config import CODE_TOOL_FAMILIES
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from config import CODE_TOOL_FAMILIES
    from schemas import CodeToolResult

from .schema import CodeWorldSpec, quality_for_tool


CORRUPTIONS: dict[str, tuple[str, ...]] = {
    "list_tree": ("entry_drop", "listing_truncation", "duplicate_entries"),
    "search_code": ("match_drop", "false_negative", "result_truncation", "stale_search"),
    "read_file": ("line_drop", "truncation", "stale_read"),
    "apply_patch": ("injected_reject", "injected_noop", "ack_loss", "delayed_acknowledgement"),
    "git_diff": ("hunk_drop", "diff_truncation", "stale_diff"),
    "run_tests": ("output_drop", "summary_truncation", "injected_timeout", "status_miscalibration"),
    "run_checks": ("diagnostic_drop", "diagnostic_truncation", "injected_timeout"),
    "run_command": ("stdout_truncation", "injected_timeout", "returncode_miscalibration"),
}


def corruption_for_call(world: CodeWorldSpec, tool_name: str, call_index: int, *, public_context: dict[str, Any] | None = None, rng: random.Random | None = None) -> str | None:
    quality = quality_for_tool(world, tool_name, public_context, call_index)
    generator = rng or random.Random(world.seed + call_index * 7919)
    # Structure and semantic accuracy are independent hidden quality axes;
    # either can produce a corrupted policy observation.
    probability = max(0.0, min(0.9, max(1.0 - quality.structure_fidelity, 1.0 - quality.semantic_accuracy)))
    if tool_name == "apply_patch":
        probability = max(0.0, min(0.75, 1.0 - quality.availability))
    if generator.random() >= probability:
        return None
    options = CORRUPTIONS[tool_name]
    return options[generator.randrange(len(options))]


def injected_unavailable(world: CodeWorldSpec, tool_name: str, call_index: int, public_context: dict[str, Any] | None = None) -> bool:
    quality = quality_for_tool(world, tool_name, public_context, call_index)
    rng = random.Random(world.seed + call_index * 104729 + sum(ord(c) for c in tool_name))
    return rng.random() > quality.availability


def corrupt_result(result: CodeToolResult, corruption_type: str | None, *, seed: int = 0) -> CodeToolResult:
    if not corruption_type or not result.observation_delivered:
        return result
    rng = random.Random(seed)
    output = result.output
    status = result.status
    returncode = result.returncode
    metadata = dict(result.metadata)
    partial = result.partial
    if corruption_type in {"entry_drop", "match_drop", "line_drop", "hunk_drop", "diagnostic_drop", "output_drop"}:
        lines = output.splitlines()
        keep = [line for index, line in enumerate(lines) if index % 3 != 0]
        output = "\n".join(keep)
        partial = True
    elif corruption_type in {"listing_truncation", "result_truncation", "truncation", "diff_truncation", "diagnostic_truncation", "summary_truncation", "stdout_truncation"}:
        output = output[: max(1, len(output) // 2)] + "\n...[world partial]"
        partial = True
    elif corruption_type == "duplicate_entries":
        lines = output.splitlines()
        output = "\n".join(lines + lines[: min(3, len(lines))])
    elif corruption_type in {"false_negative", "stale_search", "stale_read", "stale_diff"}:
        lines = output.splitlines()
        output = "\n".join(lines[1:]) if len(lines) > 1 else ""
        partial = True
    elif corruption_type == "injected_reject":
        status, output, returncode = "error", "world-injected patch rejection", 1
    elif corruption_type == "injected_noop":
        status, output, returncode = "ok", "world-injected acknowledgement; no repository mutation", 0
    elif corruption_type in {"injected_timeout", "delayed_acknowledgement"}:
        status, output, returncode = "timeout", "world-injected timeout/unknown acknowledgement", -1
    elif corruption_type == "ack_loss":
        status, output, returncode = "timeout", "world-injected unknown acknowledgement; inspect git_diff", -1
        metadata["state_may_have_changed"] = True
    elif corruption_type in {"status_miscalibration", "returncode_miscalibration"}:
        if status == "ok":
            status = "error"
        elif status == "error":
            status = "ok"
        metadata["status_miscalibrated"] = True
    metadata["corruption_type"] = corruption_type
    return replace(
        result,
        status=status,
        output=output,
        returncode=returncode,
        metadata=metadata,
        failure_origin="world_injected",
        corruption_type=corruption_type,
        partial=partial,
    )


def world_unavailable_result(tool_name: str, *, reason: str = "world-injected tool unavailable") -> CodeToolResult:
    return CodeToolResult(
        tool_name=tool_name,
        status="timeout",
        output=reason,
        returncode=-1,
        metadata={"world_injected": True},
        failure_origin="world_injected",
        corruption_type="availability",
    )


__all__ = ["CORRUPTIONS", "corruption_for_call", "corrupt_result", "injected_unavailable", "world_unavailable_result"]
