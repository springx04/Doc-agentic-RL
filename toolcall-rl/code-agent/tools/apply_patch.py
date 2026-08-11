"""Patch application tool; it never runs a verifier implicitly."""

from __future__ import annotations

from ._common import ToolExecutionContext, clean_failure, invalid_result, result_from_execution


async def execute(arguments: dict, context: ToolExecutionContext):
    patch = str(arguments.get("patch", ""))
    if "diff --git " not in patch:
        return invalid_result("apply_patch", "patch must contain unified git diff headers")
    try:
        raw = await context.client.apply_patch(context.lease_id, patch, cwd=context.cwd)
        result = result_from_execution("apply_patch", raw, limit=getattr(context.config, "max_output_chars", 16384))
        return result
    except ValueError as exc:
        return invalid_result("apply_patch", str(exc))
    except Exception as exc:
        return clean_failure("apply_patch", exc)
