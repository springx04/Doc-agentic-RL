"""Working-tree patch inspection tool."""

from __future__ import annotations

import time

from ._common import ToolExecutionContext, clean_failure, truncate_output
try:
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from schemas import CodeToolResult


async def execute(arguments: dict, context: ToolExecutionContext):
    started = time.perf_counter()
    try:
        patch = await context.client.diff(context.lease_id, cwd=context.cwd)
        output, partial = truncate_output(patch, getattr(context.config, "max_output_chars", 16384))
        return CodeToolResult(
            tool_name="git_diff",
            status="ok" if output else "empty",
            output=output,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"patch_nonempty": bool(patch), "patch_size": len(patch), "partial": partial},
            partial=partial,
        )
    except Exception as exc:
        return clean_failure("git_diff", exc)
