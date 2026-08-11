"""Strict fallback runtime command tool."""

from __future__ import annotations

from ._common import ToolExecutionContext
try:
    from ..env.command_policy import CodeCommandPolicy
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from env.command_policy import CodeCommandPolicy


async def execute(arguments: dict, context: ToolExecutionContext):
    command = str(arguments.get("command", ""))
    timeout = int(arguments.get("timeout", 180))
    return await CodeCommandPolicy.execute_guarded(
        command,
        diff=lambda: context.client.diff(context.lease_id, cwd=context.cwd),
        execute=lambda value: context.client.exec(context.lease_id, value, cwd=context.cwd, timeout=timeout),
        reset_to_patch=lambda patch: context.client.reset_to_patch(context.lease_id, patch, cwd=context.cwd),
    )
