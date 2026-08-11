"""Whitelist-based static/compile check tool."""

from __future__ import annotations

from ._common import ToolExecutionContext, clean_failure, invalid_result, quote, repo_path, result_from_execution


CHECKS = {
    "compile": "python -m compileall -q {path}",
    "ruff": "ruff check {path}",
    "mypy": "mypy {path}",
    "pyright": "pyright {path}",
    "eslint": "eslint {path}",
    "tsc": "tsc --noEmit",
}


async def execute(arguments: dict, context: ToolExecutionContext):
    check = str(arguments.get("check", "compile")).lower()
    if check not in CHECKS:
        return invalid_result("run_checks", f"unsupported check: {check}; choose from {sorted(CHECKS)}")
    try:
        path = repo_path(context.cwd, arguments.get("path", "."))
        command = CHECKS[check].format(path=quote(path))
        raw = await context.client.exec(context.lease_id, command, cwd=context.cwd, timeout=int(arguments.get("timeout", 180)))
        return result_from_execution("run_checks", raw, limit=getattr(context.config, "max_output_chars", 16384), metadata={"command": command, "check": check, "path": path})
    except ValueError as exc:
        return invalid_result("run_checks", str(exc))
    except Exception as exc:
        return clean_failure("run_checks", exc)
