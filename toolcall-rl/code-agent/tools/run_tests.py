"""Controlled test runner for Python SWE repositories."""

from __future__ import annotations

import re
import shlex

from ._common import ToolExecutionContext, clean_failure, invalid_result, quote, repo_path, result_from_execution, validate_safe_args_text


_ALLOWED_ARG = re.compile(r"^(?:-q|-x|--maxfail(?:=\d+)?|--tb=(?:short|long|no|native)|-k|[A-Za-z0-9_./:*=-]+)$")


def _test_command(target: str, args: str, root: str) -> str:
    tokens = shlex.split(args) if args.strip() else []
    if any(not _ALLOWED_ARG.fullmatch(token) for token in tokens):
        raise ValueError("run_tests args contain unsupported shell syntax")
    if any(token in {";", "&&", "||", "|", ">", "<"} for token in tokens):
        raise ValueError("run_tests args contain shell operators")
    target_part = ""
    if target:
        target_part = quote(repo_path(root, target))
    return "python -m pytest" + (f" {target_part}" if target_part else "") + (" " + " ".join(quote(token) for token in tokens) if tokens else "")


async def execute(arguments: dict, context: ToolExecutionContext):
    try:
        target = str(arguments.get("target", ""))
        args = str(arguments.get("args", ""))
        command = _test_command(target, args, context.cwd)
        raw = await context.client.exec(context.lease_id, command, cwd=context.cwd, timeout=int(arguments.get("timeout", 180)))
        return result_from_execution("run_tests", raw, limit=getattr(context.config, "max_output_chars", 16384), metadata={"command": command, "target": target})
    except ValueError as exc:
        return invalid_result("run_tests", str(exc))
    except Exception as exc:
        return clean_failure("run_tests", exc)
