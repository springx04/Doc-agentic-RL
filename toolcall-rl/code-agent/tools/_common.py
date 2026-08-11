"""Shared safe helpers for the eight logical Code tools."""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Any

try:
    from ..config import CODE_MAX_OUTPUT_CHARS, DEFAULT_CODE_CONFIG
    from ..schemas import CodeToolResult, ExecutionResult
except ImportError:  # pragma: no cover
    from config import CODE_MAX_OUTPUT_CHARS, DEFAULT_CODE_CONFIG
    from schemas import CodeToolResult, ExecutionResult


@dataclass
class ToolExecutionContext:
    client: Any
    lease_id: str
    cwd: str
    config: Any = DEFAULT_CODE_CONFIG
    task_state: Any = None


def quote(value: str) -> str:
    return shlex.quote(str(value))


def repo_path(root: str, requested: str) -> str:
    """Resolve a POSIX container path while rejecting traversal and .git."""

    raw = str(requested or ".")
    candidate = posixpath.normpath(raw if raw.startswith("/") else posixpath.join(root, raw))
    root_norm = posixpath.normpath(root)
    if candidate != root_norm and not candidate.startswith(root_norm.rstrip("/") + "/"):
        raise ValueError(f"path escapes repository root: {requested}")
    relative = posixpath.relpath(candidate, root_norm)
    if ".git" in set(relative.split("/")):
        raise ValueError(".git paths are not allowed")
    return candidate


def truncate_output(text: str, limit: int = CODE_MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    text = str(text or "")
    if limit <= 0 or len(text) <= limit:
        return text, False
    # Keep the head and tail so source context and the final test summary both
    # remain visible.  The partial flag is part of the policy observation.
    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head)
    return text[:head] + f"\n...[partial output; {len(text) - limit} chars omitted]...\n" + text[-tail:], True


def result_from_execution(tool_name: str, result: ExecutionResult, *, limit: int | None = None, metadata=None) -> CodeToolResult:
    output, partial = truncate_output(result.output, limit or CODE_MAX_OUTPUT_CHARS)
    status = "ok" if result.ok else ("timeout" if result.timed_out else "error")
    return CodeToolResult(
        tool_name=tool_name,
        status=status,
        output=output,
        latency_ms=result.duration_ms,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        metadata={**(result.metadata or {}), **(metadata or {}), "partial": partial},
        failure_origin=result.failure_origin,
        partial=partial,
    )


def invalid_result(tool_name: str, message: str) -> CodeToolResult:
    return CodeToolResult(tool_name=tool_name, status="invalid", output=str(message), failure_origin="model_action")


def clean_failure(tool_name: str, exc: BaseException) -> CodeToolResult:
    return CodeToolResult(tool_name=tool_name, status="error", output=str(exc), failure_origin="real_infrastructure")


def validate_safe_args_text(value: str) -> bool:
    return not bool(re.search(r"[;&|`$<>\n\r]", str(value)))
