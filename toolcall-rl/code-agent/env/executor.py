"""Subprocess and patch execution primitives used by Code tools.

The executor is deliberately boring: it executes only inside the lease root,
returns transport failures separately, and never injects world corruption.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping

try:
    from ..schemas import ExecutionResult
except ImportError:  # pragma: no cover
    from schemas import ExecutionResult


_PATCH_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?([^\t\n]+)", re.M)


def ensure_within(root: str | Path, path: str | Path) -> Path:
    root_path = Path(root).resolve()
    candidate = (root_path / Path(path)).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {path}") from exc
    if ".git" in candidate.relative_to(root_path).parts:
        raise ValueError(".git paths are not allowed")
    return candidate


def validate_patch_paths(patch: str, root: str | Path) -> None:
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("patch must be a non-empty string")
    if "diff --git " not in patch:
        raise ValueError("patch must contain git diff headers")
    for raw in _PATCH_PATH_RE.findall(patch):
        path = raw.strip()
        if path == "/dev/null":
            continue
        ensure_within(root, path)


def _run_sync(
    command: str,
    *,
    cwd: str | Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> ExecutionResult:
    started = time.perf_counter()
    try:
        if os.name == "nt":
            # The development/test host is Windows.  ``shell=True`` uses
            # cmd.exe, which supports the git/python commands used by the
            # local backend and keeps the remote Linux command contract
            # untouched.
            runner = command
            use_shell = True
        else:
            runner = ["bash", "-lc", command]
            use_shell = False
        completed = subprocess.run(
            runner,
            cwd=str(cwd),
            env={**os.environ, **(dict(env) if env else {})},
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout)),
            check=False,
            shell=use_shell,
        )
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        return ExecutionResult(
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            output=stdout + stderr,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return ExecutionResult(
            ok=False,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            output=stdout + stderr + f"\nCommand timed out after {timeout}s",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            timed_out=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ExecutionResult(
            ok=False,
            returncode=None,
            output=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            transport_error=str(exc),
        )


def _run_argv_sync(
    argv: list[str],
    *,
    cwd: str | Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> ExecutionResult:
    """Run a fixed executable argv without passing through a host shell."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env={**os.environ, **(dict(env) if env else {})},
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout)),
            check=False,
            shell=False,
        )
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        return ExecutionResult(
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            output=stdout + stderr,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return ExecutionResult(
            ok=False,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            output=stdout + stderr + f"\nCommand timed out after {timeout}s",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            timed_out=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ExecutionResult(
            ok=False,
            returncode=None,
            output=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            transport_error=str(exc),
        )


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class LocalCodeExecutor:
    """A local executor for tests/offline smoke runs; never contacts a server."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"repository root does not exist: {self.root}")

    async def exec(self, command: str, *, cwd: str | Path | None = None, timeout: float = 180, env=None) -> ExecutionResult:
        directory = self.root if cwd is None else ensure_within(self.root, cwd)
        return await asyncio.to_thread(_run_sync, command, cwd=directory, timeout=timeout, env=env)

    async def diff(self) -> str:
        # A patch snapshot is a protocol object, not arbitrary shell output.
        # In particular, avoid host shell startup hooks (e.g. Conda) that can
        # write warnings to stdout and turn an empty diff into bogus patch
        # text, which corrupts branch checkpoints.
        result = await asyncio.to_thread(
            _run_argv_sync,
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
            cwd=self.root,
            timeout=60,
        )
        if result.transport_error:
            raise RuntimeError(result.transport_error)
        if not result.ok:
            raise RuntimeError(result.output or "git diff failed")
        return result.stdout

    async def apply_patch(self, patch: str) -> ExecutionResult:
        validate_patch_paths(patch, self.root)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as handle:
            handle.write(patch)
            patch_path = Path(handle.name)
        try:
            # The absolute temporary path is outside the repo but is read by
            # git; the patch itself was validated to stay inside the repo.
            # ``--intent-to-add`` records a newly-created file in the index
            # without staging its content.  ``git diff HEAD`` then includes
            # it in the candidate patch, while reset_to_patch still restores
            # a clean checkout before evaluating a sibling/final result.
            return await self.exec(f"git apply --intent-to-add --whitespace=nowarn {shell_quote(str(patch_path))}", timeout=60)
        finally:
            patch_path.unlink(missing_ok=True)

    async def reset_to_patch(self, patch: str) -> ExecutionResult:
        validate_patch_paths(patch, self.root) if patch.strip() else None
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as handle:
            handle.write(patch)
            patch_path = Path(handle.name)
        try:
            command = "git reset --hard HEAD && git clean -fd"
            if patch.strip():
                command += f" && git apply --intent-to-add --whitespace=nowarn {shell_quote(str(patch_path))}"
            return await self.exec(command, timeout=120)
        finally:
            patch_path.unlink(missing_ok=True)


def shell_quote(value: str) -> str:
    """POSIX quote without relying on a shell helper package."""

    if os.name == "nt":
        return '"' + value.replace('"', '\\"') + '"'
    return "'" + value.replace("'", "'\\''") + "'"


__all__ = ["LocalCodeExecutor", "ensure_within", "shell_quote", "validate_patch_paths"]
