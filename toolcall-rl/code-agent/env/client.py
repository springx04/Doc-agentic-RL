"""Code environment client with remote HTTP and offline local backends."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

try:
    import httpx
except ImportError:  # pragma: no cover - local smoke does not require HTTP
    httpx = None

try:
    from ..config import DEFAULT_CODE_CONFIG
    from ..schemas import CodeEvalResult, CodeLease, ExecutionResult
    from .executor import LocalCodeExecutor, ensure_within, validate_patch_paths
except ImportError:  # pragma: no cover
    from config import DEFAULT_CODE_CONFIG
    from schemas import CodeEvalResult, CodeLease, ExecutionResult
    from env.executor import LocalCodeExecutor, ensure_within, validate_patch_paths


class CodeEnvError(RuntimeError):
    def __init__(self, message: str, *, failure_origin: str = "real_infrastructure"):
        super().__init__(message)
        self.failure_origin = failure_origin


class CodeEnvClient:
    """Async HTTP client for the Code pool server.

    Methods return typed records so world corruption can be applied only after
    clean execution.  The HTTP payload remains compatible with the existing
    SWE pool's ``allocate/heartbeat/exec/diff/evaluate/close`` endpoints.
    """

    def __init__(self, base_url: str | None = None, *, request_timeout: float | None = None, http_client: Any = None):
        self.base_url = (base_url or os.getenv("CODE_ENV_SERVER_URL", DEFAULT_CODE_CONFIG.env_server_url)).rstrip("/")
        self.request_timeout = request_timeout if request_timeout is not None else DEFAULT_CODE_CONFIG.request_timeout
        self._client = http_client
        self._owns_client = http_client is None
        self._leases: dict[str, CodeLease] = {}

    async def __aenter__(self) -> "CodeEnvClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_client(self) -> Any:
        if self._client is None:
            if httpx is None:
                raise CodeEnvError("httpx is required for CodeEnvClient")
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.request_timeout), trust_env=False)
        return self._client

    async def _post(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        client = await self._ensure_client()
        try:
            response = await client.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            raise CodeEnvError(f"Code environment request failed at {path}: {exc}") from exc
        if not isinstance(result, dict):
            raise CodeEnvError(f"Code environment returned a non-object response at {path}")
        return result

    async def allocate(
        self,
        image: str,
        instance_id: str = "",
        *,
        cwd: str = "/testbed",
        base_revision: str | None = None,
    ) -> CodeLease:
        result = await self._post("/allocate", {"image": image, "instance_id": instance_id, "cwd": cwd})
        if not result.get("ok", False):
            raise CodeEnvError(f"allocate failed: {result}")
        lease_id = str(result.get("lease_id") or result.get("container_id") or "")
        if not lease_id:
            raise CodeEnvError("allocate response has no lease_id")
        lease = CodeLease(
            lease_id=lease_id,
            instance_id=instance_id,
            image_name=image,
            cwd=str(result.get("cwd") or cwd),
            base_revision=base_revision,
            backend="remote",
            state="active",
            created_at=time.time(),
            last_heartbeat=time.time(),
        )
        self._leases[lease_id] = lease
        return lease

    async def heartbeat(self, lease_id: str) -> dict[str, Any]:
        result = await self._post("/heartbeat", {"lease_id": lease_id})
        if not result.get("ok", False):
            raise CodeEnvError(f"heartbeat failed: {result}")
        return result

    async def exec(
        self,
        lease_id: str,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int = 180,
        env: Mapping[str, str] | None = None,
    ) -> ExecutionResult:
        result = await self._post(
            "/exec",
            {"lease_id": lease_id, "command": command, "cwd": cwd or self._lease_cwd(lease_id), "timeout": timeout, "env": dict(env or {})},
            timeout=float(timeout) + 30.0,
        )
        if not result.get("ok", False):
            raise CodeEnvError(f"exec transport failed: {result}")
        return ExecutionResult(
            ok=int(result.get("returncode", 1)) == 0,
            returncode=result.get("returncode"),
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            output=str(result.get("output") or result.get("stdout") or "") + str(result.get("stderr") or ""),
            duration_ms=float(result.get("duration_ms", result.get("duration", 0.0)) or 0.0),
            timed_out=bool(result.get("timed_out", False) or result.get("returncode") == -1),
            metadata={key: value for key, value in result.items() if key not in {"ok", "returncode", "stdout", "stderr", "output"}},
        )

    async def diff(self, lease_id: str, *, cwd: str | None = None) -> str:
        result = await self._post("/diff", {"lease_id": lease_id, "cwd": cwd or self._lease_cwd(lease_id)})
        if not result.get("ok", False):
            raise CodeEnvError(f"diff failed: {result}")
        return str(result.get("patch") or "")

    async def apply_patch(self, lease_id: str, patch: str, *, cwd: str | None = None) -> ExecutionResult:
        """Apply only the candidate patch; no verifier is run."""

        # Validation is syntactic/path-contained and does not require the
        # remote container path to exist on this client host.
        validate_patch_paths(patch, Path(cwd or self._lease_cwd(lease_id)))
        return await self._exec_patch(lease_id, patch, cwd=cwd, reset=False)

    async def reset_to_patch(self, lease_id: str, patch: str, *, cwd: str | None = None) -> ExecutionResult:
        """Restore a clean lease to a repository snapshot, then apply patch."""

        return await self._exec_patch(lease_id, patch, cwd=cwd, reset=True)

    async def _exec_patch(self, lease_id: str, patch: str, *, cwd: str | None, reset: bool) -> ExecutionResult:
        # A base64 payload avoids shell heredoc delimiter collisions and never
        # interpolates arbitrary patch text as shell syntax.
        import base64

        encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        command = "python -c \"import base64,pathlib; pathlib.Path('/tmp/code-agent.patch').write_bytes(base64.b64decode('" + encoded + "'))\""
        if reset:
            command += " && git reset --hard HEAD && git clean -fd"
        command += " && git apply --intent-to-add --whitespace=nowarn /tmp/code-agent.patch"
        return await self.exec(lease_id, command, cwd=cwd, timeout=120)

    async def evaluate(
        self,
        lease_id: str,
        patch: str,
        eval_script: str,
        *,
        cwd: str | None = None,
        timeout: int = 300,
        evaluator_patch: str = "",
    ) -> CodeEvalResult:
        result = await self._post(
            "/evaluate",
            {"lease_id": lease_id, "patch": patch, "eval_script": eval_script, "cwd": cwd or self._lease_cwd(lease_id), "timeout": timeout, "evaluator_patch": evaluator_patch},
            timeout=float(timeout) + 120.0,
        )
        if not result.get("ok", False):
            raise CodeEnvError(f"evaluate transport failed: {result}")
        return CodeEvalResult(
            ok=True,
            resolved=bool(result.get("resolved", False)),
            returncode=result.get("returncode"),
            output=str(result.get("output") or result.get("error") or ""),
            error=str(result.get("error")) if result.get("error") else None,
            lease_id=lease_id,
            metadata=result,
        )

    async def close(self, lease_id: str) -> None:
        result = await self._post("/close", {"lease_id": lease_id})
        if not result.get("ok", False):
            raise CodeEnvError(f"close failed: {result}")
        lease = self._leases.get(lease_id)
        if lease:
            self._leases[lease_id] = CodeLease(**{**lease.__dict__, "state": "closed"})

    def _lease_cwd(self, lease_id: str) -> str:
        lease = self._leases.get(lease_id)
        return lease.cwd if lease else "/testbed"

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


class LocalCodeEnvClient:
    """Disposable local leases used by unit tests and offline smoke tests."""

    def __init__(self, repository_root: str | Path):
        self.repository_root = Path(repository_root).resolve()
        if not self.repository_root.is_dir():
            raise ValueError(f"repository root does not exist: {self.repository_root}")
        self._roots: dict[str, Path] = {}
        self._executors: dict[str, LocalCodeExecutor] = {}
        self._leases: dict[str, CodeLease] = {}

    async def allocate(self, image: str = "local", instance_id: str = "", *, cwd: str = ".", base_revision: str | None = None) -> CodeLease:
        temp_root = Path(tempfile.mkdtemp(prefix="code-agent-lease-"))
        destination = temp_root / "repo"
        # Copy source files but never copy a worktree's .git pointer: doing so
        # would make commands in the disposable lease operate on the parent
        # worktree.  Every local lease gets its own fresh repository/index.
        shutil.copytree(self.repository_root, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__"))
        subprocess.run(["git", "init", "-q"], cwd=destination, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "code-agent@localhost"], cwd=destination, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Code Agent"], cwd=destination, capture_output=True, check=True)
        # The host may have a global excludes file (for example one that
        # ignores generated-looking test names).  A lease baseline must
        # contain every file copied from the task repository: otherwise a
        # later clean evaluation can delete an ignored-but-required test or
        # source file.  ``-f`` applies only to this disposable repository.
        subprocess.run(["git", "add", "-f", "-A"], cwd=destination, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-qm", "local lease base"], cwd=destination, capture_output=True, check=True)
        lease_id = uuid.uuid4().hex
        lease = CodeLease(lease_id, instance_id, image, str(destination), base_revision, "local", "active", time.time(), time.time())
        self._roots[lease_id] = temp_root
        self._executors[lease_id] = LocalCodeExecutor(destination)
        self._leases[lease_id] = lease
        return lease

    async def heartbeat(self, lease_id: str) -> dict[str, Any]:
        self._require(lease_id)
        return {"ok": True, "lease_id": lease_id}

    async def exec(self, lease_id: str, command: str, *, cwd: str | None = None, timeout: int = 180, env=None) -> ExecutionResult:
        executor = self._require(lease_id)
        return await executor.exec(command, cwd=cwd, timeout=timeout, env=env)

    async def diff(self, lease_id: str, *, cwd: str | None = None) -> str:
        executor = self._require(lease_id)
        return await executor.diff()

    async def apply_patch(self, lease_id: str, patch: str, *, cwd: str | None = None) -> ExecutionResult:
        executor = self._require(lease_id)
        return await executor.apply_patch(patch)

    async def reset_to_patch(self, lease_id: str, patch: str, *, cwd: str | None = None) -> ExecutionResult:
        executor = self._require(lease_id)
        return await executor.reset_to_patch(patch)

    async def evaluate(self, lease_id: str, patch: str, eval_script: str, *, cwd: str | None = None, timeout: int = 300, evaluator_patch: str = "") -> CodeEvalResult:
        executor = self._require(lease_id)
        reset = await executor.reset_to_patch("")
        if not reset.ok:
            return CodeEvalResult(False, False, reset.returncode, reset.output, "reset failed", lease_id, "real_infrastructure")
        if evaluator_patch:
            evaluator_applied = await executor.apply_patch(evaluator_patch)
            if not evaluator_applied.ok:
                return CodeEvalResult(False, False, evaluator_applied.returncode, evaluator_applied.output, "evaluator-private patch apply failed", lease_id, evaluator_applied.failure_origin)
        applied = await executor.apply_patch(patch)
        if not applied.ok:
            return CodeEvalResult(True, False, applied.returncode, applied.output, "patch apply failed", lease_id)
        result = await executor.exec(eval_script, timeout=timeout)
        return CodeEvalResult(True, result.ok, result.returncode, result.output, None if result.ok else result.stderr, lease_id)

    async def close(self, lease_id: str) -> None:
        self._require(lease_id)
        root = self._roots.pop(lease_id)
        self._executors.pop(lease_id, None)
        self._leases.pop(lease_id, None)
        await asyncio.to_thread(shutil.rmtree, root, True)

    def _require(self, lease_id: str):
        if lease_id not in self._executors:
            raise CodeEnvError(f"unknown or closed local lease: {lease_id}")
        return self._executors[lease_id]

    def root_for_lease(self, lease_id: str) -> Path:
        executor = self._require(lease_id)
        return executor.root


__all__ = ["CodeEnvClient", "CodeEnvError", "LocalCodeEnvClient"]
