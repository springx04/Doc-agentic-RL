"""Docker-node HTTP server for isolated Code leases.

This server is deployment code only; local tests use ``LocalCodeEnvClient`` and
never start Docker or contact a remote node.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

try:
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover
    Flask = None


def _docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)


_PATCH_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?([^\t\n]+)", re.M)


def _patch_valid(patch: str) -> bool:
    """Reject malformed and path-escaping patches before ``git apply``."""

    if not isinstance(patch, str) or not patch.strip() or "diff --git " not in patch or "+++" not in patch:
        return False
    for raw_path in _PATCH_PATH_RE.findall(patch):
        path = raw_path.strip()
        if path == "/dev/null":
            continue
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts or ".git" in candidate.parts:
            return False
    return True


def _kill(container_id: str) -> None:
    try:
        _docker("exec", container_id, "kill", "-9", "-1", timeout=10)
    except Exception:
        pass


def create_app() -> "Flask":
    if Flask is None:
        raise RuntimeError("flask is required to run the Code exec server")
    app = Flask(__name__)
    active: dict[str, dict] = {}

    @app.get("/healthz")
    def healthz():
        result = _docker("info", "--format", "{{.ContainersRunning}}", timeout=10)
        return jsonify({"ok": result.returncode == 0, "running_containers": result.stdout.strip() if result.returncode == 0 else "?"})

    @app.get("/images")
    def images():
        result = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500
        values = [line for line in result.stdout.splitlines() if line.strip()]
        return jsonify({"ok": True, "images": values, "count": len(values)})

    @app.post("/container/create")
    def create():
        data = request.get_json(force=True) or {}
        image = str(data.get("image") or "")
        cwd = str(data.get("cwd") or "/testbed")
        if not image:
            return jsonify({"ok": False, "error": "image is required"}), 400
        name = f"code-{uuid.uuid4().hex[:12]}"
        result = _docker("run", "-d", "--init", "--name", name, "--pull", "never", "-w", cwd, image, "sleep", "infinity", timeout=120)
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500
        container_id = result.stdout.strip()
        active[container_id] = {"image": image, "cwd": cwd, "created_at": time.time()}
        return jsonify({"ok": True, "container_id": container_id, "lease_id": container_id, "cwd": cwd})

    @app.post("/container/heartbeat")
    def heartbeat():
        data = request.get_json(force=True) or {}
        lease_id = str(data.get("lease_id") or data.get("container_id") or "")
        return jsonify({"ok": bool(lease_id in active), "lease_id": lease_id})

    @app.post("/container/exec")
    def execute():
        data = request.get_json(force=True) or {}
        container_id = str(data.get("container_id") or data.get("lease_id") or "")
        command = str(data.get("command") or "")
        cwd = str(data.get("cwd") or "/testbed")
        timeout = int(data.get("timeout", 180))
        if not container_id or not command:
            return jsonify({"ok": False, "error": "container_id and command are required"}), 400
        env_args: list[str] = []
        for key, value in (data.get("env") or {}).items():
            env_args.extend(["-e", f"{key}={value}"])
        try:
            result = _docker("exec", "-w", cwd, *env_args, container_id, "bash", "-lc", command, timeout=timeout)
            return jsonify({"ok": True, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "output": result.stdout + result.stderr, "duration_ms": 0.0})
        except subprocess.TimeoutExpired:
            _kill(container_id)
            return jsonify({"ok": True, "returncode": -1, "output": f"Command timed out after {timeout}s", "timed_out": True})

    @app.post("/container/diff")
    def diff():
        data = request.get_json(force=True) or {}
        container_id = str(data.get("container_id") or data.get("lease_id") or "")
        cwd = str(data.get("cwd") or "/testbed")
        result = _docker("exec", "-w", cwd, container_id, "bash", "-lc", "git diff --binary HEAD", timeout=60)
        return jsonify({"ok": result.returncode == 0, "patch": result.stdout, "returncode": result.returncode, "error": result.stderr})

    @app.post("/container/apply_patch")
    @app.post("/container/reset_to_patch")
    def apply_patch():
        data = request.get_json(force=True) or {}
        container_id = str(data.get("container_id") or data.get("lease_id") or "")
        patch = str(data.get("patch") or "")
        cwd = str(data.get("cwd") or "/testbed")
        reset = request.path.endswith("reset_to_patch")
        if patch and not _patch_valid(patch):
            return jsonify({"ok": False, "error": "invalid unified patch"}), 400
        encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        command = f"python -c \"import base64,pathlib;pathlib.Path('/tmp/code-agent.patch').write_bytes(base64.b64decode('{encoded}'))\""
        if reset:
            command += " && git reset --hard HEAD && git clean -fd"
        if patch:
            command += " && git apply --intent-to-add --whitespace=nowarn /tmp/code-agent.patch"
        result = _docker("exec", "-w", cwd, container_id, "bash", "-lc", command, timeout=120)
        return jsonify({"ok": result.returncode == 0, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "output": result.stdout + result.stderr})

    @app.post("/container/evaluate")
    def evaluate():
        data = request.get_json(force=True) or {}
        container_id = str(data.get("container_id") or data.get("lease_id") or "")
        patch = str(data.get("patch") or "")
        script = str(data.get("eval_script") or "")
        cwd = str(data.get("cwd") or "/testbed")
        timeout = int(data.get("timeout", 300))
        if not _patch_valid(patch):
            return jsonify({"ok": True, "resolved": False, "error": "invalid patch"})
        encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        apply_command = f"git reset --hard HEAD && git clean -fd && python -c \"import base64,pathlib;pathlib.Path('/tmp/code-agent.patch').write_bytes(base64.b64decode('{encoded}'))\" && git apply --intent-to-add --whitespace=nowarn /tmp/code-agent.patch"
        applied = _docker("exec", "-w", cwd, container_id, "bash", "-lc", apply_command, timeout=120)
        if applied.returncode != 0:
            return jsonify({"ok": True, "resolved": False, "apply_returncode": applied.returncode, "error": applied.stderr})
        try:
            result = _docker("exec", "-w", cwd, container_id, "bash", "-lc", script, timeout=timeout)
            return jsonify({"ok": True, "resolved": result.returncode == 0, "returncode": result.returncode, "output": result.stdout + result.stderr})
        except subprocess.TimeoutExpired:
            _kill(container_id)
            return jsonify({"ok": True, "resolved": False, "returncode": -1, "error": "evaluation timeout"})

    @app.post("/container/destroy")
    def destroy():
        data = request.get_json(force=True) or {}
        container_id = str(data.get("container_id") or data.get("lease_id") or "")
        result = _docker("rm", "-f", container_id, timeout=30)
        active.pop(container_id, None)
        return jsonify({"ok": result.returncode == 0, "returncode": result.returncode, "error": result.stderr})

    return app


app = create_app() if Flask is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    if app is None:
        raise RuntimeError("flask is required")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
