"""Code Agent remote Docker lease pool.

The pool deliberately speaks the same container-node HTTP protocol as the
existing SWE exec service, but owns its leases, logs, configuration namespace,
and listening port.  It never starts, stops, or modifies the document-agent
pool, so a Code smoke can run alongside Doc training when configured with a
separate ``CODE_ENV_SERVER_PORT`` and a bounded container limit.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import requests
    from flask import Flask, jsonify, request
except ImportError:  # pragma: no cover - deployment dependency gate
    requests = None
    Flask = Any  # type: ignore[misc,assignment]
    jsonify = request = None


logger = logging.getLogger("code.env_pool")


@dataclass
class ExecNode:
    url: str
    max_containers: int
    active_containers: int = 0
    healthy: bool = True
    last_health_check: float = 0.0


@dataclass
class Lease:
    lease_id: str
    node_url: str
    container_id: str
    image: str
    instance_id: str
    cwd: str
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


class CodeEnvPool:
    """Thread-safe, Code-owned proxy to pre-provisioned Docker exec nodes."""

    def __init__(
        self,
        exec_server_urls: list[str],
        *,
        max_containers_per_node: int = 1,
        post: Callable[..., Any] | None = None,
        get: Callable[..., Any] | None = None,
    ) -> None:
        if not exec_server_urls:
            raise ValueError("at least one CODE_EXEC_SERVER_URLS endpoint is required")
        if max_containers_per_node < 1:
            raise ValueError("max_containers_per_node must be positive")
        self.nodes = [ExecNode(url=value.rstrip("/"), max_containers=max_containers_per_node) for value in exec_server_urls]
        self._leases: dict[str, Lease] = {}
        self._lock = threading.RLock()
        if (post is None or get is None) and requests is None:
            raise RuntimeError("Code environment pool requires requests; install it on the pool host")
        self._post = post or requests.post
        self._get = get or requests.get

    def _post_json(self, url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        response = self._post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"non-object response from {url}")
        return data

    def _get_json(self, url: str, *, timeout: int) -> dict[str, Any]:
        response = self._get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"non-object response from {url}")
        return data

    def _reserve_node(self) -> ExecNode:
        with self._lock:
            candidates = [node for node in self.nodes if node.healthy and node.active_containers < node.max_containers]
            if not candidates:
                raise RuntimeError("all Code exec nodes are unavailable or at capacity")
            node = min(candidates, key=lambda value: value.active_containers)
            node.active_containers += 1
            return node

    def _release_node(self, node_url: str) -> None:
        with self._lock:
            for node in self.nodes:
                if node.url == node_url:
                    node.active_containers = max(0, node.active_containers - 1)
                    return

    def _lease(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
        if lease is None:
            raise KeyError(f"unknown Code lease_id: {lease_id}")
        return lease

    def allocate(self, *, image: str, instance_id: str, cwd: str) -> dict[str, Any]:
        node = self._reserve_node()
        try:
            result = self._post_json(f"{node.url}/container/create", {"image": image, "cwd": cwd}, timeout=120)
            if not result.get("ok") or not result.get("container_id"):
                raise RuntimeError(f"container create failed: {result}")
        except Exception:
            self._release_node(node.url)
            raise
        lease_id = f"code-lease-{uuid.uuid4().hex[:16]}"
        lease = Lease(lease_id, node.url, str(result["container_id"]), image, instance_id, cwd)
        with self._lock:
            self._leases[lease_id] = lease
        return {"lease_id": lease.lease_id, "container_id": lease.container_id, "node_url": lease.node_url, "cwd": lease.cwd}

    def heartbeat(self, lease_id: str) -> None:
        self._lease(lease_id).last_heartbeat = time.time()

    def exec(self, *, lease_id: str, command: str, cwd: str, timeout: int, env: dict[str, str]) -> dict[str, Any]:
        lease = self._lease(lease_id)
        lease.last_heartbeat = time.time()
        return self._post_json(
            f"{lease.node_url}/container/exec",
            {"container_id": lease.container_id, "command": command, "cwd": cwd, "timeout": timeout, "env": env},
            timeout=timeout + 30,
        )

    def diff(self, *, lease_id: str, cwd: str) -> dict[str, Any]:
        lease = self._lease(lease_id)
        return self._post_json(f"{lease.node_url}/container/diff", {"container_id": lease.container_id, "cwd": cwd}, timeout=60)

    def evaluate(self, *, lease_id: str, patch: str, eval_script: str, cwd: str, timeout: int) -> dict[str, Any]:
        lease = self._lease(lease_id)
        return self._post_json(
            f"{lease.node_url}/container/evaluate",
            {"container_id": lease.container_id, "patch": patch, "eval_script": eval_script, "cwd": cwd, "timeout": timeout},
            timeout=timeout + 60,
        )

    def close(self, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.pop(lease_id, None)
        if lease is None:
            return
        try:
            self._post_json(f"{lease.node_url}/container/destroy", {"container_id": lease.container_id}, timeout=30)
        finally:
            self._release_node(lease.node_url)

    def health_check(self) -> None:
        for node in self.nodes:
            try:
                node.healthy = bool(self._get_json(f"{node.url}/healthz", timeout=5).get("ok"))
            except Exception:
                node.healthy = False
            node.last_health_check = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_leases": len(self._leases),
                "nodes": [
                    {"url": node.url, "active_containers": node.active_containers, "max_containers": node.max_containers, "healthy": node.healthy}
                    for node in self.nodes
                ],
            }


def create_app(pool: CodeEnvPool) -> Flask:
    if jsonify is None or request is None:
        raise RuntimeError("Code environment pool requires flask; install it on the pool host")
    app = Flask("code_env_pool")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "environment": "code"})

    @app.get("/status")
    def status():
        return jsonify({"ok": True, "pool": pool.status()})

    @app.post("/allocate")
    def allocate():
        data = request.get_json(force=True) or {}
        image = str(data.get("image") or "")
        if not image:
            return jsonify({"ok": False, "error": "image is required"}), 400
        try:
            return jsonify({"ok": True, **pool.allocate(image=image, instance_id=str(data.get("instance_id") or ""), cwd=str(data.get("cwd") or "/testbed"))})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/heartbeat")
    def heartbeat():
        data = request.get_json(force=True) or {}
        try:
            pool.heartbeat(str(data.get("lease_id") or ""))
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    def proxy(operation: str):
        data = request.get_json(force=True) or {}
        lease_id = str(data.get("lease_id") or "")
        if not lease_id:
            return jsonify({"ok": False, "error": "lease_id is required"}), 400
        try:
            if operation == "exec":
                result = pool.exec(lease_id=lease_id, command=str(data.get("command") or ""), cwd=str(data.get("cwd") or "/testbed"), timeout=int(data.get("timeout", 180)), env=dict(data.get("env") or {}))
            elif operation == "diff":
                result = pool.diff(lease_id=lease_id, cwd=str(data.get("cwd") or "/testbed"))
            elif operation == "evaluate":
                result = pool.evaluate(lease_id=lease_id, patch=str(data.get("patch") or ""), eval_script=str(data.get("eval_script") or ""), cwd=str(data.get("cwd") or "/testbed"), timeout=int(data.get("timeout", 300)))
            else:
                pool.close(lease_id)
                result = {"ok": True}
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    app.add_url_rule("/exec", "exec", lambda: proxy("exec"), methods=["POST"])
    app.add_url_rule("/diff", "diff", lambda: proxy("diff"), methods=["POST"])
    app.add_url_rule("/evaluate", "evaluate", lambda: proxy("evaluate"), methods=["POST"])
    app.add_url_rule("/close", "close", lambda: proxy("close"), methods=["POST"])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Code Agent environment pool server")
    parser.add_argument("--host", default=os.getenv("CODE_ENV_SERVER_BIND_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CODE_ENV_SERVER_PORT", "18091")))
    parser.add_argument("--exec-server-urls", default=os.getenv("CODE_EXEC_SERVER_URLS", ""))
    parser.add_argument("--max-containers-per-node", type=int, default=int(os.getenv("CODE_MAX_CONTAINERS_PER_NODE", "1")))
    args = parser.parse_args()
    urls = [value.strip() for value in args.exec_server_urls.split(",") if value.strip()]
    pool = CodeEnvPool(urls, max_containers_per_node=args.max_containers_per_node)
    pool.health_check()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")
    logger.info("Code pool serving %d exec nodes on %s:%s", len(pool.nodes), args.host, args.port)
    create_app(pool).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":  # pragma: no cover
    main()
