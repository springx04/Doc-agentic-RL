from __future__ import annotations

import pytest

pytest.importorskip("flask")

from env.server.pool_server import CodeEnvPool, create_app


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_code_pool_preserves_lease_cwd_and_uses_independent_prefix():
    calls = []

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/container/create"):
            return _Response({"ok": True, "container_id": "container-1"})
        return _Response({"ok": True, "returncode": 0, "stdout": ""})

    pool = CodeEnvPool(["http://exec-node:5000"], post=post, get=lambda *args, **kwargs: _Response({"ok": True}))
    app = create_app(pool)
    client = app.test_client()

    allocated = client.post("/allocate", json={"image": "swebench/demo", "instance_id": "demo", "cwd": "/workspace"}).get_json()
    assert allocated["ok"] is True
    assert allocated["lease_id"].startswith("code-lease-")
    assert allocated["cwd"] == "/workspace"
    assert calls[0][1] == {"image": "swebench/demo", "cwd": "/workspace"}

    executed = client.post("/exec", json={"lease_id": allocated["lease_id"], "command": "pwd", "cwd": "/workspace", "timeout": 12}).get_json()
    assert executed["ok"] is True
    assert calls[1][1]["container_id"] == "container-1"
    assert calls[1][1]["cwd"] == "/workspace"

    assert client.post("/close", json={"lease_id": allocated["lease_id"]}).get_json()["ok"] is True
    assert pool.status()["total_leases"] == 0
