import asyncio

from env.client import LocalCodeEnvClient
from env.command_policy import CodeCommandPolicy, CommandClass
from tools import DEFAULT_REGISTRY, ToolExecutionContext


def test_policy_routes_equivalent_commands():
    assert CodeCommandPolicy.classify("rg VALUE app.py") == CommandClass.SEARCH
    assert not CodeCommandPolicy.validate("pytest -q").allowed
    assert not CodeCommandPolicy.validate("git diff").allowed
    assert not CodeCommandPolicy.validate("pip install x").allowed
    assert CodeCommandPolicy.validate("python -c \"print(1)\"").allowed


def test_tracked_state_guard_restores_mutation(git_repo):
    async def run():
        client = LocalCodeEnvClient(git_repo)
        lease = await client.allocate("local", "test")
        context = ToolExecutionContext(client, lease.lease_id, lease.cwd)
        result = await DEFAULT_REGISTRY.execute("run_command", {"command": "python -c \"from pathlib import Path; Path('app.py').write_text('bad')\""}, context)
        assert result.failure_origin == "model_action"
        assert result.metadata["restored"] is True
        assert await client.diff(lease.lease_id, cwd=lease.cwd) == ""
        await client.close(lease.lease_id)

    asyncio.run(run())
