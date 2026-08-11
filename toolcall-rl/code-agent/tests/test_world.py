import asyncio

from bayestool.schema import CodeWorldSpec, ToolQualitySpec
from bayestool.task_state import CodeTaskStateView
from bayestool.world import CodeWorldRuntime
from env.client import LocalCodeEnvClient
from tools import DEFAULT_REGISTRY


def test_world_injected_failure_does_not_kill_lease(git_repo):
    async def run():
        qualities = {name: ToolQualitySpec() for name in ("list_tree", "search_code", "read_file", "apply_patch", "git_diff", "run_tests", "run_checks", "run_command")}
        qualities["run_tests"] = ToolQualitySpec(availability=0.0)
        world = CodeWorldSpec("i", "local", None, "coupling", "latent", "healthy", "healthy", 1, "healthy", qualities, {"inspection_core": ToolQualitySpec(), "mutation_core": ToolQualitySpec(), "validation_core": ToolQualitySpec(), "execution_core": ToolQualitySpec()})
        client = LocalCodeEnvClient(git_repo)
        lease = await client.allocate("local", "i")
        runtime = CodeWorldRuntime(world=world, client=client, lease_id=lease.lease_id, cwd=lease.cwd, task_state=CodeTaskStateView("i", "problem"), registry=DEFAULT_REGISTRY)
        failed, _, _ = await runtime.execute_tool("run_tests", {"target": "test_app.py", "args": ""})
        assert failed.failure_origin == "world_injected"
        assert failed.valid_for_rl is True
        read, _, _ = await runtime.execute_tool("read_file", {"path": "app.py", "start_line": 1, "end_line": 1})
        assert read.failure_origin == "none"
        assert (await client.heartbeat(lease.lease_id))["ok"]
        await client.close(lease.lease_id)

    asyncio.run(run())
