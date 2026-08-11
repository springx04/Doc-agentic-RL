import asyncio

from env.client import LocalCodeEnvClient
from tools import DEFAULT_REGISTRY, ToolExecutionContext


def test_logical_tools_and_path_containment(git_repo):
    async def run():
        client = LocalCodeEnvClient(git_repo)
        lease = await client.allocate("local", "test")
        context = ToolExecutionContext(client, lease.lease_id, lease.cwd)
        tree = await DEFAULT_REGISTRY.execute("list_tree", {"path": ".", "max_depth": 2}, context)
        assert tree.status == "ok"
        assert "app.py" in tree.output
        read = await DEFAULT_REGISTRY.execute("read_file", {"path": "app.py", "start_line": 1, "end_line": 1}, context)
        assert read.status == "ok" and "VALUE = 1" in read.output
        search = await DEFAULT_REGISTRY.execute("search_code", {"query": "VALUE", "path": ".", "glob": "*.py"}, context)
        assert search.status == "ok" and "app.py" in search.output
        bad = await DEFAULT_REGISTRY.execute("read_file", {"path": "../outside.py", "start_line": 1, "end_line": 1}, context)
        assert bad.status == "invalid" and bad.failure_origin == "model_action"
        await client.close(lease.lease_id)

    asyncio.run(run())
