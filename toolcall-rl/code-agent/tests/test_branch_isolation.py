import asyncio

from bayestool.branching import close_sibling_leases, create_sibling_leases
from env.checkpoint import CodeBranchCheckpoint, CodeRepoCheckpoint
from env.client import LocalCodeEnvClient


def test_siblings_use_independent_filesystems(git_repo, patch_value):
    async def run():
        client = LocalCodeEnvClient(git_repo)
        parent = await client.allocate("local", "i")
        patch = await client.diff(parent.lease_id, cwd=parent.cwd)
        checkpoint = CodeBranchCheckpoint(CodeRepoCheckpoint.create(instance_id="i", image_name="local", base_revision=None, cwd=parent.cwd, patch=patch), remaining_tool_budget=24, decision_prefix_hash="prefix")
        await client.close(parent.lease_id)
        siblings = await create_sibling_leases(client, checkpoint, image_name="local", instance_id="i", count=4)
        assert len({item.lease.lease_id for item in siblings}) == 4
        applied = await client.apply_patch(siblings[0].lease.lease_id, patch_value(), cwd=siblings[0].lease.cwd)
        assert applied.ok
        assert await client.diff(siblings[1].lease.lease_id, cwd=siblings[1].lease.cwd) == ""
        await close_sibling_leases(client, siblings)

    asyncio.run(run())
