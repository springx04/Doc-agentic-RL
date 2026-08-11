import asyncio

from bayestool.branching import close_sibling_leases, create_sibling_leases
from bayestool.world_sampler import public_sampling_context, sample_required_worlds
from config import DEFAULT_CODE_CONFIG
from env.client import LocalCodeEnvClient
from rollout import CodeRollout
from training.common import ScriptedModelClient


class RecordingFinalModel:
    def __init__(self):
        self.calls = []

    async def generate(self, messages, *, max_tokens, **kwargs):
        self.calls.append([dict(message) for message in messages])
        return "<final>continued from checkpoint</final>"


def test_sibling_continuations_restore_full_decision_state(git_repo):
    async def run():
        sample = {
            "text": "Inspect the repository",
            "metadata": {"public_instance": {"instance_id": "branch", "problem_statement": "Inspect the repository", "image_name": "local", "task_kind": "bugfix"}},
        }
        client = LocalCodeEnvClient(git_repo)
        world = sample_required_worlds(
            instance_id="branch",
            image_name="local",
            base_revision=None,
            context=public_sampling_context(tool_budget=DEFAULT_CODE_CONFIG.tool_budget, repo_file_count=2, tracked_extensions=(".py",)),
            rollout_seed=11,
        )[0]
        parent = await client.allocate("local", "branch", cwd="/testbed")
        checkpoints = []
        parent_model = ScriptedModelClient([
            '<tool_call>{"name":"list_tree","arguments":{"path":".","max_depth":1}}</tool_call>',
            "<final>parent done</final>",
        ])
        await CodeRollout().run(
            sample,
            model_client=parent_model,
            code_env_client=client,
            code_config=DEFAULT_CODE_CONFIG,
            world=world,
            interaction_lease=parent,
            on_decision_checkpoint=checkpoints.append,
        )
        checkpoint = checkpoints[1]
        assert checkpoint.call_index == 1
        assert checkpoint.remaining_tool_budget == DEFAULT_CODE_CONFIG.tool_budget - 1
        assert checkpoint.messages[-1]["role"] == "user"

        siblings = await create_sibling_leases(client, checkpoint, image_name="local", instance_id="branch", count=4, cwd="/testbed")
        try:
            roots = {str(client.root_for_lease(sibling.lease.lease_id)) for sibling in siblings}
            assert len(roots) == 4
            for sibling in siblings:
                model = RecordingFinalModel()
                result = await CodeRollout().run(
                    sample,
                    model_client=model,
                    code_env_client=client,
                    code_config=DEFAULT_CODE_CONFIG,
                    world=world,
                    interaction_lease=sibling.lease,
                    branch_checkpoint=sibling.checkpoint,
                )
                assert result.metadata["termination_reason"] == "final"
                assert result.metadata["tool_calls_used"] == 1
                assert model.calls[0] == checkpoint.messages
        finally:
            await close_sibling_leases(client, siblings)
            await client.close(parent.lease_id)

    asyncio.run(run())
