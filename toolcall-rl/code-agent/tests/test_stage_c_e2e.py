import asyncio

from data.schema import SWEInstance
from training.common import ScriptedModelClient
from training.stage_c_bayes_arpo import run_stage_c_smoke


def test_stage_c_runs_one_task_four_worlds_and_k4_siblings(git_repo):
    async def run():
        instance = SWEInstance.from_raw(
            {"instance_id": "stage-c", "problem_statement": "Inspect the repository", "image_name": "local"}
        )
        result = await run_stage_c_smoke(
            instance,
            repository_root=git_repo,
            group_size=4,
            model_factory=lambda: ScriptedModelClient([
                '<tool_call>{"name":"list_tree","arguments":{"path":".","max_depth":1}}</tool_call>',
                '<tool_call>{"name":"git_diff","arguments":{}}</tool_call>',
                "<final>inspection complete</final>",
            ]),
            seed=17,
        )
        worlds = result["worlds"]
        assert len(worlds) == 4
        assert {row["world"]["world_slot_role"] for row in worlds} == {"healthy", "local_degradation", "shared_family_fault", "change"}
        assert len({row["world"]["coupling_id"] for row in worlds}) == 1
        assert len({row["world"]["latent_world_id"] for row in worlds}) == 4
        assert len({row["parent_repo_state_digest"] for row in worlds}) == 1
        for row in worlds:
            assert row["group_validity"]["valid_for_rl"] is True
            assert len(row["siblings"]) == 4
            assert len({sibling["lease_id"] for sibling in row["siblings"]}) == 4
            assert {sibling["repo_state_digest"] for sibling in row["siblings"]} == {row["parent_repo_state_digest"]}
            assert all(sibling["valid_for_rl"] for sibling in row["siblings"])

    asyncio.run(run())
