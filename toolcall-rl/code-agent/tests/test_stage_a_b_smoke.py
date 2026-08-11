import asyncio

from data.schema import SWEInstance
from training.common import ScriptedModelClient
from training.stage_a_belief import run_stage_a
from training.stage_b_policy import run_stage_b


def test_stage_a_and_b_execute_as_independent_code_stages(git_repo, tmp_path):
    async def run():
        instance = SWEInstance.from_raw(
            {"instance_id": "stages", "problem_statement": "Inspect VALUE in app.py", "image_name": "local"}
        )
        output = tmp_path / "stage-a.json"
        stage_a = await run_stage_a([instance], repository_root=git_repo, output=output, seed=5)
        assert stage_a["environment"] == "code"
        assert len(stage_a["instances"][0]["probe_rows"]) >= 20
        assert output.exists() and output.with_name("belief_code.pt").exists()

        stage_b = await run_stage_b(
            instance,
            repository_root=git_repo,
            model_client=ScriptedModelClient([
                '<tool_call>{"name":"list_tree","arguments":{"path":".","max_depth":1}}</tool_call>',
                "<final>stage B complete</final>",
            ]),
            seed=5,
        )
        assert stage_b["environment"] == "code"
        assert stage_b["trajectory"]["metadata"]["valid_for_rl"] is True

    asyncio.run(run())
