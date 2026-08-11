import asyncio

from data.schema import SWEInstance
from training.common import ScriptedModelClient
from training.stage_d_meta import run_stage_d


def test_stage_d_carries_only_belief_across_fresh_tasks(git_repo):
    async def run():
        instances = [
            SWEInstance.from_raw({"instance_id": "one", "problem_statement": "inspect one", "image_name": "local"}),
            SWEInstance.from_raw({"instance_id": "two", "problem_statement": "inspect two", "image_name": "local"}),
        ]
        result = await run_stage_d(
            instances,
            repository_root=git_repo,
            model_factory=lambda: ScriptedModelClient([
                '<tool_call>{"name":"list_tree","arguments":{"path":".","max_depth":1}}</tool_call>',
                "<final>done</final>",
            ]),
        )
        assert result["session"]["task_count"] == 2
        assert result["session"]["belief"]["state"]["step"] >= 2
        assert [task["metadata"]["instance_id"] for task in result["tasks"]] == ["one", "two"]
        assert (git_repo / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    asyncio.run(run())
