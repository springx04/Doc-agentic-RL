import asyncio

from data.schema import PublicInstance, SWEInstance
from env.client import LocalCodeEnvClient
from rollout import generate_code_trajectory
from training.common import ScriptedModelClient


def test_single_task_multiturn_patch_validate_and_clean_evaluate(git_repo, patch_value):
    async def run():
        patch = patch_value()
        sample = {"text": "Make VALUE pass the test", "metadata": {"public_instance": {"instance_id": "i", "problem_statement": "Make VALUE pass the test", "image_name": "local", "task_kind": "bugfix"}}}
        model = ScriptedModelClient([
            '<tool_call>{"name":"list_tree","arguments":{"path":".","max_depth":1}}</tool_call>',
            '<tool_call>{"name":"read_file","arguments":{"path":"app.py","start_line":1,"end_line":1}}</tool_call>',
            f'<tool_call>{{"name":"apply_patch","arguments":{{"patch":{patch!r}}}}}</tool_call>'.replace("'", '"'),
            '<tool_call>{"name":"run_tests","arguments":{"target":"test_app.py","args":"-q"}}</tool_call>',
            '<final>VALUE updated</final>',
        ])
        # Build the patch action with JSON so embedded newlines are escaped
        # correctly instead of relying on ad-hoc model quoting.
        import json
        model.responses[2] = '<tool_call>' + json.dumps({"name": "apply_patch", "arguments": {"patch": patch}}, separators=(",", ":")) + '</tool_call>'
        result = await generate_code_trajectory(sample, model, LocalCodeEnvClient(git_repo), eval_script="python -m pytest -q")
        assert result["metadata"]["termination_reason"] == "final"
        assert result["metadata"]["valid_for_rl"] is True
        assert result["metadata"]["resolved"] is True
        assert result["metadata"]["tool_calls_used"] == 4
        assert result["trainer_only_metadata"]["world_slot_role"] == "healthy"

    asyncio.run(run())
