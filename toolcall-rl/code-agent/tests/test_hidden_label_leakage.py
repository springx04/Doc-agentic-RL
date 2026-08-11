from bayestool.belief import CodeBeliefFilter
from bayestool.belief_runtime import assert_no_hidden_labels, build_policy_visible_belief
from bayestool.task_state import CodeTaskStateView
from data.schema import SWEInstance


def test_public_instance_and_belief_do_not_expose_evaluator_labels():
    instance = SWEInstance.from_raw({"instance_id": "i", "problem_statement": "fix", "image_name": "local", "patch": "gold", "FAIL_TO_PASS": ["x"], "resolved": True})
    row = instance.to_runtime_row()
    assert "evaluator_private" not in row["metadata"]
    assert_no_hidden_labels(row)
    prompt = build_policy_visible_belief(CodeTaskStateView("i", "fix"), CodeBeliefFilter()).to_prompt()
    assert "latent_world_id" not in prompt and "gold_patch" not in prompt
