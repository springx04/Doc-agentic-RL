from bayestool.belief import CodeBeliefFilter
from bayestool.belief_runtime import build_policy_visible_belief
from bayestool.task_state import CodeTaskStateView


def test_policy_prompt_returns_the_bounded_state_it_measures():
    state = CodeTaskStateView("instance", "fix the bug")
    state.inspected_files = {f"file_{index}.py" for index in range(500)}
    prompt = build_policy_visible_belief(state, CodeBeliefFilter()).to_prompt(max_chars=7_000)
    assert "file_0.py" not in prompt
    assert "file_499.py" in prompt
    assert len(prompt) < 7_500
