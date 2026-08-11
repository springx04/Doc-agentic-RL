from bayestool.task_state import CodeTaskStateView


def test_task_local_state_resets_without_replacing_session_object():
    state = CodeTaskStateView("a", "task a")
    state.inspected_files.add("a.py")
    state.touched_files.add("a.py")
    state.patch_nonempty = True
    state.validation_since_last_edit = True
    state.reset_for_new_task(instance_id="b", problem_statement="task b", tool_budget=24)
    assert state.instance_id == "b"
    assert not state.inspected_files and not state.touched_files
    assert not state.patch_nonempty and not state.validation_since_last_edit
    assert state.phase == "LOCALIZE"
