from types import SimpleNamespace

from bayestool.belief_features import CODE_BELIEF_FEATURE_NAMES, CODE_BELIEF_FEATURE_SCHEMA_HASH, extract_code_belief_features
from bayestool.task_state import CodeTaskStateView
from schemas import CodeToolResult


def test_code_belief_schema_is_exactly_96d():
    state = CodeTaskStateView("i", "fix bug")
    result = CodeToolResult("read_file", "ok", "VALUE = 1\n", metadata={"path": "app.py"})
    features = extract_code_belief_features(tool_name="read_file", result=result, task_state=state)
    assert len(CODE_BELIEF_FEATURE_NAMES) == 96
    assert len(features) == 96
    assert len(CODE_BELIEF_FEATURE_SCHEMA_HASH) == 64
