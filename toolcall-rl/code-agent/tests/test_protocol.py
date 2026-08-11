from protocol import parse_action, render_tool_call


def test_strict_tool_call_and_terminal_actions():
    parsed = parse_action(render_tool_call("read_file", {"path": "src/a.py", "start_line": 1, "end_line": 2}))
    assert parsed.kind == "tool_call"
    assert parsed.tool_name == "read_file"
    assert parsed.arguments == {"path": "src/a.py", "start_line": 1, "end_line": 2}
    assert parse_action("<final>done</final>").is_terminal
    assert parse_action("<abstain>not enough evidence</abstain>").is_terminal


def test_protocol_rejects_extra_text_unknown_keys_and_multiple_actions():
    assert parse_action("explanation <final>done</final>").kind == "invalid"
    assert parse_action('<tool_call>{"name":"read_file","arguments":{},"extra":1}</tool_call>').kind == "invalid"
    assert parse_action("<final>x</final><abstain>y</abstain>").kind == "invalid"
