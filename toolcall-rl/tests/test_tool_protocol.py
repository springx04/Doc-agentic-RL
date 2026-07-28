from pathlib import Path
import sys


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from tool_protocol import parse_assistant_action


def test_final_requires_a_whole_assistant_turn():
    assert parse_assistant_action("<final>answer</final>").kind == "final"
    assert parse_assistant_action("Use <final> and </final> to return the answer.").kind == "protocol_error"
    assert parse_assistant_action("explanation <final>answer</final>").kind == "protocol_error"


def test_multiple_or_unclosed_actions_are_protocol_errors():
    result = parse_assistant_action(
        '<tool_call>{"name":"render_page","arguments":{}}</tool_call>'
        '<tool_call>{"name":"render_page","arguments":{}}</tool_call>'
    )
    assert result.kind == "protocol_error"
    assert result.candidate_action_count == 2
    assert parse_assistant_action("<final>answer").kind == "protocol_error"
    assert parse_assistant_action("<tool_call>{\"name\": \"render_page\"}").kind == "protocol_error"


def test_json_and_xml_tool_calls_parse_without_substring_search():
    json_result = parse_assistant_action(
        '<tool_call>\n{"name":"render_page","arguments":{"page_number":1}}\n</tool_call>'
    )
    assert json_result.kind == "tool_call"
    assert json_result.value["name"] == "render_page"
    assert json_result.value["arguments"]["page_number"] == 1

    xml_result = parse_assistant_action(
        "<tool_call><function=render_page>"
        "<parameter=page_number>1</parameter>"
        "</function></tool_call>"
    )
    assert xml_result.kind == "tool_call"
    assert xml_result.value["arguments"]["page_number"] == 1
