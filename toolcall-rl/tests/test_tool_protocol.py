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


def test_observation_metadata_cannot_prefix_a_valid_action():
    result = parse_assistant_action(
        '<task_state>{"remaining_tool_budget": 3}</task_state>\n'
        '<tool_call>{"name":"render_page","arguments":{}}</tool_call>'
    )
    assert result.kind == "protocol_error"
    assert result.candidate_action_count == 1


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


def test_function_style_tool_calls_use_only_structured_arguments_and_aliases():
    predictions = (
        '<tool_call>parse_document({"path":"/data/report.pdf","page":0})</tool_call>',
        '<tool_call>parse_document(name="/data/report.pdf", page=0)</tool_call>',
        '<tool_call>parse_document{"path":"/data/report.pdf","page":0}</tool_call>',
    )
    for prediction in predictions:
        result = parse_assistant_action(prediction)
        assert result.kind == "tool_call"
        assert result.value == {
            "name": "parse_document",
            "arguments": {"document_path": "/data/report.pdf", "page_numbers": [0]},
        }
    json_literal = parse_assistant_action(
        '<tool_call>parse_document({"path":"/data/report.pdf","use_docling":false,"value":null})</tool_call>'
    )
    assert json_literal.kind == "tool_call"
    assert json_literal.value["arguments"]["use_docling"] is False
    assert json_literal.value["arguments"]["value"] is None


def test_function_style_tool_calls_reject_code_and_ambiguous_aliases():
    assert parse_assistant_action(
        '<tool_call>parse_document(path=unknown_value)</tool_call>'
    ).kind == "protocol_error"
    duplicate = parse_assistant_action(
        '<tool_call>parse_document(document_path="/data/a.pdf", path="/data/b.pdf")</tool_call>'
    )
    assert duplicate.kind == "protocol_error"
    assert "both document_path" in (duplicate.reason or "")
    assert parse_assistant_action(
        '<tool_call>parse_document(path="/data/a.pdf") extra</tool_call>'
    ).kind == "protocol_error"


def test_abstention_is_a_strict_terminal_action():
    assert parse_assistant_action("<abstain>evidence is inconsistent</abstain>").kind == "abstain"
    assert parse_assistant_action("<abstain></abstain>").kind == "protocol_error"
    assert parse_assistant_action("prefix <abstain>reason</abstain>").kind == "protocol_error"
