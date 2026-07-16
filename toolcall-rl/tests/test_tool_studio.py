import json
import sys
from pathlib import Path


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
STUDIO_DIR = TOOLCALL_DIR / "tool_studio"
sys.path.insert(0, str(STUDIO_DIR))

import app  # noqa: E402


def test_agent_loop_executes_native_document_tool_call(monkeypatch):
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tool-1",
                                    "function": {
                                        "name": "render_page",
                                        "arguments": json.dumps({"document_path": "missing.pdf", "page_number": 1}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {"choices": [{"message": {"content": "<final>unavailable</final>"}, "finish_reason": "stop"}]},
        ]
    )
    monkeypatch.setattr(app, "openai_chat_completion", lambda config, messages: next(responses))

    events = [
        json.loads(line)
        for line in app.run_agent(
            {"prompt": "Read missing.pdf", "config": {"model": "mock", "base_url": "http://mock"}}
        )
    ]

    assert any(item["type"] == "tool_call" and item["name"] == "render_page" for item in events)
    assert any(item["type"] == "tool_started" and item["name"] == "render_page" for item in events)
    assert any(item["type"] == "tool_result" and item["status"] == "error" for item in events)
    assert events[-1] == {"type": "completed", "answer": "<final>unavailable</final>", "turns": 2}


def test_toolrl_reference_answer_separates_figure_and_table_evidence():
    answer = app._toolrl_reference_answer(
        "Task Goal: Irrelevant Tool Detection. The tool get_date is not suitable. Figure 1 caption.",
        "User: I would like to buy a movie ticket. We need the movie name and the specific date.",
    )

    assert "SFT model over-interprets" in answer
    assert "not inside Figure 1" in answer
    assert "missing movie name" in answer
