import sys
from pathlib import Path


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from export_rollout_workflows import _group_reward_ordering_metrics, build_report


def test_report_statistics_exclude_infrastructure_rollouts():
    report = build_report(
        {
            "samples": [
                {
                    "index": 1,
                    "prompt": "Question 1",
                    "response": "<final>ok</final>",
                    "label": "ok",
                    "reward": {"score": 1.0, "quality": 1.0, "valid_for_rl": True, "rollout_status": "completed"},
                    "metadata": {
                        "valid_for_rl": True,
                        "rollout_status": "completed",
                        "generation_call_count": 2,
                        "generation_turn_count": 2,
                        "visited_pages": [1, 2],
                        "answer_page_visited": True,
                        "final_supported_by_evidence": True,
                        "multi_turn_rollout": True,
                        "multi_tool_rollout": True,
                        "completed_multi_tool_rollout": True,
                        "tool_execution": {
                            "call_count": 2,
                            "unique_tools": ["parse_document", "extract_table"],
                            "calls": [
                                {"tool": "parse_document", "executed": True},
                                {"tool": "extract_table", "executed": True},
                            ],
                        },
                    },
                },
                {
                    "index": 2,
                    "prompt": "Question 2",
                    "response": "",
                    "label": "ok",
                    "reward": {"score": 0.0, "quality": 0.0, "valid_for_rl": False, "rollout_status": "generation_empty"},
                    "metadata": {
                        "valid_for_rl": False,
                        "exclude_from_group_statistics": True,
                        "rollout_status": "generation_empty",
                        "generation_called": True,
                        "generation_steps": [{"generation_called": True, "finish_reason": "stop", "raw_generation_text": ""}],
                    },
                },
            ]
        },
        limit=2,
    )
    summary = report["summary_for_exported_subset"]
    assert summary["valid_for_rl_count"] == 1
    assert summary["mean_score_over_valid_rollouts"] == 1.0
    assert summary["mean_score"] == 1.0
    assert summary["completion_rate"] == 0.5
    assert summary["generation_error_count"] == 0
    assert summary["infra_error_count"] == 1
    assert summary["multi_step_rollout_count"] == 1
    assert summary["multi_tool_rollout_count"] == 1
    assert summary["completed_multi_tool_rollout_count"] == 1
    assert report["samples"][0]["diagnostics"]["visited_pages"] == [1, 2]
    assert report["samples"][0]["diagnostics"]["final_supported_by_evidence"] is True
    assert report["samples"][0]["diagnostics"]["multi_tool_rollout"] is True
    assert report["samples"][0]["diagnostics"]["tool_call_count"] == 2
    assert report["samples"][0]["diagnostics"]["valid_tool_call_count"] == 2


def test_group_ordering_reports_unavailable_when_group_ids_are_missing():
    item = {
        "group_index": None,
        "reward": {
            "answer_correctness": 1.0,
            "evidence_localization_reward": 0.5,
            "format_validity": 1.0,
            "total_reward": 1.5,
        },
        "diagnostics": {"valid_for_rl": True},
    }
    metrics = _group_reward_ordering_metrics([item])

    assert metrics["correct_vs_incorrect_ordering_accuracy"] is None
    assert metrics["group_count"] == 0
    assert metrics["unassigned_valid_rollout_count"] == 1
    assert metrics["comparable_pair_count"] == 0


def test_group_ordering_prefers_correct_rollout_within_group():
    items = [
        {
            "group_index": 7,
            "reward": {
                "answer_correctness": 1.0,
                "evidence_localization_reward": 1.0,
                "format_validity": 1.0,
                "total_reward": 2.0,
            },
            "diagnostics": {"valid_for_rl": True},
        },
        {
            "group_index": 7,
            "reward": {
                "answer_correctness": 0.0,
                "evidence_localization_reward": 0.5,
                "format_validity": 1.0,
                "total_reward": 0.5,
            },
            "diagnostics": {"valid_for_rl": True},
        },
    ]
    metrics = _group_reward_ordering_metrics(items)

    assert metrics["correct_vs_incorrect_ordering_accuracy"] == 1.0
    assert metrics["group_count"] == 1
    assert metrics["unassigned_valid_rollout_count"] == 0
    assert metrics["comparable_pair_count"] > 0
