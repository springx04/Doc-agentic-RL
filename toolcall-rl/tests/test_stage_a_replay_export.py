import json
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export_bayestool_replay import export_replay


def _replay(trajectory_id: str) -> dict:
    return {
        "schema_version": 1,
        "trajectory_id": trajectory_id,
        "replay_id": f"replay-{trajectory_id}",
        "coupling_id": "coupling",
        "latent_world_id": "latent",
        "replica_id": 0,
        "document_hash": "doc",
        "event_count": 1,
        "events": [
            {
                "event_index": 0,
                "call_id": 0,
                "tool_id": "parse_document",
                "arguments": {},
                "task_state_before": {"remaining_tool_budget": 4},
                "observed_result": "ok",
                "world_event": {"execution_succeeded": True},
                "hidden_label": {"status": "ok"},
                "task_state_after": {"remaining_tool_budget": 3},
                "next_tool_id": None,
                "result_status": "ok",
                "execution_succeeded": True,
                "observation_delivered": True,
            }
        ],
    }


def test_export_replay_keeps_train_split_and_q_risk_metadata(tmp_path):
    q_record = {
        "task_features": [0.0] * 32,
        "particle_features": [0.0] * 32,
        "action_features": [0.0] * 32,
        "budget_features": [0.0] * 8,
        "utility": 0.25,
    }
    artifact = {
        "records": [
            {
                "source": "train.pt",
                "payload": {
                    "samples": [
                        {
                            "metadata": {
                                "rollout_id": 11,
                                "belief_replay": _replay("train-1"),
                                "bayes_branch_records": [q_record],
                            },
                            "reward": {"exact_acc": 1.0},
                        }
                    ]
                },
            },
            {
                "source": "eval.pt",
                "payload": {
                    "samples": [
                        {"metadata": {"belief_replay": _replay("eval-1")}}
                    ]
                },
            },
        ]
    }
    input_path = tmp_path / "rollout_interactions.json"
    output_path = tmp_path / "belief_replay.jsonl"
    input_path.write_text(json.dumps(artifact), encoding="utf-8")

    manifest = export_replay(input_path, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert manifest["record_count"] == 1
    assert manifest["event_count"] == 1
    assert manifest["hidden_label_count"] == 1
    assert manifest["q_source_count"] == 1
    assert rows[0]["trajectory_id"] == "train-1"
    assert rows[0]["answer_correct"] == 1.0
    assert rows[0]["metadata"]["bayes_branch_records"] == [q_record]
