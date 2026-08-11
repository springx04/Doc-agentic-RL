from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "analyze_bayestool_training_log.py"
SPEC = importlib.util.spec_from_file_location("analyze_bayestool_training_log", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_classifies_a_real_policy_update_as_effective() -> None:
    log = """
Job 'raysubmit_ok' succeeded
First rollout sample: reward: {'score': 0.0, 'tool_call_count': 1, 'valid_tool_call_count': 1, 'valid_for_rl': True, 'rollout_status': 'completed'}
Finish rollout: reward: {'score': 1.0, 'tool_call_count': 2, 'valid_tool_call_count': 2, 'valid_for_rl': True, 'rollout_status': 'completed'}
timer ref_log_probs end
timer log_probs end
rollout 0: {'rollout/rewards': 0.5, 'rollout/bayes_sibling_advantages': 0.25, 'rollout/bayes_aux_records': 1.0, 'rollout/bayes_branch_records': 1.0, 'rollout/advantages': 0.25}
step 0: {'train/loss': 0.12, 'train/pg_loss': 0.10, 'train/kl_loss': 0.02, 'train/grad_norm': 0.7}
timer update_weights end
successfully saved checkpoint from iteration 0
"""
    report = MODULE.audit_training_log(log)
    assert report["status"] == "effective"
    assert report["rl_signal_effective"] is True
    assert report["rollouts"]["unique_count"] == 2
    assert report["bayestool_action_signal_present"] is True


def test_audit_rejects_a_zero_signal_chain_only_run() -> None:
    log = """
Job 'raysubmit_zero' succeeded
First rollout sample: reward: {'score': -1.0, 'tool_call_count': 0, 'valid_tool_call_count': 0, 'protocol_error_count': 1, 'valid_for_rl': True, 'rollout_status': 'model_protocol_error'}
Finish rollout: reward: {'score': -1.0, 'tool_call_count': 0, 'valid_tool_call_count': 0, 'protocol_error_count': 1, 'valid_for_rl': True, 'rollout_status': 'model_protocol_error'}
timer ref_log_probs end
timer log_probs end
rollout 0: {'rollout/rewards': -1.0, 'rollout/bayes_sibling_advantages': 0.0, 'rollout/bayes_aux_records': 0.0, 'rollout/bayes_branch_records': 0.0, 'rollout/advantages': 0.0}
step 0: {'train/loss': 0.0, 'train/pg_loss': 0.0, 'train/kl_loss': 0.0, 'train/grad_norm': 0.0}
timer update_weights end
successfully saved checkpoint from iteration 0
"""
    report = MODULE.audit_training_log(log)
    assert report["status"] == "chain_only"
    assert report["rl_signal_effective"] is False
    assert any("identical" in diagnosis for diagnosis in report["diagnoses"])
    assert any("tool action" in diagnosis for diagnosis in report["diagnoses"])
    assert any("protocol errors" in diagnosis for diagnosis in report["diagnoses"])


def test_audit_recognizes_consumed_protocol_penalty_as_partial_signal():
    log = """
Job 'raysubmit_action_penalty' succeeded
First rollout sample: reward: {'score': -1.0, 'tool_call_count': 0, 'valid_for_rl': True, 'rollout_status': 'model_protocol_error'}
Finish rollout: reward: {'score': -1.0, 'tool_call_count': 0, 'valid_for_rl': True, 'rollout_status': 'model_protocol_error'}
timer ref_log_probs end
timer log_probs end
rollout 0: {'rollout/rewards': -1.0, 'rollout/bayes_sibling_advantages': 0.0, 'rollout/advantages': -0.5, 'rollout/action_reward_token_count': 3.0, 'rollout/action_reward_abs_sum': 1.5}
step 0: {'train/loss': 0.04, 'train/pg_loss': 0.04, 'train/grad_norm': 0.2}
timer update_weights end
successfully saved checkpoint from iteration 0
"""
    report = MODULE.audit_training_log(log)
    assert report["status"] == "partial_signal"
    assert report["rl_signal_effective"] is False
    assert report["action_reward_signal_present"] is True
    assert report["policy_update_signal_present"] is True


def test_audit_marks_attention_or_runtime_failure_before_training() -> None:
    log = """
timer ref_log_probs start
ValueError: No dot product attention backend is available for the provided inputs.
Job 'raysubmit_failed' failed
"""
    report = MODULE.audit_training_log(log)
    assert report["status"] == "failed"
    assert report["framework_chain_passed"] is False
    assert "job did not complete successfully" in report["diagnoses"]
    assert "no checkpoint save evidence was found" in report["diagnoses"]


def test_audit_accepts_smoke_result_and_on_policy_log_probs(tmp_path: Path) -> None:
    (tmp_path / "smoke_result.json").write_text(
        json.dumps({"ok": True, "global_step": 2}), encoding="utf-8"
    )
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    log = """
timer log_probs start
timer log_probs end
rollout 0: {'rollout/log_probs': -0.4, 'rollout/advantages': 0.2}
step 0: {'train/loss': 0.2, 'train/pg_loss': 0.2, 'train/grad_norm': 0.8}
timer update_weights end
"""
    report = MODULE.audit_training_log(log, tmp_path)
    assert report["framework_chain_passed"] is True
    assert report["job"]["smoke_result_succeeded"] is True
    assert report["job"]["on_policy_log_probs_seen"] is True
    assert report["job"]["reference_log_probs_seen"] is False
    assert not any("reference log-probability" in item for item in report["diagnoses"])


def test_audit_prefers_training_rollout_artifact_over_aggregate_log(tmp_path: Path) -> None:
    artifact = {
        "records": [
            {
                "source": str(tmp_path / "dump_details" / "rollout_data" / "0.pt"),
                "payload": {
                    "samples": [
                        {
                            "reward": {
                                "score": -0.1,
                                "tool_call_count": 1,
                                "valid_tool_call_count": 1,
                                "valid_for_rl": True,
                                "rollout_status": "completed",
                            }
                        },
                        {
                            "reward": {
                                "score": 0.4,
                                "tool_call_count": 2,
                                "valid_tool_call_count": 2,
                                "valid_for_rl": True,
                                "rollout_status": "completed",
                            }
                        },
                    ]
                },
            },
            {
                "source": str(tmp_path / "dump_details" / "rollout_data" / "eval_0.pt"),
                "payload": {
                    "samples": [
                        {
                            "reward": {
                                "score": 1.0,
                                "tool_call_count": 1,
                                "valid_for_rl": True,
                            }
                        }
                    ]
                },
            },
        ]
    }
    (tmp_path / "rollout_interactions.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    report = MODULE.audit_training_log("Job 'aggregate' succeeded\n", tmp_path)
    assert report["rollouts"]["observed_records"] == 2
    assert report["rollouts"]["unique_count"] == 2
    assert report["rollouts"]["tool_active_records"] == 2
    assert report["parser"]["rollout_artifact"]["evaluation_records"] == 1
