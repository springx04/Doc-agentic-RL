from __future__ import annotations

import importlib.util
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
