from schemas import CodeToolResult
from bayestool.corruption import corrupt_result


def test_ack_loss_is_world_injected_but_keeps_state_signal():
    clean = CodeToolResult("apply_patch", "ok", "applied", returncode=0)
    observed = corrupt_result(clean, "ack_loss", seed=1)
    assert observed.status == "timeout"
    assert observed.failure_origin == "world_injected"
    assert observed.metadata["state_may_have_changed"] is True
    assert observed.valid_for_rl is True
