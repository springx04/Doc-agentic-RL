from pathlib import Path


def test_stage_c_preflight_is_fail_closed_and_code_isolated():
    script = (Path(__file__).parents[1] / "scripts" / "preflight_stage_c_100.sh").read_text(encoding="utf-8")
    assert "CODE_TRAIN_MANIFEST" in script
    assert "CODE_EVALUATOR_MANIFEST" in script
    assert "len(public_rows) != 100" in script
    assert "evaluator_private" in script
    assert "OpenClaw-RL" in script
    assert "curl" in script and "/healthz" in script
    assert "docker is unavailable" in script


def test_stage_c_launcher_is_code_only_and_uses_full_expanded_batch():
    script = (Path(__file__).parents[1] / "scripts" / "run_stage_b_100.sh").read_text(encoding="utf-8")
    assert "--train-backend fsdp" in script
    assert "--use-rollout-logprobs" in script
    assert "--global-batch-size \"${CODE_GLOBAL_BATCH_SIZE:-64}\"" in script
    assert "--bayestool-enable" not in script
    assert "25 rollouts x 4 tasks" in script
    assert "--colocate" in script
    assert "--actor-num-gpus-per-node \"${CODE_GPU_COUNT}\"" in script
    assert script.index('CODE_AGENT_DIR=') < script.index('export PYTHONPATH=')
