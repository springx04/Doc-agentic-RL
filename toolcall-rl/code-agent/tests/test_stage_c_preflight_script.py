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
