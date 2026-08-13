from pathlib import Path


def test_final_eval_launcher_is_isolated_and_non_branching():
    script = (Path(__file__).parents[1] / "scripts" / "eval_final_checkpoint_50.sh").read_text(encoding="utf-8")
    assert 'export CODE_STAGE="EVAL"' in script
    assert "--eval-prompt-data code_swe_eval" in script
    assert "--eval-temperature 0" in script
    assert "--colocate" in script
    assert script.index('CODE_AGENT_DIR=') < script.index('export PYTHONPATH=')
