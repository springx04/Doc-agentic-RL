from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_eval import make_eval_only_argv


def test_eval_only_arguments_disable_rollout_updates_and_set_scheduler_floor():
    original = ["--num-rollout", "1", "--lr-decay-iters", "8"]
    result = make_eval_only_argv(original)

    assert original == ["--num-rollout", "1", "--lr-decay-iters", "8"]
    assert result == ["--num-rollout", "0", "--lr-decay-iters", "1"]


def test_eval_only_arguments_add_missing_scheduler_option():
    result = make_eval_only_argv(["--num-rollout", "4"])

    assert result == ["--num-rollout", "0", "--lr-decay-iters", "1"]
