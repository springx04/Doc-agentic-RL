"""Argument helpers for a strict, no-update RL baseline evaluation."""

from __future__ import annotations


def _replace_value(argv: list[str], option: str, value: str) -> None:
    try:
        index = argv.index(option)
    except ValueError:
        argv.extend((option, value))
        return
    if index + 1 >= len(argv):
        raise ValueError(f"missing value for {option}")
    argv[index + 1] = value


def make_eval_only_argv(argv: list[str]) -> list[str]:
    """Return arguments that evaluate an initial checkpoint without updates.

    Slime still constructs an optimizer/scheduler in eval-only mode, so its
    scheduler needs one nominal decay iteration.  ``num_rollout=0`` is the
    authoritative no-update switch in ``slime.train``.
    """
    result = list(argv)
    _replace_value(result, "--num-rollout", "0")
    _replace_value(result, "--lr-decay-iters", "1")
    return result
