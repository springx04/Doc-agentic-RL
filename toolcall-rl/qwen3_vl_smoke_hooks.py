"""Document-tool rollout hooks used by the Qwen3-VL evaluation launcher."""

from __future__ import annotations

import generate_with_retool as _implementation


# The production document agent needs room for page navigation and a final
# answer turn.  Keep the limits bounded, but do not truncate the trajectory
# after the four page reads required by the full DocVQA baseline.
_implementation.TOOL_CONFIGS.update(
    {
        "max_turns": 8,
        "max_tool_calls": 8,
        "max_obs_chars": 2048,
        "tool_concurrency": 4,
    }
)


async def generate(args, sample, sampling_params, evaluation: bool = False):
    return await _implementation.generate(
        args,
        sample,
        sampling_params,
        evaluation=evaluation,
    )


async def reward_func(args, sample, **kwargs):
    return await _implementation.reward_func(args, sample, **kwargs)
