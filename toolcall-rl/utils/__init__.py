"""Reusable local agentic-RL utilities."""

from .bayes_tool import BayesToolSelector, ToolSelection, summarize_tool_outcomes
from .replay_buffer import PriorityReplayBuffer, ReplayBatch, ReplayItem
from .rewards import (
    RewardConfig,
    compute_composite_reward,
    compute_entropy,
    compute_grpo_loss,
    compute_group_advantages,
    compute_reward,
    compute_rollout_agentic_reward,
)

__all__ = [
    "BayesToolSelector",
    "PriorityReplayBuffer",
    "ReplayBatch",
    "ReplayItem",
    "RewardConfig",
    "ToolSelection",
    "compute_composite_reward",
    "compute_entropy",
    "compute_grpo_loss",
    "compute_group_advantages",
    "compute_reward",
    "compute_rollout_agentic_reward",
    "summarize_tool_outcomes",
]
