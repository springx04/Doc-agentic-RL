"""Utilities for consuming per-action rollout training signals.

Document rollouts keep the ordinary assistant loss mask for diagnostics.  A
rejected assistant action is recorded with ``mask=0`` so it cannot inherit the
sequence-level positive reward.  The trainer consumes its negative action
reward through an explicit token-span override instead.
"""

from __future__ import annotations

from typing import Any


def build_action_training_overrides(
    response_length: int,
    assistant_token_masks: Any,
    action_rewards: Any,
) -> dict[str, Any]:
    """Build negative per-token overrides for rejected assistant actions."""
    length = max(0, int(response_length))
    advantages = [0.0] * length
    token_mask = [0] * length
    consumed_action_indices: list[int] = []
    invalid_action_indices: list[int] = []
    if not isinstance(assistant_token_masks, (list, tuple)):
        assistant_token_masks = []
    if not isinstance(action_rewards, (list, tuple)):
        action_rewards = []

    for action_index, span in enumerate(assistant_token_masks):
        if not isinstance(span, dict):
            continue
        try:
            start = max(0, int(span.get("token_start", 0) or 0))
            end = min(length, int(span.get("token_end", 0) or 0))
        except (TypeError, ValueError):
            continue
        reward = 0.0
        if action_index < len(action_rewards):
            try:
                reward = float(action_rewards[action_index] or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
        mask_value = int(span.get("mask", 1) or 0)
        # Only rejected/invalid actions receive an explicit override.  Valid
        # actions continue to learn from the ordinary sequence advantage.
        if start >= end or mask_value != 0 or reward >= 0.0:
            continue
        for position in range(start, end):
            advantages[position] = reward
            token_mask[position] = 1
        consumed_action_indices.append(action_index)
        invalid_action_indices.append(action_index)

    return {
        "action_advantages": advantages,
        "action_token_mask": token_mask,
        "consumed_action_indices": consumed_action_indices,
        "invalid_action_indices": invalid_action_indices,
        "action_reward_consumed": bool(consumed_action_indices),
    }


def action_reward_loss(log_probs: Any, action_advantages: Any, action_token_mask: Any) -> Any:
    """Return a small differentiable diagnostic/training loss for action spans.

    The production actors apply the same override through their normal PPO/GRPO
    policy loss.  This helper is intentionally tiny so a CPU unit test can
    prove that a negative action reward produces a non-zero gradient without a
    model or distributed runtime.
    """
    import torch

    log_probs = torch.as_tensor(log_probs)
    advantages = torch.as_tensor(action_advantages, dtype=log_probs.dtype, device=log_probs.device)
    mask = torch.as_tensor(action_token_mask, dtype=log_probs.dtype, device=log_probs.device)
    denominator = mask.sum().clamp_min(1.0)
    return -(log_probs * advantages * mask).sum() / denominator
