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


def _clone_training_value(value: Any) -> Any:
    """Clone a tensor-like value without importing the training backend."""
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _training_value_length(value: Any) -> int:
    """Return the first-dimension length for tensors and Python sequences."""
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    try:
        return len(value)
    except TypeError:
        return 0


def apply_action_training_overrides(
    *,
    response_length: int,
    advantages: Any,
    returns: Any,
    loss_mask: Any,
    assistant_token_masks: Any,
    action_rewards: Any,
) -> dict[str, Any]:
    """Apply rejected-action penalties to the actual policy-training arrays.

    The rollout keeps rejected assistant actions masked out of the ordinary
    sequence objective.  This function is the single bridge from the
    diagnostic action metadata to the trainer-owned advantage and loss-mask
    arrays.  It works with both CPU Python lists (used by FSDP tests) and
    PyTorch tensors (used by the Megatron actor), and returns cloned values so
    the caller never mutates the original rollout tensors accidentally.
    """
    signal = build_action_training_overrides(
        response_length,
        assistant_token_masks,
        action_rewards,
    )
    if not signal["action_reward_consumed"]:
        return {
            "advantages": advantages,
            "returns": returns,
            "loss_mask": loss_mask,
            **signal,
            "applied_action_indices": [],
            "applied_action_token_count": 0,
            "applied_action_reward_abs_sum": 0.0,
        }

    updated_advantages = _clone_training_value(advantages)
    updated_returns = _clone_training_value(returns) if returns is not None else None
    updated_loss_mask = _clone_training_value(loss_mask)
    max_positions = min(
        max(0, int(response_length)),
        _training_value_length(updated_advantages),
        _training_value_length(updated_loss_mask),
    )
    if updated_returns is not None:
        max_positions = min(max_positions, _training_value_length(updated_returns))

    spans = assistant_token_masks if isinstance(assistant_token_masks, (list, tuple)) else []
    applied_action_indices: list[int] = []
    applied_token_count = 0
    applied_reward_abs_sum = 0.0
    for action_index in signal["consumed_action_indices"]:
        if action_index >= len(spans) or not isinstance(spans[action_index], dict):
            continue
        span = spans[action_index]
        try:
            start = max(0, int(span.get("token_start", 0) or 0))
            end = min(max_positions, int(span.get("token_end", 0) or 0))
        except (TypeError, ValueError):
            continue
        if start >= end:
            continue
        reward = float(signal["action_advantages"][start])
        for position in range(start, end):
            updated_advantages[position] = reward
            if updated_returns is not None:
                updated_returns[position] = reward
            updated_loss_mask[position] = 1
        applied_action_indices.append(action_index)
        applied_token_count += end - start
        applied_reward_abs_sum += abs(reward) * (end - start)

    return {
        "advantages": updated_advantages,
        "returns": updated_returns,
        "loss_mask": updated_loss_mask,
        "action_advantages": signal["action_advantages"],
        "action_token_mask": signal["action_token_mask"],
        "consumed_action_indices": applied_action_indices,
        "invalid_action_indices": signal["invalid_action_indices"],
        "action_reward_consumed": bool(applied_action_indices),
        "applied_action_indices": applied_action_indices,
        "applied_action_token_count": applied_token_count,
        "applied_action_reward_abs_sum": applied_reward_abs_sum,
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
