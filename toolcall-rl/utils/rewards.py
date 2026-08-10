"""Composite rewards and GRPO math for the local agentic-RL harness.

The existing document scorer remains the source of truth for answer metrics.
This module consumes its quality/format signals and adds the BayesTool-RL
terms described in the implementation plan, so the new reward can be enabled
without changing the document-tool protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any, Mapping

try:
    from .bayes_tool import BayesToolSelector, summarize_tool_outcomes
except ImportError:  # Running this module with toolcall-rl/utils on sys.path.
    from bayes_tool import BayesToolSelector, summarize_tool_outcomes


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the scalar agentic reward."""

    task_weight: float = 1.0
    correct_reward: float = 1.0
    format_bonus: float = 0.0
    format_error_penalty: float = -1.0
    multi_tool_bonus: float = 0.1
    information_gain_weight: float = 0.05
    tool_cost: float = 0.01
    kl_beta: float = 0.1
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_weight",
            "correct_reward",
            "format_bonus",
            "multi_tool_bonus",
            "information_gain_weight",
            "tool_cost",
            "kl_beta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(float(self.format_error_penalty)):
            raise ValueError("format_error_penalty must be finite")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError("clip_min must not exceed clip_max")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tool_names(tools_used: Iterable[Any] | None) -> list[str]:
    if tools_used is None:
        return []
    if isinstance(tools_used, (str, bytes)):
        tools_used = [tools_used]
    result: list[str] = []
    for tool in tools_used:
        name = str(tool).strip()
        if name:
            result.append(name)
    return result


def _accuracy(output: Any, ground_truth: Any, task_score: float | None) -> float:
    if task_score is not None:
        return max(0.0, min(1.0, _finite(task_score, "task_score")))
    return 1.0 if output == ground_truth else 0.0


def compute_composite_reward(
    output: Any,
    ground_truth: Any,
    format_ok: bool,
    tools_used: Iterable[Any] | None,
    *,
    information_gain: float = 0.0,
    kl_divergence: float = 0.0,
    task_score: float | None = None,
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    """Return each reward component and the combined scalar.

    With the default configuration this follows the design document's
    behavior: a correct, well-formed answer receives ``1``; a well-formed
    wrong answer receives ``0``; malformed output receives ``-1``; every tool
    call costs ``0.01``; and a second distinct tool adds ``0.1`` on a correct
    task.  Information gain and KL are explicit, auditable terms.
    """

    cfg = config or RewardConfig()
    tool_names = _tool_names(tools_used)
    unique_tools = list(dict.fromkeys(tool_names))
    call_count = len(tool_names)
    accuracy = _accuracy(output, ground_truth, task_score)
    well_formed = bool(format_ok)

    if not well_formed:
        task_reward = cfg.format_error_penalty
    elif accuracy > 0.0:
        task_reward = cfg.correct_reward * accuracy
        task_reward += cfg.format_bonus
    else:
        task_reward = 0.0

    multi_tool_reward = cfg.multi_tool_bonus if accuracy >= 1.0 and len(unique_tools) > 1 else 0.0
    info_gain_value = _finite(information_gain, "information_gain")
    kl_value = max(0.0, _finite(kl_divergence, "kl_divergence"))
    information_gain_reward = cfg.information_gain_weight * info_gain_value
    tool_cost_penalty = -cfg.tool_cost * call_count
    kl_penalty = -cfg.kl_beta * kl_value
    total = (
        cfg.task_weight * task_reward
        + multi_tool_reward
        + information_gain_reward
        + tool_cost_penalty
        + kl_penalty
    )
    if cfg.clip_min is not None:
        total = max(float(cfg.clip_min), total)
    if cfg.clip_max is not None:
        total = min(float(cfg.clip_max), total)
    return {
        "reward": float(total),
        "total_reward": float(total),
        "task_score": float(accuracy),
        "task_reward": float(task_reward),
        "format_ok": well_formed,
        "format_reward": float(cfg.format_bonus if well_formed and accuracy > 0.0 else 0.0),
        "multi_tool_reward": float(multi_tool_reward),
        "information_gain": info_gain_value,
        "information_gain_reward": float(information_gain_reward),
        "kl_divergence": kl_value,
        "kl_penalty": float(kl_penalty),
        "tool_cost_penalty": float(tool_cost_penalty),
        "tools_used": list(tool_names),
        "unique_tools": unique_tools,
        "tool_count": call_count,
    }


def compute_reward(
    output: Any,
    ground_truth: Any,
    format_ok: bool,
    tools_used: Iterable[Any] | None,
    *,
    information_gain: float = 0.0,
    kl_divergence: float = 0.0,
    task_score: float | None = None,
    config: RewardConfig | None = None,
) -> float:
    """Compatibility scalar API matching the design document's sketch."""

    return float(
        compute_composite_reward(
            output,
            ground_truth,
            format_ok,
            tools_used,
            information_gain=information_gain,
            kl_divergence=kl_divergence,
            task_score=task_score,
            config=config,
        )["total_reward"]
    )


def compute_rollout_agentic_reward(
    *,
    task_score: float,
    format_ok: bool,
    tool_calls: Iterable[Mapping[str, Any]] = (),
    tool_names: Sequence[str] | None = None,
    kl_divergence: float = 0.0,
    config: RewardConfig | None = None,
    selector: BayesToolSelector | None = None,
) -> dict[str, Any]:
    """Adapt an existing document-tool execution trace to the new reward."""

    calls = list(tool_calls)
    bayes = summarize_tool_outcomes(calls, tool_names=tool_names, selector=selector)
    used = [
        str(call.get("tool") or call.get("tool_name"))
        for call in calls
        if (call.get("tool") or call.get("tool_name")) and call.get("executed") is not False
    ]
    result = compute_composite_reward(
        output=None,
        ground_truth=None,
        format_ok=format_ok,
        tools_used=used,
        information_gain=float(bayes["information_gain"]),
        kl_divergence=kl_divergence,
        task_score=task_score,
        config=config,
    )
    result["bayes"] = bayes
    return result


def compute_group_advantages(
    rewards: Sequence[float],
    *,
    valid_mask: Sequence[bool] | None = None,
    epsilon: float = 1e-8,
) -> list[float]:
    """Compute normalized group-relative advantages, excluding invalid rollouts."""

    values = [_finite(reward, "reward") for reward in rewards]
    if valid_mask is None:
        valid = [True] * len(values)
    else:
        if len(valid_mask) != len(values):
            raise ValueError("valid_mask must have the same length as rewards")
        valid = [bool(value) for value in valid_mask]
    eligible = [value for value, include in zip(values, valid) if include]
    if not eligible:
        return [0.0] * len(values)
    mean = sum(eligible) / len(eligible)
    variance = sum((value - mean) ** 2 for value in eligible) / len(eligible)
    std = math.sqrt(max(variance, 0.0))
    if std <= float(epsilon):
        return [0.0 if not include else value - mean for value, include in zip(values, valid)]
    return [0.0 if not include else (value - mean) / std for value, include in zip(values, valid)]


def compute_grpo_loss(
    old_log_probs: Any,
    new_log_probs: Any,
    advantages: Any,
    *,
    clip_epsilon: float = 0.2,
) -> Any:
    """Compute the clipped GRPO surrogate loss for scalars or torch tensors."""

    epsilon = _finite(clip_epsilon, "clip_epsilon")
    if epsilon < 0.0 or epsilon >= 1.0:
        raise ValueError("clip_epsilon must be in [0, 1)")
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only in minimal installs
        torch = None
    if torch is not None and any(isinstance(value, torch.Tensor) for value in (old_log_probs, new_log_probs, advantages)):
        old = old_log_probs if isinstance(old_log_probs, torch.Tensor) else torch.as_tensor(old_log_probs, dtype=torch.float32)
        new = new_log_probs if isinstance(new_log_probs, torch.Tensor) else torch.as_tensor(new_log_probs, dtype=old.dtype, device=old.device)
        adv = advantages if isinstance(advantages, torch.Tensor) else torch.as_tensor(advantages, dtype=old.dtype, device=old.device)
        ratio = torch.exp(new - old)
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        return -torch.minimum(ratio * adv, clipped * adv).mean()

    old_values = [float(value) for value in old_log_probs]
    new_values = [float(value) for value in new_log_probs]
    advantage_values = [float(value) for value in advantages]
    if not (len(old_values) == len(new_values) == len(advantage_values)):
        raise ValueError("log-probability and advantage sequences must have equal lengths")
    if not old_values:
        return 0.0
    objectives = []
    for old, new, advantage in zip(old_values, new_values, advantage_values):
        ratio = math.exp(max(-60.0, min(60.0, new - old)))
        clipped_ratio = min(max(ratio, 1.0 - epsilon), 1.0 + epsilon)
        objectives.append(min(ratio * advantage, clipped_ratio * advantage))
    return -sum(objectives) / len(objectives)


def compute_entropy(probabilities: Any) -> Any:
    """Compute categorical entropy for a sequence or torch tensor."""

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(probabilities, torch.Tensor):
        probs = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        return -(probs * probs.log()).sum(dim=-1).mean()
    values = [max(0.0, float(value)) for value in probabilities]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((value / total) * math.log(max(value / total, 1e-12)) for value in values if value > 0.0)


__all__ = [
    "RewardConfig",
    "compute_composite_reward",
    "compute_entropy",
    "compute_grpo_loss",
    "compute_group_advantages",
    "compute_reward",
    "compute_rollout_agentic_reward",
]
