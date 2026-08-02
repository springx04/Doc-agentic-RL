"""Reward normalization helpers shared by rollout code and focused tests."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def normalize_group_rewards(
    raw_rewards: Iterable[float],
    group_indices: Iterable[int],
    excluded: Iterable[bool],
    std_normalization: bool,
) -> list[float]:
    """Normalize each group using only RL-valid trajectories.

    Infrastructure-invalid samples are returned as zero and never contribute
    to a group's mean or standard deviation. The sample standard deviation
    matches the torch.Tensor.std() behavior used by the existing GRPO path.
    """
    rewards = [float(value) for value in raw_rewards]
    groups = [int(value) for value in group_indices]
    excluded_values = [bool(value) for value in excluded]
    if not (len(rewards) == len(groups) == len(excluded_values)):
        raise ValueError("reward, group and exclusion arrays must have equal length")

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, group_index in enumerate(groups):
        if not excluded_values[index]:
            grouped_indices[group_index].append(index)

    normalized = [0.0 if excluded_values[index] else rewards[index] for index in range(len(rewards))]
    for indices in grouped_indices.values():
        values = [rewards[index] for index in indices]
        mean = sum(values) / len(values)
        for index in indices:
            value = rewards[index] - mean
            if std_normalization:
                if len(values) > 1:
                    variance = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
                    value /= math.sqrt(variance) + 1e-6
                else:
                    value = 0.0
            normalized[index] = value
    return normalized
