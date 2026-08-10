"""Small dependency-free prioritized replay buffer for agentic trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
from typing import Any


@dataclass
class ReplayItem:
    trajectory: Any
    reward: float
    priority: float
    metadata: dict[str, Any] = field(default_factory=dict)
    insertion_id: int = 0


@dataclass
class ReplayBatch:
    """A sampled batch plus indices and importance-sampling weights.

    The iterator yields ``(items, indices, weights)`` for compatibility with
    the common prioritized-replay calling convention.  Attribute access is
    preferred when the sampled probabilities are also needed.
    """

    items: list[ReplayItem]
    indices: list[int]
    weights: list[float]
    probabilities: list[float]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        yield self.items
        yield self.indices
        yield self.weights


class PriorityReplayBuffer:
    """Ring-buffer storage with reward/priority-proportional sampling."""

    def __init__(
        self,
        capacity: int = 10_000,
        *,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be greater than zero")
        if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and greater than zero")
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self._rng = rng if rng is not None else random.Random(seed)
        self._storage: list[ReplayItem] = []
        self._next_index = 0
        self._next_insertion_id = 0

    def __len__(self) -> int:
        return len(self._storage)

    def __iter__(self):
        return iter(tuple(self._storage))

    def __getitem__(self, index: int) -> ReplayItem:
        return self._storage[index]

    def _priority_from_reward(self, reward: float) -> float:
        value = max(abs(float(reward)), self.epsilon)
        return value**self.alpha

    def add(
        self,
        trajectory: Any,
        reward: float,
        *,
        priority: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("reward must be finite")
        raw_priority = self._priority_from_reward(value) if priority is None else float(priority)
        if not math.isfinite(raw_priority) or raw_priority <= 0.0:
            raise ValueError("priority must be finite and greater than zero")
        item = ReplayItem(
            trajectory=trajectory,
            reward=value,
            priority=raw_priority,
            metadata=dict(metadata or {}),
            insertion_id=self._next_insertion_id,
        )
        self._next_insertion_id += 1
        if len(self._storage) < self.capacity:
            self._storage.append(item)
            index = len(self._storage) - 1
        else:
            index = self._next_index
            self._storage[index] = item
        self._next_index = (index + 1) % self.capacity
        return index

    def extend(self, records: Sequence[tuple[Any, float]]) -> list[int]:
        return [self.add(trajectory, reward) for trajectory, reward in records]

    def _probabilities(self) -> list[float]:
        if not self._storage:
            return []
        weights = [max(float(item.priority), self.epsilon) for item in self._storage]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            return [1.0 / len(weights)] * len(weights)
        return [weight / total for weight in weights]

    def _weighted_choice(self, indices: list[int], weights: list[float]) -> int:
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return self._rng.choice(indices)
        threshold = self._rng.random() * total
        running = 0.0
        for index, weight in zip(indices, weights):
            running += weight
            if running >= threshold:
                return index
        return indices[-1]

    def sample(
        self,
        batch_size: int,
        *,
        beta: float = 0.4,
        replace: bool | None = None,
    ) -> ReplayBatch:
        """Sample items and return normalized importance weights."""

        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not self._storage:
            raise ValueError("cannot sample from an empty replay buffer")
        beta_value = float(beta)
        if not math.isfinite(beta_value) or beta_value < 0.0:
            raise ValueError("beta must be finite and non-negative")
        probabilities = self._probabilities()
        if replace is None:
            replace = size > len(self._storage)
        if not replace and size > len(self._storage):
            raise ValueError("batch_size exceeds buffer size when replace=False")

        if replace:
            selected = [self._weighted_choice(list(range(len(self._storage))), probabilities) for _ in range(size)]
        else:
            available = list(range(len(self._storage)))
            available_weights = list(probabilities)
            selected = []
            for _ in range(size):
                chosen_position = self._weighted_choice(list(range(len(available))), available_weights)
                selected.append(available.pop(chosen_position))
                available_weights.pop(chosen_position)

        selected_probabilities = [probabilities[index] for index in selected]
        raw_importance = [
            (len(self._storage) * probability) ** (-beta_value)
            for probability in selected_probabilities
        ]
        max_weight = max(raw_importance, default=1.0)
        normalized_weights = [weight / max_weight for weight in raw_importance]
        return ReplayBatch(
            items=[self._storage[index] for index in selected],
            indices=selected,
            weights=normalized_weights,
            probabilities=selected_probabilities,
        )

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")
        for index, priority in zip(indices, priorities):
            position = int(index)
            if position < 0 or position >= len(self._storage):
                raise IndexError(f"replay index out of range: {position}")
            value = float(priority)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("priority must be finite and greater than zero")
            self._storage[position].priority = value

    def clear(self) -> None:
        self._storage.clear()
        self._next_index = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "next_index": self._next_index,
            "next_insertion_id": self._next_insertion_id,
            "items": [
                {
                    "trajectory": item.trajectory,
                    "reward": item.reward,
                    "priority": item.priority,
                    "metadata": dict(item.metadata),
                    "insertion_id": item.insertion_id,
                }
                for item in self._storage
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError("checkpoint capacity does not match buffer capacity")
        raw_items = state.get("items", [])
        if not isinstance(raw_items, list) or len(raw_items) > self.capacity:
            raise ValueError("invalid replay checkpoint items")
        storage: list[ReplayItem] = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid replay checkpoint item")
            priority = float(raw["priority"])
            if not math.isfinite(priority) or priority <= 0.0:
                raise ValueError("invalid replay checkpoint priority")
            storage.append(
                ReplayItem(
                    trajectory=raw.get("trajectory"),
                    reward=float(raw["reward"]),
                    priority=priority,
                    metadata=dict(raw.get("metadata") or {}),
                    insertion_id=int(raw.get("insertion_id", 0)),
                )
            )
        self._storage = storage
        self._next_index = int(state.get("next_index", len(storage) % self.capacity)) % self.capacity
        self._next_insertion_id = int(state.get("next_insertion_id", len(storage)))

    def stats(self) -> dict[str, float | int]:
        rewards = [item.reward for item in self._storage]
        return {
            "size": len(self._storage),
            "capacity": self.capacity,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_priority": max((item.priority for item in self._storage), default=0.0),
        }


__all__ = ["PriorityReplayBuffer", "ReplayBatch", "ReplayItem"]
