"""Pure BayesTool question batching and rank assignment helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


QuestionBatch = tuple[str, list[int]]


def pack_questions_by_cost(
    ready_questions: Sequence[QuestionBatch],
    question_costs: Mapping[str, float],
    *,
    questions_per_step: int,
    max_questions_per_step: int,
    target_global_train_cost: float | None,
) -> tuple[list[list[QuestionBatch]], float, str]:
    """Pack whole questions into bounded optimizer rounds by estimated cost.

    Question records remain atomic: this helper never splits a question across
    rounds and never uses another question to repair an incomplete one.
    """

    if not ready_questions:
        return [], 0.0, "no_ready_questions"
    max_questions = max(1, int(max_questions_per_step))
    count_target = max(1, min(int(questions_per_step), max_questions))

    def _cost(question_id: str) -> float:
        try:
            value = float(question_costs.get(question_id, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value >= 0.0 else 0.0

    explicit_target = target_global_train_cost is not None and float(target_global_train_cost) > 0.0
    target = (
        float(target_global_train_cost)
        if explicit_target
        else (sum(_cost(question_id) for question_id, _ in ready_questions) / max(1, len(ready_questions)))
        * count_target
    )
    target = max(1.0, target)

    pending = sorted(
        ((str(question_id), list(indices)) for question_id, indices in ready_questions),
        key=lambda item: (-_cost(item[0]), item[0]),
    )
    batches: list[list[QuestionBatch]] = []
    current: list[QuestionBatch] = []
    current_cost = 0.0
    for question_id, indices in pending:
        cost = _cost(question_id)
        if current and (current_cost + cost > target or len(current) >= max_questions):
            batches.append(current)
            current = []
            current_cost = 0.0
        current.append((question_id, indices))
        current_cost += cost
    if current:
        batches.append(current)
    source = "explicit_cost_target" if explicit_target else "derived_cost_target"
    return batches, target, source


def assign_equal_cardinality_lpt(
    indices: Sequence[int],
    costs: Mapping[int, float],
    *,
    dp_size: int,
) -> tuple[list[list[int]], list[float]]:
    """Assign samples with descending-cost LPT and equal local cardinality."""

    dp_size = int(dp_size)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    indices = [int(index) for index in indices]
    if not indices or len(indices) % dp_size:
        raise ValueError(
            "LPT assignment requires a non-empty sample list divisible by dp_size: "
            f"samples={len(indices)} dp_size={dp_size}"
        )

    def _cost(index: int) -> float:
        try:
            value = float(costs.get(index, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value >= 0.0 else 0.0

    local_count = len(indices) // dp_size
    rank_indices: list[list[int]] = [[] for _ in range(dp_size)]
    rank_costs = [0.0] * dp_size
    for index in sorted(indices, key=lambda value: (-_cost(value), value)):
        available = [rank for rank in range(dp_size) if len(rank_indices[rank]) < local_count]
        if not available:
            raise AssertionError("LPT assignment exceeded a rank's local cardinality")
        rank = min(available, key=lambda value: (rank_costs[value], value))
        rank_indices[rank].append(index)
        rank_costs[rank] += _cost(index)
    return rank_indices, rank_costs


__all__ = ["QuestionBatch", "pack_questions_by_cost", "assign_equal_cardinality_lpt"]
