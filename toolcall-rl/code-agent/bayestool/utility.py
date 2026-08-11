"""Code-specific information gain, cost, inefficiency, and utility."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping


def information_gain(
    *,
    tool_name: str,
    output: str,
    history: Iterable[Mapping[str, Any]] = (),
    task_state: Any = None,
    patch_changed: bool = False,
) -> float:
    """Estimate only new public evidence; no gold/verifier inputs are accepted."""

    text = str(output or "")
    prior = [str(row.get("output", "")) for row in history]
    if not text:
        return 0.0
    if text in prior:
        return 0.0
    prior_text = "\n".join(prior[-8:])
    lines = text.splitlines()
    new_lines = [line for line in lines if line not in prior_text]
    paths = set(re.findall(r"(?:^|\s)([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?)", text))
    trace_paths = set(re.findall(r"(?:File|at)\s+([A-Za-z0-9_./-]+)", text))
    tests = len(re.findall(r"(?:FAILED|PASSED|ERROR|FAIL|PASS)", text, re.I))
    hunks = text.count("@@")
    score = 0.10 * min(10, len(new_lines)) + 0.12 * min(8, len(paths)) + 0.14 * min(5, len(trace_paths))
    score += 0.15 * min(4, tests) + 0.18 * min(4, hunks)
    if patch_changed:
        score += 0.25
    if tool_name in {"list_tree", "search_code", "read_file"}:
        score += 0.10
    return max(0.0, min(1.0, score))


def normalized_cost(*, tool_calls: int, latency_ms: float, observation_chars: int, validation_runtime_ms: float, budget: int) -> float:
    budget_norm = max(1.0, float(budget))
    return (
        0.40 * min(1.0, max(0.0, tool_calls / budget_norm))
        + 0.25 * min(1.0, max(0.0, latency_ms / max(1.0, budget_norm * 1000.0)))
        + 0.20 * min(1.0, max(0.0, observation_chars / max(1.0, budget_norm * 4096.0)))
        + 0.15 * min(1.0, max(0.0, validation_runtime_ms / max(1.0, budget_norm * 5000.0)))
    )


def inefficiency_score(events: Iterable[Mapping[str, Any]]) -> float:
    rows = list(events)
    if not rows:
        return 0.0
    signatures = [
        (row.get("tool_name") or row.get("tool"), str(row.get("arguments") or row.get("output") or ""))
        for row in rows
    ]
    duplicate = sum(count - 1 for count in Counter(signatures).values() if count > 1)
    low_info = sum(1 for row in rows if float(row.get("information_gain", 0.0) or 0.0) < 0.02)
    return min(1.0, 0.6 * duplicate / max(1, len(rows)) + 0.4 * low_info / max(1, len(rows)))


def failure_penalty(*, failure_origin: str, protocol_error: bool = False, invalid_action: bool = False, premature_final: bool = False, budget_exhausted: bool = False) -> float:
    # World-injected failures are deliberately absent from this score.
    if failure_origin == "world_injected":
        return 0.0
    return min(1.0, 0.35 * protocol_error + 0.30 * invalid_action + 0.20 * premature_final + 0.15 * budget_exhausted)


def compute_utility(*, resolved: bool, cost: float, inefficiency: float, failure_penalty_value: float) -> float:
    task_score = 2.0 * (1.0 if resolved else 0.0) - 1.0
    return task_score - 0.20 * cost - 0.20 * inefficiency - 0.35 * failure_penalty_value


__all__ = ["compute_utility", "failure_penalty", "information_gain", "inefficiency_score", "normalized_cost"]
