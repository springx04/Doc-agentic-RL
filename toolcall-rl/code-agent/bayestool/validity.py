"""RL validity and decision-group fail-closed rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


REAL_INFRA_FAILURES = frozenset({"real_infrastructure"})


def valid_for_rl(*, failure_origin: str, termination_reason: str | None = None) -> bool:
    """World/model/task failures are valid samples; infrastructure failures are not."""

    return str(failure_origin) not in REAL_INFRA_FAILURES


@dataclass(frozen=True)
class GroupValidity:
    valid_for_rl: bool
    reason: str
    expected_k: int
    actual_k: int
    invalid_variants: tuple[str, ...] = ()


def validate_group_records(records: Iterable[Mapping[str, Any]], *, expected_k: int) -> GroupValidity:
    rows = list(records)
    if expected_k not in {4, 8}:
        return GroupValidity(False, "unsupported K", expected_k, len(rows))
    if len(rows) != expected_k:
        return GroupValidity(False, "incomplete sibling group", expected_k, len(rows))
    invalid: list[str] = []
    for index, row in enumerate(rows):
        origin = str(row.get("failure_origin", "none"))
        if origin == "real_infrastructure" or row.get("valid_for_rl") is False:
            invalid.append(str(row.get("variant_id", index)))
    if invalid:
        return GroupValidity(False, "real infrastructure failure in sibling group", expected_k, len(rows), tuple(invalid))
    fields = ("instance_id", "latent_world_id", "repo_state_digest", "decision_prefix_hash")
    for field in fields:
        if len({str(row.get(field)) for row in rows}) != 1:
            return GroupValidity(False, f"group field mismatch: {field}", expected_k, len(rows))
    return GroupValidity(True, "complete", expected_k, len(rows))


def fail_closed_advantage(rewards: Iterable[float], records: Iterable[Mapping[str, Any]], *, expected_k: int) -> tuple[float, ...]:
    rows = list(records)
    values = tuple(float(value) for value in rewards)
    validity = validate_group_records(rows, expected_k=expected_k)
    if not validity.valid_for_rl or len(values) != len(rows):
        return tuple(0.0 for _ in values)
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


__all__ = ["GroupValidity", "fail_closed_advantage", "valid_for_rl", "validate_group_records"]
