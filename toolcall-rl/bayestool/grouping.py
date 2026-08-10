"""Strict question/realization/group planning for BayesTool RL.

The rollout sampler may use replicas internally, but replicas are not an RL
grouping primitive.  This module is the single place that defines the
question -> world realization -> decision group -> continuation hierarchy.
Keeping the planner independent of Ray and torch makes it usable by the
manifest builder, rollout producer, and both trainer backends.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


WORLD_SLOT_ROLES: tuple[str, ...] = (
    "healthy",
    "local_degradation",
    "shared_family_fault",
    "change",
)
ALLOWED_GROUP_SIZES: frozenset[int] = frozenset({4, 8})
GROUPING_SCHEMA_VERSION = "bayestool-question-grouping-v2"
DEFAULT_POLICY_VERSION = "bayestool-policy-v1"


def _nested(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata.get("bayestool")
    return value if isinstance(value, Mapping) else {}


def metadata_value(metadata: Mapping[str, Any], key: str, default: Any = "") -> Any:
    value = metadata.get(key)
    if value is None or value == "":
        value = _nested(metadata).get(key, default)
    return default if value is None else value


def question_id_from_metadata(metadata: Mapping[str, Any], default: str = "unknown") -> str:
    nested = _nested(metadata)
    for key in ("question_id", "meta_trajectory_id", "task_id"):
        value = metadata.get(key) or nested.get(key)
        if value is not None and str(value).strip():
            return str(value)
    episode = metadata.get("episode_content_id") or nested.get("episode_content_id")
    question_index = metadata.get("meta_question_index", nested.get("meta_question_index"))
    if episode is not None and question_index is not None:
        return f"{episode}:q{question_index}"
    value = metadata.get("coupling_id") or nested.get("coupling_id")
    return str(value) if value is not None and str(value).strip() else str(default)


def slot_role_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Return a stable role, with a compatibility mapping for old manifests."""

    value = metadata_value(metadata, "world_slot_role", "")
    if value:
        return str(value)
    world_type = str(metadata_value(metadata, "world_type", "")).casefold()
    if world_type == "healthy":
        return "healthy"
    if world_type in {"single_tool_degradation", "context_degradation"}:
        return "local_degradation"
    if world_type == "shared_family_fault":
        return "shared_family_fault"
    if world_type in {"abrupt_change", "gradual_change"}:
        return "change"
    return ""


def _stable_id(payload: Mapping[str, Any], prefix: str) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:24]
    return f"{prefix}:{digest}"


@dataclass(frozen=True)
class RealizationPlan:
    """One world realization and its exactly-one decision group."""

    world_slot_role: str
    variant_id: str = "base"
    latent_world_id: str = ""
    selected_decision_event: str = "root"
    decision_prefix_hash: str = ""
    runtime_state_digest: str = ""
    k: int = 4
    slot_weight: float = 0.25
    policy_version: str = DEFAULT_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.world_slot_role not in WORLD_SLOT_ROLES:
            raise ValueError(f"unknown BayesTool world_slot_role: {self.world_slot_role!r}")
        if int(self.k) not in ALLOWED_GROUP_SIZES:
            raise ValueError(f"decision group K must be 4 or 8, got {self.k!r}")
        if not self.variant_id:
            raise ValueError("variant_id must be non-empty")
        if float(self.slot_weight) < 0.0:
            raise ValueError("slot_weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_slot_role": self.world_slot_role,
            "variant_id": self.variant_id,
            "latent_world_id": self.latent_world_id,
            "selected_decision_event": self.selected_decision_event,
            "decision_prefix_hash": self.decision_prefix_hash,
            "runtime_state_digest": self.runtime_state_digest,
            "k": int(self.k),
            "slot_weight": float(self.slot_weight),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class QuestionRolloutPlan:
    """A complete, ready-to-train plan for one original question."""

    question_id: str
    policy_version: str = DEFAULT_POLICY_VERSION
    realizations: tuple[RealizationPlan, ...] = ()
    schema_version: str = GROUPING_SCHEMA_VERSION
    max_records: int = 48
    slot_weights: tuple[tuple[str, float], ...] = tuple((role, 0.25) for role in WORLD_SLOT_ROLES)

    def __post_init__(self) -> None:
        if not str(self.question_id).strip():
            raise ValueError("question_id must be non-empty")
        if not 4 <= len(self.realizations) <= 6:
            raise ValueError(f"a question requires 4-6 realizations, got {len(self.realizations)}")
        roles = {item.world_slot_role for item in self.realizations}
        missing = set(WORLD_SLOT_ROLES) - roles
        if missing:
            raise ValueError(f"required world slot roles are missing: {sorted(missing)}")
        if len({(item.world_slot_role, item.variant_id) for item in self.realizations}) != len(self.realizations):
            raise ValueError("a question cannot contain duplicate role/variant realizations")
        if sum(int(item.k) for item in self.realizations) > int(self.max_records):
            raise ValueError("question realization groups exceed the 48-record safety limit")
        weights = dict(self.slot_weights)
        if any(role not in weights for role in WORLD_SLOT_ROLES):
            raise ValueError("slot_weights must contain all four required roles")
        total = sum(float(weights[role]) for role in WORLD_SLOT_ROLES)
        if total <= 0.0:
            raise ValueError("required world slot weights must have positive total")

    @property
    def group_count(self) -> int:
        return len(self.realizations)

    @property
    def record_count(self) -> int:
        return sum(int(item.k) for item in self.realizations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question_id": self.question_id,
            "policy_version": self.policy_version,
            "realizations": [item.to_dict() for item in self.realizations],
            "group_count": self.group_count,
            "record_count": self.record_count,
            "max_records": int(self.max_records),
            "slot_weights": {role: float(weight) for role, weight in self.slot_weights},
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QuestionRolloutPlan":
        raw_weights = value.get("slot_weights", {})
        if isinstance(raw_weights, Mapping):
            slot_weights = tuple((role, float(raw_weights.get(role, 0.25))) for role in WORLD_SLOT_ROLES)
        else:
            slot_weights = tuple((role, 0.25) for role in WORLD_SLOT_ROLES)
        realizations = tuple(
            RealizationPlan(
                world_slot_role=str(item.get("world_slot_role", "")),
                variant_id=str(item.get("variant_id", "base")),
                latent_world_id=str(item.get("latent_world_id", "")),
                selected_decision_event=str(item.get("selected_decision_event", "root")),
                decision_prefix_hash=str(item.get("decision_prefix_hash", "")),
                runtime_state_digest=str(item.get("runtime_state_digest", "")),
                k=int(item.get("k", 4)),
                slot_weight=float(item.get("slot_weight", dict(slot_weights).get(str(item.get("world_slot_role", "")), 0.25))),
                policy_version=str(item.get("policy_version", value.get("policy_version", DEFAULT_POLICY_VERSION))),
            )
            for item in value.get("realizations", ())
            if isinstance(item, Mapping)
        )
        return cls(
            question_id=str(value.get("question_id", "")),
            policy_version=str(value.get("policy_version", DEFAULT_POLICY_VERSION)),
            realizations=realizations,
            schema_version=str(value.get("schema_version", GROUPING_SCHEMA_VERSION)),
            max_records=int(value.get("max_records", 48)),
            slot_weights=slot_weights,
        )


def make_question_rollout_plan(
    question_id: str,
    *,
    policy_version: str = DEFAULT_POLICY_VERSION,
    seed: int | str = 0,
    group_size: int = 4,
    extra_variants: Sequence[Mapping[str, Any] | str] = (),
    k_by_role: Mapping[str, int] | None = None,
) -> QuestionRolloutPlan:
    """Create the mandatory H/L/F/C plan plus at most two variants.

    The returned order is stable.  ``seed`` only chooses variant identities;
    it never changes the required four roles or the loss weights.
    """

    if int(group_size) not in ALLOWED_GROUP_SIZES:
        raise ValueError(f"group_size must be 4 or 8, got {group_size!r}")
    variants: list[tuple[str, str]] = []
    for index, value in enumerate(extra_variants):
        if index >= 2:
            break
        if isinstance(value, Mapping):
            role = str(value.get("world_slot_role", value.get("slot_role", "local_degradation")))
            variant = str(value.get("variant_id", f"variant-{index + 1}"))
        else:
            role = "local_degradation"
            variant = str(value or f"variant-{index + 1}")
        if role not in WORLD_SLOT_ROLES or role == "healthy":
            raise ValueError(f"extra variant must target a non-healthy required role, got {role!r}")
        variants.append((role, variant))
    entries: list[RealizationPlan] = []
    k_by_role = {str(key): int(value) for key, value in (k_by_role or {}).items()}
    for role in WORLD_SLOT_ROLES:
        k = k_by_role.get(role, int(group_size))
        latent_id = _stable_id({"question_id": question_id, "role": role, "seed": str(seed)}, "latent")
        entries.append(
            RealizationPlan(
                world_slot_role=role,
                latent_world_id=latent_id,
                k=k,
                slot_weight=0.25,
                policy_version=policy_version,
            )
        )
    for role, variant in variants:
        latent_id = _stable_id(
            {"question_id": question_id, "role": role, "variant": variant, "seed": str(seed)},
            "latent",
        )
        entries.append(
            RealizationPlan(
                world_slot_role=role,
                variant_id=variant,
                latent_world_id=latent_id,
                k=group_size,
                slot_weight=0.25,
                policy_version=policy_version,
            )
        )
    return QuestionRolloutPlan(question_id=str(question_id), policy_version=policy_version, realizations=tuple(entries))


def _record_metadata(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata
        return record
    metadata = getattr(record, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _record_sample_identity(record: Any, index: int) -> str:
    metadata = _record_metadata(record)
    for key in ("sample_id", "trajectory_id", "rollout_index", "sample_index"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    value = getattr(record, "index", None)
    if value is not None:
        return str(value)
    # ``rollout_id`` identifies a generation batch in several launchers, so
    # it is only a last-resort identity and never outranks the sample index.
    value = metadata.get("rollout_id")
    return str(value) if value is not None and str(value).strip() else f"record:{index}"


def _group_id(metadata: Mapping[str, Any]) -> str:
    return str(metadata_value(metadata, "decision_group_id", ""))


def _group_report(items: Sequence[tuple[int, Any]], allowed: frozenset[int]) -> tuple[dict[str, Any], list[str]]:
    metadata = [_record_metadata(record) for _, record in items]
    errors: list[str] = []
    questions = {question_id_from_metadata(row) for row in metadata}
    worlds = {str(metadata_value(row, "latent_world_id", "")) for row in metadata}
    events = {str(metadata_value(row, "selected_decision_event", "")) for row in metadata}
    prefixes = {str(metadata_value(row, "decision_prefix_hash", "")) for row in metadata}
    runtimes = {str(metadata_value(row, "runtime_state_digest", "")) for row in metadata}
    policies = {str(metadata_value(row, "policy_version", "")) for row in metadata}
    roles = {slot_role_from_metadata(row) for row in metadata}
    variants_by_role: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        variants_by_role[slot_role_from_metadata(row)].add(str(metadata_value(row, "variant_id", "base")))
    if len(items) not in allowed:
        errors.append(f"group_size={len(items)}")
    if len(questions) != 1:
        errors.append("cross_question")
    if len(worlds) != 1 or "" in worlds:
        errors.append("cross_latent_world")
    if len(events) != 1 or "" in events:
        errors.append("selected_decision_event_not_frozen")
    if len(prefixes) != 1 or "" in prefixes:
        errors.append("cross_decision_prefix")
    if len(runtimes) != 1 or "" in runtimes:
        errors.append("runtime_state_not_frozen")
    if len(policies) != 1 or "" in policies:
        errors.append("policy_version_not_frozen")
    if len(roles) != 1 or "" in roles:
        errors.append("world_slot_role_missing")
    identities = [_record_sample_identity(record, index) for index, record in items]
    if len(set(identities)) != len(identities):
        errors.append("duplicate_independent_sample")
    if any(
        str(metadata_value(row, "rollout_status", "")) in {"infra_error", "context_overflow", "tool_error"}
        or bool(row.get("infra_error"))
        or bool(row.get("context_overflow"))
        for row in metadata
    ):
        errors.append("infrastructure_invalid")
    return {
        "group_id": _group_id(metadata[0]) if metadata else "",
        "size": len(items),
        "question_ids": sorted(questions),
        "latent_world_ids": sorted(worlds),
        "decision_prefix_hashes": sorted(prefixes),
        "runtime_state_digests": sorted(runtimes),
        "policy_versions": sorted(policies),
        "world_slot_roles": sorted(roles),
        "variants_by_role": {role: sorted(values) for role, values in variants_by_role.items()},
        "sample_identities": identities,
        "valid": not errors,
    }, errors


def validate_bayestool_question_records(
    records: Sequence[Any],
    *,
    allowed_group_sizes: Sequence[int] = (4, 8),
    require_four_roles: bool = True,
    max_records: int = 48,
) -> dict[str, Any]:
    """Validate complete question/group/slot invariants without repairing data."""

    allowed = frozenset(int(value) for value in allowed_group_sizes)
    groups: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    questions: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    violations: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        metadata = _record_metadata(record)
        if bool(metadata.get("dummy_removed_sample")) or bool(metadata.get("exclude_from_group_statistics")):
            continue
        question = question_id_from_metadata(metadata, default=f"record:{index}")
        group_id = _group_id(metadata)
        if not group_id:
            violations.append({"index": index, "errors": ["missing_decision_group_id"]})
            continue
        groups[group_id].append((index, record))
        questions[question].append((index, record))

    group_summaries: list[dict[str, Any]] = []
    valid_group_ids: set[str] = set()
    for group_id, items in groups.items():
        summary, errors = _group_report(items, allowed)
        summary["group_id"] = group_id
        if errors:
            summary["errors"] = errors
            violations.append({"group_id": group_id, "size": len(items), "errors": errors})
        else:
            valid_group_ids.add(group_id)
        group_summaries.append(summary)

    question_summaries: list[dict[str, Any]] = []
    valid_questions: set[str] = set()
    for question, items in questions.items():
        q_groups = {_group_id(_record_metadata(record)) for _, record in items}
        q_roles = {slot_role_from_metadata(_record_metadata(record)) for _, record in items}
        q_variants = {
            (slot_role_from_metadata(_record_metadata(record)), str(metadata_value(_record_metadata(record), "variant_id", "base")))
            for _, record in items
        }
        errors: list[str] = []
        if len(items) > int(max_records):
            errors.append(f"records={len(items)} exceeds {max_records}")
        if not 4 <= len(q_groups) <= 6:
            errors.append(f"realization_count={len(q_groups)}")
        if require_four_roles and not set(WORLD_SLOT_ROLES).issubset(q_roles):
            errors.append("missing_required_world_slot_role")
        if len(q_groups) != len(q_variants):
            errors.append("group_count_not_equal_realizations")
        if any(group_id not in valid_group_ids for group_id in q_groups):
            errors.append("invalid_group")
        summary = {
            "question_id": question,
            "record_count": len(items),
            "group_count": len(q_groups),
            "world_slot_roles": sorted(q_roles),
            "group_ids": sorted(q_groups),
            "valid": not errors,
        }
        if errors:
            summary["errors"] = errors
            violations.append({"question_id": question, "errors": errors})
        else:
            valid_questions.add(question)
        question_summaries.append(summary)

    return {
        "valid": not violations,
        "question_count": len(questions),
        "valid_question_count": len(valid_questions),
        "group_count": len(groups),
        "valid_group_count": len(valid_group_ids),
        "group_size_histogram": {
            str(size): sum(1 for summary in group_summaries if summary.get("size") == size)
            for size in sorted({int(summary.get("size", 0)) for summary in group_summaries})
        },
        "violations": violations,
        "groups": group_summaries,
        "questions": question_summaries,
    }


def compute_hierarchical_loss_weights(
    records: Sequence[Any],
    *,
    question_count: int | None = None,
    slot_weights: Mapping[str, float] | None = None,
    validate: bool = True,
) -> tuple[list[float], dict[str, Any]]:
    """Compute group -> variant -> slot -> question normalized weights."""

    report = validate_bayestool_question_records(records) if validate else {"valid": True, "violations": []}
    if validate and not report["valid"]:
        raise ValueError(f"invalid BayesTool records for weighted loss: {report['violations'][:8]}")
    weights_by_role = {role: 0.25 for role in WORLD_SLOT_ROLES}
    if slot_weights:
        weights_by_role.update({str(key): float(value) for key, value in slot_weights.items()})
    active = [
        (index, record)
        for index, record in enumerate(records)
        if not bool(_record_metadata(record).get("dummy_removed_sample"))
        and not bool(_record_metadata(record).get("exclude_from_group_statistics"))
    ]
    questions = sorted({question_id_from_metadata(_record_metadata(record)) for _, record in active})
    q_count = max(1, int(question_count if question_count is not None else len(questions)))
    groups: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for index, record in active:
        groups[_group_id(_record_metadata(record))].append((index, record))
    variants_by_question_slot: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, record in active:
        metadata = _record_metadata(record)
        variants_by_question_slot[(question_id_from_metadata(metadata), slot_role_from_metadata(metadata))].add(
            str(metadata_value(metadata, "variant_id", "base"))
        )
    result = [0.0] * len(records)
    for items in groups.values():
        if not items:
            continue
        metadata = _record_metadata(items[0][1])
        question = question_id_from_metadata(metadata)
        role = slot_role_from_metadata(metadata)
        variant = str(metadata_value(metadata, "variant_id", "base"))
        variant_count = max(1, len(variants_by_question_slot[(question, role)]))
        group_k = len(items)
        weight = (1.0 / q_count) * float(weights_by_role.get(role, 0.0)) / variant_count / group_k
        for index, _ in items:
            result[index] = weight
    sums_by_question: dict[str, float] = defaultdict(float)
    sums_by_slot: dict[str, float] = defaultdict(float)
    for index, value in enumerate(result):
        if value <= 0.0:
            continue
        metadata = _record_metadata(records[index])
        question = question_id_from_metadata(metadata)
        role = slot_role_from_metadata(metadata)
        sums_by_question[question] += value
        sums_by_slot[f"{question}:{role}"] += value
    expected_question = 1.0 / q_count
    for question, value in sums_by_question.items():
        if abs(value - expected_question) > 1e-6:
            raise AssertionError(f"question weight is not normalized: {question} -> {value}")
    for key, value in sums_by_slot.items():
        if abs(value - expected_question * float(weights_by_role.get(key.rsplit(":", 1)[-1], 0.0))) > 1e-6:
            raise AssertionError(f"slot weight is not normalized: {key} -> {value}")
    return result, {
        "question_count": q_count,
        "question_weight_sums": dict(sums_by_question),
        "slot_weight_sums": dict(sums_by_slot),
        "slot_weights": dict(weights_by_role),
        "group_weight_sums": {
            group_id: sum(result[index] for index, _ in items) for group_id, items in groups.items()
        },
    }


def make_runtime_state_digest(
    state: Mapping[str, Any],
    *,
    return_definition_version: str = "bayestool-return-v2",
) -> str:
    """Hash the frozen decision-node state used by parent and child groups."""

    payload = {
        "remaining_tool_budget": int(state.get("remaining_tool_budget", state.get("tool_budget", 0)) or 0),
        "tool_call_count": int(state.get("tool_call_count", 0) or 0),
        "world_schedule": state.get("world_schedule", state.get("schedule_metadata", [])),
        "world_call_counters": state.get("world_call_counters", state.get("world_events", [])),
        "restore_state_id": str(state.get("restore_state_id", state.get("branch_resume_id", "root"))),
        "branch_horizon": int(state.get("branch_horizon", 0) or 0),
        "bootstrap": state.get("bootstrap", "terminal_reward"),
        "return_definition_version": return_definition_version,
    }
    return _stable_id(payload, "runtime")


__all__ = [
    "WORLD_SLOT_ROLES",
    "ALLOWED_GROUP_SIZES",
    "GROUPING_SCHEMA_VERSION",
    "DEFAULT_POLICY_VERSION",
    "RealizationPlan",
    "QuestionRolloutPlan",
    "make_question_rollout_plan",
    "question_id_from_metadata",
    "slot_role_from_metadata",
    "validate_bayestool_question_records",
    "compute_hierarchical_loss_weights",
    "make_runtime_state_digest",
]
