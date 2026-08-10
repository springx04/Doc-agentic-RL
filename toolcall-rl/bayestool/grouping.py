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
import math
import random
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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _variant_score(candidate: Mapping[str, Any]) -> float:
    """Score observable variant value; hidden labels are never consulted."""

    explicit = candidate.get("score")
    if explicit is not None:
        return _finite_float(explicit)
    return (
        0.35 * _finite_float(candidate.get("coverage_deficit"))
        + 0.25 * _finite_float(candidate.get("learning_progress"))
        + 0.25 * _finite_float(candidate.get("calibration_gap"))
        + 0.15 * _finite_float(candidate.get("diagnostic_value"))
    )


def select_extra_variants(
    question_id: str,
    candidates: Sequence[Mapping[str, Any] | str],
    *,
    seed: int | str = 0,
    max_extra: int = 2,
    exploration_probability: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select at most two observable world variants with a fixed seed.

    The caller may provide more than two candidates; selection is explicit and
    auditable.  ``make_question_rollout_plan`` still rejects more than two
    already-selected variants so malformed manifests fail closed.
    """

    if int(max_extra) < 0 or int(max_extra) > 2:
        raise ValueError(f"max_extra must be in [0, 2], got {max_extra!r}")
    probability = max(0.0, min(1.0, _finite_float(exploration_probability)))
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(candidates):
        if isinstance(value, Mapping):
            role = str(value.get("world_slot_role", value.get("slot_role", "local_degradation")))
            variant = str(value.get("variant_id", f"variant-{index + 1}"))
            item = dict(value)
        else:
            role = "local_degradation"
            variant = str(value or f"variant-{index + 1}")
            item = {}
        if role not in WORLD_SLOT_ROLES or role == "healthy":
            raise ValueError(f"extra variant must target a non-healthy required role, got {role!r}")
        key = (role, variant)
        if key in seen:
            raise ValueError(f"duplicate extra variant candidate: {role}:{variant}")
        seen.add(key)
        item.update({"world_slot_role": role, "variant_id": variant})
        item["score"] = _variant_score(item)
        normalized.append(item)

    if not normalized or max_extra == 0:
        return [], {
            "question_id": str(question_id),
            "seed": str(seed),
            "candidates": normalized,
            "selected": [],
            "selection_reason": "no_variant_candidates",
        }

    temperature = max(0.05, _finite_float(normalized[0].get("temperature"), 1.0))
    max_score = max(float(item["score"]) for item in normalized)
    exp_scores = [math.exp((float(item["score"]) - max_score) / temperature) for item in normalized]
    normalizer = sum(exp_scores) or 1.0
    for item, value in zip(normalized, exp_scores, strict=True):
        item["sampling_probability"] = float(value / normalizer)

    rng = random.Random(f"{question_id}:{seed}:bayestool-variant-selection")
    remaining = list(normalized)
    selected: list[dict[str, Any]] = []
    for _ in range(min(int(max_extra), len(remaining))):
        explore = rng.random() < probability
        if explore:
            weights = [max(0.0, float(item.get("sampling_probability", 0.0))) for item in remaining]
            total = sum(weights)
            if total <= 0.0:
                chosen_index = rng.randrange(len(remaining))
            else:
                draw = rng.random() * total
                chosen_index = len(remaining) - 1
                for index, weight in enumerate(weights):
                    draw -= weight
                    if draw < 0.0:
                        chosen_index = index
                        break
            reason = "seeded_exploration"
        else:
            chosen_index = max(
                range(len(remaining)),
                key=lambda index: (float(remaining[index]["score"]), -index),
            )
            reason = "highest_observable_value"
        chosen = dict(remaining.pop(chosen_index))
        chosen["selection_reason"] = reason
        chosen["selected"] = True
        selected.append(chosen)

    selected_keys = {(item["world_slot_role"], item["variant_id"]) for item in selected}
    for item in normalized:
        item["selected"] = (item["world_slot_role"], item["variant_id"]) in selected_keys
        item.setdefault("selection_reason", "not_selected")
    return selected, {
        "question_id": str(question_id),
        "seed": str(seed),
        "candidates": normalized,
        "selected": [dict(item) for item in selected],
        "selection_reason": "seeded_exploration" if any(item["selection_reason"] == "seeded_exploration" for item in selected) else "highest_observable_value",
    }


@dataclass
class K8RollingScheduler:
    """Bound K=8 usage while keeping an auditable rolling target."""

    target_ratio: float = 0.25
    floor: float = 0.0
    ceiling: float = 1.0
    window: int = 32
    seed: int | str = 0
    history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.target_ratio = max(0.0, min(1.0, _finite_float(self.target_ratio)))
        self.floor = max(0.0, min(1.0, _finite_float(self.floor)))
        self.ceiling = max(self.floor, min(1.0, _finite_float(self.ceiling, 1.0)))
        self.window = max(1, int(self.window))

    @property
    def ratio(self) -> float:
        recent = self.history[-self.window :]
        return sum(recent) / len(recent) if recent else 0.0

    def choose(self, question_id: str, *, requested_k: int = 4, signal: float = 0.0) -> tuple[int, dict[str, Any]]:
        requested_k = int(requested_k)
        if requested_k not in ALLOWED_GROUP_SIZES:
            raise ValueError(f"requested_k must be 4 or 8, got {requested_k!r}")
        before = self.ratio
        if requested_k == 8:
            selected_k = 8
            reason = "manifest_requested_k8"
        elif before < self.floor:
            selected_k = 8
            reason = "rolling_floor_recovery"
        elif before >= self.ceiling:
            selected_k = 4
            reason = "rolling_ceiling_guard"
        else:
            signal_adjustment = max(-0.25, min(0.25, _finite_float(signal)))
            probability = max(self.floor, min(self.ceiling, self.target_ratio + signal_adjustment))
            draw = random.Random(f"{question_id}:{self.seed}:k8").random()
            selected_k = 8 if draw < probability else 4
            reason = "rolling_target_exploration" if selected_k == 8 else "rolling_target_k4"
        self.history.append(1 if selected_k == 8 else 0)
        if len(self.history) > self.window:
            del self.history[:-self.window]
        return selected_k, {
            "question_id": str(question_id),
            "selected_k": selected_k,
            "reason": reason,
            "ratio_before": before,
            "ratio_after": self.ratio,
            "target_ratio": self.target_ratio,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "window": self.window,
        }


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
        slot_weight = float(self.slot_weight)
        if not math.isfinite(slot_weight) or slot_weight < 0.0:
            raise ValueError("slot_weight must be a finite non-negative number")
        if not str(self.selected_decision_event).strip():
            raise ValueError("selected_decision_event must be non-empty")

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
    # Data-builder manifests set this after freezing one latent world spec per
    # realization.  Runtime-created fallback plans keep it false until their
    # first world is materialized; this prevents a synthetic planning ID from
    # being mistaken for a finalized world identity.
    latent_ids_finalized: bool = False
    # Candidate/selection diagnostics are part of the frozen manifest so a
    # dynamic variant decision can be audited without changing group weights.
    variant_selection: tuple[Mapping[str, Any], ...] = ()

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
        if not 1 <= int(self.max_records) <= 48:
            raise ValueError(f"max_records must be in [1, 48], got {self.max_records!r}")
        if sum(int(item.k) for item in self.realizations) > int(self.max_records):
            raise ValueError("question realization groups exceed the 48-record safety limit")
        weights = dict(self.slot_weights)
        if set(weights) != set(WORLD_SLOT_ROLES):
            raise ValueError("slot_weights must contain exactly the four required roles")
        for role in WORLD_SLOT_ROLES:
            weight = float(weights[role])
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"slot weight for {role!r} must be finite and non-negative")
        total = sum(float(weights[role]) for role in WORLD_SLOT_ROLES)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"slot weights must sum to one, got {total}")
        for realization in self.realizations:
            if self.latent_ids_finalized and not str(realization.latent_world_id).strip():
                raise ValueError(
                    "a finalized question plan must declare latent_world_id for every realization"
                )
            expected = float(weights[realization.world_slot_role])
            if not math.isclose(float(realization.slot_weight), expected, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"realization slot weight disagrees with plan for {realization.world_slot_role!r}: "
                    f"{realization.slot_weight} != {expected}"
                )
            if realization.policy_version != self.policy_version:
                raise ValueError("all realizations must use the question policy_version")

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
            "latent_ids_finalized": bool(self.latent_ids_finalized),
            "variant_selection": [dict(item) for item in self.variant_selection if isinstance(item, Mapping)],
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
            latent_ids_finalized=bool(value.get("latent_ids_finalized", False)),
            variant_selection=tuple(
                dict(item) for item in (value.get("variant_selection", ()) or ()) if isinstance(item, Mapping)
            ),
        )


def make_question_rollout_plan(
    question_id: str,
    *,
    policy_version: str = DEFAULT_POLICY_VERSION,
    seed: int | str = 0,
    group_size: int | str = 4,
    extra_variants: Sequence[Mapping[str, Any] | str] = (),
    k_by_role: Mapping[str, int] | None = None,
    realization_count: int | None = None,
    k8_scheduler: K8RollingScheduler | None = None,
    selection_signal: float = 0.0,
) -> QuestionRolloutPlan:
    """Create the mandatory H/L/F/C plan plus at most two variants.

    The returned order is stable.  ``seed`` only chooses variant identities;
    it never changes the required four roles or the loss weights.
    """

    if isinstance(group_size, str) and group_size.casefold() == "rolling":
        if k8_scheduler is None:
            raise ValueError("group_size='rolling' requires a K8RollingScheduler")
        group_size, _ = k8_scheduler.choose(
            str(question_id), requested_k=4, signal=selection_signal
        )
    if int(group_size) not in ALLOWED_GROUP_SIZES:
        raise ValueError(f"group_size must be 4 or 8, got {group_size!r}")
    if realization_count is not None and not 4 <= int(realization_count) <= 6:
        raise ValueError(f"realization_count must be in [4, 6], got {realization_count!r}")
    requested_extra_count = max(0, int(realization_count) - 4) if realization_count is not None else None
    if requested_extra_count and not extra_variants:
        defaults = (
            {"world_slot_role": "local_degradation", "variant_id": "fallback-local"},
            {"world_slot_role": "change", "variant_id": "fallback-change"},
        )
        extra_variants = defaults[:requested_extra_count]
    if requested_extra_count is not None and len(extra_variants) != requested_extra_count:
        raise ValueError(
            "realization_count and extra_variants disagree: "
            f"expected {requested_extra_count} extras, got {len(extra_variants)}"
        )
    variants: list[tuple[str, str]] = []
    selection_records: list[Mapping[str, Any]] = []
    if len(extra_variants) > 2:
        raise ValueError(
            "at most two extra BayesTool variants may enter a question plan; "
            "select candidates before calling make_question_rollout_plan"
        )
    for index, value in enumerate(extra_variants):
        if isinstance(value, Mapping):
            role = str(value.get("world_slot_role", value.get("slot_role", "local_degradation")))
            variant = str(value.get("variant_id", f"variant-{index + 1}"))
            selection = value.get("selection")
            if isinstance(selection, Mapping):
                selection_records.append(dict(selection))
            elif any(key in value for key in ("score", "sampling_probability", "selection_reason")):
                selection_records.append(
                    {
                        key: value[key]
                        for key in (
                            "world_slot_role",
                            "variant_id",
                            "score",
                            "sampling_probability",
                            "selection_reason",
                        )
                        if key in value
                    }
                )
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
    return QuestionRolloutPlan(
        question_id=str(question_id),
        policy_version=policy_version,
        realizations=tuple(entries),
        variant_selection=tuple(selection_records),
    )


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
    initial_inputs = {str(metadata_value(row, "initial_input_hash", "")) for row in metadata}
    roles = {slot_role_from_metadata(row) for row in metadata}
    variants = {str(metadata_value(row, "variant_id", "base")) for row in metadata}
    declared_group_sizes: set[int] = set()
    for row in metadata:
        value = metadata_value(row, "decision_group_size", None)
        if value is not None and str(value).strip():
            try:
                declared_group_sizes.add(int(value))
            except (TypeError, ValueError):
                errors.append("invalid_declared_group_size")
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
    if len(initial_inputs) != 1 or "" in initial_inputs:
        errors.append("initial_input_not_frozen")
    if len(roles) != 1 or "" in roles:
        errors.append("world_slot_role_missing")
    if len(variants) != 1 or "" in variants:
        errors.append("variant_not_frozen")
    if declared_group_sizes and declared_group_sizes != {len(items)}:
        errors.append("declared_group_size_mismatch")
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
        "initial_input_hashes": sorted(initial_inputs),
        "world_slot_roles": sorted(roles),
        "variants": sorted(variants),
        "declared_group_sizes": sorted(declared_group_sizes),
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


def validate_question_rollout_plan_records(
    records: Sequence[Any],
    plan: QuestionRolloutPlan | Mapping[str, Any],
    *,
    max_records: int = 48,
) -> dict[str, Any]:
    """Validate records against the explicit per-question realization plan.

    The generic validator checks group-local identity.  This companion check
    verifies the stronger manifest contract: exactly one group for every
    ``(world_slot_role, variant_id)`` realization and exactly the declared K
    records in each group.  It is intentionally read-only and never repairs a
    partial question with records from another question.
    """

    parsed_plan = plan if isinstance(plan, QuestionRolloutPlan) else QuestionRolloutPlan.from_mapping(plan)
    report = validate_bayestool_question_records(
        records,
        allowed_group_sizes=tuple(sorted(ALLOWED_GROUP_SIZES)),
        max_records=min(int(max_records), int(parsed_plan.max_records), 48),
    )
    errors: list[str] = []
    active_records = [
        record
        for record in records
        if not bool(_record_metadata(record).get("dummy_removed_sample"))
        and not bool(_record_metadata(record).get("exclude_from_group_statistics"))
    ]
    question_ids = {question_id_from_metadata(_record_metadata(record)) for record in active_records}
    if question_ids != {parsed_plan.question_id}:
        errors.append("plan_question_id_mismatch")
    if len(active_records) > int(parsed_plan.max_records):
        errors.append("plan_record_limit_exceeded")
    if len(active_records) != int(parsed_plan.record_count):
        errors.append(
            f"plan_record_count_mismatch:{len(active_records)}!={parsed_plan.record_count}"
        )

    groups: dict[str, list[Any]] = defaultdict(list)
    for record in active_records:
        groups[_group_id(_record_metadata(record))].append(record)
    expected = {(item.world_slot_role, item.variant_id): item for item in parsed_plan.realizations}
    actual: dict[tuple[str, str], tuple[str, int]] = {}
    for group_id, group_records in groups.items():
        metadata = _record_metadata(group_records[0])
        pair = (slot_role_from_metadata(metadata), str(metadata_value(metadata, "variant_id", "base")))
        if pair in actual:
            errors.append(f"duplicate_realization_group:{pair[0]}:{pair[1]}")
        actual[pair] = (group_id, len(group_records))
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            errors.append(f"missing_plan_realizations:{missing}")
        if extra:
            errors.append(f"unexpected_plan_realizations:{extra}")

    def _plan_structure_signature(plan_value: QuestionRolloutPlan) -> str:
        payload = plan_value.to_dict()
        for realization in payload.get("realizations", []):
            if isinstance(realization, dict):
                for field in ("selected_decision_event", "decision_prefix_hash", "runtime_state_digest"):
                    realization.pop(field, None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    parsed_plan_structure = _plan_structure_signature(parsed_plan)

    def _record_plan(record: Any) -> tuple[QuestionRolloutPlan, bool] | None:
        metadata = _record_metadata(record)
        raw_plan = metadata.get("question_rollout_plan")
        if not isinstance(raw_plan, Mapping):
            return parsed_plan, False
        try:
            return QuestionRolloutPlan.from_mapping(raw_plan), True
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"record_question_rollout_plan_invalid:{exc}")
            return None

    def _plan_realization(
        plan_value: QuestionRolloutPlan,
        pair: tuple[str, str],
    ) -> RealizationPlan | None:
        return next(
            (
                realization
                for realization in plan_value.realizations
                if (realization.world_slot_role, realization.variant_id) == pair
            ),
            None,
        )

    def _check_record_binding(
        record: Any,
        pair: tuple[str, str],
        *,
        group_id: str,
    ) -> None:
        metadata = _record_metadata(record)
        parsed_record_plan = _record_plan(record)
        if parsed_record_plan is None:
            return
        record_plan, has_explicit_plan = parsed_record_plan
        if record_plan.question_id != parsed_plan.question_id:
            errors.append(
                f"record_plan_question_id_mismatch:{record_plan.question_id}!={parsed_plan.question_id}"
            )
        if has_explicit_plan and _plan_structure_signature(record_plan) != parsed_plan_structure:
            errors.append(f"record_plan_structure_mismatch:{pair[0]}:{pair[1]}:{group_id}")
        realization = _plan_realization(record_plan, pair)
        if realization is None:
            errors.append(f"record_plan_realization_missing:{pair[0]}:{pair[1]}")
            return
        expected_fields = {
            "latent_world_id": realization.latent_world_id,
            "selected_decision_event": realization.selected_decision_event,
            "decision_prefix_hash": realization.decision_prefix_hash,
            "runtime_state_digest": realization.runtime_state_digest,
            "policy_version": realization.policy_version,
        }
        for field, expected_value in expected_fields.items():
            actual_value = str(metadata_value(metadata, field, ""))
            if has_explicit_plan and field in {
                "selected_decision_event",
                "decision_prefix_hash",
                "runtime_state_digest",
            } and not str(expected_value).strip():
                errors.append(
                    f"record_plan_{field}_not_frozen:{pair[0]}:{pair[1]}:{group_id}"
                )
            elif str(expected_value).strip() and actual_value != str(expected_value):
                errors.append(
                    f"record_plan_{field}_mismatch:{pair[0]}:{pair[1]}:{actual_value}!={expected_value}"
                )
        if int(realization.k) != len(groups.get(group_id, ())):
            errors.append(
                f"record_plan_group_size_mismatch:{pair[0]}:{pair[1]}:"
                f"{len(groups.get(group_id, ()))}!={realization.k}"
            )
        actual_slot_weight = metadata_value(metadata, "slot_weight", None)
        if actual_slot_weight is not None:
            try:
                if not math.isclose(
                    float(actual_slot_weight),
                    float(realization.slot_weight),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    errors.append(
                        f"record_plan_slot_weight_mismatch:{pair[0]}:{pair[1]}:"
                        f"{actual_slot_weight}!={realization.slot_weight}"
                    )
            except (TypeError, ValueError):
                errors.append(f"record_plan_slot_weight_invalid:{pair[0]}:{pair[1]}")

    for pair, expected_k in expected.items():
        if pair in actual and actual[pair][1] != int(expected[pair].k):
            errors.append(
                f"plan_group_size_mismatch:{pair[0]}:{pair[1]}:{actual[pair][1]}!={expected[pair].k}"
            )
        if pair not in actual:
            continue
        actual_group_id = actual[pair][0]
        metadata = _record_metadata(groups[actual_group_id][0])
        realization = expected[pair]
        expected_identity = {
            "latent_world_id": realization.latent_world_id,
            "selected_decision_event": realization.selected_decision_event,
            "decision_prefix_hash": realization.decision_prefix_hash,
            "runtime_state_digest": realization.runtime_state_digest,
            "policy_version": realization.policy_version,
        }
        for field, expected_value in expected_identity.items():
            # Root plans created before a runtime checkpoint exists leave the
            # node-specific fields blank.  Once a plan declares one, a record
            # must match it exactly; silently accepting a different node would
            # mix advantages from different prefixes.
            if not str(expected_value).strip():
                continue
            actual_value = str(metadata_value(metadata, field, ""))
            if actual_value != str(expected_value):
                errors.append(
                    f"plan_{field}_mismatch:{pair[0]}:{pair[1]}:{actual_value}!={expected_value}"
                )
        actual_policy = str(metadata_value(metadata, "policy_version", ""))
        if actual_policy != str(realization.policy_version):
            errors.append(
                f"plan_policy_version_mismatch:{pair[0]}:{pair[1]}:{actual_policy}!={realization.policy_version}"
            )
        actual_slot_weight = metadata_value(metadata, "slot_weight", None)
        if actual_slot_weight is None:
            if parsed_plan.latent_ids_finalized:
                errors.append(f"plan_slot_weight_missing:{pair[0]}:{pair[1]}")
        else:
            try:
                if not math.isclose(
                    float(actual_slot_weight),
                    float(realization.slot_weight),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    errors.append(
                        f"plan_slot_weight_mismatch:{pair[0]}:{pair[1]}:{actual_slot_weight}!={realization.slot_weight}"
                    )
            except (TypeError, ValueError):
                errors.append(f"plan_slot_weight_invalid:{pair[0]}:{pair[1]}")
        for record in groups[actual_group_id]:
            _check_record_binding(record, pair, group_id=actual_group_id)
    return {
        **report,
        "plan_valid": not errors,
        "plan_errors": errors,
        "expected_realizations": {
            f"{role}:{variant}": realization.k
            for (role, variant), realization in expected.items()
        },
        "actual_realizations": {
            f"{role}:{variant}": {"group_id": group_id, "size": size}
            for (role, variant), (group_id, size) in actual.items()
        },
        "valid": bool(report.get("valid")) and not errors,
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
    else:
        # Prefer the frozen manifest carried by the records when a caller did
        # not pass an override.  This keeps non-uniform future slot weights
        # faithful to the question plan instead of silently reverting to four
        # equal slots.
        manifest_weights: dict[str, float] = {}
        for _, record in enumerate(records):
            metadata = _record_metadata(record)
            raw_plan = metadata.get("question_rollout_plan")
            if not isinstance(raw_plan, Mapping):
                continue
            raw_slot_weights = raw_plan.get("slot_weights")
            if isinstance(raw_slot_weights, Mapping):
                for role in WORLD_SLOT_ROLES:
                    if role in raw_slot_weights:
                        value = float(raw_slot_weights[role])
                        previous = manifest_weights.get(role)
                        if previous is not None and not math.isclose(previous, value, rel_tol=0.0, abs_tol=1e-6):
                            raise ValueError(f"conflicting BayesTool slot weights for role {role!r}")
                        manifest_weights[role] = value
        if manifest_weights:
            weights_by_role.update(manifest_weights)
    if set(weights_by_role) != set(WORLD_SLOT_ROLES):
        raise ValueError("BayesTool slot weights must contain exactly the four required roles")
    if any(not math.isfinite(value) or value < 0.0 for value in weights_by_role.values()):
        raise ValueError("BayesTool slot weights must be finite and non-negative")
    if not math.isclose(sum(weights_by_role.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"BayesTool slot weights must sum to one, got {weights_by_role}")
    active = [
        (index, record)
        for index, record in enumerate(records)
        if not bool(_record_metadata(record).get("dummy_removed_sample"))
        and not bool(_record_metadata(record).get("exclude_from_group_statistics"))
    ]
    questions = sorted({question_id_from_metadata(_record_metadata(record)) for _, record in active})
    q_count = max(1, int(question_count if question_count is not None else len(questions)))
    if questions and question_count is not None and q_count != len(questions):
        raise ValueError(
            "question_count must equal the active questions in the manifest batch: "
            f"declared={q_count} actual={len(questions)}"
        )
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
    group_weight_sums = {
        group_id: sum(result[index] for index, _ in items) for group_id, items in groups.items()
    }
    group_expected_weights: dict[str, float] = {}
    for group_id, items in groups.items():
        metadata = _record_metadata(items[0][1])
        question = question_id_from_metadata(metadata)
        role = slot_role_from_metadata(metadata)
        variant_count = len(variants_by_question_slot[(question, role)])
        expected = (1.0 / q_count) * float(weights_by_role[role]) / max(1, variant_count)
        group_expected_weights[group_id] = expected
        if abs(group_weight_sums[group_id] - expected) > 1e-6:
            raise AssertionError(
                f"group weight is not invariant to K: {group_id} -> "
                f"{group_weight_sums[group_id]} expected {expected}"
            )
    return result, {
        "question_count": q_count,
        "question_weight_sums": dict(sums_by_question),
        "slot_weight_sums": dict(sums_by_slot),
        "slot_weights": dict(weights_by_role),
        "group_weight_sums": group_weight_sums,
        "group_expected_weight_sums": group_expected_weights,
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
    "K8RollingScheduler",
    "RealizationPlan",
    "QuestionRolloutPlan",
    "make_question_rollout_plan",
    "select_extra_variants",
    "question_id_from_metadata",
    "slot_role_from_metadata",
    "validate_bayestool_question_records",
    "validate_question_rollout_plan_records",
    "compute_hierarchical_loss_weights",
    "make_runtime_state_digest",
]
