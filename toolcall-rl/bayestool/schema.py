"""Typed records shared by the environment, belief, decision, and trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .config import CONTENT_TYPES, TOOL_FAMILIES, TOOL_NAMES


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _finite(value, low)))


@dataclass(frozen=True)
class ToolQualitySpecPatch:
    availability: float | None = None
    semantic_accuracy: float | None = None
    structure_fidelity: float | None = None
    calibration_temperature: float | None = None
    calibration_bias: float | None = None
    relative_cost: float | None = None
    latency_scale: float | None = None

    def apply(self, base: "ToolQualitySpec", weight: float = 1.0) -> "ToolQualitySpec":
        values = base.to_dict()
        weight = _clip(weight)
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                values[name] = values[name] + weight * (_finite(value) - values[name])
        return ToolQualitySpec(**values)

    def to_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ToolQualitySpec:
    availability: float
    semantic_accuracy: float
    structure_fidelity: float
    calibration_temperature: float = 1.0
    calibration_bias: float = 0.0
    relative_cost: float = 1.0
    latency_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "availability", _clip(self.availability))
        object.__setattr__(self, "semantic_accuracy", _clip(self.semantic_accuracy))
        object.__setattr__(self, "structure_fidelity", _clip(self.structure_fidelity))
        object.__setattr__(self, "calibration_temperature", max(0.05, _finite(self.calibration_temperature, 1.0)))
        object.__setattr__(self, "calibration_bias", max(-1.0, min(1.0, _finite(self.calibration_bias))))
        object.__setattr__(self, "relative_cost", max(0.05, _finite(self.relative_cost, 1.0)))
        object.__setattr__(self, "latency_scale", max(0.05, _finite(self.latency_scale, 1.0)))

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SessionStateSpec:
    state: Literal["healthy", "degraded", "overloaded", "outage"] = "healthy"
    latency_scale: float = 1.0
    availability_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "latency_scale": float(self.latency_scale),
            "availability_scale": float(self.availability_scale),
        }


@dataclass(frozen=True)
class SharedFactorSpec:
    family: str
    state: Literal["healthy", "degraded", "down"] = "healthy"
    severity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "state": self.state, "severity": float(self.severity)}


@dataclass(frozen=True)
class ContextRule:
    scope: Literal["document", "page", "region", "content_type"]
    tool_names: tuple[str, ...]
    page_numbers: tuple[int, ...] | None = None
    region: tuple[float, float, float, float] | None = None
    content_types: tuple[str, ...] | None = None
    overrides: ToolQualitySpecPatch = field(default_factory=ToolQualitySpecPatch)
    pending_context_binding: bool = False

    def matches(
        self,
        tool_name: str,
        *,
        page_number: int | None = None,
        page_numbers: Sequence[int] | None = None,
        region: tuple[float, float, float, float] | None = None,
        content_type: str | None = None,
    ) -> bool:
        if tool_name not in self.tool_names:
            return False
        if self.pending_context_binding:
            return False
        if self.scope == "document":
            return True
        if self.scope == "page":
            requested_pages = set(self.page_numbers or ())
            observed_pages = set(int(value) for value in (page_numbers or ()) if value is not None)
            if page_number is not None:
                observed_pages.add(int(page_number))
            return bool(requested_pages & observed_pages)
        if self.scope == "content_type":
            return content_type in (self.content_types or ())
        if region is None or self.region is None:
            return False
        return all(abs(float(a) - float(b)) <= 0.25 for a, b in zip(region, self.region, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "tool_names": list(self.tool_names),
            "page_numbers": list(self.page_numbers) if self.page_numbers is not None else None,
            "region": list(self.region) if self.region is not None else None,
            "content_types": list(self.content_types) if self.content_types is not None else None,
            "overrides": self.overrides.to_dict(),
            "pending_context_binding": bool(self.pending_context_binding),
        }


@dataclass(frozen=True)
class RegimeSegment:
    start_call: int
    end_call: int | None
    transition: Literal["stable", "abrupt", "linear"]
    tool_overrides: dict[str, ToolQualitySpecPatch] = field(default_factory=dict)
    shared_overrides: dict[str, SharedFactorSpec] = field(default_factory=dict)

    def progress(self, call_index: int) -> float:
        if self.end_call is None or self.end_call <= self.start_call:
            return 1.0 if call_index >= self.start_call else 0.0
        # The first call in a gradual schedule is already an effective
        # post-change observation.  Using ``+1`` avoids a zero-effect first
        # step in short rollouts while remaining monotonic and bounded.
        if call_index < self.start_call:
            return 0.0
        if call_index >= self.end_call:
            return 1.0
        return max(0.0, min(1.0, (call_index - self.start_call + 1) / (self.end_call - self.start_call + 1)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_call": self.start_call,
            "end_call": self.end_call,
            "transition": self.transition,
            "tool_overrides": {name: patch.to_dict() for name, patch in self.tool_overrides.items()},
            "shared_overrides": {name: value.to_dict() for name, value in self.shared_overrides.items()},
        }


@dataclass(frozen=True)
class ToolWorldSpec:
    coupling_id: str
    world_id: str
    seed: int
    world_type: str
    session_state: SessionStateSpec
    tool_states: dict[str, ToolQualitySpec]
    shared_factors: dict[str, SharedFactorSpec]
    context_rules: tuple[ContextRule, ...] = ()
    regime_schedule: tuple[RegimeSegment, ...] = ()
    # New identity fields are optional only for loading old manifests.  New
    # samples are required to populate them in ``sample_tool_world``.
    latent_world_id: str = ""
    world_slot: int = 0
    replica_id: int = 0
    latent_seed: int = 0
    # Stable realization identity.  ``replica_id`` is retained only as a
    # continuation index for old manifests; it is not an RL grouping key.
    world_slot_role: str = ""
    variant_id: str = "base"

    def __post_init__(self) -> None:
        missing = set(TOOL_NAMES) - set(self.tool_states)
        if missing:
            raise ValueError(f"tool world is missing tool states: {sorted(missing)}")
        unknown_families = set(self.shared_factors) - set(TOOL_FAMILIES)
        if unknown_families:
            raise ValueError(f"unknown shared tool families: {sorted(unknown_families)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "coupling_id": self.coupling_id,
            "world_id": self.world_id,
            "seed": int(self.seed),
            "world_type": self.world_type,
            "session_state": self.session_state.to_dict(),
            "tool_states": {name: quality.to_dict() for name, quality in self.tool_states.items()},
            "shared_factors": {name: factor.to_dict() for name, factor in self.shared_factors.items()},
            "context_rules": [rule.to_dict() for rule in self.context_rules],
            "regime_schedule": [segment.to_dict() for segment in self.regime_schedule],
            "latent_world_id": self.latent_world_id,
            "world_slot": int(self.world_slot),
            "replica_id": int(self.replica_id),
            "latent_seed": int(self.latent_seed),
            "world_slot_role": self.world_slot_role,
            "variant_id": self.variant_id,
        }

    def to_latent_dict(self) -> dict[str, Any]:
        """Serialize only the hidden world shared by all replicas."""

        return {
            "coupling_id": self.coupling_id,
            "latent_world_id": self.latent_world_id,
            "world_slot": int(self.world_slot),
            "latent_seed": int(self.latent_seed),
            "world_slot_role": self.world_slot_role,
            "variant_id": self.variant_id,
            "world_type": self.world_type,
            "session_state": self.session_state.to_dict(),
            "tool_states": {name: quality.to_dict() for name, quality in self.tool_states.items()},
            "shared_factors": {name: factor.to_dict() for name, factor in self.shared_factors.items()},
            "context_rules": [rule.to_dict() for rule in self.context_rules],
            "regime_schedule": [segment.to_dict() for segment in self.regime_schedule],
        }


@dataclass(frozen=True)
class ToolStateLabel:
    tool_name: str
    quality: ToolQualitySpec
    session_state: str
    shared_states: dict[str, str]
    regime_state: str
    world_id: str
    latent_world_id: str = ""
    replica_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "quality": self.quality.to_dict(),
            "session_state": self.session_state,
            "shared_states": dict(self.shared_states),
            "regime_state": self.regime_state,
            "world_id": self.world_id,
            "latent_world_id": self.latent_world_id,
            "replica_id": int(self.replica_id),
        }


@dataclass(frozen=True)
class WorldEvent:
    call_id: int
    tool_name: str
    status: str
    latency: float
    information_gain: float
    semantic_agreement: float
    schema_valid: bool
    image_valid: bool
    error_family: str = "none"
    corruption_type: str | None = None
    failure_origin: Literal["real_infrastructure", "world_injected", "model_action", "none"] = "none"
    relative_cost: float = 1.0
    page_number: int | None = None
    page_numbers: tuple[int, ...] | None = None
    region: tuple[float, float, float, float] | None = None
    content_type: str | None = None
    predictive_surprise: float = 0.0
    change_detected: bool = False
    observation_status: str = "ok"
    corruption_applied: bool = False
    execution_succeeded: bool = True
    observation_delivered: bool = True

    @property
    def valid_for_rl(self) -> bool:
        return self.failure_origin != "real_infrastructure"

    def visible_dict(self) -> dict[str, Any]:
        """Only the fields that may be shown to the policy."""

        return {
            "tool": self.tool_name,
            "status": self.status,
            "latency": round(float(self.latency), 4),
            "information_gain": round(float(self.information_gain), 4),
            "semantic_agreement": round(float(self.semantic_agreement), 4),
            "schema_valid": bool(self.schema_valid),
            "image_valid": bool(self.image_valid),
            "error_family": self.error_family,
            "page_number": self.page_number,
            "page_numbers": list(self.page_numbers) if self.page_numbers is not None else None,
            "region": list(self.region) if self.region is not None else None,
            "content_type": self.content_type,
            "observation_status": self.observation_status,
            "corruption_applied": bool(self.corruption_applied),
            "execution_succeeded": bool(self.execution_succeeded),
            "observation_delivered": bool(self.observation_delivered),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.visible_dict()
        value.update(
            {
                "call_id": int(self.call_id),
                "corruption_type": self.corruption_type,
                "failure_origin": self.failure_origin,
                "relative_cost": float(self.relative_cost),
                "predictive_surprise": float(self.predictive_surprise),
                "change_detected": bool(self.change_detected),
                "observation_status": self.observation_status,
                "corruption_applied": bool(self.corruption_applied),
                "execution_succeeded": bool(self.execution_succeeded),
                "observation_delivered": bool(self.observation_delivered),
                "valid_for_rl": self.valid_for_rl,
            }
        )
        return value


@dataclass(frozen=True)
class ObservationFeatures:
    values: tuple[float, ...]
    feature_names: tuple[str, ...]
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.values) != len(self.feature_names):
            raise ValueError("observation feature values and names must have the same length")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.feature_names, self.values, strict=True)}


@dataclass(frozen=True)
class TaskStateView:
    question_type: str
    current_page: int | None
    visited_pages: tuple[int, ...]
    unvisited_page_count: int | None
    table_candidate_pages: tuple[int, ...]
    supporting_pages: tuple[int, ...]
    evidence_sufficient: bool
    visual_input_required: bool
    remaining_tool_budget: int
    last_tool: str | None
    last_result_status: str | None
    phase: str = "search"
    content_type: str | None = None
    question: str = ""

    @classmethod
    def from_navigation_state(cls, state: Mapping[str, Any] | None, *, tool_budget: int | None = None) -> "TaskStateView":
        state = state or {}
        visited = tuple(sorted({int(value) for value in state.get("visited_pages", ()) if str(value).isdigit()}))
        unvisited = state.get("unvisited_pages", ())
        unvisited_count = len(unvisited) if isinstance(unvisited, (list, tuple, set)) else None
        budget = state.get("remaining_tool_budget")
        if budget is None:
            budget = tool_budget if tool_budget is not None else state.get("tool_budget", 0)
        return cls(
            question_type=str(state.get("question_type") or "text"),
            current_page=int(state["current_page"]) if str(state.get("current_page", "")).isdigit() else None,
            visited_pages=visited,
            unvisited_page_count=unvisited_count,
            table_candidate_pages=tuple(
                sorted({int(value) for value in state.get("table_candidate_pages", ()) if str(value).isdigit()})
            ),
            supporting_pages=tuple(
                sorted({int(value) for value in state.get("supporting_pages", ()) if str(value).isdigit()})
            ),
            evidence_sufficient=bool(state.get("evidence_sufficient", False)),
            visual_input_required=bool(state.get("visual_input_required", False)),
            remaining_tool_budget=max(0, int(_finite(budget))),
            last_tool=str(state.get("last_tool")) if state.get("last_tool") else None,
            last_result_status=str(state.get("last_result_status")) if state.get("last_result_status") else None,
            phase=str(state.get("phase") or "search"),
            content_type=str(state.get("content_type")) if state.get("content_type") else None,
            question=str(state.get("question") or ""),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "current_page": self.current_page,
            "visited_pages": list(self.visited_pages),
            "unvisited_page_count": self.unvisited_page_count,
            "table_candidate_pages": list(self.table_candidate_pages),
            "supporting_pages": list(self.supporting_pages),
            "evidence_sufficient": self.evidence_sufficient,
            "visual_input_required": self.visual_input_required,
            "remaining_tool_budget": self.remaining_tool_budget,
            "last_tool": self.last_tool,
            "last_result_status": self.last_result_status,
            "phase": self.phase,
            "question": self.question,
        }


@dataclass(frozen=True)
class ToolQualityPosterior:
    availability_mean: float
    availability_std: float
    semantic_mean: float
    semantic_std: float
    structure_mean: float
    structure_std: float
    calibration_mean: float
    calibration_std: float
    cost_mean: float
    cost_std: float

    def to_prompt_dict(self) -> dict[str, list[float]]:
        return {
            "availability": [round(_clip(self.availability_mean), 2), round(max(0.0, self.availability_std), 2)],
            "semantic": [round(_clip(self.semantic_mean), 2), round(max(0.0, self.semantic_std), 2)],
            "structure": [round(_clip(self.structure_mean), 2), round(max(0.0, self.structure_std), 2)],
            "calibration": [round(_clip(self.calibration_mean), 2), round(max(0.0, self.calibration_std), 2)],
            "cost": [round(max(0.05, self.cost_mean), 2), round(max(0.0, self.cost_std), 2)],
        }


@dataclass(frozen=True)
class BeliefSnapshot:
    version: int
    step: int
    session_probs: tuple[float, ...]
    shared_family_probs: dict[str, tuple[float, ...]]
    regime_probs: tuple[float, ...]
    change_probability: float
    tool_quality: dict[str, ToolQualityPosterior]
    posterior_entropy: float
    ood_score: float
    document_hash: str = ""

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "change_probability": round(_clip(self.change_probability), 2),
            "posterior_entropy": round(max(0.0, self.posterior_entropy), 2),
            "ood_score": round(_clip(self.ood_score), 2),
            "tools": {name: self.tool_quality[name].to_prompt_dict() for name in TOOL_NAMES if name in self.tool_quality},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "step": self.step,
            "session_probs": list(self.session_probs),
            "shared_family_probs": {name: list(values) for name, values in self.shared_family_probs.items()},
            "regime_probs": list(self.regime_probs),
            "change_probability": self.change_probability,
            "tool_quality": {name: value.to_prompt_dict() for name, value in self.tool_quality.items()},
            "posterior_entropy": self.posterior_entropy,
            "ood_score": self.ood_score,
            "document_hash": self.document_hash,
        }


@dataclass(frozen=True)
class ObservationPrediction:
    status_probs: tuple[float, ...]
    latency_probs: tuple[float, ...]
    information_gain_probs: tuple[float, ...]
    semantic_agreement_probs: tuple[float, ...]
    schema_valid_prob: float
    image_valid_prob: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": list(self.status_probs),
            "latency_bin": list(self.latency_probs),
            "information_gain": list(self.information_gain_probs),
            "semantic_agreement": list(self.semantic_agreement_probs),
            "schema_valid": self.schema_valid_prob,
            "image_valid": self.image_valid_prob,
        }


@dataclass(frozen=True)
class PosteriorParticle:
    particle_id: int
    weight: float
    session_state: str
    regime_state: str
    shared_states: dict[str, str]
    tool_quality: dict[str, ToolQualitySpec]


@dataclass(frozen=True)
class ActionValue:
    action_key: str
    value_mean: float
    value_log_variance: float
    particle_values: tuple[float, ...] = ()

    @property
    def value_variance(self) -> float:
        import math

        return math.exp(max(-20.0, min(20.0, self.value_log_variance)))


@dataclass(frozen=True)
class BranchRecord:
    coupling_id: str
    sibling_group_id: str
    prefix_hash: str
    world_id: str
    belief_version: int
    action_key: str
    utility: float
    oracle_action: str | None = None
    oracle_utility: float | None = None
    policy_utility: float | None = None
    relative_regret: float | None = None
    horizon: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SwitchPair:
    content_signature: str
    coupling_id: str
    state_u: dict[str, Any]
    state_v: dict[str, Any]
    action_u: str
    action_v: str
    return_u: float
    return_v: float
    belief_js: float
    ood_u: float
    ood_v: float
    first_distinguishing_event_step: int | None
    action_u_text: str | None = None
    action_v_text: str | None = None


__all__ = [
    "ToolQualitySpecPatch",
    "ToolQualitySpec",
    "SessionStateSpec",
    "SharedFactorSpec",
    "ContextRule",
    "RegimeSegment",
    "ToolWorldSpec",
    "ToolStateLabel",
    "WorldEvent",
    "ObservationFeatures",
    "TaskStateView",
    "ToolQualityPosterior",
    "BeliefSnapshot",
    "ObservationPrediction",
    "PosteriorParticle",
    "ActionValue",
    "BranchRecord",
    "SwitchPair",
]
