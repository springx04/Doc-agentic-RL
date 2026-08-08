"""Coupled, target-independent non-stationary tool worlds."""

from __future__ import annotations

import copy
import asyncio
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import (
    CONTENT_TYPES,
    SESSION_STATES,
    TOOL_FAMILIES,
    TOOL_NAMES,
    BayesToolConfig,
    default_config,
)
from .schema import (
    ContextRule,
    RegimeSegment,
    SessionStateSpec,
    SharedFactorSpec,
    ToolQualitySpec,
    ToolQualitySpecPatch,
    ToolStateLabel,
    ToolWorldSpec,
    WorldEvent,
)


HIDDEN_OBSERVATION_KEYS = {
    "world_id",
    "true_quality",
    "corruption_type",
    "clean_result",
    "clean_confidence",
    "ground_truth_state",
    "answer_page",
    "answer_bbox",
    "label",
    "answers",
}

WORLD_TYPES: tuple[str, ...] = (
    "healthy",
    "single_tool_degradation",
    "context_degradation",
    "shared_family_fault",
    "abrupt_change",
    "gradual_change",
)


@dataclass(frozen=True)
class WorldSamplingContext:
    """Public task information allowed to influence world sampling.

    This deliberately contains capabilities and budgets, never labels,
    answer locations, or answer text.  Keeping it as a typed record makes it
    difficult for a caller to accidentally pass simulator ground truth into
    the sampler.
    """

    page_count: int | None = None
    tool_argument_capabilities: dict[str, frozenset[str]] | Mapping[str, Sequence[str]] = ()
    tool_budget: int = 8

    def __post_init__(self) -> None:
        page_count = None if self.page_count is None else max(1, int(self.page_count))
        object.__setattr__(self, "page_count", page_count)
        raw = self.tool_argument_capabilities
        if isinstance(raw, Mapping):
            capabilities = {
                str(tool): frozenset(str(argument) for argument in arguments)
                for tool, arguments in raw.items()
            }
        else:
            capabilities = {}
        object.__setattr__(self, "tool_argument_capabilities", capabilities)
        object.__setattr__(self, "tool_budget", max(1, int(self.tool_budget)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, tool_budget: int = 8) -> "WorldSamplingContext":
        value = value or {}
        return cls(
            page_count=value.get("page_count"),
            tool_argument_capabilities=value.get("tool_argument_capabilities", {}),
            tool_budget=int(value.get("tool_budget", tool_budget) or tool_budget),
        )

    def supports(self, tool_name: str, *arguments: str) -> bool:
        available = self.tool_argument_capabilities.get(tool_name, frozenset())
        return all(argument in available for argument in arguments)


DEFAULT_TOOL_ARGUMENT_CAPABILITIES: dict[str, frozenset[str]] = {
    "parse_document": frozenset({"page", "page_number", "page_numbers", "region", "bbox"}),
    "detect_layout": frozenset({"page", "page_number", "region", "bbox"}),
    "render_page": frozenset({"page", "page_number"}),
    "crop_region": frozenset({"page", "page_number", "region", "bbox"}),
    "zoom_region": frozenset({"page", "page_number", "region", "bbox"}),
    "ocr_region": frozenset({"page", "page_number", "region", "bbox"}),
    "extract_table": frozenset({"page", "page_number", "region", "bbox"}),
    "chart_to_table": frozenset({"page", "page_number", "region", "bbox"}),
}


def _quality_patch_from_dict(value: Any) -> ToolQualitySpecPatch:
    raw = value if isinstance(value, Mapping) else {}
    return ToolQualitySpecPatch(
        **{
            name: raw.get(name)
            for name in ToolQualitySpecPatch.__dataclass_fields__
            if name in raw
        }
    )


def tool_world_spec_from_dict(value: Mapping[str, Any]) -> ToolWorldSpec:
    """Rehydrate a manifest world without resampling its hidden state."""

    if not isinstance(value, Mapping):
        raise TypeError("world spec must be a mapping")
    session_raw = value.get("session_state") if isinstance(value.get("session_state"), Mapping) else {}
    session_state = SessionStateSpec(
        state=str(session_raw.get("state", "healthy")),
        latency_scale=float(session_raw.get("latency_scale", 1.0)),
        availability_scale=float(session_raw.get("availability_scale", 1.0)),
    )
    tool_states_raw = value.get("tool_states") if isinstance(value.get("tool_states"), Mapping) else {}
    tool_states = {
        str(name): ToolQualitySpec(**dict(spec))
        for name, spec in tool_states_raw.items()
        if isinstance(spec, Mapping)
    }
    shared_raw = value.get("shared_factors") if isinstance(value.get("shared_factors"), Mapping) else {}
    shared_factors = {
        str(name): SharedFactorSpec(
            family=str(spec.get("family", name)),
            state=str(spec.get("state", "healthy")),
            severity=float(spec.get("severity", 0.0)),
        )
        for name, spec in shared_raw.items()
        if isinstance(spec, Mapping)
    }
    context_rules: list[ContextRule] = []
    for raw_rule in value.get("context_rules", ()) or ():
        if not isinstance(raw_rule, Mapping):
            continue
        region = raw_rule.get("region")
        context_rules.append(
            ContextRule(
                scope=str(raw_rule.get("scope", "document")),
                tool_names=tuple(str(item) for item in raw_rule.get("tool_names", ()) or ()),
                page_numbers=(
                    tuple(int(item) for item in raw_rule.get("page_numbers", ()) or ())
                    if raw_rule.get("page_numbers") is not None
                    else None
                ),
                region=tuple(float(item) for item in region) if isinstance(region, (list, tuple)) else None,
                content_types=(
                    tuple(str(item) for item in raw_rule.get("content_types", ()) or ())
                    if raw_rule.get("content_types") is not None
                    else None
                ),
                overrides=_quality_patch_from_dict(raw_rule.get("overrides")),
                pending_context_binding=bool(raw_rule.get("pending_context_binding", False)),
            )
        )
    regime_schedule: list[RegimeSegment] = []
    for raw_segment in value.get("regime_schedule", ()) or ():
        if not isinstance(raw_segment, Mapping):
            continue
        tool_overrides_raw = raw_segment.get("tool_overrides")
        tool_overrides = {
            str(name): _quality_patch_from_dict(spec)
            for name, spec in (tool_overrides_raw.items() if isinstance(tool_overrides_raw, Mapping) else ())
        }
        shared_overrides_raw = raw_segment.get("shared_overrides")
        shared_overrides = {
            str(name): SharedFactorSpec(
                family=str(spec.get("family", name)),
                state=str(spec.get("state", "healthy")),
                severity=float(spec.get("severity", 0.0)),
            )
            for name, spec in (shared_overrides_raw.items() if isinstance(shared_overrides_raw, Mapping) else ())
            if isinstance(spec, Mapping)
        }
        regime_schedule.append(
            RegimeSegment(
                start_call=int(raw_segment.get("start_call", 0)),
                end_call=(
                    int(raw_segment["end_call"])
                    if raw_segment.get("end_call") is not None
                    else None
                ),
                transition=str(raw_segment.get("transition", "abrupt")),
                tool_overrides=tool_overrides,
                shared_overrides=shared_overrides,
            )
        )
    return ToolWorldSpec(
        coupling_id=str(value.get("coupling_id", "")),
        world_id=str(value.get("world_id", "")),
        seed=int(value.get("seed", 0)),
        world_type=str(value.get("world_type", "healthy")),
        session_state=session_state,
        tool_states=tool_states,
        shared_factors=shared_factors,
        context_rules=tuple(context_rules),
        regime_schedule=tuple(regime_schedule),
        latent_world_id=str(value.get("latent_world_id", "")),
        world_slot=int(value.get("world_slot", 0)),
        replica_id=int(value.get("replica_id", 0)),
        latent_seed=int(value.get("latent_seed", value.get("seed", 0))),
    )


def sample_session_state(
    coupling_id: str,
    *,
    rollout_id: int | str = 0,
    world_slot: int = 0,
    config: BayesToolConfig | None = None,
    force_healthy: bool = False,
) -> SessionStateSpec:
    """Sample the session-level backend regime shared by all tool calls.

    The state is keyed by world slot rather than replica, so stochastic
    replicas observe the same session regime while retaining independent
    quality/corruption draws.  A healthy coverage slot is kept genuinely
    healthy; other slots use the configured four-state prior.
    """

    config = config or default_config(enabled=True)
    if force_healthy:
        return SessionStateSpec()
    configured = dict(getattr(config, "session_state_probabilities", ()) or ())
    weights = [max(0.0, float(configured.get(name, 0.0))) for name in SESSION_STATES]
    if sum(weights) <= 0.0:
        weights = [0.70, 0.20, 0.08, 0.02]
    rng = random.Random(stable_seed(coupling_id, rollout_id, world_slot, "session-state"))
    state = str(rng.choices(SESSION_STATES, weights=weights, k=1)[0])
    if state == "degraded":
        return SessionStateSpec(
            state=state,
            latency_scale=_uniform(rng, 1.25, 1.80),
            availability_scale=_uniform(rng, 0.70, 0.90),
        )
    if state == "overloaded":
        return SessionStateSpec(
            state=state,
            latency_scale=_uniform(rng, 1.80, 3.00),
            availability_scale=_uniform(rng, 0.45, 0.75),
        )
    if state == "outage":
        return SessionStateSpec(
            state=state,
            latency_scale=_uniform(rng, 3.00, 5.00),
            availability_scale=_uniform(rng, 0.15, 0.45),
        )
    return SessionStateSpec()


def stable_seed(*parts: Any) -> int:
    """Derive a process-independent 64-bit seed from public rollout identity."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def document_hash(document: str | Path | None) -> str:
    """Hash a document path's bytes when available, otherwise hash its identity."""

    if document is None:
        return hashlib.sha256(b"<missing-document>").hexdigest()
    path = Path(str(document))
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    return hashlib.sha256(str(document).encode("utf-8", "surrogatepass")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return float(low + (high - low) * rng.random())


def _quality(
    rng: random.Random,
    *,
    healthy: bool = False,
    degraded: bool = False,
) -> ToolQualitySpec:
    if healthy:
        availability = _uniform(rng, 0.99, 1.0)
        semantic = _uniform(rng, 0.95, 1.0)
        structure = _uniform(rng, 0.95, 1.0)
        cost = _uniform(rng, 0.90, 1.10)
    elif degraded:
        availability = _uniform(rng, 0.35, 0.85)
        semantic = _uniform(rng, 0.35, 0.85)
        structure = _uniform(rng, 0.35, 0.85)
        cost = _uniform(rng, 1.05, 2.2)
    else:
        availability = _uniform(rng, 0.82, 0.98)
        semantic = _uniform(rng, 0.82, 0.98)
        structure = _uniform(rng, 0.82, 0.98)
        cost = _uniform(rng, 0.95, 1.25)
    return ToolQualitySpec(
        availability=availability,
        semantic_accuracy=semantic,
        structure_fidelity=structure,
        calibration_temperature=_uniform(rng, 0.85, 1.20),
        calibration_bias=_uniform(rng, -0.08, 0.08),
        relative_cost=cost,
        latency_scale=_uniform(rng, 0.85, 1.25) if not degraded else _uniform(rng, 1.0, 3.0),
    )


def _clean_quality() -> ToolQualitySpec:
    """Return the shared clean baseline used before world faults are applied."""

    return ToolQualitySpec(
        availability=1.0,
        semantic_accuracy=1.0,
        structure_fidelity=1.0,
        calibration_temperature=1.0,
        calibration_bias=0.0,
        relative_cost=1.0,
        latency_scale=1.0,
    )


def _latent_identity(coupling_id: str, rollout_id: int | str, world_slot: int, world_type: str) -> str:
    payload = f"{coupling_id}\x1f{rollout_id}\x1f{int(world_slot)}\x1f{world_type}".encode(
        "utf-8", "surrogatepass"
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _context_candidate_tools(sampling_context: WorldSamplingContext) -> tuple[str, ...]:
    capabilities = sampling_context.tool_argument_capabilities
    if not capabilities:
        capabilities = DEFAULT_TOOL_ARGUMENT_CAPABILITIES
    page_capable = tuple(
        name
        for name in TOOL_NAMES
        if name in capabilities
        and bool(capabilities[name] & frozenset({"page", "page_number", "page_numbers", "region", "bbox"}))
    )
    return page_capable or tuple(TOOL_NAMES)


def _multiply_quality(base: ToolQualitySpec, factor: float) -> ToolQualitySpec:
    return ToolQualitySpec(
        availability=max(0.01, min(1.0, base.availability * factor)),
        semantic_accuracy=max(0.01, min(1.0, base.semantic_accuracy * factor)),
        structure_fidelity=max(0.01, min(1.0, base.structure_fidelity * factor)),
        calibration_temperature=base.calibration_temperature + (1.0 - factor) * 0.25,
        calibration_bias=base.calibration_bias,
        relative_cost=base.relative_cost * (1.0 + (1.0 - factor) * 0.8),
        latency_scale=base.latency_scale * (1.0 + (1.0 - factor) * 1.5),
    )


def sample_world_type(
    coupling_id: str,
    *,
    rollout_id: int | str = 0,
    world_slot: int = 0,
    replica_id: int = 0,
    config: BayesToolConfig | None = None,
) -> str:
    """Sample a deterministic world regime from the configured distribution."""

    config = config or default_config(enabled=True)
    configured = dict(getattr(config, "world_type_probabilities", ()) or ())
    weights = [max(0.0, float(configured.get(name, 0.0))) for name in WORLD_TYPES]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(WORLD_TYPES)
    # A replica is a stochastic re-run of the same world slot.  Its hidden
    # regime must therefore be shared with the slot; only the later quality
    # and corruption draws use replica_id.
    rng = random.Random(stable_seed(coupling_id, rollout_id, world_slot, "world-type"))
    return str(rng.choices(WORLD_TYPES, weights=weights, k=1)[0])


def _coverage_world_types(
    coupling_id: str,
    *,
    rollout_id: int | str,
    config: BayesToolConfig,
) -> tuple[str, str, str, str]:
    """Build four covered slots while using configured local/change weights."""

    configured = dict(getattr(config, "world_type_probabilities", ()) or ())
    local = random.Random(stable_seed(coupling_id, rollout_id, "local-world-type"))
    change = random.Random(stable_seed(coupling_id, rollout_id, "change-world-type"))
    local_weights = (
        max(0.0, float(configured.get("single_tool_degradation", 0.0))),
        max(0.0, float(configured.get("context_degradation", 0.0))),
    )
    change_weights = (
        max(0.0, float(configured.get("abrupt_change", 0.0))),
        max(0.0, float(configured.get("gradual_change", 0.0))),
    )
    if sum(local_weights) <= 0.0:
        local_weights = (1.0, 1.0)
    if sum(change_weights) <= 0.0:
        change_weights = (1.0, 1.0)
    local_type = str(
        local.choices(
            ("single_tool_degradation", "context_degradation"),
            weights=local_weights,
            k=1,
        )[0]
    )
    change_type = str(
        change.choices(
            ("abrupt_change", "gradual_change"),
            weights=change_weights,
            k=1,
        )[0]
    )
    slots = ["healthy", local_type, "shared_family_fault", change_type]
    random.Random(stable_seed(coupling_id, rollout_id, "world-slot-order")).shuffle(slots)
    return tuple(slots)  # type: ignore[return-value]


def _default_shared_factors() -> dict[str, SharedFactorSpec]:
    return {name: SharedFactorSpec(name, "healthy", 0.0) for name in TOOL_FAMILIES}


def sample_tool_world(
    coupling_id: str,
    *,
    world_slot: int,
    replica_id: int = 0,
    rollout_id: int | str = 0,
    config: BayesToolConfig | None = None,
    world_type: str | None = None,
    sampling_context: WorldSamplingContext | Mapping[str, Any] | None = None,
    tool_budget: int | None = None,
) -> ToolWorldSpec:
    """Create one deterministic world without accessing task labels or answers."""

    config = config or default_config(enabled=True)
    public_context = (
        sampling_context
        if isinstance(sampling_context, WorldSamplingContext)
        else WorldSamplingContext.from_mapping(sampling_context, tool_budget=tool_budget or 8)
    )
    if tool_budget is not None and not isinstance(sampling_context, WorldSamplingContext):
        public_context = WorldSamplingContext(
            page_count=public_context.page_count,
            tool_argument_capabilities=public_context.tool_argument_capabilities,
            tool_budget=tool_budget,
        )
    latent_seed = stable_seed(coupling_id, rollout_id, world_slot, "latent-world")
    # ``seed`` remains replica-specific for backwards-compatible manifests;
    # hidden world state below is sampled exclusively from ``latent_seed``.
    seed = stable_seed(latent_seed, "replica-observation", replica_id)
    rng = random.Random(latent_seed)
    world_types = WORLD_TYPES
    if world_type is not None:
        selected_type = sample_world_type(
            coupling_id,
            rollout_id=rollout_id,
            world_slot=world_slot,
            replica_id=replica_id,
            config=config,
        ) if str(world_type).casefold() == "sampled" else world_type
    else:
        # The default four slots are deliberately heterogeneous: one healthy
        # world, one local/context world, one shared-family world, and one
        # non-stationary world. The local/change choices and slot ordering are
        # deterministic but use the configured training distribution; replicas
        # keep the same slot semantics while varying only corruption draws.
        if int(world_slot) >= 4:
            selected_type = sample_world_type(
                coupling_id,
                rollout_id=rollout_id,
                world_slot=world_slot,
                replica_id=replica_id,
                config=config,
            )
        else:
            selected_type = _coverage_world_types(
                coupling_id,
                rollout_id=rollout_id,
                config=config,
            )[int(world_slot)]
    if selected_type not in world_types:
        raise ValueError(f"unknown BayesTool world type: {selected_type}")

    qualities = {name: _clean_quality() for name in TOOL_NAMES}
    shared = _default_shared_factors()
    context_rules: list[ContextRule] = []
    schedule: list[RegimeSegment] = []
    session = sample_session_state(
        coupling_id,
        rollout_id=rollout_id,
        world_slot=world_slot,
        config=config,
        force_healthy=selected_type == "healthy",
    )

    if selected_type == "single_tool_degradation":
        target = TOOL_NAMES[rng.randrange(len(TOOL_NAMES))]
        qualities[target] = _quality(rng, degraded=True)
    elif selected_type == "context_degradation":
        candidates = _context_candidate_tools(public_context)
        target = candidates[rng.randrange(len(candidates))]
        page_count = public_context.page_count or 1
        page = rng.randint(1, page_count)
        capabilities = public_context.tool_argument_capabilities or DEFAULT_TOOL_ARGUMENT_CAPABILITIES
        can_bind_page = bool(
            capabilities.get(target, frozenset())
            & frozenset({"page", "page_number", "page_numbers"})
        )
        context_rules.append(
            ContextRule(
                scope="page",
                tool_names=(target,),
                page_numbers=(page,) if can_bind_page else None,
                overrides=ToolQualitySpecPatch(
                    availability=_uniform(rng, 0.35, 0.75),
                    semantic_accuracy=_uniform(rng, 0.35, 0.85),
                    structure_fidelity=_uniform(rng, 0.35, 0.85),
                    relative_cost=_uniform(rng, 1.2, 2.0),
                ),
                pending_context_binding=not can_bind_page,
            )
        )
    elif selected_type == "shared_family_fault":
        family = tuple(TOOL_FAMILIES)[rng.randrange(len(TOOL_FAMILIES))]
        severity = _uniform(rng, 0.35, 0.75)
        shared[family] = SharedFactorSpec(family, "degraded" if severity < 0.65 else "down", severity)
    elif selected_type == "abrupt_change":
        # Keep one post-change decision opportunity in the normal budgeted
        # rollout.  Call ids are zero-based, so starting at budget-2 leaves a
        # later call available for the policy to react to the change.
        latest_start = max(1, public_context.tool_budget - 2)
        call = rng.randint(1, latest_start)
        target = TOOL_NAMES[rng.randrange(len(TOOL_NAMES))]
        schedule.append(
            RegimeSegment(
                start_call=call,
                end_call=None,
                transition="abrupt",
                tool_overrides={
                    target: ToolQualitySpecPatch(
                        availability=_uniform(rng, 0.35, 0.75),
                        semantic_accuracy=_uniform(rng, 0.35, 0.80),
                        structure_fidelity=_uniform(rng, 0.35, 0.80),
                        relative_cost=_uniform(rng, 1.2, 2.0),
                    )
                },
            )
        )
    elif selected_type == "gradual_change":
        latest_start = max(1, public_context.tool_budget - 2)
        start = rng.randint(1, latest_start)
        latest_end = max(start, public_context.tool_budget - 1)
        end = rng.randint(start, latest_end)
        family = tuple(TOOL_FAMILIES)[rng.randrange(len(TOOL_FAMILIES))]
        schedule.append(
            RegimeSegment(
                start_call=start,
                end_call=end,
                transition="linear",
                shared_overrides={
                    family: SharedFactorSpec(family, "degraded", _uniform(rng, 0.45, 0.80))
                },
            )
        )

    return ToolWorldSpec(
        coupling_id=str(coupling_id),
        world_id=f"{coupling_id}:slot={world_slot}:replica={replica_id}:type={selected_type}",
        seed=seed,
        world_type=selected_type,
        session_state=session,
        tool_states=qualities,
        shared_factors=shared,
        context_rules=tuple(context_rules),
        regime_schedule=tuple(schedule),
        latent_world_id=_latent_identity(coupling_id, rollout_id, world_slot, selected_type),
        world_slot=int(world_slot),
        replica_id=int(replica_id),
        latent_seed=int(latent_seed),
    )


class CleanResultCache:
    """Process and shared-disk cache for one clean tool execution.

    The in-memory layer removes duplicate awaits inside one rollout worker.
    When ``root`` is configured, the async path also uses an atomic lock file
    and an atomic result replace so coupled Ray workers share the same clean
    result instead of executing the real tool independently.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self._values: dict[str, Any] = {}
        self.lock_timeout_seconds = 30.0 * 60.0
        self.poll_interval_seconds = 0.05

    @staticmethod
    def key(document_digest: str, tool_name: str, arguments: Mapping[str, Any], backend_version: str) -> str:
        return "|".join((document_digest, str(tool_name), canonical_json(arguments), str(backend_version)))

    def set_root(self, root: str | Path | None) -> None:
        """Set the shared cache root once for a rollout worker."""

        if self.root is None and root:
            self.root = Path(root)

    def _disk_paths(self, key: str) -> tuple[Path, Path] | None:
        if self.root is None:
            return None
        digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()
        return self.root / f"{digest}.json", self.root / f"{digest}.lock"

    def _read_disk(self, key: str) -> tuple[bool, Any]:
        paths = self._disk_paths(key)
        if paths is None:
            return False, None
        result_path, _ = paths
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and "value" in payload:
                return True, payload["value"]
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
            return False, None
        return False, None

    def _write_disk(self, key: str, value: Any) -> None:
        paths = self._disk_paths(key)
        if paths is None:
            return
        result_path, _ = paths
        temporary_path = result_path.with_name(
            f"{result_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(
                json.dumps({"value": value}, ensure_ascii=False, separators=(",", ":"), default=str),
                encoding="utf-8",
            )
            os.replace(temporary_path, result_path)
        except (OSError, TypeError, ValueError):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _try_acquire_disk_lock(self, lock_path: Path) -> bool:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            return True
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > self.lock_timeout_seconds:
                    lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        except OSError:
            # A cache write must never make a valid tool call fail.  Falling
            # back to the process-local layer is safer than blocking forever
            # on an unavailable output filesystem.
            return True

    @staticmethod
    def _release_disk_lock(lock_path: Path) -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def get_or_set(
        self,
        document_digest: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        backend_version: str,
        producer: Callable[[], Any],
    ) -> Any:
        key = self.key(document_digest, tool_name, arguments, backend_version)
        if key in self._values:
            return self._values[key]
        hit, value = self._read_disk(key)
        if hit:
            self._values[key] = value
            return value
        self._values[key] = producer()
        self._write_disk(key, self._values[key])
        return self._values[key]

    async def get_or_set_async(
        self,
        document_digest: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        backend_version: str,
        producer: Callable[[], Any],
    ) -> Any:
        """Async variant that holds an inter-process lock across production."""

        key = self.key(document_digest, tool_name, arguments, backend_version)
        if key in self._values:
            return self._values[key]
        hit, value = self._read_disk(key)
        if hit:
            self._values[key] = value
            return value
        paths = self._disk_paths(key)
        if paths is None:
            produced = producer()
            value = await produced if inspect.isawaitable(produced) else produced
            self._values[key] = value
            return value

        _, lock_path = paths
        acquired = False
        while not acquired:
            hit, value = self._read_disk(key)
            if hit:
                self._values[key] = value
                return value
            acquired = self._try_acquire_disk_lock(lock_path)
            if not acquired:
                await asyncio.sleep(self.poll_interval_seconds)
        try:
            # Another producer may have finished between the initial read and
            # lock acquisition.  Always re-check before invoking the tool.
            hit, value = self._read_disk(key)
            if hit:
                self._values[key] = value
                return value
            produced = producer()
            value = await produced if inspect.isawaitable(produced) else produced
            self._values[key] = value
            self._write_disk(key, value)
            return value
        finally:
            self._release_disk_lock(lock_path)


def _value_error_family(tool_name: str, corruption: str | None) -> str:
    if corruption:
        if tool_name in {"render_page", "crop_region", "zoom_region"}:
            return "render"
        if tool_name == "ocr_region":
            return "ocr"
        if tool_name in {"detect_layout", "extract_table", "chart_to_table"}:
            return "structure"
        return "semantic"
    return "none"


def _strip_hidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_hidden(item)
            for key, item in value.items()
            if str(key).casefold() not in {item.casefold() for item in HIDDEN_OBSERVATION_KEYS}
        }
    if isinstance(value, list):
        return [_strip_hidden(item) for item in value]
    return value


def sanitize_observed_result(result: str) -> str:
    """Remove simulator-only fields while preserving ordinary document evidence."""

    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        cleaned = str(result)
        for key in HIDDEN_OBSERVATION_KEYS:
            cleaned = re.sub(rf"(?im)^\s*{re.escape(key)}\s*:\s*[^\n]+\n?", "", cleaned)
        return cleaned
    return json.dumps(_strip_hidden(parsed), ensure_ascii=False, sort_keys=True)


def _replace_digits(text: str, rng: random.Random, probability: float) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        chars = list(value)
        for index, char in enumerate(chars):
            if char.isdigit() and rng.random() < probability:
                chars[index] = str(rng.randrange(10))
        return "".join(chars)

    return re.sub(r"[-+]?\d+(?:[.,]\d+)*", replace, text)


def _mutate_structured_observation(value: Any, corruption: str, severity: float, rng: random.Random, *, key: str = "") -> Any:
    """Apply target-independent mutations to every matching structured field."""

    key_name = str(key).casefold()
    if corruption == "bbox_jitter" and key_name in {"bbox", "bounding_box", "box", "region", "coordinates"}:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [float(item) + rng.uniform(-0.08, 0.08) * max(1.0, abs(float(item))) for item in value]
            except (TypeError, ValueError):
                return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for name, item in value.items():
            name_key = str(name).casefold()
            if corruption == "type_misclassification" and name_key in {"type", "kind", "category", "content_type"}:
                candidates = ["text", "table", "figure", "formula", "chart"]
                current = str(item).casefold()
                alternatives = [candidate for candidate in candidates if candidate != current]
                result[name] = rng.choice(alternatives) if alternatives and rng.random() < min(0.9, severity + 0.2) else item
                continue
            if corruption == "confidence_miscalibration" and ("confidence" in name_key or name_key in {"score", "probability"}):
                try:
                    numeric = float(item)
                except (TypeError, ValueError):
                    result[name] = _mutate_structured_observation(item, corruption, severity, rng, key=name_key)
                else:
                    delta = rng.uniform(-max(0.05, severity), max(0.05, severity))
                    result[name] = max(0.0, min(1.0, numeric + delta))
                continue
            result[name] = _mutate_structured_observation(item, corruption, severity, rng, key=name_key)
        if corruption == "region_miss":
            # Remove a random subset of region/layout records, never a
            # label-selected record.  If there is only one record, retain it
            # but mark it as unavailable through the surrounding event.
            for name, item in list(result.items()):
                if isinstance(item, list) and len(item) > 1 and all(isinstance(entry, Mapping) for entry in item):
                    result[name] = [entry for entry in item if rng.random() > min(0.75, severity)] or item[:1]
        return result
    if isinstance(value, list):
        return [_mutate_structured_observation(item, corruption, severity, rng, key=key_name) for item in value]
    return value


def _mutate_observation_text(value: Any, rng: random.Random, corruption: str, severity: float) -> Any:
    """Mutate evidence fields without changing the surrounding JSON schema."""

    if isinstance(value, dict):
        return {
            name: _mutate_observation_text(item, rng, corruption, severity)
            if str(name).casefold() in {"text", "markdown", "value", "content", "label", "name"}
            else _mutate_observation_text(item, rng, corruption, severity)
            for name, item in value.items()
        }
    if isinstance(value, list):
        items = [_mutate_observation_text(item, rng, corruption, severity) for item in value]
        if corruption in {"page_order_jitter", "line_order", "row_column_swap", "category_swap"} and len(items) > 1:
            rng.shuffle(items)
        return items
    if not isinstance(value, str):
        return value
    if corruption in {"character_deletion", "paragraph_omission", "page_miss"}:
        if len(value) <= 12:
            return value
        remove = max(1, min(len(value) // 4, int(len(value) * severity * 0.25)))
        indexes = set(rng.sample(range(len(value)), min(remove, len(value))))
        return "".join(char for index, char in enumerate(value) if index not in indexes)
    if corruption in {"character_substitution", "digit_confusion", "numeric_noise", "scale_bias"}:
        return _replace_digits(value, rng, min(0.55, severity * 0.9))
    if corruption == "text_fallback":
        return value.replace("|", " ")
    return value


class ObservationCorruptionAdapter:
    """Tool-schema-aware, target-independent observation corruption.

    The adapter mutates only fields that are already present in a clean tool
    response.  In particular it never creates an answer-like field, and JSON
    responses remain JSON after corruption so protocol errors are not silently
    confused with a bad tool observation.
    """

    @staticmethod
    def _choose(tool_name: str, rng: random.Random) -> str:
        if tool_name == "parse_document":
            return rng.choice(("paragraph_omission", "character_substitution", "page_order_jitter", "page_miss"))
        if tool_name == "render_page":
            return rng.choice(("blur", "downsample", "occlusion", "compression_noise", "rotation"))
        if tool_name in {"crop_region", "zoom_region"}:
            return rng.choice(("bbox_shift", "scale_error", "edge_crop", "downsample"))
        if tool_name == "ocr_region":
            return rng.choice(("character_deletion", "character_substitution", "digit_confusion", "line_order", "confidence_miscalibration"))
        if tool_name == "detect_layout":
            return rng.choice(("bbox_jitter", "region_miss", "region_duplicate", "type_misclassification"))
        if tool_name == "extract_table":
            return rng.choice(("row_omission", "column_omission", "row_column_swap", "cell_merge", "text_fallback"))
        return rng.choice(("series_omission", "scale_bias", "category_swap", "numeric_noise"))

    @classmethod
    def corrupt(
        cls,
        tool_name: str,
        text: str,
        quality: ToolQualitySpec,
        rng: random.Random,
    ) -> tuple[str, str | None, bool, bool]:
        severity = max(0.0, min(0.85, 1.0 - min(quality.semantic_accuracy, quality.structure_fidelity)))
        if severity <= 0.025:
            return text, None, True, False
        corruption = cls._choose(tool_name, rng)
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            parsed = None

        if parsed is not None:
            if corruption in {"bbox_jitter", "region_miss", "type_misclassification", "confidence_miscalibration"}:
                mutated = _mutate_structured_observation(parsed, corruption, severity, rng)
            elif corruption in {
                "paragraph_omission", "character_substitution", "page_order_jitter", "page_miss",
                "character_deletion", "digit_confusion", "line_order", "row_column_swap",
                "row_omission", "column_omission", "cell_merge", "text_fallback",
                "series_omission", "scale_bias", "category_swap", "numeric_noise",
            }:
                mutated = _mutate_observation_text(parsed, rng, corruption, severity)
            else:
                # Render/image corruptions are applied to image artifacts;
                # preserve the JSON metadata exactly.
                mutated = parsed
            return json.dumps(mutated, ensure_ascii=False, sort_keys=True), corruption, True, True

        # Plain-text tool results have no schema to break.  Keep this path
        # deterministic and avoid appending simulator-only diagnostics.
        if corruption in {"character_deletion", "paragraph_omission", "page_miss"} and len(text) > 12:
            remove = max(1, min(len(text) // 4, int(len(text) * severity * 0.25)))
            indexes = set(rng.sample(range(len(text)), min(remove, len(text))))
            mutated_text = "".join(char for index, char in enumerate(text) if index not in indexes)
        elif corruption in {"character_substitution", "digit_confusion", "numeric_noise", "scale_bias"}:
            mutated_text = _replace_digits(text, rng, min(0.55, severity * 0.9))
        else:
            mutated_text = text
        return mutated_text, corruption, True, mutated_text != text or corruption in {"blur", "downsample", "occlusion", "compression_noise", "rotation", "bbox_shift", "scale_error", "edge_crop"}


def _corrupt_text(tool_name: str, text: str, quality: ToolQualitySpec, rng: random.Random) -> tuple[str, str | None]:
    return ObservationCorruptionAdapter.corrupt(tool_name, text, quality, rng)[:2]

    # Kept below as a compatibility reference for old manifests and tests;
    # new runtime calls use ObservationCorruptionAdapter above.
    severity = max(0.0, min(0.85, 1.0 - min(quality.semantic_accuracy, quality.structure_fidelity)))
    if severity <= 0.025:
        return text, None
    if tool_name == "parse_document":
        corruption = rng.choice(("paragraph_omission", "character_substitution", "page_order_jitter", "page_miss"))
    elif tool_name == "render_page":
        corruption = rng.choice(("blur", "downsample", "occlusion", "compression_noise", "rotation"))
    elif tool_name in {"crop_region", "zoom_region"}:
        corruption = rng.choice(("bbox_shift", "scale_error", "edge_crop", "downsample"))
    elif tool_name == "ocr_region":
        corruption = rng.choice(("character_deletion", "character_substitution", "digit_confusion", "line_order", "confidence_miscalibration"))
    elif tool_name == "detect_layout":
        corruption = rng.choice(("bbox_jitter", "region_miss", "region_duplicate", "type_misclassification"))
    elif tool_name == "extract_table":
        corruption = rng.choice(("row_omission", "column_omission", "row_column_swap", "cell_merge", "text_fallback"))
    else:
        corruption = rng.choice(("series_omission", "scale_bias", "category_swap", "numeric_noise"))

    value = text
    if corruption in {"character_deletion", "paragraph_omission", "page_miss", "row_omission", "column_omission", "series_omission"}:
        pieces = value.splitlines(keepends=True)
        if len(pieces) > 1:
            keep = [piece for piece in pieces if rng.random() > min(0.6, severity * 0.9)]
            value = "".join(keep or pieces[:1])
        elif len(value) > 12:
            count = max(1, int(len(value) * min(0.3, severity * 0.35)))
            indexes = sorted(rng.sample(range(len(value)), min(count, len(value))))
            value = "".join(char for index, char in enumerate(value) if index not in set(indexes))
    elif corruption in {"character_substitution", "digit_confusion", "numeric_noise", "scale_bias"}:
        value = _replace_digits(value, rng, min(0.55, severity * 0.9))
        if corruption == "character_substitution":
            substitutions = {"0": "O", "1": "I", "5": "S", "8": "B", "a": "e", "e": "a"}
            value = "".join(substitutions.get(char, char) if rng.random() < severity * 0.22 else char for char in value)
    elif corruption in {"page_order_jitter", "line_order", "row_column_swap", "category_swap"}:
        pieces = value.splitlines(keepends=True)
        if len(pieces) >= 2:
            rng.shuffle(pieces)
            value = "".join(pieces)
    elif corruption in {"region_duplicate", "cell_merge", "text_fallback"}:
        pieces = value.splitlines(keepends=True)
        if len(pieces) >= 2:
            if corruption == "region_duplicate":
                value = value + pieces[rng.randrange(len(pieces))]
            elif corruption == "cell_merge":
                value = value.replace("|", " ", max(1, int(len(value) * 0.01)))
            else:
                value = re.sub(r"[{}\[\]]", "", value)
    elif corruption in {"bbox_jitter", "region_miss", "type_misclassification", "confidence_miscalibration"}:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            value = value + f"\n[observation quality degraded: {corruption}]"
        else:
            value = json.dumps(
                _mutate_structured_observation(parsed, corruption, severity, rng),
                ensure_ascii=False,
                sort_keys=True,
            )
    else:
        # Layout/render corruptions are represented by target-independent
        # metadata noise.  They do not inspect an answer location.
        value = value + f"\n[observation quality degraded: {corruption}]"
    return value, corruption


def _extract_page_numbers(arguments: Mapping[str, Any]) -> tuple[int, ...]:
    raw_pages = arguments.get("page_numbers")
    if isinstance(raw_pages, (list, tuple, set)):
        pages: list[int] = []
        for value in raw_pages:
            try:
                pages.append(int(value))
            except (TypeError, ValueError):
                continue
        if pages:
            return tuple(dict.fromkeys(pages))
    page = arguments.get("page_number", arguments.get("page"))
    try:
        return (int(page),) if page is not None else ()
    except (TypeError, ValueError):
        return ()


def _extract_page_region(arguments: Mapping[str, Any]) -> tuple[int | None, tuple[float, float, float, float] | None]:
    page = arguments.get("page_number", arguments.get("page"))
    try:
        page_number = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_number = None
    bbox = arguments.get("bbox", arguments.get("region"))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            region = tuple(max(0.0, min(1.0, float(value))) for value in bbox)
        except (TypeError, ValueError):
            region = None
    else:
        region = None
    return page_number, region


@dataclass(frozen=True)
class TransformResult:
    observed_result: str
    world_event: WorldEvent
    hidden_supervision: ToolStateLabel

    def __iter__(self):
        yield self.observed_result
        yield self.world_event
        yield self.hidden_supervision


class WorldRuntime:
    """Apply hidden world state between a clean tool result and observation."""

    def __init__(
        self,
        spec: ToolWorldSpec,
        *,
        config: BayesToolConfig | None = None,
        output_root: str | Path | None = None,
        clean_cache: CleanResultCache | None = None,
        tool_backend_version: str = "document-tools-v1",
        document_digest: str = "",
        sampling_context: WorldSamplingContext | Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.config = config or default_config(enabled=True)
        self.output_root = Path(output_root) if output_root else None
        self.clean_cache = clean_cache or CleanResultCache()
        self.tool_backend_version = tool_backend_version
        self.document_digest = str(document_digest)
        self.sampling_context = (
            sampling_context
            if isinstance(sampling_context, WorldSamplingContext)
            else WorldSamplingContext.from_mapping(sampling_context)
        )
        self.tool_budget = int(self.sampling_context.tool_budget)
        self.call_count = 0
        self.events: list[WorldEvent] = []
        self.supervision: list[ToolStateLabel] = []
        self.context_rule_match_count = 0
        self.first_context_match_call: int | None = None
        self.context_affected_call_count = 0
        self.schedule_call_counts: dict[int, int] = {index: 0 for index, _ in enumerate(spec.regime_schedule)}
        self.schedule_affected_call_counts: dict[int, int] = {
            index: 0 for index, _ in enumerate(spec.regime_schedule)
        }
        self.schedule_first_effective_call: dict[int, int | None] = {
            index: None for index, _ in enumerate(spec.regime_schedule)
        }
        self.schedule_last_effective_call: dict[int, int | None] = {
            index: None for index, _ in enumerate(spec.regime_schedule)
        }

    @classmethod
    def for_sample(
        cls,
        *,
        coupling_id: str,
        sample_index: int,
        rollout_id: int | str = 0,
        config: BayesToolConfig | None = None,
        output_root: str | Path | None = None,
        world_type: str | None = None,
        document_digest: str = "",
        clean_cache: CleanResultCache | None = None,
        fixed_world_specs: Sequence[Mapping[str, Any] | ToolWorldSpec] | None = None,
        sampling_context: WorldSamplingContext | Mapping[str, Any] | None = None,
        tool_budget: int | None = None,
    ) -> "WorldRuntime":
        config = config or default_config(enabled=True)
        # Use the same public context for sampling and runtime accounting.  An
        # explicit budget must not be lost when the caller supplied a mapping
        # (or no context at all), otherwise schedules are sampled with one
        # budget and executed/reported with the default budget of eight.
        if isinstance(sampling_context, WorldSamplingContext):
            runtime_context = sampling_context
            if tool_budget is not None and int(tool_budget) != runtime_context.tool_budget:
                runtime_context = WorldSamplingContext(
                    page_count=runtime_context.page_count,
                    tool_argument_capabilities=runtime_context.tool_argument_capabilities,
                    tool_budget=tool_budget,
                )
        else:
            runtime_context = WorldSamplingContext.from_mapping(
                sampling_context,
                tool_budget=tool_budget or 8,
            )
            if tool_budget is not None:
                runtime_context = WorldSamplingContext(
                    page_count=runtime_context.page_count,
                    tool_argument_capabilities=runtime_context.tool_argument_capabilities,
                    tool_budget=tool_budget,
                )
        world_slot = int(sample_index) % max(1, int(config.worlds_per_prompt))
        replica_id = (int(sample_index) // max(1, int(config.worlds_per_prompt))) % max(
            1, int(config.replicas_per_world)
        )
        fixed_index = world_slot * max(1, int(config.replicas_per_world)) + replica_id
        fixed_value = (
            fixed_world_specs[fixed_index]
            if fixed_world_specs is not None and fixed_index < len(fixed_world_specs)
            else None
        )
        if fixed_value is not None and world_type is None:
            spec = (
                fixed_value
                if isinstance(fixed_value, ToolWorldSpec)
                else tool_world_spec_from_dict(fixed_value)
            )
            if spec.coupling_id != str(coupling_id):
                raise ValueError(
                    "fixed world spec coupling_id does not match the rollout coupling_id: "
                    f"{spec.coupling_id!r} != {coupling_id!r}"
                )
        else:
            spec = sample_tool_world(
                coupling_id,
                world_slot=world_slot,
                replica_id=replica_id,
                rollout_id=rollout_id,
                config=config,
                world_type=world_type,
                sampling_context=runtime_context,
                tool_budget=tool_budget,
            )
        return cls(
            spec,
            config=config,
            output_root=output_root,
            document_digest=document_digest,
            clean_cache=clean_cache,
            sampling_context=runtime_context,
        )

    def _shared_multiplier(self, tool_name: str, call_index: int) -> float:
        multiplier = 1.0
        for family, members in TOOL_FAMILIES.items():
            if tool_name not in members:
                continue
            factor = self.spec.shared_factors.get(family)
            if factor and factor.state != "healthy":
                multiplier *= max(0.10, 1.0 - 0.65 * float(factor.severity))
            for segment in self.spec.regime_schedule:
                if call_index < segment.start_call:
                    continue
                override = segment.shared_overrides.get(family)
                if override is not None:
                    progress = segment.progress(call_index) if segment.transition == "linear" else 1.0
                    multiplier *= max(0.10, 1.0 - 0.65 * float(override.severity) * progress)
        return max(0.05, min(1.0, multiplier))

    def quality_for(
        self,
        tool_name: str,
        *,
        page_number: int | None = None,
        page_numbers: Sequence[int] | None = None,
        region: tuple[float, float, float, float] | None = None,
        content_type: str | None = None,
        call_index: int | None = None,
    ) -> ToolQualitySpec:
        call_index = self.call_count if call_index is None else int(call_index)
        base = self.spec.tool_states.get(tool_name)
        if base is None:
            raise KeyError(f"unknown tool in world: {tool_name}")
        quality = _multiply_quality(base, self._shared_multiplier(tool_name, call_index))
        quality = ToolQualitySpec(
            availability=quality.availability * self.spec.session_state.availability_scale,
            semantic_accuracy=quality.semantic_accuracy,
            structure_fidelity=quality.structure_fidelity,
            calibration_temperature=quality.calibration_temperature,
            calibration_bias=quality.calibration_bias,
            relative_cost=quality.relative_cost,
            latency_scale=quality.latency_scale * self.spec.session_state.latency_scale,
        )
        for rule in self.spec.context_rules:
            if rule.matches(
                tool_name,
                page_number=page_number,
                page_numbers=page_numbers,
                region=region,
                content_type=content_type,
            ):
                quality = rule.overrides.apply(quality)
        for segment in self.spec.regime_schedule:
            if call_index < segment.start_call:
                continue
            patch = segment.tool_overrides.get(tool_name)
            if patch is not None:
                progress = segment.progress(call_index) if segment.transition == "linear" else 1.0
                quality = patch.apply(quality, progress)
        return quality

    def _make_image_outputs(
        self,
        image_paths: list[str],
        *,
        call_id: int,
        corruption: str | None,
        rng: random.Random,
    ) -> dict[str, str]:
        if not image_paths or self.output_root is None:
            return {}
        # Image paths are sent back through the model context.  Do not put
        # coupling/world identifiers in a visible path because that would
        # turn a filesystem detail into a world-state label.  The diagnostic
        # metadata still retains the exact hidden world identity separately.
        coupling_key = hashlib.sha256(self.spec.coupling_id.encode("utf-8")).hexdigest()[:16]
        world_key = hashlib.sha256(self.spec.world_id.encode("utf-8")).hexdigest()[:16]
        target_dir = self.output_root / "tool_outputs" / "bayestool" / coupling_key / world_key / str(call_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        replacements: dict[str, str] = {}
        for image_index, raw_path in enumerate(image_paths):
            source = Path(raw_path)
            if not source.is_file():
                continue
            target_suffix = ".jpg" if corruption == "compression_noise" else (source.suffix or ".png")
            target = target_dir / f"image_{image_index}{target_suffix}"
            try:
                if corruption and corruption in {
                    "blur",
                    "downsample",
                    "occlusion",
                    "compression_noise",
                    "rotation",
                    "bbox_shift",
                    "scale_error",
                    "edge_crop",
                }:
                    from PIL import Image, ImageDraw, ImageFilter

                    image = Image.open(source).convert("RGB")
                    if corruption == "blur":
                        image = image.filter(ImageFilter.GaussianBlur(radius=1.5 + rng.random() * 2.0))
                    elif corruption == "downsample":
                        small = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
                        image = small.resize((image.width, image.height))
                    elif corruption == "occlusion":
                        draw = ImageDraw.Draw(image)
                        width = max(1, int(image.width * (0.10 + 0.20 * rng.random())))
                        height = max(1, int(image.height * (0.10 + 0.20 * rng.random())))
                        left = rng.randint(0, max(0, image.width - width))
                        top = rng.randint(0, max(0, image.height - height))
                        draw.rectangle((left, top, left + width, top + height), fill=(128, 128, 128))
                    elif corruption == "edge_crop":
                        left = max(0, int(image.width * 0.02))
                        top = max(0, int(image.height * 0.02))
                        image = image.crop((left, top, max(left + 1, image.width - left), max(top + 1, image.height - top)))
                    elif corruption == "rotation":
                        angle = rng.uniform(-6.0, 6.0)
                        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
                        image = image.rotate(angle, resample=resampling, expand=False, fillcolor=(255, 255, 255))
                    elif corruption == "bbox_shift":
                        dx = int(rng.uniform(-0.08, 0.08) * image.width)
                        dy = int(rng.uniform(-0.08, 0.08) * image.height)
                        shifted = Image.new("RGB", image.size, (255, 255, 255))
                        shifted.paste(image, (dx, dy))
                        image = shifted
                    elif corruption == "scale_error":
                        factor = rng.uniform(0.82, 1.18)
                        scaled = image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))))
                        canvas = Image.new("RGB", image.size, (255, 255, 255))
                        left = (canvas.width - scaled.width) // 2
                        top = (canvas.height - scaled.height) // 2
                        canvas.paste(scaled, (left, top))
                        image = canvas
                    elif corruption == "compression_noise":
                        image.save(target, format="JPEG", quality=rng.randint(25, 55), optimize=True)
                        replacements[str(source)] = str(target)
                        continue
                    image.save(target)
                else:
                    shutil.copy2(source, target)
            except Exception:
                shutil.copy2(source, target)
            replacements[str(source)] = str(target)
        return replacements

    def transform_result(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        clean_result: str,
        *,
        result_status: str = "ok",
        image_paths: list[str] | None = None,
        image_valid: bool = True,
        failure_origin: str = "none",
    ) -> TransformResult:
        """Return observed output, public event, and hidden supervision.

        ``clean_result`` is treated as an opaque tool result.  This function
        never receives the dataset label, answer page, or answer bbox.
        """

        call_id = self.call_count
        self.call_count += 1
        page_number, region = _extract_page_region(arguments)
        page_numbers = _extract_page_numbers(arguments)
        content_type = str(arguments.get("content_type")) if arguments.get("content_type") else None
        rng = random.Random(
            stable_seed(
                self.spec.latent_seed or self.spec.seed,
                self.spec.replica_id,
                call_id,
                tool_name,
                canonical_json(arguments),
            )
        )
        quality = self.quality_for(
            tool_name,
            page_number=page_number,
            page_numbers=page_numbers,
            region=region,
            content_type=content_type,
            call_index=call_id,
        )
        matched_rules = [
            rule
            for rule in self.spec.context_rules
            if rule.matches(
                tool_name,
                page_number=page_number,
                page_numbers=page_numbers,
                region=region,
                content_type=content_type,
            )
        ]
        self.context_rule_match_count += len(matched_rules)
        if matched_rules:
            if self.first_context_match_call is None:
                self.first_context_match_call = int(call_id)
            self.context_affected_call_count += 1
        for index, segment in enumerate(self.spec.regime_schedule):
            if call_id >= int(segment.start_call) and (
                segment.end_call is None or call_id <= int(segment.end_call)
            ):
                self.schedule_call_counts[index] = self.schedule_call_counts.get(index, 0) + 1
                affected = tool_name in segment.tool_overrides
                if not affected and segment.shared_overrides:
                    affected = any(
                        tool_name in TOOL_FAMILIES.get(family, ())
                        for family in segment.shared_overrides
                    )
                if affected:
                    self.schedule_affected_call_counts[index] = (
                        self.schedule_affected_call_counts.get(index, 0) + 1
                    )
                    if self.schedule_first_effective_call.get(index) is None:
                        self.schedule_first_effective_call[index] = int(call_id)
                    self.schedule_last_effective_call[index] = int(call_id)
        change_detected = any(
            int(call_id) == int(segment.start_call)
            for segment in self.spec.regime_schedule
        )
        if failure_origin == "real_infrastructure":
            observed = sanitize_observed_result(clean_result)
            event = WorldEvent(
                call_id=call_id,
                tool_name=tool_name,
                status="error",
                latency=0.0,
                information_gain=0.0,
                semantic_agreement=0.0,
                schema_valid=False,
                image_valid=False,
                error_family="availability",
                failure_origin="real_infrastructure",
                relative_cost=quality.relative_cost,
                page_number=page_number,
                page_numbers=page_numbers or None,
                region=region,
                content_type=content_type,
                change_detected=change_detected,
                observation_status="infrastructure_failure",
                corruption_applied=False,
                execution_succeeded=False,
                observation_delivered=False,
            )
            label = ToolStateLabel(
                tool_name,
                quality,
                self.spec.session_state.state,
                {},
                "stable",
                self.spec.world_id,
                self.spec.latent_world_id,
                self.spec.replica_id,
            )
            self.events.append(event)
            self.supervision.append(label)
            return TransformResult(observed, event, label)

        availability_draw = rng.random()
        if availability_draw > quality.availability or result_status in {"timeout", "error"}:
            status = "timeout" if result_status == "timeout" or rng.random() < 0.15 else "error"
            corruption = "availability_failure"
            observed = json.dumps(
                {
                    "status": status,
                    "error": f"Tool {tool_name} returned no reliable observation.",
                    "execution_succeeded": True,
                    "observation_delivered": False,
                    "observation_status": "unavailable",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            event = WorldEvent(
                call_id=call_id,
                tool_name=tool_name,
                status=status,
                latency=max(0.01, quality.latency_scale * _uniform(rng, 0.5, 2.0)),
                information_gain=0.0,
                semantic_agreement=0.0,
                schema_valid=False,
                image_valid=False,
                error_family="availability",
                corruption_type=corruption,
                failure_origin="world_injected",
                relative_cost=quality.relative_cost,
                page_number=page_number,
                page_numbers=page_numbers or None,
                region=region,
                content_type=content_type,
                change_detected=change_detected,
                observation_status="unavailable",
                corruption_applied=False,
                execution_succeeded=True,
                observation_delivered=False,
            )
        else:
            observed, corruption, schema_valid, corruption_applied = ObservationCorruptionAdapter.corrupt(
                tool_name,
                sanitize_observed_result(str(clean_result)),
                quality,
                rng,
            )
            replacements = self._make_image_outputs(image_paths or [], call_id=call_id, corruption=corruption, rng=rng)
            for source, target in replacements.items():
                observed = observed.replace(source, target)
            status = "partial" if corruption and rng.random() < 0.45 else "ok"
            # This metric is computed from the delivered observation, not from
            # hidden quality.  It is therefore safe to expose as policy input.
            semantic_agreement = 1.0
            if corruption_applied:
                semantic_agreement = 0.65 if status == "partial" else 0.80
            if not str(observed).strip():
                semantic_agreement = 0.0
            event = WorldEvent(
                call_id=call_id,
                tool_name=tool_name,
                status=status,
                latency=max(0.01, quality.latency_scale * _uniform(rng, 0.5, 2.0)),
                information_gain=max(0.0, min(1.0, math.log1p(len(observed)) / 10.0)),
                semantic_agreement=semantic_agreement,
                schema_valid=schema_valid,
                image_valid=bool(image_valid),
                error_family=_value_error_family(tool_name, corruption),
                corruption_type=corruption,
                failure_origin="world_injected" if corruption else "none",
                relative_cost=quality.relative_cost,
                page_number=page_number,
                page_numbers=page_numbers or None,
                region=region,
                content_type=content_type,
                change_detected=change_detected,
                observation_status="corrupted" if corruption_applied else "ok",
                corruption_applied=bool(corruption_applied),
                execution_succeeded=True,
                observation_delivered=True,
            )

        shared_states = {
            family: self.spec.shared_factors[family].state
            for family, members in TOOL_FAMILIES.items()
            if tool_name in members and family in self.spec.shared_factors
        }
        regime_state = "stable"
        for segment in self.spec.regime_schedule:
            if call_id >= segment.start_call:
                regime_state = "abrupt_transition" if segment.transition == "abrupt" else "gradual_transition"
        label = ToolStateLabel(
            tool_name=tool_name,
            quality=quality,
            session_state=self.spec.session_state.state,
            shared_states=shared_states,
            regime_state=regime_state,
            world_id=self.spec.world_id,
            latent_world_id=self.spec.latent_world_id,
            replica_id=self.spec.replica_id,
        )
        self.events.append(event)
        self.supervision.append(label)
        return TransformResult(sanitize_observed_result(observed), event, label)

    def record_real_infrastructure_failure(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        error: str,
    ) -> TransformResult:
        return self.transform_result(
            tool_name,
            arguments,
            str(error),
            result_status="error",
            failure_origin="real_infrastructure",
        )

    def hidden_supervision_metadata(self) -> list[dict[str, Any]]:
        return [label.to_dict() for label in self.supervision]

    def public_event_metadata(self) -> list[dict[str, Any]]:
        return [event.visible_dict() for event in self.events]

    def schedule_metadata(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for index, segment in enumerate(self.spec.regime_schedule):
            end = self.tool_budget if segment.end_call is None else int(segment.end_call)
            values.append(
                {
                    "index": index,
                    "start_call": int(segment.start_call),
                    "end_call": int(end),
                    "transition": segment.transition,
                    "calls_observed": int(self.schedule_call_counts.get(index, 0)),
                    "first_effective_call": self.schedule_first_effective_call.get(index),
                    "last_effective_call": self.schedule_last_effective_call.get(index),
                    "affected_call_count": int(self.schedule_affected_call_counts.get(index, 0)),
                    "effective_within_budget": bool(
                        int(segment.start_call) <= self.tool_budget
                        and int(end) >= int(segment.start_call)
                    ),
                    "schedule_not_exercised": self.schedule_affected_call_counts.get(index, 0) <= 0,
                }
            )
        return values

    def context_metadata(self) -> dict[str, Any]:
        return {
            "page_count": self.sampling_context.page_count,
            "tool_budget": self.tool_budget,
            "context_rule_count": len(self.spec.context_rules),
            "context_rule_match_count": int(self.context_rule_match_count),
            "first_context_match_call": self.first_context_match_call,
            "affected_call_count": int(self.context_affected_call_count),
            "context_rule_not_exercised": bool(
                self.spec.context_rules and self.context_rule_match_count <= 0
            ),
        }


__all__ = [
    "HIDDEN_OBSERVATION_KEYS",
    "stable_seed",
    "document_hash",
    "canonical_json",
    "WORLD_TYPES",
    "SESSION_STATES",
    "WorldSamplingContext",
    "DEFAULT_TOOL_ARGUMENT_CAPABILITIES",
    "sample_session_state",
    "sample_world_type",
    "sample_tool_world",
    "tool_world_spec_from_dict",
    "CleanResultCache",
    "TransformResult",
    "ObservationCorruptionAdapter",
    "WorldRuntime",
    "sanitize_observed_result",
]
