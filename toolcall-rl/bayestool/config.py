"""Configuration and constants for the complete BayesTool-RL runtime."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


TOOL_NAMES: tuple[str, ...] = (
    "parse_document",
    "detect_layout",
    "render_page",
    "crop_region",
    "zoom_region",
    "ocr_region",
    "extract_table",
    "chart_to_table",
)

TOOL_FAMILIES: dict[str, tuple[str, ...]] = {
    "render_core": ("render_page", "crop_region", "zoom_region", "ocr_region", "detect_layout"),
    "text_core": ("parse_document", "ocr_region"),
    "structure_core": ("detect_layout", "extract_table", "chart_to_table"),
}

STATUS_NAMES: tuple[str, ...] = ("ok", "partial", "error", "timeout", "invalid", "empty")
ERROR_FAMILIES: tuple[str, ...] = (
    "none",
    "availability",
    "render",
    "ocr",
    "structure",
    "semantic",
    "latency",
    "protocol",
)
CONTENT_TYPES: tuple[str, ...] = ("text", "table", "chart", "figure", "formula", "mixed")
STAGE_NAMES: tuple[str, ...] = ("a", "b", "c", "d")
SESSION_STATES: tuple[str, ...] = ("healthy", "degraded", "overloaded", "outage")


@dataclass(frozen=True)
class UtilityConfig:
    cost_weight: float = 0.15
    inefficiency_weight: float = 0.20
    failure_weight: float = 0.35


@dataclass(frozen=True)
class PairingConfig:
    max_ood_score: float = 0.15
    min_belief_js: float = 0.10
    max_belief_js: float = 0.80
    min_action_margin: float = 0.05
    preinv_max_js: float = 0.02


@dataclass(frozen=True)
class AuxiliaryConfig:
    interval: int = 2
    max_switch_bundles_per_rank: int = 8
    max_preinv_bundles_per_rank: int = 8
    switch_loss_weight: float = 0.20
    preinv_loss_weight: float = 0.05
    kl_loss_weight: float = 0.01
    micro_batch_size: int = 4
    use_switch_loss: bool = True
    use_pre_invariance: bool = True


@dataclass(frozen=True)
class MetaConfig:
    questions_per_episode_min: int = 2
    questions_per_episode_max: int = 4
    discount: float = 0.95


@dataclass(frozen=True)
class BeliefConfig:
    feature_dim: int = 96
    session_hidden: int = 256
    context_hidden: int = 256
    shared_hidden: int = 128
    tool_embedding_dim: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 256
    sequence_length: int = 16
    gradient_clip: float = 1.0
    early_stop_patience: int = 5


@dataclass(frozen=True)
class StageDefinition:
    """Executable policy for one phase in the four-stage training schedule."""

    name: str
    objective: str
    runtime_mode: str
    branch_probability: float
    use_meta_episode: bool
    use_persistent_session_belief: bool
    expected_update_fraction: float


@dataclass(frozen=True)
class BayesToolConfig:
    """All runtime, decision, and training knobs in the implementation plan."""

    enabled: bool = False
    # Explicit question -> realization -> decision-group planning.  The
    # legacy world/replica fields below remain loadable for old manifests, but
    # they must never determine RL group cardinality or loss weighting.
    default_group_size: int = 4
    min_realizations: int = 4
    max_realizations: int = 6
    max_records_per_question: int = 48
    k8_target_ratio: float = 0.25
    k8_floor: float = 0.0
    k8_ceiling: float = 1.0
    k8_window: int = 32
    policy_version: str = "bayestool-policy-v1"
    worlds_per_prompt: int = 4
    # Compatibility metadata only.  New plans materialize K continuations
    # from one frozen realization; keeping the default at one avoids implying
    # the retired worlds x replicas (4 x 2 -> K2) training semantics.
    replicas_per_world: int = 1
    posterior_particles: int = 8
    max_action_candidates: int = 4
    max_siblings: int = 4
    branch_horizon: int = 3
    branch_probability_when_eligible: float = 0.25
    consensus_threshold: float = 0.75
    decision_regret_threshold: float = 0.08
    max_observation_hypotheses: int = 6
    dvoi_minimum: float = 0.0
    cvar_alpha: float = 0.20
    local_surprise_threshold: float = 4.0
    family_surprise_threshold: float = 6.0
    global_surprise_threshold: float = 8.0
    change_probability_threshold: float = 0.80
    max_belief_prompt_tokens: int = 1200
    posterior_particles_seed_offset: int = 100_003
    stage: str = "c"
    use_dvoi: bool = True
    use_regret_branching: bool = True
    use_reopen: bool = True
    # The library default is the Stage-C policy.  Stage D explicitly enables
    # both fields through config_from_args/stage_definition; keeping them
    # true here would silently run Meta episodes during the default paired
    # world stage.
    use_meta_episode: bool = False
    use_persistent_session_belief: bool = False
    world_type_probabilities: tuple[tuple[str, float], ...] = (
        ("healthy", 0.20),
        ("single_tool_degradation", 0.25),
        ("context_degradation", 0.20),
        ("shared_family_fault", 0.20),
        ("abrupt_change", 0.10),
        ("gradual_change", 0.05),
    )
    session_state_probabilities: tuple[tuple[str, float], ...] = (
        ("healthy", 0.70),
        ("degraded", 0.20),
        ("overloaded", 0.08),
        ("outage", 0.02),
    )
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BayesToolConfig":
        if not value:
            return cls()
        data = dict(value)
        nested = {
            "utility": UtilityConfig,
            "pairing": PairingConfig,
            "auxiliary": AuxiliaryConfig,
            "meta": MetaConfig,
            "belief": BeliefConfig,
        }
        for key, type_ in nested.items():
            raw = data.get(key)
            if isinstance(raw, Mapping):
                data[key] = type_(**dict(raw))
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: data[key] for key in data if key in known})

    def with_enabled(self, enabled: bool) -> "BayesToolConfig":
        return replace(self, enabled=bool(enabled))

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if hasattr(value, "__dataclass_fields__"):
                return {name: convert(getattr(value, name)) for name in value.__dataclass_fields__}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            return value

        return convert(self)


def default_config(*, enabled: bool = False) -> BayesToolConfig:
    """Return a fresh default configuration.

    The library default is disabled to preserve existing baseline invocations;
    the BayesTool shell entry points explicitly enable it.
    """

    return BayesToolConfig(enabled=enabled)


STAGE_DEFINITIONS: dict[str, StageDefinition] = {
    "a": StageDefinition(
        name="a",
        objective="tool-world belief pretraining",
        runtime_mode="belief_pretraining",
        branch_probability=0.0,
        use_meta_episode=False,
        use_persistent_session_belief=False,
        expected_update_fraction=0.0,
    ),
    "b": StageDefinition(
        name="b",
        objective="single-task belief-conditioned policy training",
        runtime_mode="single_task_belief_conditioned",
        branch_probability=0.10,
        use_meta_episode=False,
        use_persistent_session_belief=False,
        expected_update_fraction=0.20,
    ),
    "c": StageDefinition(
        name="c",
        objective="paired-world belief switching and Bayes-ARPO",
        runtime_mode="paired_world_belief_switch",
        branch_probability=0.25,
        use_meta_episode=False,
        use_persistent_session_belief=False,
        expected_update_fraction=0.60,
    ),
    "d": StageDefinition(
        name="d",
        objective="cross-task persistent-session meta-policy training",
        runtime_mode="cross_task_meta_policy",
        branch_probability=0.25,
        use_meta_episode=True,
        use_persistent_session_belief=True,
        expected_update_fraction=0.20,
    ),
}


def stage_definition(stage: str | BayesToolConfig) -> StageDefinition:
    """Return the explicit schedule entry used by runtime and launchers."""

    value = stage.stage if isinstance(stage, BayesToolConfig) else stage
    key = str(value or "c").casefold()
    if key not in STAGE_DEFINITIONS:
        raise ValueError(f"unknown BayesTool stage: {value!r}")
    return STAGE_DEFINITIONS[key]


def stage_schedule() -> tuple[StageDefinition, ...]:
    """Return the recommended A→B→C→D schedule in execution order."""

    return tuple(STAGE_DEFINITIONS[name] for name in STAGE_NAMES)


def validate_stage_capabilities(
    stage: str,
    *,
    belief_checkpoint: str | None = None,
    q_checkpoint: str | None = None,
    risk_checkpoint: str | None = None,
    meta_manifest: str | None = None,
    allow_heuristic_belief: bool = False,
    allow_heuristic_q: bool = False,
    allow_heuristic_risk: bool = False,
) -> dict[str, Any]:
    """Fail closed when a stage would silently fall back to heuristics."""

    definition = stage_definition(stage)
    missing: dict[str, str] = {}

    def require(name: str, path: str | None, allowed: bool) -> None:
        if path and os.path.exists(path):
            return
        if allowed:
            return
        missing[name] = (
            "checkpoint is missing (or path does not exist); pass the explicit "
            "heuristic-ablation flag only for a smoke test"
        )

    if definition.name in {"b", "c", "d"}:
        require("belief", belief_checkpoint, allow_heuristic_belief)
    if definition.name in {"c", "d"}:
        require("q", q_checkpoint, allow_heuristic_q)
        require("risk", risk_checkpoint, allow_heuristic_risk)
    if definition.name == "d" and not (meta_manifest and os.path.exists(meta_manifest)):
        missing["meta_manifest"] = "Stage D requires a dedicated meta-episode manifest"
    if missing:
        raise ValueError(
            f"BayesTool Stage {definition.name.upper()} capability gate failed: "
            + "; ".join(f"{name}: {reason}" for name, reason in missing.items())
        )
    return {
        "stage": definition.name,
        "belief": bool(belief_checkpoint and os.path.exists(belief_checkpoint)),
        "q": bool(q_checkpoint and os.path.exists(q_checkpoint)),
        "risk": bool(risk_checkpoint and os.path.exists(risk_checkpoint)),
        "meta_manifest": bool(meta_manifest and os.path.exists(meta_manifest)),
        "heuristic_belief": bool(allow_heuristic_belief),
        "heuristic_q": bool(allow_heuristic_q),
        "heuristic_risk": bool(allow_heuristic_risk),
    }


def config_from_args(args: Any, *, enabled: bool | None = None) -> BayesToolConfig:
    """Build a config from a slime argparse namespace without requiring slime."""

    configured = getattr(args, "bayestool_config", None)
    config = BayesToolConfig.from_mapping(configured if isinstance(configured, Mapping) else None)
    updates: dict[str, Any] = {}
    aliases = {
        "branch_probability_when_eligible": "bayestool_branch_probability",
        "stage": "bayestool_stage",
    }
    for name in BayesToolConfig.__dataclass_fields__:
        if name in {"utility", "pairing", "auxiliary", "meta", "belief"}:
            continue
        attr = aliases.get(name, f"bayestool_{name}")
        if hasattr(args, attr) and getattr(args, attr) is not None:
            updates[name] = getattr(args, attr)
    auxiliary_updates: dict[str, Any] = {}
    auxiliary_aliases = {
        "interval": "bayestool_aux_interval",
        "max_switch_bundles_per_rank": "bayestool_max_switch_bundles_per_rank",
        "max_preinv_bundles_per_rank": "bayestool_max_preinv_bundles_per_rank",
        "switch_loss_weight": "bayestool_switch_loss_weight",
        "preinv_loss_weight": "bayestool_preinv_loss_weight",
        "micro_batch_size": "bayestool_aux_micro_batch_size",
    }
    for name, attr in auxiliary_aliases.items():
        if hasattr(args, attr) and getattr(args, attr) is not None:
            auxiliary_updates[name] = getattr(args, attr)
    auxiliary_boolean_aliases = {
        "use_switch_loss": "bayestool_without_switch_loss",
        "use_pre_invariance": "bayestool_without_pre_invariance",
    }
    for name, attr in auxiliary_boolean_aliases.items():
        # These are opt-out flags.  Their argparse default is False, which
        # must not overwrite an explicit value supplied by a config mapping.
        if hasattr(args, attr) and bool(getattr(args, attr)):
            auxiliary_updates[name] = False
    if auxiliary_updates:
        updates["auxiliary"] = replace(config.auxiliary, **auxiliary_updates)
    meta_updates: dict[str, Any] = {}
    if hasattr(args, "bayestool_meta_discount") and getattr(args, "bayestool_meta_discount") is not None:
        meta_updates["discount"] = float(getattr(args, "bayestool_meta_discount"))
    if meta_updates:
        updates["meta"] = replace(config.meta, **meta_updates)
    def _probability_pairs(value: Any, *, label: str) -> tuple[tuple[str, float], ...]:
        raw_probabilities: Any = value
        if isinstance(raw_probabilities, str):
            raw_probabilities = json.loads(raw_probabilities)
        if isinstance(raw_probabilities, Mapping):
            raw_probabilities = tuple((str(key), float(value)) for key, value in raw_probabilities.items())
        elif isinstance(raw_probabilities, (list, tuple)):
            raw_probabilities = tuple((str(item[0]), float(item[1])) for item in raw_probabilities)
        else:
            raise ValueError(f"bayestool {label} probabilities must be a JSON object or pair list")
        if not raw_probabilities:
            raise ValueError(f"bayestool {label} probabilities cannot be empty")
        return raw_probabilities

    world_probability_value = getattr(args, "bayestool_world_type_probabilities", None)
    if world_probability_value is not None:
        updates["world_type_probabilities"] = _probability_pairs(world_probability_value, label="world type")
    session_probability_value = getattr(args, "bayestool_session_state_probabilities", None)
    if session_probability_value is not None:
        updates["session_state_probabilities"] = _probability_pairs(session_probability_value, label="session state")
    boolean_aliases = {
        "use_dvoi": "bayestool_without_dvoi",
        "use_regret_branching": "bayestool_without_regret_branching",
        "use_reopen": "bayestool_without_reopen",
        "use_meta_episode": "bayestool_without_meta_episode",
        "use_persistent_session_belief": "bayestool_without_persistent_session_belief",
    }
    for name, attr in boolean_aliases.items():
        # Do not turn an opt-out flag into an opt-in flag merely because the
        # parser materialized its default False value.
        if hasattr(args, attr) and bool(getattr(args, attr)):
            updates[name] = False
    if enabled is None and hasattr(args, "bayestool_enable"):
        enabled = bool(getattr(args, "bayestool_enable"))
    if enabled is not None:
        updates["enabled"] = bool(enabled)
    config = replace(config, **updates)
    # Stage definitions provide the exact A/B/C/D defaults.  A launcher may
    # explicitly override only the branch gate for a hardware profile (the
    # plan uses .25 for 4B and .20 for 8B in Stage C); meta/persistent policy
    # remains owned by the stage.  The opt-out flags below stay authoritative.
    stage = str(config.stage or "c").casefold()
    definition = stage_definition(stage)
    config = replace(
        config,
        branch_probability_when_eligible=(
            config.branch_probability_when_eligible
            if "branch_probability_when_eligible" in updates
            else definition.branch_probability
        ),
        use_meta_episode=definition.use_meta_episode,
        use_persistent_session_belief=definition.use_persistent_session_belief,
    )
    # Explicit ablations are intentionally reapplied after the stage policy.
    # This keeps e.g. Stage D + --without-meta-episode a valid experiment
    # without allowing parser defaults to silently enable a feature.
    for name, attr in boolean_aliases.items():
        if hasattr(args, attr) and bool(getattr(args, attr)):
            config = replace(config, **{name: False})
    return config


def add_bayestool_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the public BayesTool flags to an existing training parser."""

    parser.add_argument("--bayestool-enable", action="store_true", default=False)
    parser.add_argument("--bayestool-group-size", dest="bayestool_default_group_size", type=int, default=None)
    parser.add_argument("--bayestool-min-realizations", type=int, default=None)
    parser.add_argument("--bayestool-max-realizations", type=int, default=None)
    parser.add_argument("--bayestool-max-records-per-question", type=int, default=None)
    parser.add_argument("--bayestool-k8-target-ratio", type=float, default=None)
    parser.add_argument("--bayestool-k8-floor", type=float, default=None)
    parser.add_argument("--bayestool-k8-ceiling", type=float, default=None)
    parser.add_argument("--bayestool-k8-window", type=int, default=None)
    parser.add_argument("--bayestool-policy-version", type=str, default=None)
    parser.add_argument("--bayestool-worlds-per-prompt", type=int, default=None)
    parser.add_argument("--bayestool-replicas-per-world", type=int, default=None)
    parser.add_argument("--bayestool-questions-per-step", type=int, default=None)
    parser.add_argument("--bayestool-max-questions-per-step", type=int, default=None)
    parser.add_argument("--bayestool-posterior-particles", type=int, default=None)
    parser.add_argument("--bayestool-max-action-candidates", type=int, default=None)
    parser.add_argument("--bayestool-max-siblings", type=int, default=None)
    parser.add_argument("--bayestool-branch-horizon", type=int, default=None)
    parser.add_argument("--bayestool-branch-probability", type=float, default=None)
    parser.add_argument("--bayestool-consensus-threshold", type=float, default=None)
    parser.add_argument("--bayestool-decision-regret-threshold", type=float, default=None)
    parser.add_argument("--bayestool-max-observation-hypotheses", type=int, default=None)
    parser.add_argument("--bayestool-dvoi-minimum", type=float, default=None)
    parser.add_argument("--bayestool-cvar-alpha", type=float, default=None)
    parser.add_argument("--bayestool-local-surprise-threshold", type=float, default=None)
    parser.add_argument("--bayestool-family-surprise-threshold", type=float, default=None)
    parser.add_argument("--bayestool-global-surprise-threshold", type=float, default=None)
    parser.add_argument("--bayestool-change-probability-threshold", type=float, default=None)
    parser.add_argument("--bayestool-max-belief-prompt-tokens", type=int, default=None)
    parser.add_argument("--bayestool-meta-discount", type=float, default=None)
    parser.add_argument("--bayestool-world-type-probabilities", type=str, default=None)
    parser.add_argument("--bayestool-session-state-probabilities", type=str, default=None)
    parser.add_argument("--bayestool-stage", choices=["a", "b", "c", "d"], default=None)
    parser.add_argument("--bayestool-without-dvoi", action="store_true", default=None)
    parser.add_argument("--bayestool-without-regret-branching", action="store_true", default=None)
    parser.add_argument("--bayestool-without-reopen", action="store_true", default=None)
    parser.add_argument("--bayestool-without-meta-episode", action="store_true", default=None)
    parser.add_argument("--bayestool-without-persistent-session-belief", action="store_true", default=None)
    parser.add_argument("--bayestool-without-switch-loss", action="store_true", default=None)
    parser.add_argument("--bayestool-without-pre-invariance", action="store_true", default=None)
    parser.add_argument("--bayestool-aux-interval", type=int, default=None)
    parser.add_argument("--bayestool-max-switch-bundles-per-rank", type=int, default=None)
    parser.add_argument("--bayestool-max-preinv-bundles-per-rank", type=int, default=None)
    parser.add_argument("--bayestool-switch-loss-weight", type=float, default=None)
    parser.add_argument("--bayestool-preinv-loss-weight", type=float, default=None)
    parser.add_argument("--bayestool-aux-micro-batch-size", type=int, default=None)
    parser.add_argument("--bayestool-belief-checkpoint", type=str, default=None)
    parser.add_argument("--bayestool-q-checkpoint", type=str, default=None)
    parser.add_argument("--bayestool-risk-checkpoint", type=str, default=None)
    parser.add_argument("--bayestool-meta-manifest", type=str, default=None)
    parser.add_argument("--bayestool-checkpoint-interval-questions", type=int, default=None)
    parser.add_argument("--bayestool-checkpoint-retention", type=int, default=None)
    parser.add_argument("--bayestool-allow-heuristic-belief", action="store_true", default=False)
    parser.add_argument("--bayestool-allow-heuristic-q", action="store_true", default=False)
    parser.add_argument("--bayestool-allow-heuristic-risk", action="store_true", default=False)
    return parser


__all__ = [
    "TOOL_NAMES",
    "TOOL_FAMILIES",
    "STATUS_NAMES",
    "ERROR_FAMILIES",
    "CONTENT_TYPES",
    "STAGE_NAMES",
    "SESSION_STATES",
    "UtilityConfig",
    "PairingConfig",
    "AuxiliaryConfig",
    "MetaConfig",
    "BeliefConfig",
    "StageDefinition",
    "STAGE_DEFINITIONS",
    "BayesToolConfig",
    "default_config",
    "stage_definition",
    "stage_schedule",
    "validate_stage_capabilities",
    "config_from_args",
    "add_bayestool_arguments",
]
