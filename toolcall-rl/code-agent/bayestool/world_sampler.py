"""Deterministic, public-context-only Code world sampling."""

from __future__ import annotations

import random
from pathlib import PurePosixPath
from typing import Any, Mapping

try:
    from ..config import CODE_TOOL_FAMILIES, CODE_TOOL_NAMES, stable_hash
except ImportError:  # pragma: no cover
    from config import CODE_TOOL_FAMILIES, CODE_TOOL_NAMES, stable_hash

from .schema import CodeContextRule, CodeWorldSamplingContext, CodeWorldSpec, ToolQualitySpec


def public_sampling_context(
    *,
    tool_budget: int,
    repo_file_count: int,
    tracked_extensions: list[str] | tuple[str, ...] = (),
    top_level_dirs: list[str] | tuple[str, ...] = (),
    languages: list[str] | tuple[str, ...] = (),
    detected_frameworks: list[str] | tuple[str, ...] = (),
    tool_argument_capabilities: Mapping[str, Any] | None = None,
) -> CodeWorldSamplingContext:
    """Build the only context a world sampler may inspect."""

    return CodeWorldSamplingContext(
        tool_budget=int(tool_budget),
        repo_file_count=max(0, int(repo_file_count)),
        tracked_extensions=tuple(sorted(set(map(str, tracked_extensions)))),
        top_level_dirs=tuple(sorted(set(map(str, top_level_dirs)))),
        languages=tuple(sorted(set(map(str, languages)))),
        detected_frameworks=tuple(sorted(set(map(str, detected_frameworks)))),
        tool_argument_capabilities=dict(tool_argument_capabilities or {}),
    )


def _base_quality() -> dict[str, ToolQualitySpec]:
    return {name: ToolQualitySpec() for name in CODE_TOOL_NAMES}


def _family_quality() -> dict[str, ToolQualitySpec]:
    return {name: ToolQualitySpec() for name in CODE_TOOL_FAMILIES}


def sample_required_worlds(
    *,
    instance_id: str,
    image_name: str,
    base_revision: str | None,
    context: CodeWorldSamplingContext,
    rollout_seed: int,
    coupling_id: str | None = None,
) -> tuple[CodeWorldSpec, ...]:
    """Construct healthy/local/shared/change worlds for one task/repository."""

    coupling = coupling_id or stable_hash(
        {"environment": "code", "instance_id": instance_id, "image_name": image_name, "base_revision": base_revision, "rollout_seed": rollout_seed},
        prefix="code-coupling-v1",
    )
    root_rng = random.Random(int(rollout_seed))
    local_tool = _choose_local_tool(context, root_rng)
    local_prefix = _choose_prefix(context, root_rng)
    family = _choose_family(root_rng)
    change_point = root_rng.randint(6, max(6, min(16, max(6, context.tool_budget - 4))))
    worlds: list[CodeWorldSpec] = []
    for slot, role in enumerate(("healthy", "local_degradation", "shared_family_fault", "change")):
        seed = root_rng.randrange(0, 2**31 - 1)
        tool_states = _base_quality()
        family_states = _family_quality()
        rules: list[CodeContextRule] = []
        world_type = role
        world_change = None
        transition = None
        if role == "local_degradation":
            if local_prefix:
                world_type = "context_degradation"
                rules.append(
                    CodeContextRule(
                        tool_names=("read_file", "search_code"),
                        path_prefixes=(local_prefix,),
                        quality_override=ToolQualitySpec(availability=0.72, semantic_accuracy=0.78, structure_fidelity=0.75, latency_scale=1.35),
                    )
                )
            else:
                world_type = "single_tool_degradation"
                tool_states[local_tool] = ToolQualitySpec(availability=0.65, semantic_accuracy=0.75, structure_fidelity=0.72, latency_scale=1.4)
        elif role == "shared_family_fault":
            family_states[family] = ToolQualitySpec(availability=0.68, semantic_accuracy=0.74, structure_fidelity=0.70, latency_scale=1.35)
            world_type = "shared_family_fault"
        elif role == "change":
            world_type = "abrupt_change" if root_rng.random() < 0.65 else "gradual_change"
            transition = "abrupt" if world_type == "abrupt_change" else "gradual"
            world_change = change_point
        worlds.append(
            CodeWorldSpec(
                instance_id=instance_id,
                image_name=image_name,
                base_revision=base_revision,
                coupling_id=coupling,
                latent_world_id=stable_hash({"coupling": coupling, "slot": slot, "seed": seed}, prefix="code-latent-world-v1"),
                world_slot_role=role,
                world_type=world_type,
                seed=seed,
                session_state="healthy",
                tool_states=tool_states,
                family_states=family_states,
                context_rules=tuple(rules),
                change_point=world_change,
                change_transition=transition,
                variant_id="base",
            )
        )
    _validate_required_worlds(worlds, instance_id, image_name, base_revision, coupling)
    return tuple(worlds)


def _choose_local_tool(context: CodeWorldSamplingContext, rng: random.Random) -> str:
    candidates = [name for name in CODE_TOOL_NAMES if name not in {"apply_patch", "run_command"}]
    return candidates[rng.randrange(len(candidates))]


def _choose_prefix(context: CodeWorldSamplingContext, rng: random.Random) -> str | None:
    dirs = [value.strip("/") for value in context.top_level_dirs if value not in {".git", ""}]
    return (dirs[rng.randrange(len(dirs))] if dirs and rng.random() < 0.5 else None)


def _choose_family(rng: random.Random) -> str:
    return ("inspection_core", "validation_core")[rng.randrange(2)]


def _validate_required_worlds(worlds, instance_id, image_name, base_revision, coupling):
    if len(worlds) != 4 or {world.world_slot_role for world in worlds} != {"healthy", "local_degradation", "shared_family_fault", "change"}:
        raise AssertionError("required Code world set must contain exactly four roles")
    if any(world.instance_id != instance_id or world.image_name != image_name or world.base_revision != base_revision or world.coupling_id != coupling for world in worlds):
        raise AssertionError("required Code worlds must share task/repository/coupling")
    if len({world.latent_world_id for world in worlds}) != 4:
        raise AssertionError("required Code worlds need distinct latent ids")


__all__ = ["public_sampling_context", "sample_required_worlds"]
