"""Hidden-world schema for the Code environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

try:
    from ..config import CODE_TOOL_FAMILIES, CODE_TOOL_NAMES, stable_hash
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from config import CODE_TOOL_FAMILIES, CODE_TOOL_NAMES, stable_hash
    from schemas import CodeToolResult


def _clip(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(low, min(high, number))


@dataclass(frozen=True)
class ToolQualitySpec:
    availability: float = 1.0
    semantic_accuracy: float = 1.0
    structure_fidelity: float = 1.0
    calibration_temperature: float = 1.0
    calibration_bias: float = 0.0
    relative_cost: float = 1.0
    latency_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "availability", _clip(self.availability))
        object.__setattr__(self, "semantic_accuracy", _clip(self.semantic_accuracy))
        object.__setattr__(self, "structure_fidelity", _clip(self.structure_fidelity))
        object.__setattr__(self, "calibration_temperature", max(0.05, float(self.calibration_temperature)))
        object.__setattr__(self, "calibration_bias", max(-1.0, min(1.0, float(self.calibration_bias))))
        object.__setattr__(self, "relative_cost", max(0.05, float(self.relative_cost)))
        object.__setattr__(self, "latency_scale", max(0.05, float(self.latency_scale)))

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class CodeContextRule:
    tool_names: tuple[str, ...]
    repository: bool = False
    path_prefixes: tuple[str, ...] = ()
    file_extensions: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    command_classes: tuple[str, ...] = ()
    quality_override: ToolQualitySpec | None = None

    def matches(self, tool_name: str, context: Mapping[str, Any] | None = None) -> bool:
        if tool_name not in self.tool_names:
            return False
        public = context or {}
        if self.repository:
            return True
        path = str(public.get("path") or public.get("requested_path") or "")
        if self.path_prefixes and not any(path.replace("\\", "/").startswith(prefix.rstrip("/") + "/") or path == prefix for prefix in self.path_prefixes):
            return False
        extension = str(public.get("file_extension") or "")
        if self.file_extensions and extension not in self.file_extensions:
            return False
        language = str(public.get("language") or "")
        if self.languages and language not in self.languages:
            return False
        command_class = str(public.get("command_class") or "")
        if self.command_classes and command_class not in self.command_classes:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_names": list(self.tool_names),
            "repository": self.repository,
            "path_prefixes": list(self.path_prefixes),
            "file_extensions": list(self.file_extensions),
            "languages": list(self.languages),
            "command_classes": list(self.command_classes),
            "quality_override": self.quality_override.to_dict() if self.quality_override else None,
        }


@dataclass(frozen=True)
class CodeWorldSamplingContext:
    tool_budget: int
    repo_file_count: int
    tracked_extensions: tuple[str, ...]
    top_level_dirs: tuple[str, ...]
    languages: tuple[str, ...]
    detected_frameworks: tuple[str, ...]
    tool_argument_capabilities: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "tool_budget": int(self.tool_budget),
            "repo_file_count": int(self.repo_file_count),
            "tracked_extensions": list(self.tracked_extensions),
            "top_level_dirs": list(self.top_level_dirs),
            "languages": list(self.languages),
            "detected_frameworks": list(self.detected_frameworks),
            "tool_argument_capabilities": dict(self.tool_argument_capabilities),
        }


@dataclass(frozen=True)
class CodeWorldSpec:
    instance_id: str
    image_name: str
    base_revision: str | None
    coupling_id: str
    latent_world_id: str
    world_slot_role: Literal["healthy", "local_degradation", "shared_family_fault", "change"]
    world_type: str
    seed: int
    session_state: str
    tool_states: dict[str, ToolQualitySpec]
    family_states: dict[str, ToolQualitySpec]
    context_rules: tuple[CodeContextRule, ...] = ()
    change_point: int | None = None
    change_transition: str | None = None
    variant_id: str = "base"

    def __post_init__(self) -> None:
        missing = set(CODE_TOOL_NAMES) - set(self.tool_states)
        if missing:
            raise ValueError(f"world is missing tool states: {sorted(missing)}")
        if self.world_slot_role not in {"healthy", "local_degradation", "shared_family_fault", "change"}:
            raise ValueError(f"invalid world slot role: {self.world_slot_role}")

    def to_latent_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "image_name": self.image_name,
            "base_revision": self.base_revision,
            "coupling_id": self.coupling_id,
            "latent_world_id": self.latent_world_id,
            "world_slot_role": self.world_slot_role,
            "world_type": self.world_type,
            "seed": self.seed,
            "session_state": self.session_state,
            "tool_states": {name: value.to_dict() for name, value in self.tool_states.items()},
            "family_states": {name: value.to_dict() for name, value in self.family_states.items()},
            "context_rules": [rule.to_dict() for rule in self.context_rules],
            "change_point": self.change_point,
            "change_transition": self.change_transition,
            "variant_id": self.variant_id,
        }

    @property
    def runtime_digest(self) -> str:
        return stable_hash(self.to_latent_dict(), prefix="code-world-runtime-v1")


@dataclass(frozen=True)
class ToolStateLabel:
    """Trainer-only hidden label.  Never call ``to_dict`` for policy context."""

    tool_name: str
    availability: float
    semantic_accuracy: float
    structure_fidelity: float
    session_state: str
    world_slot_role: str
    latent_world_id: str
    corruption_type: str | None = None

    def to_trainer_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class WorldEvent:
    call_index: int
    tool_name: str
    result: CodeToolResult
    information_gain: float
    public_context: dict[str, Any] = field(default_factory=dict)
    label: ToolStateLabel | None = None

    def visible_dict(self) -> dict[str, Any]:
        value = self.result.visible_dict()
        value.update({"call_index": self.call_index, "information_gain": self.information_gain, "public_context": dict(self.public_context)})
        return value

    def trainer_dict(self) -> dict[str, Any]:
        value = self.visible_dict()
        value.update({"failure_origin": self.result.failure_origin, "valid_for_rl": self.result.valid_for_rl, "label": self.label.to_trainer_dict() if self.label else None})
        return value


def quality_for_tool(spec: CodeWorldSpec, tool_name: str, public_context: Mapping[str, Any] | None = None, call_index: int = 0) -> ToolQualitySpec:
    quality = spec.tool_states[tool_name]
    for family, names in CODE_TOOL_FAMILIES.items():
        if tool_name in names and family in spec.family_states:
            family_quality = spec.family_states[family]
            quality = ToolQualitySpec(
                availability=quality.availability * family_quality.availability,
                semantic_accuracy=quality.semantic_accuracy * family_quality.semantic_accuracy,
                structure_fidelity=quality.structure_fidelity * family_quality.structure_fidelity,
                calibration_temperature=quality.calibration_temperature,
                calibration_bias=quality.calibration_bias,
                relative_cost=max(quality.relative_cost, family_quality.relative_cost),
                latency_scale=quality.latency_scale * family_quality.latency_scale,
            )
            break
    for rule in spec.context_rules:
        if rule.matches(tool_name, public_context) and rule.quality_override is not None:
            override = rule.quality_override
            quality = ToolQualitySpec(
                availability=quality.availability * override.availability,
                semantic_accuracy=quality.semantic_accuracy * override.semantic_accuracy,
                structure_fidelity=quality.structure_fidelity * override.structure_fidelity,
                calibration_temperature=quality.calibration_temperature * override.calibration_temperature,
                calibration_bias=quality.calibration_bias + override.calibration_bias,
                relative_cost=max(quality.relative_cost, override.relative_cost),
                latency_scale=quality.latency_scale * override.latency_scale,
            )
    if spec.change_point is not None and call_index >= spec.change_point:
        factor = 0.85 if spec.change_transition == "gradual" else 0.65
        quality = ToolQualitySpec(
            availability=quality.availability * factor,
            semantic_accuracy=quality.semantic_accuracy * factor,
            structure_fidelity=quality.structure_fidelity * factor,
            calibration_temperature=quality.calibration_temperature,
            calibration_bias=quality.calibration_bias,
            relative_cost=quality.relative_cost,
            latency_scale=quality.latency_scale * (1.1 if factor < 0.8 else 1.03),
        )
    return quality


__all__ = ["CodeContextRule", "CodeWorldSamplingContext", "CodeWorldSpec", "ToolQualitySpec", "ToolStateLabel", "WorldEvent", "quality_for_tool"]
