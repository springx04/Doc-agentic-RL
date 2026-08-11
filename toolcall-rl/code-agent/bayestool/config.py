"""Code-specific BayesTool configuration; never imports Doc configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from ..config import (
        CODE_ALLOWED_GROUP_SIZES,
        CODE_BELIEF_FEATURE_DIM,
        CODE_BELIEF_SCHEMA_VERSION,
        CODE_DEFAULT_BRANCH_HORIZON,
        CODE_DEFAULT_TOOL_BUDGET,
        CODE_ERROR_FAMILIES,
        CODE_SESSION_STATES,
        CODE_STATUS_NAMES,
        CODE_TOOL_FAMILIES,
        CODE_TOOL_NAMES,
        CODE_TOOL_SCHEMA_VERSION,
        CODE_WORLD_SCHEMA_VERSION,
        CODE_WORLD_SLOT_ROLES,
    )
except ImportError:  # pragma: no cover
    from config import (
        CODE_ALLOWED_GROUP_SIZES,
        CODE_BELIEF_FEATURE_DIM,
        CODE_BELIEF_SCHEMA_VERSION,
        CODE_DEFAULT_BRANCH_HORIZON,
        CODE_DEFAULT_TOOL_BUDGET,
        CODE_ERROR_FAMILIES,
        CODE_SESSION_STATES,
        CODE_STATUS_NAMES,
        CODE_TOOL_FAMILIES,
        CODE_TOOL_NAMES,
        CODE_TOOL_SCHEMA_VERSION,
        CODE_WORLD_SCHEMA_VERSION,
        CODE_WORLD_SLOT_ROLES,
    )


@dataclass(frozen=True)
class CodeBayesConfig:
    tool_budget: int = CODE_DEFAULT_TOOL_BUDGET
    max_tool_budget: int = 30
    group_size: int = 4
    required_worlds: tuple[str, ...] = CODE_WORLD_SLOT_ROLES
    branch_horizon: int = CODE_DEFAULT_BRANCH_HORIZON
    change_call_range: tuple[int, int] = (6, 16)
    belief_feature_dim: int = CODE_BELIEF_FEATURE_DIM
    belief_schema_version: str = CODE_BELIEF_SCHEMA_VERSION
    tool_schema_version: str = CODE_TOOL_SCHEMA_VERSION
    world_schema_version: str = CODE_WORLD_SCHEMA_VERSION
    world_prior: dict[str, float] = field(
        default_factory=lambda: {
            "healthy": 0.20,
            "single_tool_degradation": 0.25,
            "context_degradation": 0.20,
            "shared_family_fault": 0.20,
            "abrupt_change": 0.10,
            "gradual_change": 0.05,
        }
    )

    def __post_init__(self) -> None:
        if self.group_size not in CODE_ALLOWED_GROUP_SIZES:
            raise ValueError("Code group_size must be 4 or 8")
        if self.belief_feature_dim != CODE_BELIEF_FEATURE_DIM:
            raise ValueError("Code belief feature dimension must be exactly 96")
        if tuple(self.required_worlds) != tuple(CODE_WORLD_SLOT_ROLES):
            raise ValueError("Code required world roles are fixed")


DEFAULT_CODE_BAYES_CONFIG = CodeBayesConfig()

__all__ = [
    "CODE_ALLOWED_GROUP_SIZES",
    "CODE_BELIEF_FEATURE_DIM",
    "CODE_BELIEF_SCHEMA_VERSION",
    "CODE_ERROR_FAMILIES",
    "CODE_SESSION_STATES",
    "CODE_STATUS_NAMES",
    "CODE_TOOL_FAMILIES",
    "CODE_TOOL_NAMES",
    "CODE_WORLD_SLOT_ROLES",
    "CodeBayesConfig",
    "DEFAULT_CODE_BAYES_CONFIG",
]
