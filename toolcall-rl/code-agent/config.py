"""Configuration for the isolated Code Agent / SWE environment.

This module intentionally has no imports from the document-agent runtime.  The
directory is placed on ``PYTHONPATH`` by the Code entry points, just like the
existing ``toolcall-rl`` runtime is.  All environment variables use the
``CODE_`` namespace so a Doc run cannot silently redirect a Code run to its
checkpoints or caches.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


CODE_TOOL_NAMES: tuple[str, ...] = (
    "list_tree",
    "search_code",
    "read_file",
    "apply_patch",
    "git_diff",
    "run_tests",
    "run_checks",
    "run_command",
)

CODE_TOOL_FAMILIES: dict[str, tuple[str, ...]] = {
    "inspection_core": ("list_tree", "search_code", "read_file", "git_diff"),
    "mutation_core": ("apply_patch",),
    "validation_core": ("run_tests", "run_checks"),
    "execution_core": ("run_tests", "run_checks", "run_command"),
}

CODE_STATUS_NAMES: tuple[str, ...] = ("ok", "partial", "error", "timeout", "invalid", "empty")
CODE_ERROR_FAMILIES: tuple[str, ...] = (
    "none",
    "availability",
    "inspection",
    "mutation",
    "validation",
    "execution",
    "latency",
    "protocol",
)
CODE_SESSION_STATES: tuple[str, ...] = ("healthy", "degraded", "overloaded", "outage")
CODE_WORLD_SLOT_ROLES: tuple[str, ...] = (
    "healthy",
    "local_degradation",
    "shared_family_fault",
    "change",
)
CODE_ALLOWED_GROUP_SIZES: frozenset[int] = frozenset({4, 8})
CODE_DEFAULT_TOOL_BUDGET = 24
CODE_MAX_TOOL_BUDGET = 30
CODE_DEFAULT_BRANCH_HORIZON = 3
CODE_MAX_OUTPUT_CHARS = 16_384
CODE_MAX_READ_LINES = 240
CODE_MAX_TREE_ENTRIES = 2_000
CODE_MAX_SEARCH_RESULTS = 50
CODE_BELIEF_FEATURE_DIM = 96
CODE_BELIEF_SCHEMA_VERSION = "code-belief-v1"
CODE_TOOL_SCHEMA_VERSION = "code-tools-v1"
CODE_WORLD_SCHEMA_VERSION = "code-world-v1"
CODE_CHECKPOINT_SCHEMA_VERSION = "code-checkpoint-v1"
CODE_MANIFEST_SCHEMA_VERSION = "code-manifest-v1"

CODE_WORLD_PRIORS: tuple[tuple[str, float], ...] = (
    ("healthy", 0.20),
    ("single_tool_degradation", 0.25),
    ("context_degradation", 0.20),
    ("shared_family_fault", 0.20),
    ("abrupt_change", 0.10),
    ("gradual_change", 0.05),
)


def stable_hash(value: Any, *, prefix: str = "") -> str:
    """Hash canonical JSON without leaking object repr or dictionary order."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(prefix.encode("utf-8") + encoded).hexdigest()


def normalize_patch(patch: str) -> str:
    """Normalize only for digest purposes; return the original patch elsewhere."""

    if not isinstance(patch, str):
        return ""
    lines = patch.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines) + ("\n" if lines else "")


@dataclass(frozen=True)
class CodeConfig:
    """Runtime knobs for one Code rollout/training process."""

    env_server_url: str = "http://localhost:18090"
    docker_registry: str = ""
    tool_budget: int = CODE_DEFAULT_TOOL_BUDGET
    max_tool_budget: int = CODE_MAX_TOOL_BUDGET
    request_timeout: float = 180.0
    heartbeat_interval: float = 30.0
    max_output_chars: int = CODE_MAX_OUTPUT_CHARS
    max_read_lines: int = CODE_MAX_READ_LINES
    max_tree_entries: int = CODE_MAX_TREE_ENTRIES
    max_search_results: int = CODE_MAX_SEARCH_RESULTS
    save_traj_dir: Path = Path("outputs/code/trajectories")
    output_dir: Path = Path("outputs/code")
    cache_dir: Path = Path("outputs/code/cache")
    log_dir: Path = Path("outputs/code/logs")
    belief_checkpoint: Path | None = None
    q_checkpoint: Path | None = None
    risk_checkpoint: Path | None = None
    group_size: int = 4
    branch_horizon: int = CODE_DEFAULT_BRANCH_HORIZON
    max_siblings: int = 4
    context_max_chars: int = 120_000
    belief_prompt_max_chars: int = 8_000
    evaluation_timeout: int = 300
    environment: str = "code"
    tool_schema_version: str = CODE_TOOL_SCHEMA_VERSION
    world_schema_version: str = CODE_WORLD_SCHEMA_VERSION
    belief_schema_version: str = CODE_BELIEF_SCHEMA_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.environment != "code":
            raise ValueError("CodeConfig.environment must be 'code'")
        if not (1 <= int(self.tool_budget) <= int(self.max_tool_budget) <= CODE_MAX_TOOL_BUDGET):
            raise ValueError("tool_budget must be within [1, CODE_MAX_TOOL_BUDGET]")
        if int(self.group_size) not in CODE_ALLOWED_GROUP_SIZES:
            raise ValueError("group_size must be 4 or 8")
        if int(self.max_siblings) < 1 or int(self.max_siblings) > int(self.group_size):
            raise ValueError("max_siblings must be between 1 and group_size")
        if int(self.branch_horizon) < 1:
            raise ValueError("branch_horizon must be positive")

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir)

    def ensure_output_dirs(self) -> None:
        for path in (self.output_dir, self.save_traj_dir, self.cache_dir, self.log_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints/belief", "checkpoints/q", "checkpoints/risk", "checkpoints/policy", "eval", "manifests"):
            (Path(self.output_dir) / name).mkdir(parents=True, exist_ok=True)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "tool_schema_version": self.tool_schema_version,
            "world_schema_version": self.world_schema_version,
            "belief_schema_version": self.belief_schema_version,
            "belief_feature_dim": CODE_BELIEF_FEATURE_DIM,
            "group_size": self.group_size,
            "tool_budget": self.tool_budget,
            "branch_horizon": self.branch_horizon,
            "max_siblings": self.max_siblings,
            "belief_checkpoint": str(self.belief_checkpoint) if self.belief_checkpoint else None,
            "q_checkpoint": str(self.q_checkpoint) if self.q_checkpoint else None,
            "risk_checkpoint": str(self.risk_checkpoint) if self.risk_checkpoint else None,
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: Any) -> "CodeConfig":
        env = dict(os.environ if environ is None else environ)

        def value(name: str, default: Any) -> Any:
            return env.get(name, default)

        def integer(name: str, default: int) -> int:
            raw = value(name, default)
            try:
                return int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer") from exc

        def number(name: str, default: float) -> float:
            raw = value(name, default)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc

        data: dict[str, Any] = {
            "env_server_url": value("CODE_ENV_SERVER_URL", cls.env_server_url),
            "docker_registry": value("CODE_DOCKER_REGISTRY", cls.docker_registry),
            "tool_budget": integer("CODE_TOOL_BUDGET", cls.tool_budget),
            "request_timeout": number("CODE_ENV_HTTP_TIMEOUT", cls.request_timeout),
            "heartbeat_interval": number("CODE_HEARTBEAT_INTERVAL", cls.heartbeat_interval),
            "save_traj_dir": Path(value("CODE_SAVE_TRAJ_DIR", cls.save_traj_dir)),
            "output_dir": Path(value("CODE_OUTPUT_DIR", cls.output_dir)),
            "cache_dir": Path(value("CODE_CACHE_DIR", cls.cache_dir)),
            "log_dir": Path(value("CODE_LOG_DIR", cls.log_dir)),
            "belief_checkpoint": _optional_path(value("CODE_BELIEF_CHECKPOINT", None)),
            "q_checkpoint": _optional_path(value("CODE_Q_CHECKPOINT", None)),
            "risk_checkpoint": _optional_path(value("CODE_RISK_CHECKPOINT", None)),
            "group_size": integer("CODE_GROUP_SIZE", cls.group_size),
            "branch_horizon": integer("CODE_BRANCH_HORIZON", cls.branch_horizon),
        }
        data.update(overrides)
        return cls(**data)


def _optional_path(value: Any) -> Path | None:
    if value in (None, "", "none", "None"):
        return None
    return Path(str(value))


DEFAULT_CODE_CONFIG = CodeConfig()


def tool_schema_hash() -> str:
    schemas = {
        "list_tree": {"path": "string", "max_depth": "integer"},
        "search_code": {"query": "string", "path": "string", "glob": "string", "max_results": "integer"},
        "read_file": {"path": "string", "start_line": "integer", "end_line": "integer"},
        "apply_patch": {"patch": "string"},
        "git_diff": {},
        "run_tests": {"target": "string", "args": "string"},
        "run_checks": {"check": "string", "path": "string"},
        "run_command": {"command": "string"},
    }
    return stable_hash(schemas, prefix=CODE_TOOL_SCHEMA_VERSION)


CODE_TOOL_SCHEMA_HASH = tool_schema_hash()


__all__ = [
    "CODE_ALLOWED_GROUP_SIZES",
    "CODE_BELIEF_FEATURE_DIM",
    "CODE_BELIEF_SCHEMA_VERSION",
    "CODE_CHECKPOINT_SCHEMA_VERSION",
    "CODE_DEFAULT_BRANCH_HORIZON",
    "CODE_DEFAULT_TOOL_BUDGET",
    "CODE_ERROR_FAMILIES",
    "CODE_MANIFEST_SCHEMA_VERSION",
    "CODE_MAX_OUTPUT_CHARS",
    "CODE_MAX_TOOL_BUDGET",
    "CODE_SESSION_STATES",
    "CODE_STATUS_NAMES",
    "CODE_TOOL_FAMILIES",
    "CODE_TOOL_NAMES",
    "CODE_TOOL_SCHEMA_HASH",
    "CODE_TOOL_SCHEMA_VERSION",
    "CODE_WORLD_PRIORS",
    "CODE_WORLD_SCHEMA_VERSION",
    "CODE_WORLD_SLOT_ROLES",
    "CodeConfig",
    "DEFAULT_CODE_CONFIG",
    "normalize_patch",
    "stable_hash",
]
