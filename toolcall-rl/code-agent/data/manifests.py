"""Code manifest and checkpoint capability validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    from ..config import CODE_BELIEF_FEATURE_DIM, CODE_BELIEF_SCHEMA_VERSION, CODE_MANIFEST_SCHEMA_VERSION, CODE_TOOL_SCHEMA_VERSION, CODE_WORLD_SCHEMA_VERSION, CODE_ALLOWED_GROUP_SIZES
except ImportError:  # pragma: no cover
    from config import CODE_BELIEF_FEATURE_DIM, CODE_BELIEF_SCHEMA_VERSION, CODE_MANIFEST_SCHEMA_VERSION, CODE_TOOL_SCHEMA_VERSION, CODE_WORLD_SCHEMA_VERSION, CODE_ALLOWED_GROUP_SIZES


@dataclass(frozen=True)
class CodeManifest:
    stage: str
    base_model: str
    tool_schema_version: str = CODE_TOOL_SCHEMA_VERSION
    world_schema_version: str = CODE_WORLD_SCHEMA_VERSION
    belief_schema_version: str = CODE_BELIEF_SCHEMA_VERSION
    belief_feature_dim: int = CODE_BELIEF_FEATURE_DIM
    belief_checkpoint: str | None = None
    q_checkpoint: str | None = None
    risk_checkpoint: str | None = None
    group_size: int = 4
    tool_budget: int = 24
    environment: str = "code"
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.environment != "code":
            raise ValueError("manifest environment must be code")
        if self.tool_schema_version != CODE_TOOL_SCHEMA_VERSION or self.world_schema_version != CODE_WORLD_SCHEMA_VERSION or self.belief_schema_version != CODE_BELIEF_SCHEMA_VERSION:
            raise ValueError("Code manifest schema version mismatch")
        if self.belief_feature_dim != CODE_BELIEF_FEATURE_DIM:
            raise ValueError("Code manifest belief feature dimension mismatch")
        if self.group_size not in CODE_ALLOWED_GROUP_SIZES:
            raise ValueError("Code manifest group_size must be 4 or 8")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"manifest_schema_version": CODE_MANIFEST_SCHEMA_VERSION, "environment": self.environment, "stage": self.stage, "base_model": self.base_model, "tool_schema_version": self.tool_schema_version, "world_schema_version": self.world_schema_version, "belief_schema_version": self.belief_schema_version, "belief_feature_dim": self.belief_feature_dim, "belief_checkpoint": self.belief_checkpoint, "q_checkpoint": self.q_checkpoint, "risk_checkpoint": self.risk_checkpoint, "group_size": self.group_size, "tool_budget": self.tool_budget, **self.extra}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeManifest":
        if value.get("manifest_schema_version") not in (None, CODE_MANIFEST_SCHEMA_VERSION):
            raise ValueError("unsupported Code manifest schema")
        known = {"manifest_schema_version", "environment", "stage", "base_model", "tool_schema_version", "world_schema_version", "belief_schema_version", "belief_feature_dim", "belief_checkpoint", "q_checkpoint", "risk_checkpoint", "group_size", "tool_budget"}
        result = cls(stage=str(value.get("stage") or ""), base_model=str(value.get("base_model") or ""), tool_schema_version=str(value.get("tool_schema_version") or ""), world_schema_version=str(value.get("world_schema_version") or ""), belief_schema_version=str(value.get("belief_schema_version") or ""), belief_feature_dim=int(value.get("belief_feature_dim", 0)), belief_checkpoint=value.get("belief_checkpoint"), q_checkpoint=value.get("q_checkpoint"), risk_checkpoint=value.get("risk_checkpoint"), group_size=int(value.get("group_size", 0)), tool_budget=int(value.get("tool_budget", 0)), environment=str(value.get("environment") or ""), extra={key: item for key, item in value.items() if key not in known})
        result.validate()
        return result

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CodeManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_capabilities(manifest: CodeManifest, *, stage: str, required_checkpoints: bool = True) -> None:
    manifest.validate()
    if manifest.stage != stage:
        raise ValueError(f"manifest stage mismatch: expected {stage}, got {manifest.stage}")
    if required_checkpoints:
        missing = [name for name in ("belief_checkpoint", "q_checkpoint", "risk_checkpoint") if not getattr(manifest, name)]
        if missing:
            raise ValueError(f"stage {stage} requires Code checkpoints: {', '.join(missing)}")


__all__ = ["CodeManifest", "validate_capabilities"]
