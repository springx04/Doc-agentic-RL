"""Repository-aware branch checkpoints for Code sibling rollouts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

try:
    from ..config import CODE_CHECKPOINT_SCHEMA_VERSION, normalize_patch, stable_hash
except ImportError:  # pragma: no cover
    from config import CODE_CHECKPOINT_SCHEMA_VERSION, normalize_patch, stable_hash


def patch_sha256(patch: str) -> str:
    return hashlib.sha256(normalize_patch(patch).encode("utf-8")).hexdigest()


def repo_state_digest(
    instance_id: str,
    image_name: str,
    base_revision: str | None,
    patch: str,
) -> str:
    return stable_hash(
        {
            "instance_id": str(instance_id),
            "image_name": str(image_name),
            "base_revision": base_revision,
            "patch_sha256": patch_sha256(patch),
        },
        prefix="code-repo-state-v1",
    )


@dataclass(frozen=True)
class CodeRepoCheckpoint:
    instance_id: str
    image_name: str
    base_revision: str | None
    cwd: str
    patch: str
    patch_sha256: str
    repo_state_digest: str

    @classmethod
    def create(cls, *, instance_id: str, image_name: str, base_revision: str | None, cwd: str, patch: str) -> "CodeRepoCheckpoint":
        return cls(
            instance_id=str(instance_id),
            image_name=str(image_name),
            base_revision=base_revision,
            cwd=str(cwd),
            patch=patch,
            patch_sha256=patch_sha256(patch),
            repo_state_digest=repo_state_digest(instance_id, image_name, base_revision, patch),
        )

    def validate(self) -> None:
        expected_patch = patch_sha256(self.patch)
        expected_state = repo_state_digest(self.instance_id, self.image_name, self.base_revision, self.patch)
        if self.patch_sha256 != expected_patch:
            raise ValueError("repository checkpoint patch digest mismatch")
        if self.repo_state_digest != expected_state:
            raise ValueError("repository checkpoint state digest mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": CODE_CHECKPOINT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeRepoCheckpoint":
        if value.get("schema_version") not in (None, CODE_CHECKPOINT_SCHEMA_VERSION):
            raise ValueError("unsupported Code repository checkpoint schema")
        record = cls(
            instance_id=str(value["instance_id"]),
            image_name=str(value["image_name"]),
            base_revision=value.get("base_revision"),
            cwd=str(value.get("cwd") or "."),
            patch=str(value.get("patch") or ""),
            patch_sha256=str(value["patch_sha256"]),
            repo_state_digest=str(value["repo_state_digest"]),
        )
        record.validate()
        return record


@dataclass
class CodeBranchCheckpoint:
    repo: CodeRepoCheckpoint
    messages: list[Any] = field(default_factory=list)
    token_state: Any = None
    task_state: Any = None
    world_state: Any = None
    belief_state: Any = None
    decision_state: Any = None
    call_index: int = 0
    remaining_tool_budget: int = 0
    decision_prefix_hash: str = ""
    runtime_state_digest: str = ""

    def clone(self) -> "CodeBranchCheckpoint":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        self.repo.validate()
        return {
            "schema_version": CODE_CHECKPOINT_SCHEMA_VERSION,
            "repo": self.repo.to_dict(),
            "messages": copy.deepcopy(self.messages),
            "token_state": copy.deepcopy(self.token_state),
            "task_state": _serialize(self.task_state),
            "world_state": _serialize(self.world_state),
            "belief_state": _serialize(self.belief_state),
            "decision_state": _serialize(self.decision_state),
            "call_index": int(self.call_index),
            "remaining_tool_budget": int(self.remaining_tool_budget),
            "decision_prefix_hash": self.decision_prefix_hash,
            "runtime_state_digest": self.runtime_state_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeBranchCheckpoint":
        if value.get("schema_version") not in (None, CODE_CHECKPOINT_SCHEMA_VERSION):
            raise ValueError("unsupported Code branch checkpoint schema")
        return cls(
            repo=CodeRepoCheckpoint.from_dict(value["repo"]),
            messages=copy.deepcopy(list(value.get("messages") or [])),
            token_state=copy.deepcopy(value.get("token_state")),
            task_state=copy.deepcopy(value.get("task_state")),
            world_state=copy.deepcopy(value.get("world_state")),
            belief_state=copy.deepcopy(value.get("belief_state")),
            decision_state=copy.deepcopy(value.get("decision_state")),
            call_index=int(value.get("call_index", 0)),
            remaining_tool_budget=int(value.get("remaining_tool_budget", 0)),
            decision_prefix_hash=str(value.get("decision_prefix_hash") or ""),
            runtime_state_digest=str(value.get("runtime_state_digest") or ""),
        )


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serialize(value.to_dict())
    if hasattr(value, "__dict__"):
        return _serialize(vars(value))
    return repr(value)


async def make_repo_checkpoint(client: Any, lease: Any) -> CodeRepoCheckpoint:
    patch = await client.diff(lease.lease_id, cwd=lease.cwd)
    return CodeRepoCheckpoint.create(
        instance_id=lease.instance_id,
        image_name=lease.image_name,
        base_revision=lease.base_revision,
        cwd=lease.cwd,
        patch=patch,
    )


__all__ = [
    "CodeBranchCheckpoint",
    "CodeRepoCheckpoint",
    "make_repo_checkpoint",
    "normalize_patch",
    "patch_sha256",
    "repo_state_digest",
]
