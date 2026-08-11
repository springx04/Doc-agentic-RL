"""Independent lease cloning for decision-level sibling continuations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

try:
    from ..env.checkpoint import CodeBranchCheckpoint
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from env.checkpoint import CodeBranchCheckpoint


@dataclass
class SiblingLease:
    variant_id: str
    lease: Any
    checkpoint: CodeBranchCheckpoint


async def create_sibling_leases(client: Any, parent: CodeBranchCheckpoint, *, image_name: str, instance_id: str, count: int, cwd: str = "/testbed") -> list[SiblingLease]:
    if count not in {4, 8}:
        raise ValueError("sibling count must be 4 or 8")
    parent.repo.validate()
    siblings: list[SiblingLease] = []
    try:
        for index in range(count):
            lease = await client.allocate(image_name, instance_id, cwd=cwd, base_revision=parent.repo.base_revision)
            restored = await client.reset_to_patch(lease.lease_id, parent.repo.patch, cwd=lease.cwd)
            if not restored.ok:
                await client.close(lease.lease_id)
                raise RuntimeError(f"failed to restore sibling {index}: {restored.output}")
            sibling = parent.clone()
            sibling.repo = parent.repo
            sibling.decision_state = copy.deepcopy(parent.decision_state)
            siblings.append(SiblingLease(str(index), lease, sibling))
        if len({item.lease.lease_id for item in siblings}) != count:
            raise AssertionError("sibling continuations must use independent leases")
        return siblings
    except Exception:
        for sibling in siblings:
            try:
                await client.close(sibling.lease.lease_id)
            except Exception:
                pass
        raise


async def close_sibling_leases(client: Any, siblings: list[SiblingLease]) -> None:
    for sibling in siblings:
        await client.close(sibling.lease.lease_id)


__all__ = ["SiblingLease", "close_sibling_leases", "create_sibling_leases"]
