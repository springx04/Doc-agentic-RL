"""Decision-level sibling grouping for Code Bayes-ARPO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .validity import GroupValidity, fail_closed_advantage, validate_group_records


GROUPING_SCHEMA_VERSION = "code-grouping-v1"


def decision_prefix_hash(messages: Iterable[Any]) -> str:
    payload = json.dumps(list(messages), sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decision_group_id(*, instance_id: str, latent_world_id: str, decision_event_id: str, decision_prefix_hash_value: str, repo_state_digest: str, belief_state_digest: str, world_runtime_digest: str, coupling_id: str, world_slot_role: str, variant_id: str = "base", policy_version: str = "code-policy-v1", return_definition: str = "terminal-utility-v1") -> str:
    payload = {
        "environment": "code", "instance_id": instance_id, "latent_world_id": latent_world_id, "decision_event_id": decision_event_id,
        "decision_prefix_hash": decision_prefix_hash_value, "repo_state_digest": repo_state_digest, "belief_state_digest": belief_state_digest,
        "world_runtime_digest": world_runtime_digest, "coupling_id": coupling_id, "return_definition": return_definition,
        "world_slot_role": world_slot_role, "variant_id": variant_id, "policy_version": policy_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def coupling_id(*, instance_id: str, image_name: str, base_revision: str | None, rollout_seed: int) -> str:
    return hashlib.sha256(json.dumps({"environment": "code", "instance_id": instance_id, "image_name": image_name, "base_revision": base_revision, "rollout_seed": rollout_seed}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class DecisionSiblingGroup:
    group_id: str
    instance_id: str
    latent_world_id: str
    world_slot_role: str
    repo_state_digest: str
    decision_prefix_hash: str
    expected_k: int
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))

    def validate(self) -> GroupValidity:
        return validate_group_records(self.records, expected_k=self.expected_k)

    def advantages(self, rewards: Iterable[float]) -> tuple[float, ...]:
        return fail_closed_advantage(rewards, self.records, expected_k=self.expected_k)


def validate_decision_group(records: Iterable[Mapping[str, Any]], *, expected_k: int) -> GroupValidity:
    return validate_group_records(records, expected_k=expected_k)


__all__ = ["DecisionSiblingGroup", "GROUPING_SCHEMA_VERSION", "coupling_id", "decision_group_id", "decision_prefix_hash", "validate_decision_group"]
