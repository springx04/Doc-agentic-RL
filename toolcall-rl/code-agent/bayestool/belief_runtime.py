"""Policy-visible rendering of Code task state and belief."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from ..config import CODE_BELIEF_SCHEMA_VERSION
except ImportError:  # pragma: no cover
    from config import CODE_BELIEF_SCHEMA_VERSION

from .belief import CodeBeliefFilter, CodeBeliefSnapshot
from .task_state import CodeTaskStateView


HIDDEN_LABEL_KEYS = frozenset({"world_type", "world_slot_role", "latent_world_id", "gold_patch", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "test_patch", "official_resolved", "corruption_type", "true_tool_quality"})


@dataclass(frozen=True)
class PolicyVisibleBelief:
    task_state: dict[str, Any]
    belief: dict[str, Any]
    schema_version: str = CODE_BELIEF_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = {"task_state": self.task_state, "tool_belief": self.belief, "schema_version": self.schema_version}
        assert_no_hidden_labels(value)
        return value

    def to_prompt(self, *, max_chars: int = 8_000) -> str:
        task_payload = dict(self.task_state)
        belief_payload = dict(self.belief)
        payload = {"task_state": task_payload, "tool_belief": belief_payload, "schema_version": self.schema_version}
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(serialized) > max_chars:
            # Keep the state semantically complete while bounding only the
            # inspect/search histories that can grow with a long episode.
            for key, limit in (("inspected_files", 64), ("touched_files", 64), ("search_queries", 20)):
                value = task_payload.get(key)
                if isinstance(value, list) and len(value) > limit:
                    task_payload[key] = value[-limit:]
            payload = {"task_state": task_payload, "tool_belief": belief_payload, "schema_version": self.schema_version}
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(serialized) > max_chars:
            raise ValueError("policy belief prompt would exceed its fixed limit")
        assert_no_hidden_labels(payload)
        return f"<task_state>{json.dumps(task_payload, ensure_ascii=False, sort_keys=True)}</task_state>\n<tool_belief>{json.dumps(belief_payload, ensure_ascii=False, sort_keys=True)}</tool_belief>"


def assert_no_hidden_labels(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in HIDDEN_LABEL_KEYS or re.search(r"gold|fail_to_pass|pass_to_pass|verifier|official", str(key), re.I):
                raise ValueError(f"hidden Code label leaked into policy-visible data: {key}")
            assert_no_hidden_labels(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            assert_no_hidden_labels(item)
    elif isinstance(value, str) and re.search(r"latent_world_id|gold_patch|FAIL_TO_PASS|PASS_TO_PASS", value):
        raise ValueError("hidden Code label leaked into policy-visible text")


def build_policy_visible_belief(task_state: CodeTaskStateView, belief: CodeBeliefFilter | CodeBeliefSnapshot) -> PolicyVisibleBelief:
    snapshot = belief.snapshot() if isinstance(belief, CodeBeliefFilter) else belief
    value = PolicyVisibleBelief(task_state.to_prompt_dict(), snapshot.to_prompt_dict())
    assert_no_hidden_labels(value.to_dict())
    return value


__all__ = ["HIDDEN_LABEL_KEYS", "PolicyVisibleBelief", "assert_no_hidden_labels", "build_policy_visible_belief"]
