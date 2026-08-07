"""Canonical, causally ordered BayesTool belief replay records."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


CANONICAL_REPLAY_SCHEMA_VERSION = 1


def _stable_id(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8", "surrogatepass")
    return hashlib.sha256(payload).hexdigest()[:24]


def _trace_events(trace: Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    return [
        item
        for item in (trace or ())
        if isinstance(item, Mapping)
        and item.get("kind") == "tool_call"
        and bool(item.get("executed"))
    ]


def export_canonical_replay(
    metadata: Mapping[str, Any],
    *,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Build one replay row from the post-rollout trace.

    The event list is the sole source of belief supervision.  It contains the
    public observation, explicit before/after task state, the hidden label in
    a separate field for offline supervision, and the next action identifier.
    No event is synthesized when a tool was not actually executed.
    """

    tool_execution = metadata.get("tool_execution")
    tool_calls = tool_execution.get("calls", []) if isinstance(tool_execution, Mapping) else []
    trace = metadata.get("execution_trace") or tool_calls
    events = _trace_events(trace if isinstance(trace, Sequence) else ())
    bayes = metadata.get("bayestool") if isinstance(metadata.get("bayestool"), Mapping) else {}
    coupling_id = str(metadata.get("coupling_id") or bayes.get("coupling_id") or "")
    latent_world_id = str(metadata.get("latent_world_id") or bayes.get("latent_world_id") or "")
    replica_id = int(metadata.get("replica_id", bayes.get("replica_id", 0)) or 0)
    document_hash = str(metadata.get("document_hash") or bayes.get("document_hash") or "")
    base_id = trajectory_id or metadata.get("rollout_id") or metadata.get("sample_id") or "trajectory"
    canonical_id = str(base_id)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(events):
        tool_name = str(item.get("tool") or item.get("parsed_tool_name") or "")
        arguments = item.get("arguments") if isinstance(item.get("arguments"), Mapping) else {}
        world_event = item.get("world_event") if isinstance(item.get("world_event"), Mapping) else {}
        hidden = item.get("bayes_supervision") if isinstance(item.get("bayes_supervision"), Mapping) else {}
        observed_result = item.get("observed_result")
        if observed_result is None:
            observed_result = item.get("result", "")
        before = item.get("task_state_before") if isinstance(item.get("task_state_before"), Mapping) else {}
        after = item.get("task_state_after") if isinstance(item.get("task_state_after"), Mapping) else {}
        next_tool_id = None
        if index + 1 < len(events):
            next_tool_id = str(events[index + 1].get("tool") or events[index + 1].get("parsed_tool_name") or "")
        rows.append(
            {
                "event_index": index,
                "call_id": int(world_event.get("call_id", item.get("call_id", index)) or index),
                "tool_id": tool_name,
                "arguments": dict(arguments),
                "task_state_before": dict(before),
                "observed_result": str(observed_result or ""),
                "world_event": dict(world_event),
                "hidden_label": dict(hidden),
                "task_state_after": dict(after),
                "next_tool_id": next_tool_id,
                "result_status": item.get("result_status"),
                "execution_succeeded": bool(world_event.get("execution_succeeded", item.get("success", False))),
                "observation_delivered": bool(world_event.get("observation_delivered", True)),
            }
        )
    return {
        "schema_version": CANONICAL_REPLAY_SCHEMA_VERSION,
        "trajectory_id": canonical_id,
        "replay_id": _stable_id(canonical_id, coupling_id, latent_world_id, replica_id),
        "coupling_id": coupling_id,
        "latent_world_id": latent_world_id,
        "replica_id": replica_id,
        "document_hash": document_hash,
        "event_count": len(rows),
        "events": rows,
    }


def validate_canonical_replay(record: Mapping[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the row is trainable."""

    errors: list[str] = []
    if int(record.get("schema_version", -1)) != CANONICAL_REPLAY_SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not str(record.get("trajectory_id") or ""):
        errors.append("missing trajectory_id")
    events = record.get("events")
    if not isinstance(events, list):
        return errors + ["events must be a list"]
    if int(record.get("event_count", len(events))) != len(events):
        errors.append("event_count mismatch")
    previous_call = -1
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"event {index} is not an object")
            continue
        if int(event.get("event_index", -1)) != index:
            errors.append(f"event {index} has non-canonical event_index")
        call_id = int(event.get("call_id", -1))
        if call_id <= previous_call:
            errors.append(f"event {index} call_id is not strictly increasing")
        previous_call = call_id
        if not str(event.get("tool_id") or ""):
            errors.append(f"event {index} missing tool_id")
        for required in ("task_state_before", "observed_result", "world_event", "hidden_label", "task_state_after"):
            if required not in event:
                errors.append(f"event {index} missing {required}")
        expected_next = events[index + 1].get("tool_id") if index + 1 < len(events) and isinstance(events[index + 1], Mapping) else None
        if event.get("next_tool_id") != expected_next:
            errors.append(f"event {index} next_tool_id mismatch")
    return errors


def canonical_replay_is_valid(record: Mapping[str, Any]) -> bool:
    return not validate_canonical_replay(record)


__all__ = [
    "CANONICAL_REPLAY_SCHEMA_VERSION",
    "export_canonical_replay",
    "validate_canonical_replay",
    "canonical_replay_is_valid",
]
