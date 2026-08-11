"""Runtime that separates hidden Code world effects from real infrastructure."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from ..config import CODE_TOOL_FAMILIES, stable_hash
    from ..schemas import CodeToolResult
    from ..tools import DEFAULT_REGISTRY, ToolExecutionContext
except ImportError:  # pragma: no cover
    from config import CODE_TOOL_FAMILIES, stable_hash
    from schemas import CodeToolResult
    from tools import DEFAULT_REGISTRY, ToolExecutionContext

from .belief_features import extract_code_belief_features
from .corruption import corruption_for_call, corrupt_result, injected_unavailable, world_unavailable_result
from .schema import CodeWorldSpec, ToolStateLabel, WorldEvent, quality_for_tool
from .task_state import CodeTaskStateView
from .utility import information_gain


@dataclass
class WorldRuntimeState:
    call_index: int = 0
    events: list[WorldEvent] | None = None
    runtime_digest: str = ""

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = []


class CodeWorldRuntime:
    """Execute one hidden world around a clean tool registry."""

    def __init__(self, *, world: CodeWorldSpec, client: Any, lease_id: str, cwd: str, task_state: CodeTaskStateView, registry=DEFAULT_REGISTRY):
        self.world = world
        self.client = client
        self.lease_id = lease_id
        self.cwd = cwd
        self.task_state = task_state
        self.registry = registry
        self.state = WorldRuntimeState()
        self.public_history: list[dict[str, Any]] = []
        self.trainer_events: list[dict[str, Any]] = []

    @property
    def call_index(self) -> int:
        return self.state.call_index

    @property
    def runtime_state_digest(self) -> str:
        return stable_hash(
            {"world_runtime": self.world.runtime_digest, "call_index": self.state.call_index, "event_count": len(self.state.events or [])},
            prefix="code-runtime-state-v1",
        )

    async def execute_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> tuple[CodeToolResult, tuple[float, ...], WorldEvent]:
        call_index = self.state.call_index
        public_context = self._public_context(tool_name, arguments)
        quality = quality_for_tool(self.world, tool_name, public_context, call_index)
        corruption = corruption_for_call(self.world, tool_name, call_index, rng=random.Random(self.world.seed + call_index))

        # Availability and explicitly injected timeout/reject failures are
        # world observations.  They must not kill/close the actual lease.
        if injected_unavailable(self.world, tool_name, call_index, public_context) or corruption in {"injected_timeout", "injected_reject", "injected_noop"}:
            result = world_unavailable_result(tool_name, reason=f"world-injected {tool_name} unavailable")
            result = corrupt_result(result, corruption if corruption != "injected_reject" else None, seed=self.world.seed + call_index)
        else:
            context = ToolExecutionContext(self.client, self.lease_id, self.cwd)
            result = await self.registry.execute(tool_name, dict(arguments), context)
            if result.failure_origin == "real_infrastructure":
                # A true transport/backend failure is not rewritten as a
                # hidden world event and makes the eventual group invalid.
                corruption = None
            elif corruption:
                result = corrupt_result(result, corruption, seed=self.world.seed + call_index)

        gain = information_gain(
            tool_name=tool_name,
            output=result.output,
            history=self.public_history,
            task_state=self.task_state,
            patch_changed=tool_name == "apply_patch" and result.status == "ok",
        )
        metadata = dict(result.metadata)
        metadata["information_gain"] = gain
        metadata["relative_cost"] = quality.relative_cost
        metadata["error_family"] = self._error_family(tool_name, result)
        result = result.__class__(**{**result.__dict__, "metadata": metadata})
        label = ToolStateLabel(
            tool_name=tool_name,
            availability=quality.availability,
            semantic_accuracy=quality.semantic_accuracy,
            structure_fidelity=quality.structure_fidelity,
            session_state=self.world.session_state,
            world_slot_role=self.world.world_slot_role,
            latent_world_id=self.world.latent_world_id,
            corruption_type=corruption,
        )
        event = WorldEvent(call_index, tool_name, result, gain, public_context, label)
        self.state.events.append(event)
        self.public_history.append({**event.visible_dict(), "tool": tool_name, "family": self._family(tool_name)})
        self.trainer_events.append(event.trainer_dict())
        self.state.call_index += 1

        patch = None
        if tool_name in {"apply_patch", "git_diff"}:
            try:
                patch = await self.client.diff(self.lease_id, cwd=self.cwd)
            except Exception:
                patch = None
        self.task_state.update(tool_name, result, patch=patch)
        features = extract_code_belief_features(
            tool_name=tool_name,
            result=result,
            task_state=self.task_state,
            history=self.public_history[:-1],
            public_context=public_context,
        )
        self.state.runtime_digest = self.runtime_state_digest
        return result, features, event

    def _public_context(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        value = {"tool_name": tool_name}
        value.update({key: item for key, item in arguments.items() if key not in {"patch", "command"}})
        if "path" in arguments:
            path = str(arguments["path"])
            value["file_extension"] = "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
            value["language"] = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust"}.get(value["file_extension"], "")
        value["family"] = self._family(tool_name)
        return value

    @staticmethod
    def _family(tool_name: str) -> str:
        for family, names in CODE_TOOL_FAMILIES.items():
            if tool_name in names:
                return family
        return "unknown"

    def _error_family(self, tool_name: str, result: CodeToolResult) -> str:
        if result.failure_origin == "model_action":
            return "protocol"
        if result.status == "timeout":
            return "latency"
        if result.status == "invalid":
            return "protocol"
        if result.status == "error":
            if tool_name in {"apply_patch"}:
                return "mutation"
            if tool_name in {"run_tests", "run_checks"}:
                return "validation"
            if tool_name in {"run_command"}:
                return "execution"
            return "inspection"
        return "none"

    def snapshot(self) -> dict[str, Any]:
        return {"call_index": self.state.call_index, "events": copy.deepcopy(self.trainer_events), "public_history": copy.deepcopy(self.public_history), "runtime_state_digest": self.runtime_state_digest}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.state.call_index = int(snapshot.get("call_index", 0))
        self.public_history = copy.deepcopy(list(snapshot.get("public_history") or []))
        self.trainer_events = copy.deepcopy(list(snapshot.get("events") or []))
        # The public and trainer histories are sufficient for continuation;
        # retain the count so the runtime digest remains the same state
        # identity that the checkpoint recorded without reconstructing hidden
        # label objects solely for bookkeeping.
        self.state.events = [None] * len(self.trainer_events)  # type: ignore[list-item]


__all__ = ["CodeWorldRuntime", "WorldRuntimeState"]
