"""Independent Code Belief Filter and optional recurrent network."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    from ..config import CODE_BELIEF_SCHEMA_VERSION, CODE_ERROR_FAMILIES, CODE_SESSION_STATES, CODE_STATUS_NAMES, CODE_TOOL_FAMILIES, CODE_TOOL_NAMES, stable_hash
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from config import CODE_BELIEF_SCHEMA_VERSION, CODE_ERROR_FAMILIES, CODE_SESSION_STATES, CODE_STATUS_NAMES, CODE_TOOL_FAMILIES, CODE_TOOL_NAMES, stable_hash
    from schemas import CodeToolResult

from .belief_features import CODE_BELIEF_FEATURE_NAMES, CODE_BELIEF_FEATURE_SCHEMA_HASH

try:  # Optional for lightweight tooling/tests.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class CodeToolQualityPosterior:
    availability_mean: float = 0.8
    availability_std: float = 0.2
    semantic_mean: float = 0.8
    semantic_std: float = 0.2
    structure_mean: float = 0.8
    structure_std: float = 0.2
    cost_mean: float = 1.0
    cost_std: float = 0.2

    def to_prompt_dict(self) -> dict[str, list[float]]:
        return {
            "availability": [round(self.availability_mean, 3), round(self.availability_std, 3)],
            "semantic": [round(self.semantic_mean, 3), round(self.semantic_std, 3)],
            "structure": [round(self.structure_mean, 3), round(self.structure_std, 3)],
            "cost": [round(self.cost_mean, 3), round(self.cost_std, 3)],
        }


@dataclass(frozen=True)
class CodeBeliefSnapshot:
    version: int
    step: int
    session_probs: tuple[float, ...]
    regime_probs: tuple[float, ...]
    change_probability: float
    family_quality: dict[str, tuple[float, float]]
    tool_quality: dict[str, CodeToolQualityPosterior]
    posterior_entropy: float
    ood_score: float
    environment: str = "code"
    feature_schema_version: str = CODE_BELIEF_SCHEMA_VERSION
    feature_schema_hash: str = CODE_BELIEF_FEATURE_SCHEMA_HASH

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "session": [round(value, 3) for value in self.session_probs],
            "regime": [round(value, 3) for value in self.regime_probs],
            "change_probability": round(self.change_probability, 3),
            "family_quality": {name: [round(value, 3) for value in values] for name, values in self.family_quality.items()},
            "tools": {name: value.to_prompt_dict() for name, value in self.tool_quality.items()},
            "posterior_entropy": round(self.posterior_entropy, 3),
            "ood_score": round(self.ood_score, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "version": self.version,
            "step": self.step,
            "session_probs": list(self.session_probs),
            "regime_probs": list(self.regime_probs),
            "change_probability": self.change_probability,
            "family_quality": {name: list(values) for name, values in self.family_quality.items()},
            "tool_quality": {name: value.__dict__ for name, value in self.tool_quality.items()},
            "posterior_entropy": self.posterior_entropy,
            "ood_score": self.ood_score,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_hash": self.feature_schema_hash,
        }


@dataclass
class CodeBeliefState:
    step: int = 0
    session_counts: list[float] = field(default_factory=lambda: [1.0] * len(CODE_SESSION_STATES))
    regime_counts: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    change_evidence: float = 0.0
    tool_success: dict[str, float] = field(default_factory=lambda: {name: 4.0 for name in CODE_TOOL_NAMES})
    tool_failure: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in CODE_TOOL_NAMES})
    tool_semantic: dict[str, float] = field(default_factory=lambda: {name: 4.0 for name in CODE_TOOL_NAMES})
    tool_structure: dict[str, float] = field(default_factory=lambda: {name: 4.0 for name in CODE_TOOL_NAMES})
    tool_cost: dict[str, list[float]] = field(default_factory=lambda: {name: [1.0, 1.0] for name in CODE_TOOL_NAMES})
    family_failures: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in CODE_TOOL_FAMILIES})
    history: list[dict[str, Any]] = field(default_factory=list)


class CodeBeliefFilter:
    """Online public-observation filter used by rollout and policy prompts."""

    def __init__(self, state: CodeBeliefState | None = None):
        self.state = state or CodeBeliefState()

    def update(self, tool_name: str, result: CodeToolResult | Mapping[str, Any], *, features: tuple[float, ...] | None = None, family: str | None = None) -> CodeBeliefSnapshot:
        if tool_name not in CODE_TOOL_NAMES:
            return self.snapshot()
        status = str(result.status if hasattr(result, "status") else result.get("status", "error"))
        output = str(result.output if hasattr(result, "output") else result.get("output", ""))
        latency = float(result.latency_ms if hasattr(result, "latency_ms") else result.get("latency_ms", 0.0) or 0.0)
        success = status in {"ok", "partial"}
        if success:
            self.state.tool_success[tool_name] += 1.0
        else:
            self.state.tool_failure[tool_name] += 1.0
        if status in {"error", "timeout", "invalid"} and family in self.state.family_failures:
            self.state.family_failures[family] += 1.0
        semantic_signal = 1.0 if output and not status in {"error", "timeout", "invalid"} else 0.0
        structure_signal = 1.0 if output and not bool(getattr(result, "partial", False)) else 0.0
        self.state.tool_semantic[tool_name] += semantic_signal
        self.state.tool_structure[tool_name] += structure_signal
        cost = self.state.tool_cost[tool_name]
        cost[0] += max(0.05, latency / 1000.0)
        cost[1] += 1.0
        if status == "timeout":
            self.state.change_evidence += 0.25
        elif status == "error":
            self.state.change_evidence += 0.15
        elif success:
            self.state.change_evidence = max(0.0, self.state.change_evidence - 0.03)
        self.state.step += 1
        row = {"tool": tool_name, "status": status, "output": output, "latency_ms": latency, "family": family or ""}
        self.state.history.append(row)
        self.state.history = self.state.history[-64:]
        return self.snapshot()

    def snapshot(self) -> CodeBeliefSnapshot:
        session = _normalize(self.state.session_counts)
        regime = _normalize(self.state.regime_counts)
        change = max(0.0, min(1.0, self.state.change_evidence / max(1.0, self.state.step * 0.35)))
        tools = {}
        for name in CODE_TOOL_NAMES:
            availability = self.state.tool_success[name] / max(1.0, self.state.tool_success[name] + self.state.tool_failure[name])
            semantic = self.state.tool_semantic[name] / max(1.0, self.state.tool_success[name] + self.state.tool_failure[name])
            structure = self.state.tool_structure[name] / max(1.0, self.state.tool_success[name] + self.state.tool_failure[name])
            cost_sum, cost_count = self.state.tool_cost[name]
            tools[name] = CodeToolQualityPosterior(availability, 0.2, semantic, 0.2, structure, 0.2, cost_sum / max(1.0, cost_count), 0.2)
        family = {name: (max(0.0, 1.0 - min(1.0, failures / max(1.0, self.state.step))), 0.2) for name, failures in self.state.family_failures.items()}
        entropy = _entropy(session) + _entropy(regime)
        ood = min(1.0, max(0.0, sum(value.availability_std for value in tools.values()) / len(tools)))
        return CodeBeliefSnapshot(self.state.step, self.state.step, tuple(session), tuple(regime), change, family, tools, entropy, ood)

    def clone(self) -> "CodeBeliefFilter":
        import copy

        return CodeBeliefFilter(copy.deepcopy(self.state))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeBeliefFilter":
        """Restore Code-only belief state from a branch checkpoint."""

        if value.get("environment") not in (None, "code"):
            raise ValueError("cannot restore a non-Code belief checkpoint")
        if value.get("feature_schema_hash") not in (None, CODE_BELIEF_FEATURE_SCHEMA_HASH):
            raise ValueError("Code belief checkpoint feature schema mismatch")
        raw_state = value.get("state")
        if not isinstance(raw_state, Mapping):
            raise ValueError("Code belief checkpoint is missing state")
        state = CodeBeliefState()
        state.step = max(0, int(raw_state.get("step", 0)))
        for name, expected in (("session_counts", len(CODE_SESSION_STATES)), ("regime_counts", 3)):
            values = list(raw_state.get(name, getattr(state, name)))
            if len(values) != expected:
                raise ValueError(f"Code belief checkpoint has invalid {name}")
            setattr(state, name, [max(0.0, float(item)) for item in values])
        state.change_evidence = max(0.0, float(raw_state.get("change_evidence", 0.0) or 0.0))
        for name, default in (("tool_success", state.tool_success), ("tool_failure", state.tool_failure), ("tool_semantic", state.tool_semantic), ("tool_structure", state.tool_structure), ("family_failures", state.family_failures)):
            raw_values = raw_state.get(name, {})
            if not isinstance(raw_values, Mapping):
                raise ValueError(f"Code belief checkpoint has invalid {name}")
            setattr(state, name, {key: max(0.0, float(raw_values.get(key, default[key]))) for key in default})
        raw_cost = raw_state.get("tool_cost", {})
        if not isinstance(raw_cost, Mapping):
            raise ValueError("Code belief checkpoint has invalid tool_cost")
        state.tool_cost = {}
        for tool, fallback in CodeBeliefState().tool_cost.items():
            values = list(raw_cost.get(tool, fallback))
            if len(values) != 2:
                raise ValueError("Code belief checkpoint tool_cost must have two values")
            state.tool_cost[tool] = [max(0.0, float(item)) for item in values]
        state.history = [dict(item) for item in raw_state.get("history", ()) if isinstance(item, Mapping)][-64:]
        return cls(state)

    def to_dict(self) -> dict[str, Any]:
        return {"environment": "code", "feature_schema_version": CODE_BELIEF_SCHEMA_VERSION, "feature_schema_hash": CODE_BELIEF_FEATURE_SCHEMA_HASH, "state": self.state.__dict__}


def _normalize(values: list[float]) -> list[float]:
    total = sum(max(0.0, value) for value in values) or 1.0
    return [max(0.0, value) / total for value in values]


def _entropy(values: list[float] | tuple[float, ...]) -> float:
    return -sum(value * math.log(max(1e-8, value)) for value in values)


if nn is not None:
    class CodeBeliefNetwork(nn.Module):
        """Compact encoder + recurrent heads for Code's 96-D observations."""

        def __init__(self, feature_dim: int = 96, hidden_dim: int = 128):
            super().__init__()
            if feature_dim != 96:
                raise ValueError("Code Belief Network feature_dim must be 96")
            self.observation_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.session_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.context_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.family_head = nn.Linear(hidden_dim, len(CODE_TOOL_FAMILIES) * 2)
            self.tool_quality_head = nn.Linear(hidden_dim, len(CODE_TOOL_NAMES) * 5)
            self.session_head = nn.Linear(hidden_dim, len(CODE_SESSION_STATES))
            self.regime_head = nn.Linear(hidden_dim, 3)
            self.change_head = nn.Linear(hidden_dim, 1)
            self.cost_head = nn.Linear(hidden_dim, len(CODE_TOOL_NAMES))

        def forward(self, features):
            encoded = self.observation_encoder(features)
            session, _ = self.session_gru(encoded)
            context, _ = self.context_gru(session)
            last = context[:, -1]
            return {"session": self.session_head(last), "regime": self.regime_head(last), "change": self.change_head(last), "family": self.family_head(last), "tool_quality": self.tool_quality_head(last), "cost": self.cost_head(last)}
else:  # pragma: no cover
    class CodeBeliefNetwork:  # type: ignore[no-redef]
        def __init__(self, feature_dim: int = 96, hidden_dim: int = 128):
            if feature_dim != 96:
                raise ValueError("Code Belief Network feature_dim must be 96")


def checkpoint_metadata() -> dict[str, Any]:
    return {
        "environment": "code",
        "checkpoint_type": "belief",
        "feature_dim": 96,
        "feature_schema_version": CODE_BELIEF_SCHEMA_VERSION,
        "feature_schema_hash": CODE_BELIEF_FEATURE_SCHEMA_HASH,
        "feature_names": list(CODE_BELIEF_FEATURE_NAMES),
        "tool_names": list(CODE_TOOL_NAMES),
        "tool_families": {name: list(values) for name, values in CODE_TOOL_FAMILIES.items()},
    }


def save_belief_checkpoint(path: str | Path, *, belief: CodeBeliefFilter | None = None, model: Any = None, extra: Mapping[str, Any] | None = None) -> None:
    payload = {"metadata": {**checkpoint_metadata(), **dict(extra or {})}, "belief": belief.to_dict() if belief else None}
    if model is not None and hasattr(model, "state_dict"):
        if torch is None:
            raise RuntimeError("torch is required to save network weights")
        payload["model_state"] = model.state_dict()
        torch.save(payload, str(path))
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_belief_checkpoint(path: str | Path, *, model: Any = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        if torch is not None:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    expected = checkpoint_metadata()
    if metadata.get("environment") != "code" or metadata.get("checkpoint_type") != "belief":
        raise ValueError("Code Belief checkpoint environment/type mismatch")
    if metadata.get("feature_dim") != expected["feature_dim"] or metadata.get("feature_schema_hash") != expected["feature_schema_hash"]:
        raise ValueError("Code Belief checkpoint feature schema mismatch")
    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])
    return payload


__all__ = ["CodeBeliefFilter", "CodeBeliefNetwork", "CodeBeliefSnapshot", "CodeBeliefState", "CodeToolQualityPosterior", "CODE_BELIEF_FEATURE_SCHEMA_HASH", "checkpoint_metadata", "load_belief_checkpoint", "save_belief_checkpoint"]
