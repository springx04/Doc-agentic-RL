"""Code-only Q model and checkpoint gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from ..config import CODE_TOOL_NAMES, stable_hash
except ImportError:  # pragma: no cover
    from config import CODE_TOOL_NAMES, stable_hash

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


CODE_Q_SCHEMA_VERSION = "code-q-v1"


def action_key(action: Any) -> str:
    if isinstance(action, Mapping):
        kind = action.get("kind", "tool")
        if kind == "tool":
            return f"tool:{action.get('name') or action.get('tool')}:{json.dumps(action.get('arguments', {}), sort_keys=True, separators=(',', ':'))}"
        return f"{kind}:{action.get('text') or action.get('answer') or action.get('reason') or ''}"
    return str(action)


def action_features(action: Any, *, remaining_budget: int = 0) -> tuple[float, ...]:
    key = action_key(action)
    tool = ""
    if key.startswith("tool:"):
        tool = key.split(":", 2)[1]
    return tuple(float(tool == name) for name in CODE_TOOL_NAMES) + (
        float(key.startswith("final:")), float(key.startswith("abstain:")), min(1.0, remaining_budget / 30.0)
    )


class CodeQModel:
    def __init__(self, feature_dim: int = 8 + 3, hidden_dim: int = 128):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.weights = [0.0] * feature_dim
        self.bias = 0.0
        if nn is not None:
            self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        else:
            self.network = None

    def predict(self, features: Any) -> float:
        if self.network is not None and torch is not None:
            with torch.no_grad():
                tensor = features if hasattr(features, "shape") else torch.tensor([features], dtype=torch.float32)
                return float(self.network(tensor.float()).reshape(-1)[0].item())
        values = list(features)
        return self.bias + sum(weight * float(value) for weight, value in zip(self.weights, values))

    def fit_linear(self, examples: list[tuple[list[float], float]], learning_rate: float = 1e-3) -> None:
        for features, target in examples:
            prediction = self.predict(features)
            error = float(target) - prediction
            self.bias += learning_rate * error
            for index, value in enumerate(features[: len(self.weights)]):
                self.weights[index] += learning_rate * error * float(value)


def q_checkpoint_metadata() -> dict[str, Any]:
    return {"environment": "code", "checkpoint_type": "q", "schema_version": CODE_Q_SCHEMA_VERSION, "tool_names": list(CODE_TOOL_NAMES), "feature_schema_hash": stable_hash(list(CODE_TOOL_NAMES), prefix="code-q-features-v1")}


def save_q_checkpoint(path: str | Path, model: CodeQModel, *, extra: Mapping[str, Any] | None = None) -> None:
    payload = {"metadata": {**q_checkpoint_metadata(), **dict(extra or {})}, "weights": model.weights, "bias": model.bias}
    if model.network is not None and torch is not None:
        payload["model_state"] = model.network.state_dict()
        torch.save(payload, str(path))
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def load_q_checkpoint(path: str | Path, model: CodeQModel | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False) if torch is not None else json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    expected = q_checkpoint_metadata()
    if metadata.get("environment") != "code" or metadata.get("checkpoint_type") != "q" or metadata.get("schema_version") != expected["schema_version"]:
        raise ValueError("Code Q checkpoint environment/type/schema mismatch")
    if metadata.get("feature_schema_hash") != expected["feature_schema_hash"] or metadata.get("tool_names") != expected["tool_names"]:
        raise ValueError("Code Q checkpoint feature schema mismatch")
    if model is not None:
        model.weights = list(payload.get("weights") or model.weights)
        model.bias = float(payload.get("bias", model.bias))
        if model.network is not None and payload.get("model_state") is not None:
            model.network.load_state_dict(payload["model_state"])
    return payload


__all__ = ["CODE_Q_SCHEMA_VERSION", "CodeQModel", "action_features", "action_key", "load_q_checkpoint", "q_checkpoint_metadata", "save_q_checkpoint"]
