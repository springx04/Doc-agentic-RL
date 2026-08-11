"""Code-only stop-risk model and checkpoint gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from ..config import stable_hash
except ImportError:  # pragma: no cover
    from config import stable_hash


CODE_RISK_SCHEMA_VERSION = "code-risk-v1"


class CodeRiskModel:
    def __init__(self):
        self.bias = 0.0
        self.weights: dict[str, float] = {"remaining_budget": -0.02, "evidence_sufficient": -0.7, "change_probability": 0.2, "patch_nonempty": -0.15, "validation_failed": 0.2}

    def predict(self, features: Mapping[str, Any]) -> float:
        value = self.bias + sum(float(self.weights.get(key, 0.0)) * float(item) for key, item in features.items())
        return max(0.0, min(1.0, 1.0 / (1.0 + __import__("math").exp(-value))))


def risk_checkpoint_metadata() -> dict[str, Any]:
    return {"environment": "code", "checkpoint_type": "risk", "schema_version": CODE_RISK_SCHEMA_VERSION, "feature_schema_hash": stable_hash(sorted(CodeRiskModel().weights), prefix="code-risk-features-v1")}


def save_risk_checkpoint(path: str | Path, model: CodeRiskModel, *, extra: Mapping[str, Any] | None = None) -> None:
    payload = {"metadata": {**risk_checkpoint_metadata(), **dict(extra or {})}, "bias": model.bias, "weights": model.weights}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def load_risk_checkpoint(path: str | Path, model: CodeRiskModel | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    expected = risk_checkpoint_metadata()
    if metadata.get("environment") != "code" or metadata.get("checkpoint_type") != "risk" or metadata.get("schema_version") != expected["schema_version"]:
        raise ValueError("Code Risk checkpoint environment/type/schema mismatch")
    if metadata.get("feature_schema_hash") != expected["feature_schema_hash"]:
        raise ValueError("Code Risk checkpoint feature schema mismatch")
    if model is not None:
        model.bias = float(payload.get("bias", model.bias))
        model.weights.update({str(key): float(value) for key, value in (payload.get("weights") or {}).items()})
    return payload


__all__ = ["CODE_RISK_SCHEMA_VERSION", "CodeRiskModel", "load_risk_checkpoint", "risk_checkpoint_metadata", "save_risk_checkpoint"]
