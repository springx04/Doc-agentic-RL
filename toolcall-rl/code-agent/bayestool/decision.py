"""Belief-conditioned Code action selection with independent Q/Risk models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

try:
    from ..config import stable_hash
except ImportError:  # pragma: no cover
    from config import stable_hash

from .belief import CodeBeliefFilter, CodeBeliefSnapshot
from .q_model import CodeQModel, action_features, action_key
from .risk_model import CodeRiskModel
from .task_state import CodeTaskStateView


@dataclass(frozen=True)
class DecisionCandidate:
    action: dict[str, Any]
    value: float
    risk: float
    score: float


@dataclass(frozen=True)
class DecisionReport:
    selected: dict[str, Any] | None
    mode: str
    candidates: tuple[DecisionCandidate, ...]
    decision_regret: float
    stop_risk: float
    belief_digest: str


class CodeDecisionController:
    def __init__(self, q_model: CodeQModel | None = None, risk_model: CodeRiskModel | None = None):
        self.q_model = q_model or CodeQModel()
        self.risk_model = risk_model or CodeRiskModel()

    def evaluate(self, task_state: CodeTaskStateView, belief: CodeBeliefFilter | CodeBeliefSnapshot, candidates: Iterable[Mapping[str, Any]]) -> DecisionReport:
        snapshot = belief.snapshot() if isinstance(belief, CodeBeliefFilter) else belief
        rows: list[DecisionCandidate] = []
        risk_features = {
            "remaining_budget": task_state.remaining_tool_budget / 30.0,
            "evidence_sufficient": float(task_state.evidence_sufficient),
            "change_probability": snapshot.change_probability,
            "patch_nonempty": float(task_state.patch_nonempty),
            "validation_failed": float(task_state.last_validation_returncode not in (None, 0)),
        }
        stop_risk = self.risk_model.predict(risk_features)
        for action in candidates:
            action_dict = dict(action)
            value = self.q_model.predict(action_features(action_dict, remaining_budget=task_state.remaining_tool_budget))
            if action_dict.get("kind", "tool") == "tool":
                tool = str(action_dict.get("name") or action_dict.get("tool") or "")
                posterior = snapshot.tool_quality.get(tool)
                if posterior:
                    value += 0.2 * posterior.availability_mean + 0.1 * posterior.semantic_mean - 0.05 * posterior.cost_mean
            action_risk = stop_risk if action_dict.get("kind") in {"final", "abstain"} else 0.0
            rows.append(DecisionCandidate(action_dict, value, action_risk, value - action_risk))
        rows.sort(key=lambda row: (-row.score, action_key(row.action)))
        if not rows:
            selected = None
            mode = "empty"
            regret = 0.0
        else:
            selected = rows[0].action
            mode = "policy"
            if task_state.evidence_sufficient and any(row.action.get("kind") == "final" for row in rows):
                selected = next(row.action for row in rows if row.action.get("kind") == "final")
                mode = "submit"
            regret = max(0.0, rows[0].score - rows[-1].score)
        digest = stable_hash(
            {
                "task_state": task_state.to_prompt_dict(),
                "belief": snapshot.to_dict(),
                "candidates": [action_key(row.action) for row in rows],
            },
            prefix="code-decision-state-v1",
        )
        return DecisionReport(selected, mode, tuple(rows), regret, stop_risk, digest)


__all__ = ["CodeDecisionController", "DecisionCandidate", "DecisionReport"]
