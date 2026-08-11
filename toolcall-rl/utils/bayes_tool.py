"""Bayesian tool selection utilities for document-agent rollouts.

The production rollout already lets the language model emit a tool action.  This
module is deliberately independent of that runtime so it can be used in three
places without importing the model server: as a policy-head selector, as an
offline trajectory analyser, and as a small stateful component in a local
training harness.

Each tool has a Beta posterior.  Thompson sampling supplies an uncertainty-
aware score which is multiplied by the policy probability for that tool.  The
selector never executes a tool itself; callers remain responsible for applying
the selected action and reporting its outcome through :meth:`update`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import random
from typing import Any


def _digamma(value: float) -> float:
    """Approximate the digamma function without a SciPy dependency."""

    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("digamma is only defined here for finite positive values")
    result = 0.0
    while value < 6.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    inverse_square = inverse * inverse
    # The asymptotic expansion is more than sufficient for reward-scale
    # information-gain estimates and remains stable for the small priors used
    # by the selector.
    result += (
        math.log(value)
        - 0.5 * inverse
        - inverse_square * (1.0 / 12.0 - inverse_square * (1.0 / 120.0 - inverse_square / 252.0))
    )
    return result


def _beta_entropy(alpha: float, beta: float) -> float:
    """Return the differential entropy of ``Beta(alpha, beta)``."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta parameters must be positive")
    total = alpha + beta
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(total)
    entropy = (
        log_beta
        - (alpha - 1.0) * _digamma(alpha)
        - (beta - 1.0) * _digamma(beta)
        + (total - 2.0) * _digamma(total)
    )
    # Floating point round-off can produce a tiny negative value for the
    # uniform prior, whose entropy is exactly zero.
    return 0.0 if abs(entropy) < 1e-12 else float(entropy)


@dataclass(frozen=True)
class ToolSelection:
    """Auditable result of one Bayesian tool-selection decision."""

    tool_name: str | None
    should_call: bool
    sampled_confidence: float | None
    policy_probability: float
    score: float
    threshold: float
    posterior_mean: float | None
    posterior_entropy: float | None
    candidates: tuple[dict[str, Any], ...] = ()
    call_probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "should_call": self.should_call,
            "sampled_confidence": self.sampled_confidence,
            "policy_probability": self.policy_probability,
            "score": self.score,
            "threshold": self.threshold,
            "posterior_mean": self.posterior_mean,
            "posterior_entropy": self.posterior_entropy,
            "candidates": [dict(candidate) for candidate in self.candidates],
            "call_probability": self.call_probability,
        }


class BayesToolSelector:
    """Maintain Beta posteriors and select tools with Thompson sampling.

    Parameters are intentionally kept as public ``alpha`` and ``beta`` maps to
    make checkpoint inspection straightforward and to remain compatible with
    the implementation sketch in the project design document.
    """

    def __init__(
        self,
        tool_names: Sequence[str] | Iterable[str],
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        confidence_threshold: float = 0.0,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if isinstance(tool_names, (str, bytes)):
            raise TypeError("tool_names must be an iterable of names, not one string")
        names = [str(name).strip() for name in tool_names]
        if any(not name for name in names):
            raise ValueError("tool names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")
        self._validate_positive(prior_alpha, "prior_alpha")
        self._validate_positive(prior_beta, "prior_beta")
        self._validate_threshold(confidence_threshold)
        if rng is not None and seed is not None:
            raise ValueError("pass either seed or rng, not both")

        self.tool_names = tuple(names)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.confidence_threshold = float(confidence_threshold)
        self.alpha = {name: self.prior_alpha for name in self.tool_names}
        self.beta = {name: self.prior_beta for name in self.tool_names}
        self._rng = rng if rng is not None else random.Random(seed)
        self.total_updates = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_information_gain = 0.0

    @staticmethod
    def _validate_positive(value: float, name: str) -> None:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")

    @staticmethod
    def _validate_threshold(value: float) -> None:
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")

    def _require_tool(self, tool_name: str) -> str:
        name = str(tool_name)
        if name not in self.alpha:
            raise KeyError(f"unknown tool: {name}")
        return name

    def add_tool(self, tool_name: str, *, prior_alpha: float | None = None, prior_beta: float | None = None) -> None:
        """Register a tool discovered after selector construction."""

        name = str(tool_name).strip()
        if not name:
            raise ValueError("tool name must be non-empty")
        if name in self.alpha:
            return
        alpha = self.prior_alpha if prior_alpha is None else float(prior_alpha)
        beta = self.prior_beta if prior_beta is None else float(prior_beta)
        self._validate_positive(alpha, "prior_alpha")
        self._validate_positive(beta, "prior_beta")
        self.tool_names = (*self.tool_names, name)
        self.alpha[name] = alpha
        self.beta[name] = beta

    def ensure_tools(self, tool_names: Iterable[str]) -> None:
        for name in tool_names:
            self.add_tool(name)

    def posterior(self, tool_name: str) -> tuple[float, float]:
        name = self._require_tool(tool_name)
        return float(self.alpha[name]), float(self.beta[name])

    def confidence(self, tool_name: str) -> float:
        alpha, beta = self.posterior(tool_name)
        return alpha / (alpha + beta)

    def variance(self, tool_name: str) -> float:
        alpha, beta = self.posterior(tool_name)
        total = alpha + beta
        return alpha * beta / (total * total * (total + 1.0))

    def entropy(self, tool_name: str) -> float:
        alpha, beta = self.posterior(tool_name)
        return _beta_entropy(alpha, beta)

    def sample(self, tool_name: str) -> float:
        """Draw one confidence value from a tool's posterior."""

        alpha, beta = self.posterior(tool_name)
        return float(self._rng.betavariate(alpha, beta))

    def update(self, tool_name: str, success: bool, *, weight: float = 1.0) -> float:
        """Update one posterior and return the resulting information gain.

        ``success`` is intentionally binary.  Fractional evidence can be
        represented through ``weight`` when a caller has a calibrated outcome
        confidence, while preserving conjugate Beta updates.
        """

        name = self._require_tool(tool_name)
        self._validate_positive(weight, "weight")
        before = self.entropy(name)
        if bool(success):
            self.alpha[name] += float(weight)
            self.success_count += 1
        else:
            self.beta[name] += float(weight)
            self.failure_count += 1
        self.total_updates += 1
        gain = before - self.entropy(name)
        self.total_information_gain += gain
        return float(gain)

    def update_from_reward(
        self,
        tool_name: str,
        reward: float,
        *,
        success_threshold: float = 0.0,
        weight: float = 1.0,
    ) -> float:
        """Convert a scalar outcome into a binary posterior observation."""

        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("reward must be finite")
        return self.update(tool_name, value > float(success_threshold), weight=weight)

    def select(
        self,
        policy_probs: Mapping[str, float],
        *,
        threshold: float | None = None,
        sample: bool = True,
    ) -> ToolSelection:
        """Combine policy probabilities and Bayesian confidence.

        Missing tools receive probability zero.  A result below ``threshold``
        means that the caller should continue text generation instead of
        issuing a tool call.
        """

        effective_threshold = self.confidence_threshold if threshold is None else float(threshold)
        self._validate_threshold(effective_threshold)
        candidates: list[dict[str, Any]] = []
        for name in self.tool_names:
            raw_probability = float(policy_probs.get(name, 0.0))
            if not math.isfinite(raw_probability) or raw_probability < 0.0:
                raise ValueError(f"policy probability for {name!r} must be finite and non-negative")
            probability = min(1.0, raw_probability)
            sampled_confidence = self.sample(name) if sample else self.confidence(name)
            candidates.append(
                {
                    "tool_name": name,
                    "policy_probability": probability,
                    "sampled_confidence": sampled_confidence,
                    "posterior_mean": self.confidence(name),
                    "posterior_entropy": self.entropy(name),
                    "score": probability * sampled_confidence,
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["tool_name"])))
        if not candidates:
            return ToolSelection(None, False, None, 0.0, 0.0, effective_threshold, None, None, ())

        best = candidates[0]
        should_call = float(best["score"]) >= effective_threshold and float(best["policy_probability"]) > 0.0
        selected_name = str(best["tool_name"]) if should_call else None
        return ToolSelection(
            tool_name=selected_name,
            should_call=should_call,
            sampled_confidence=float(best["sampled_confidence"]),
            policy_probability=float(best["policy_probability"]),
            score=float(best["score"]),
            threshold=effective_threshold,
            posterior_mean=float(best["posterior_mean"]),
            posterior_entropy=float(best["posterior_entropy"]),
            candidates=tuple(dict(item) for item in candidates),
        )

    def select_tool(
        self,
        policy_probs: Mapping[str, float],
        *,
        threshold: float | None = None,
        sample: bool = True,
    ) -> str | None:
        """Return only the chosen tool name, or ``None`` for text generation."""

        return self.select(policy_probs, threshold=threshold, sample=sample).tool_name

    def posterior_stats(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "alpha": float(self.alpha[name]),
                "beta": float(self.beta[name]),
                "mean": self.confidence(name),
                "variance": self.variance(name),
                "entropy": self.entropy(name),
            }
            for name in self.tool_names
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable checkpoint payload."""

        return {
            "tool_names": list(self.tool_names),
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "confidence_threshold": self.confidence_threshold,
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "total_updates": self.total_updates,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_information_gain": self.total_information_gain,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        names = tuple(str(name) for name in state.get("tool_names", self.tool_names))
        if names != self.tool_names:
            raise ValueError("checkpoint tool names do not match selector tool names")
        alpha = state.get("alpha", {})
        beta = state.get("beta", {})
        for name in self.tool_names:
            self._validate_positive(float(alpha[name]), f"alpha[{name}]")
            self._validate_positive(float(beta[name]), f"beta[{name}]")
        self.alpha = {name: float(alpha[name]) for name in self.tool_names}
        self.beta = {name: float(beta[name]) for name in self.tool_names}
        self.total_updates = int(state.get("total_updates", 0))
        self.success_count = int(state.get("success_count", 0))
        self.failure_count = int(state.get("failure_count", 0))
        self.total_information_gain = float(state.get("total_information_gain", 0.0))

    def reset(self) -> None:
        self.alpha = {name: self.prior_alpha for name in self.tool_names}
        self.beta = {name: self.prior_beta for name in self.tool_names}
        self.total_updates = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_information_gain = 0.0


def summarize_tool_outcomes(
    tool_calls: Iterable[Mapping[str, Any]],
    *,
    tool_names: Sequence[str] | None = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    selector: BayesToolSelector | None = None,
) -> dict[str, Any]:
    """Build Bayesian statistics from an existing rollout execution trace.

    Calls that were parsed but not executed (for example, a call blocked by a
    search budget) do not update a posterior.  A failed executed call does.
    """

    names = [str(name) for name in (tool_names or [])]
    for call in tool_calls:
        tool = call.get("tool") or call.get("tool_name")
        if tool and str(tool) not in names:
            names.append(str(tool))
    if selector is None:
        selector = BayesToolSelector(names, prior_alpha=prior_alpha, prior_beta=prior_beta)
    else:
        selector.ensure_tools(names)
    gains: list[float] = []
    executed_calls = 0
    successes = 0
    failures = 0
    for call in tool_calls:
        tool = call.get("tool") or call.get("tool_name")
        if not tool or call.get("executed") is False:
            continue
        executed_calls += 1
        success = bool(call.get("success", False))
        if success:
            successes += 1
        else:
            failures += 1
        gains.append(selector.update(str(tool), success))
    return {
        "tool_names": list(selector.tool_names),
        "executed_calls": executed_calls,
        "successes": successes,
        "failures": failures,
        "information_gain": float(sum(gains)),
        "step_information_gain": gains,
        "posteriors": selector.posterior_stats(),
        "selector_state": selector.state_dict(),
    }


__all__ = ["BayesToolSelector", "ToolSelection", "summarize_tool_outcomes"]
