"""Bayesian particles, action values, DVOI, reopen policy, and risk control."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .belief import BeliefRuntime
from .config import STATUS_NAMES, TOOL_NAMES, BayesToolConfig, default_config
from .schema import ActionValue, BeliefSnapshot, PosteriorParticle, TaskStateView
from .world import stable_seed

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]

try:
    from tool_protocol import parse_assistant_action
except ImportError:  # pragma: no cover
    parse_assistant_action = None


ACTION_KINDS = ("tool", "final", "abstain")


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    temperature = max(1e-5, float(temperature))
    scaled = [float(value) / temperature for value in values]
    max_value = max(scaled, default=0.0)
    exps = [math.exp(max(-50.0, min(50.0, value - max_value))) for value in scaled]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    """Jensen-Shannon divergence in nats for equal-length categorical vectors."""

    if len(left) != len(right):
        raise ValueError("JS inputs must have the same length")
    left_total = sum(max(0.0, float(value)) for value in left) or 1.0
    right_total = sum(max(0.0, float(value)) for value in right) or 1.0
    p = [max(1e-12, float(value) / left_total) for value in left]
    q = [max(1e-12, float(value) / right_total) for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q, strict=True)]
    return 0.5 * sum(a * math.log(a / m) for a, m in zip(p, midpoint, strict=True)) + 0.5 * sum(
        b * math.log(b / m) for b, m in zip(q, midpoint, strict=True)
    )


def canonical_action(action: Any) -> dict[str, Any]:
    """Normalize a policy action into a stable keyable representation."""

    if isinstance(action, Mapping):
        kind = str(action.get("kind") or action.get("type") or "").casefold()
        if kind in {"tool_call", "tool"}:
            tool_name = str(action.get("tool") or action.get("name") or "")
            arguments = action.get("arguments") or {}
            return {"kind": "tool", "tool": tool_name, "arguments": arguments if isinstance(arguments, Mapping) else {}}
        if kind in {"final", "answer"}:
            return {"kind": "final", "answer": str(action.get("answer") or action.get("value") or "")}
        if kind in {"abstain", "reject", "refuse"}:
            return {"kind": "abstain", "reason": str(action.get("reason") or action.get("value") or "")}
    text = str(action or "").strip()
    if parse_assistant_action is not None:
        parsed = parse_assistant_action(text)
        if parsed.kind == "tool_call" and isinstance(parsed.value, Mapping):
            return canonical_action({"kind": "tool", **dict(parsed.value)})
        if parsed.kind == "final":
            return {"kind": "final", "answer": str(parsed.value or "")}
        if parsed.kind == "abstain":
            return {"kind": "abstain", "reason": str(parsed.value or "")}
    return {"kind": "unknown", "value": text}


def canonical_action_key(action: Any) -> str:
    value = canonical_action(action)
    kind = value.get("kind")
    if kind == "tool":
        import json

        arguments = json.dumps(value.get("arguments") or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"tool:{value.get('tool', '')}:{arguments}"
    if kind == "final":
        return f"final:{' '.join(str(value.get('answer', '')).casefold().split())}"
    if kind == "abstain":
        return f"abstain:{' '.join(str(value.get('reason', '')).casefold().split())}"
    return f"unknown:{value.get('value', '')}"


def action_kind(action: Any) -> str:
    return str(canonical_action(action).get("kind") or "unknown")


def action_is_diagnostic(action: Any) -> bool:
    value = canonical_action(action)
    if value.get("kind") != "tool":
        return False
    return str(value.get("tool")) in {"detect_layout", "render_page", "ocr_region", "zoom_region", "crop_region"}


if torch is not None:

    class BayesQHead(nn.Module):
        """Small value estimator used for finite branch selection only."""

        def __init__(self, task_dim: int = 32, particle_dim: int = 64, action_dim: int = 64, budget_dim: int = 8) -> None:
            super().__init__()
            self.task_projection = nn.Linear(32, task_dim)
            self.particle_projection = nn.Linear(32, particle_dim)
            self.action_projection = nn.Linear(32, action_dim)
            self.net = nn.Sequential(
                nn.Linear(task_dim + particle_dim + action_dim + budget_dim, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )

        def forward(self, task_features: Tensor, particle_features: Tensor, action_features: Tensor, budget_features: Tensor) -> Tensor:
            return self.net(
                torch.cat(
                    (
                        self.task_projection(task_features),
                        self.particle_projection(particle_features),
                        self.action_projection(action_features),
                        budget_features,
                    ),
                    dim=-1,
                )
            )

else:  # pragma: no cover

    class BayesQHead:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for BayesQHead")


def sample_posterior_particles(
    snapshot: BeliefSnapshot,
    *,
    count: int,
    seed: int,
) -> tuple[PosteriorParticle, ...]:
    rng = random.Random(seed)
    particles: list[PosteriorParticle] = []
    for particle_id in range(max(1, int(count))):
        qualities = {}
        for tool_name, posterior in snapshot.tool_quality.items():
            qualities[tool_name] = _sample_quality(posterior, rng)
        session_state = rng.choices(
            ("healthy", "degraded", "overloaded", "outage"),
            weights=list(snapshot.session_probs),
            k=1,
        )[0]
        regime_state = rng.choices(
            ("stable", "abrupt_transition", "gradual_transition"),
            weights=list(snapshot.regime_probs),
            k=1,
        )[0]
        shared = {
            family: rng.choices(("healthy", "degraded", "down"), weights=list(values), k=1)[0]
            for family, values in snapshot.shared_family_probs.items()
        }
        particles.append(
            PosteriorParticle(
                particle_id=particle_id,
                weight=1.0 / max(1, count),
                session_state=session_state,
                regime_state=regime_state,
                shared_states=shared,
                tool_quality=qualities,
            )
        )
    return tuple(particles)


def _sample_beta(mean: float, std: float, rng: random.Random) -> float:
    mean = _clip(mean, 0.001, 0.999)
    variance = max(1e-5, std * std)
    concentration = max(2.0, mean * (1.0 - mean) / variance - 1.0)
    alpha = max(0.1, mean * concentration)
    beta = max(0.1, (1.0 - mean) * concentration)
    return rng.betavariate(alpha, beta)


def _sample_quality(posterior: Any, rng: random.Random) -> Any:
    from .schema import ToolQualitySpec

    return ToolQualitySpec(
        availability=_sample_beta(posterior.availability_mean, posterior.availability_std, rng),
        semantic_accuracy=_sample_beta(posterior.semantic_mean, posterior.semantic_std, rng),
        structure_fidelity=_sample_beta(posterior.structure_mean, posterior.structure_std, rng),
        calibration_temperature=max(0.05, _sample_beta(posterior.calibration_mean, posterior.calibration_std, rng) * 1.5),
        calibration_bias=_clip(rng.gauss(0.0, posterior.calibration_std), -1.0, 1.0),
        relative_cost=max(0.05, rng.lognormvariate(math.log(max(0.05, posterior.cost_mean)), max(0.05, posterior.cost_std / max(0.05, posterior.cost_mean)))),
        latency_scale=1.0,
    )


def _task_feature_vector(task_state: TaskStateView) -> list[float]:
    question_type = task_state.question_type.casefold()
    return [
        float(question_type == "text"),
        float(question_type == "table"),
        float(question_type == "chart"),
        float(question_type in {"visual", "figure"}),
        math.log1p(task_state.current_page or 0),
        math.log1p(len(task_state.visited_pages)),
        math.log1p(task_state.unvisited_page_count or 0),
        math.log1p(task_state.remaining_tool_budget),
        float(task_state.evidence_sufficient),
        float(task_state.visual_input_required),
        float(task_state.phase == "search"),
        float(task_state.phase == "probe"),
        float(task_state.phase == "commit"),
        math.log1p(len(task_state.table_candidate_pages)),
        math.log1p(len(task_state.supporting_pages)),
        float(task_state.last_tool is not None),
    ] + [0.0] * 16


def _action_feature_vector(action: Any, task_state: TaskStateView, particle: PosteriorParticle) -> list[float]:
    value = canonical_action(action)
    kind = value.get("kind")
    tool = str(value.get("tool") or "")
    numbers = []
    arguments = value.get("arguments") if isinstance(value.get("arguments"), Mapping) else {}
    for key in ("page", "page_number"):
        try:
            numbers.append(float(arguments.get(key)))
            break
        except (TypeError, ValueError):
            continue
    bbox = arguments.get("bbox", arguments.get("region"))
    page_numbers = arguments.get("page_numbers")
    page_span = len(page_numbers) if isinstance(page_numbers, (list, tuple, set)) else (1 if numbers else 0)
    area = 0.0
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            area = abs((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
        except (TypeError, ValueError):
            area = 0.0
    digest = hashlib.sha256(canonical_action_key(action).encode("utf-8")).digest()
    hash_features = [byte / 255.0 for byte in digest[:8]]
    return (
        [float(kind == name) for name in ACTION_KINDS]
        + [float(tool == name) for name in TOOL_NAMES]
        + [
            numbers[0] / 100.0 if numbers else 0.0,
            min(1.0, float(page_span) / 8.0),
            area,
            float(particle.tool_quality.get(tool).relative_cost if tool in particle.tool_quality else 1.0),
            float(task_state.last_tool == tool),
            float(action_is_diagnostic(action)),
        ]
        + hash_features
        + [0.0] * 7
    )


def _particle_feature_vector(particle: PosteriorParticle) -> list[float]:
    values = [
        float(particle.session_state == name)
        for name in ("healthy", "degraded", "overloaded", "outage")
    ] + [
        float(particle.regime_state == name)
        for name in ("stable", "abrupt_transition", "gradual_transition")
    ]
    for family in sorted(particle.shared_states):
        values.extend(
            float(particle.shared_states[family] == name)
            for name in ("healthy", "degraded", "down")
        )
    # Keep the particle branch factorized: the action feature carries the
    # selected tool, while this vector carries pooled evidence from *all*
    # tool posteriors and shared family states.  Do not truncate a flattened
    # per-tool list, which silently dropped the later tools from Q training.
    quality_dimensions = ("availability", "semantic_accuracy", "structure_fidelity", "relative_cost")
    pooled: list[float] = []
    for dimension in quality_dimensions:
        values_for_dimension = [
            float(getattr(particle.tool_quality[name], dimension))
            for name in TOOL_NAMES
            if name in particle.tool_quality
        ]
        if not values_for_dimension:
            values_for_dimension = [0.0]
        mean = sum(values_for_dimension) / len(values_for_dimension)
        minimum = min(values_for_dimension)
        maximum = max(values_for_dimension)
        variance = sum((value - mean) ** 2 for value in values_for_dimension) / len(values_for_dimension)
        pooled.extend([mean, minimum, maximum, math.sqrt(max(0.0, variance))])
    result = values + pooled
    if len(result) != 32:
        raise RuntimeError(f"factorized particle feature contract must be 32-D, got {len(result)}")
    return result


def q_feature_vectors(
    task_state: TaskStateView,
    particle: PosteriorParticle,
    action: Any,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return the fixed-width feature contract used by BayesQHead replay."""

    return (
        _task_feature_vector(task_state),
        _particle_feature_vector(particle),
        _action_feature_vector(action, task_state, particle),
        [math.log1p(task_state.remaining_tool_budget)] + [0.0] * 7,
    )


def heuristic_action_value(action: Any, task_state: TaskStateView, particle: PosteriorParticle) -> float:
    value = canonical_action(action)
    kind = value.get("kind")
    if kind == "final":
        return 0.85 if task_state.evidence_sufficient else 0.05
    if kind == "abstain":
        return 0.25 if not task_state.evidence_sufficient else -0.25
    tool = str(value.get("tool") or "")
    quality = particle.tool_quality.get(tool)
    if quality is None:
        return -0.25
    expected = 0.65 * quality.semantic_accuracy + 0.30 * quality.structure_fidelity + 0.05 * quality.availability
    cost_penalty = 0.05 * quality.relative_cost
    return expected - cost_penalty + (0.08 if action_is_diagnostic(action) and not task_state.evidence_sufficient else 0.0)


@dataclass(frozen=True)
class DecisionReport:
    action_values: tuple[ActionValue, ...]
    consensus: float
    consensus_action: str | None
    bayes_action: str | None
    decision_regret: float
    particles: tuple[PosteriorParticle, ...]
    selected_action: str | None = None
    selected_mode: str = "bayes"
    dvoi: dict[str, float] | None = None
    stop_decision: dict[str, Any] | None = None
    candidate_actions: tuple[tuple[str, Any], ...] = ()
    candidate_degenerate: bool = False


class AnswerRiskCalibrator:
    """A small logistic risk calibrator; coefficients can be fitted offline."""

    FEATURE_NAMES = (
        "final_evidence_support",
        "independent_tool_count",
        "semantic_posterior",
        "structure_posterior",
        "unvisited_page_ratio",
        "recent_surprise",
        "answer_self_consistency",
        "remaining_budget",
    )

    def __init__(self, weights: Sequence[float] | None = None, bias: float = 0.0) -> None:
        self.weights = tuple(float(value) for value in (weights or (-2.0, -0.25, -1.0, -0.75, 0.7, 0.35, -0.5, -0.05)))
        if len(self.weights) != len(self.FEATURE_NAMES):
            raise ValueError("risk calibrator weight count mismatch")
        self.bias = float(bias)

    def fit(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        labels: Sequence[float],
        *,
        epochs: int = 400,
        learning_rate: float = 0.05,
        l2: float = 1.0e-3,
    ) -> dict[str, float]:
        """Fit the calibrator on validation-time binary error labels.

        This is deliberately a small in-process logistic regression instead
        of a dependency on sklearn.  The runtime feature schema is fixed, so
        the fitted coefficients can be serialized and loaded by every rollout
        worker without importing a second ML framework.
        """

        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must have equal length")
        if not feature_rows:
            raise ValueError("risk calibrator needs at least one validation row")
        weights = [float(value) for value in self.weights]
        bias = float(self.bias)
        rows = [
            [float(row.get(name, 0.0) or 0.0) for name in self.FEATURE_NAMES]
            for row in feature_rows
        ]
        targets = [_clip(float(label), 0.0, 1.0) for label in labels]
        count = float(len(rows))
        for _ in range(max(1, int(epochs))):
            grad_w = [0.0] * len(weights)
            grad_b = 0.0
            for row, target in zip(rows, targets, strict=True):
                score = bias + sum(weight * value for weight, value in zip(weights, row, strict=True))
                probability = 1.0 / (1.0 + math.exp(-_clip(score, -30.0, 30.0)))
                error = probability - target
                grad_b += error
                for index, value in enumerate(row):
                    grad_w[index] += error * value
            grad_b /= count
            bias -= float(learning_rate) * grad_b
            for index in range(len(weights)):
                grad_w[index] = grad_w[index] / count + float(l2) * weights[index]
                weights[index] -= float(learning_rate) * grad_w[index]
        self.weights = tuple(weights)
        self.bias = bias
        probabilities = [self.predict(row) for row in feature_rows]
        log_loss = -sum(
            target * math.log(max(1e-8, probability))
            + (1.0 - target) * math.log(max(1e-8, 1.0 - probability))
            for target, probability in zip(targets, probabilities, strict=True)
        ) / count
        return {"samples": count, "log_loss": float(log_loss)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerRiskCalibrator":
        names = payload.get("feature_names")
        if names is not None and tuple(str(value) for value in names) != cls.FEATURE_NAMES:
            raise ValueError("risk calibrator feature schema mismatch")
        return cls(weights=payload.get("weights"), bias=float(payload.get("bias", 0.0) or 0.0))

    def predict(self, features: Mapping[str, float]) -> float:
        score = self.bias + sum(weight * float(features.get(name, 0.0)) for name, weight in zip(self.FEATURE_NAMES, self.weights, strict=True))
        return 1.0 / (1.0 + math.exp(-_clip(score, -30.0, 30.0)))

    def risk_features(self, task_state: TaskStateView, metadata: Mapping[str, Any] | None = None) -> dict[str, float]:
        metadata = metadata or {}
        nested_bayes = metadata.get("bayestool", {})
        nested_bayes = nested_bayes if isinstance(nested_bayes, Mapping) else {}
        snapshot = metadata.get("belief_snapshot", nested_bayes.get("belief_snapshot", {}))
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        quality = metadata.get("tool_quality", snapshot.get("tool_quality", snapshot.get("tools", {})))
        quality = quality if isinstance(quality, Mapping) else {}
        last_tool = str(metadata.get("last_tool") or task_state.last_tool or "")
        selected_quality = quality.get(last_tool, {}) if last_tool else {}
        if not isinstance(selected_quality, Mapping):
            selected_quality = {}

        def _quality_mean(name: str) -> float:
            value = selected_quality.get(name, 0.0)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = value[0] if value else 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        semantic_default = _quality_mean("semantic")
        structure_default = _quality_mean("structure")
        semantic = float(metadata.get("semantic_posterior", semantic_default) or 0.0)
        structure = float(metadata.get("structure_posterior", structure_default) or 0.0)
        navigation = metadata.get("navigation_state", {})
        navigation = navigation if isinstance(navigation, Mapping) else {}
        page_count_value = metadata.get(
            "page_count",
            navigation.get("page_count", len(task_state.visited_pages) + (task_state.unvisited_page_count or 0)),
        )
        try:
            page_count = max(1, int(page_count_value or 0))
        except (TypeError, ValueError):
            page_count = max(1, len(task_state.visited_pages) + (task_state.unvisited_page_count or 0))
        independent_default = metadata.get("independent_tool_count")
        if independent_default is None:
            unique_tools = metadata.get("unique_tools", ())
            independent_default = len(unique_tools) if isinstance(unique_tools, Sequence) and not isinstance(unique_tools, (str, bytes)) else 0.0
        surprise_default = metadata.get("recent_surprise", snapshot.get("recent_surprise", 0.0))
        consistency_default = metadata.get("answer_self_consistency", 0.0)
        return {
            "final_evidence_support": float(
                metadata.get("final_evidence_support", task_state.evidence_sufficient)
            ),
            "independent_tool_count": float(independent_default or 0.0),
            "semantic_posterior": semantic,
            "structure_posterior": structure,
            "unvisited_page_ratio": float(task_state.unvisited_page_count or 0) / page_count,
            "recent_surprise": float(surprise_default or 0.0),
            "answer_self_consistency": float(consistency_default or 0.0),
            "remaining_budget": math.log1p(task_state.remaining_tool_budget),
        }


class DecisionController:
    """Implement Persistent Consensus–Probe–Reopen–Robust Commit."""

    def __init__(
        self,
        config: BayesToolConfig | None = None,
        *,
        q_head: BayesQHead | None = None,
        risk_calibrator: AnswerRiskCalibrator | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or default_config(enabled=True)
        self.q_head = q_head
        self.risk_calibrator = risk_calibrator or AnswerRiskCalibrator()
        self.seed = int(seed)
        self.q_head_ready = False
        self.reports: list[DecisionReport] = []

    def enable_q_head(self, enabled: bool = True) -> None:
        """Enable learned Q values after a warm-up replay has been fitted."""

        self.q_head_ready = bool(enabled)

    def evaluate(
        self,
        task_state: TaskStateView,
        belief: BeliefRuntime | BeliefSnapshot,
        candidates: Iterable[Any],
        *,
        seed: int | None = None,
    ) -> DecisionReport:
        snapshot = belief.snapshot() if isinstance(belief, BeliefRuntime) else belief
        keys: list[str] = []
        actions: dict[str, Any] = {}
        for action in candidates:
            key = canonical_action_key(action)
            if key not in actions and len(actions) < self.config.max_action_candidates:
                actions[key] = action
                keys.append(key)
        particles = sample_posterior_particles(
            snapshot,
            count=self.config.posterior_particles,
            seed=seed if seed is not None else stable_seed(self.seed, snapshot.step, snapshot.version),
        )
        candidate_degenerate = len(keys) < 2
        action_values: list[ActionValue] = []
        particle_best: list[str] = []
        values_by_action: dict[str, list[float]] = {key: [] for key in keys}
        for particle in particles:
            values = {}
            for key in keys:
                value = heuristic_action_value(actions[key], task_state, particle)
                if self.q_head is not None and self.q_head_ready and torch is not None:
                    with torch.no_grad():
                        q_output = self.q_head(
                            torch.tensor([_task_feature_vector(task_state)], dtype=torch.float32),
                            torch.tensor([_particle_feature_vector(particle)], dtype=torch.float32),
                            torch.tensor([_action_feature_vector(actions[key], task_state, particle)], dtype=torch.float32),
                            torch.tensor([[math.log1p(task_state.remaining_tool_budget)] + [0.0] * 7], dtype=torch.float32),
                        )
                        value = float(q_output[0, 0].item())
                values[key] = value
            if values:
                particle_best.append(max(values, key=lambda key: (values[key], key)))
            for key, value in values.items():
                values_by_action[key].append(value)
        for key in keys:
            values = values_by_action[key]
            mean = sum(values) / len(values) if values else -1.0
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
            action_values.append(ActionValue(key, mean, math.log(max(1e-8, variance)), tuple(values)))
        action_values.sort(key=lambda item: (-item.value_mean, item.action_key))
        weighted_votes: dict[str, float] = {key: 0.0 for key in keys}
        for key in particle_best:
            weighted_votes[key] = weighted_votes.get(key, 0.0) + 1.0 / max(1, len(particles))
        consensus_action = (
            max(weighted_votes, key=lambda key: (weighted_votes[key], key))
            if weighted_votes and not candidate_degenerate
            else None
        )
        consensus = weighted_votes.get(consensus_action, 0.0) if consensus_action else 0.0
        bayes_action = action_values[0].action_key if action_values else None
        regret = 0.0
        if action_values and particles and not candidate_degenerate:
            for particle_index, particle in enumerate(particles):
                particle_values = {key: heuristic_action_value(actions[key], task_state, particle) for key in keys}
                if self.q_head is not None and self.q_head_ready and torch is not None:
                    with torch.no_grad():
                        for key in keys:
                            q_output = self.q_head(
                                torch.tensor([_task_feature_vector(task_state)], dtype=torch.float32),
                                torch.tensor([_particle_feature_vector(particle)], dtype=torch.float32),
                                torch.tensor([_action_feature_vector(actions[key], task_state, particle)], dtype=torch.float32),
                                torch.tensor([[math.log1p(task_state.remaining_tool_budget)] + [0.0] * 7], dtype=torch.float32),
                            )
                            particle_values[key] = float(q_output[0, 0].item())
                best = max(particle_values.values(), default=0.0)
                chosen = particle_values.get(bayes_action, 0.0)
                regret += particle.weight * (best - chosen)
        report = DecisionReport(
            tuple(action_values),
            consensus,
            consensus_action,
            bayes_action,
            regret,
            particles,
            candidate_actions=tuple(actions.items()),
            candidate_degenerate=candidate_degenerate,
        )
        self.reports.append(report)
        return report

    def _robust_action(self, report: DecisionReport) -> str | None:
        if not report.action_values:
            return None
        alpha = _clip(self.config.cvar_alpha, 0.01, 0.99)
        scores = []
        for value in report.action_values:
            ordered = sorted(value.particle_values)
            count = max(1, int(math.ceil(len(ordered) * alpha)))
            scores.append((sum(ordered[:count]) / count, value.action_key))
        return max(scores, key=lambda item: (item[0], item[1]))[1]

    def dvoi(
        self,
        task_state: TaskStateView,
        belief: BeliefRuntime,
        report: DecisionReport,
        diagnostic_actions: Iterable[Any],
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        for action in diagnostic_actions:
            if not action_is_diagnostic(action):
                continue
            action_key = canonical_action_key(action)
            tool = str(canonical_action(action).get("tool") or "")
            prediction = belief.predict_observation(tool, task_state)
            # The diagnostic tool itself consumes one call.  All hypothetical
            # continuation values and the stop controller must therefore see
            # the remaining budget after that observation, not the pre-action
            # budget that selected the probe.
            next_task_state = replace(
                task_state,
                remaining_tool_budget=max(0, task_state.remaining_tool_budget - 1),
                last_tool=tool or task_state.last_tool,
            )
            # Enumerate the full factorized observation distribution, then
            # retain only the six highest-probability joint atoms required by
            # the method.  Every predicted observation head participates in
            # the atom: status, latency, information gain, semantic
            # agreement, schema validity, and image validity.
            schema_probability = _clip(float(prediction.schema_valid_prob), 1e-6, 1.0 - 1e-6)
            image_probability = _clip(float(prediction.image_valid_prob), 1e-6, 1.0 - 1e-6)
            status_values = [
                (status, float(prediction.status_probs[index]))
                for index, status in enumerate(STATUS_NAMES)
                if index < len(prediction.status_probs) and float(prediction.status_probs[index]) > 0.0
            ]
            latency_values = [
                ((index + 0.5) / 2.0, float(value))
                for index, value in enumerate(prediction.latency_probs)
                if float(value) > 0.0
            ]
            information_values = [
                ((index + 0.5) / 5.0, float(value))
                for index, value in enumerate(prediction.information_gain_probs)
                if float(value) > 0.0
            ]
            semantic_values = [
                ((index + 0.5) / 5.0, float(value))
                for index, value in enumerate(prediction.semantic_agreement_probs)
                if float(value) > 0.0
            ]
            hypotheses: list[tuple[str, float, float, float, float, bool, bool]] = []
            for (
                (status, status_probability),
                (latency, latency_probability),
                (information_gain, information_probability),
                (semantic_agreement, semantic_probability),
                (schema_valid, schema_factor),
                (image_valid, image_factor),
            ) in itertools.product(
                status_values,
                latency_values,
                information_values,
                semantic_values,
                ((True, schema_probability), (False, 1.0 - schema_probability)),
                ((True, image_probability), (False, 1.0 - image_probability)),
            ):
                hypotheses.append(
                    (
                        status,
                        status_probability * latency_probability * information_probability * semantic_probability * schema_factor * image_factor,
                        latency,
                        information_gain,
                        semantic_agreement,
                        schema_valid,
                        image_valid,
                    )
                )
            hypotheses.sort(key=lambda item: item[1], reverse=True)
            expected_next_regret = 0.0
            total_weight = sum(item[1] for item in hypotheses[: self.config.max_observation_hypotheses]) or 1.0
            for status, probability, latency, information_gain, semantic_agreement, schema_valid, image_valid in hypotheses[: self.config.max_observation_hypotheses]:
                clone = belief.hypothetical_update(
                    tool,
                    {
                        "tool": tool,
                        "status": status,
                        "latency": latency,
                        "information_gain": information_gain,
                        "semantic_agreement": semantic_agreement,
                        "schema_valid": schema_valid,
                        "image_valid": image_valid,
                    },
                    task_state=next_task_state,
                )
                candidate_actions = [action for _, action in report.candidate_actions]
                if not candidate_actions:
                    # Reports created by older callers may not carry the
                    # original action objects. Reconstruct the canonical key
                    # where possible so DVOI still evaluates the same finite
                    # action set instead of treating every branch as empty.
                    for value in report.action_values:
                        value_key = value.action_key
                        if value_key.startswith("tool:"):
                            _, tool_name, raw_arguments = value_key.split(":", 2)
                            try:
                                candidate_actions.append(
                                    {"kind": "tool", "tool": tool_name, "arguments": json.loads(raw_arguments)}
                                )
                            except json.JSONDecodeError:
                                continue
                        elif value_key.startswith("final:"):
                            candidate_actions.append({"kind": "final", "answer": value_key[6:]})
                        elif value_key.startswith("abstain:"):
                            candidate_actions.append({"kind": "abstain", "reason": value_key[8:]})
                # Hypothetical branches must not mutate the controller's
                # public report history (or alter the later random branch
                # gate through its report count).
                hypothetical_controller = DecisionController(
                    self.config,
                    q_head=self.q_head,
                    risk_calibrator=self.risk_calibrator,
                    seed=stable_seed(
                        self.seed,
                        action_key,
                        status,
                        latency,
                        information_gain,
                        semantic_agreement,
                        schema_valid,
                        image_valid,
                    ),
                )
                hypothetical_controller.enable_q_head(self.q_head_ready)
                next_report = hypothetical_controller.evaluate(next_task_state, clone, candidate_actions)
                expected_next_regret += probability / total_weight * next_report.decision_regret
            tool_quality = belief.snapshot().tool_quality.get(tool)
            cost = float(tool_quality.cost_mean if tool_quality else 1.0)
            budget_pressure = 1.0 / max(1, task_state.remaining_tool_budget)
            output[action_key] = (
                report.decision_regret
                - expected_next_regret
                - 0.05 * cost * (1.0 + budget_pressure)
            )
        return output

    def select(
        self,
        task_state: TaskStateView,
        belief: BeliefRuntime,
        candidates: Iterable[Any],
        *,
        diagnostic_actions: Iterable[Any] = (),
        risk_metadata: Mapping[str, Any] | None = None,
    ) -> DecisionReport:
        report = self.evaluate(task_state, belief, candidates)
        dvoi_values: dict[str, float] = {}
        if report.consensus >= self.config.consensus_threshold:
            selected = report.consensus_action
            mode = "consensus"
        elif report.decision_regret >= self.config.decision_regret_threshold and task_state.remaining_tool_budget >= 2:
            if self.config.use_dvoi:
                dvoi_values = self.dvoi(task_state, belief, report, diagnostic_actions)
            positive = [(value, key) for key, value in dvoi_values.items() if value > self.config.dvoi_minimum]
            if positive:
                selected = max(positive, key=lambda item: (item[0], item[1]))[1]
                mode = "probe"
            else:
                selected = self._robust_action(report)
                mode = "robust"
        else:
            selected = report.bayes_action
            mode = "bayes"

        risk = self.risk_calibrator.predict(self.risk_calibrator.risk_features(task_state, risk_metadata))
        stop_risk = risk
        continue_cost = min((0.05 + 0.05 * (task_state.remaining_tool_budget <= 1)), 1.0)
        continue_risk = min(1.0, continue_cost + max(0.0, report.decision_regret))
        abstain_risk = 0.35
        stop_mode = "stop" if stop_risk <= min(continue_risk, abstain_risk) else ("abstain" if abstain_risk < continue_risk else "continue")
        final_key = next(
            (
                key
                for key, action in report.candidate_actions
                if canonical_action(action).get("kind") == "final"
            ),
            None,
        )
        abstain_key = next(
            (
                key
                for key, action in report.candidate_actions
                if canonical_action(action).get("kind") == "abstain"
            ),
            None,
        )
        if stop_mode == "abstain":
            if abstain_key is not None:
                selected = abstain_key
                mode = "abstain"
            else:
                # A controller may only select actions that the policy
                # actually generated for this turn.  If no abstain action was
                # proposed, keep the selected policy action and continue.
                stop_mode = "continue"
        elif stop_mode == "stop" and final_key is not None:
            # Stop is executable only when the policy supplied a concrete
            # final action in this candidate set.
            selected = final_key
            mode = "stop"
        elif stop_mode == "stop":
            stop_mode = "continue"
        result = DecisionReport(
            report.action_values,
            report.consensus,
            report.consensus_action,
            report.bayes_action,
            report.decision_regret,
            report.particles,
            selected_action=selected,
            selected_mode=mode,
            dvoi=dvoi_values,
            stop_decision={
                "stop_risk": stop_risk,
                "abstain_risk": abstain_risk,
                "continue_risk": continue_risk,
                "mode": stop_mode,
            },
            candidate_actions=report.candidate_actions,
            candidate_degenerate=report.candidate_degenerate,
        )
        self.reports[-1] = result
        return result

    def branch_gate(
        self,
        task_state: TaskStateView,
        report: DecisionReport,
        *,
        belief: BeliefRuntime | BeliefSnapshot | None = None,
        rng: random.Random | None = None,
    ) -> bool:
        """Apply the exact eligibility and unbiased branch-probability gate."""

        if not self.config.use_regret_branching:
            return False
        if report.candidate_degenerate:
            return False
        if report.decision_regret <= self.config.decision_regret_threshold:
            return False
        if task_state.remaining_tool_budget < 2:
            return False
        if belief is not None:
            snapshot = belief.snapshot() if isinstance(belief, BeliefRuntime) else belief
            if snapshot.ood_score >= 0.15:
                return False
        if report.stop_decision and float(report.stop_decision.get("ood_score", 0.0)) >= 0.15:
            return False
        generator = rng or random.Random(stable_seed(self.seed, task_state.remaining_tool_budget, len(self.reports)))
        return generator.random() < self.config.branch_probability_when_eligible


__all__ = [
    "ACTION_KINDS",
    "js_divergence",
    "canonical_action",
    "canonical_action_key",
    "action_kind",
    "action_is_diagnostic",
    "sample_posterior_particles",
    "BayesQHead",
    "q_feature_vectors",
    "AnswerRiskCalibrator",
    "DecisionReport",
    "DecisionController",
]
