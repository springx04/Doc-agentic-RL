"""Local, framework-agnostic Agentic RL training harness.

This entry point intentionally does not start Ray, SGLang, a remote model, or
any document backend.  It provides the local contract that the eventual slime
launcher can call: collect a group of rollouts, compute valid group-relative
advantages, add trajectories to prioritized replay, and evaluate the clipped
GRPO objective with KL/entropy terms.

Run ``python toolcall-rl/train.py --smoke`` for a deterministic CPU check.
Production model generation remains in the existing slime rollout entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from config.args import AgenticRLConfig, build_parser
from utils.bayes_tool import BayesToolSelector, ToolSelection
from utils.replay_buffer import PriorityReplayBuffer
from utils.rewards import (
    RewardConfig,
    compute_composite_reward,
    compute_group_advantages,
    compute_grpo_loss,
)


def train(args: Any) -> Any:
    """Proxy for the existing Slime train entry point."""
    slime_train_path = (Path.cwd() / "train.py").resolve()
    if slime_train_path == Path(__file__).resolve() or not slime_train_path.is_file():
        raise ImportError(f"cannot locate the existing Slime train module at {slime_train_path}")
    spec = importlib.util.spec_from_file_location("_openclaw_slime_train", slime_train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the existing Slime train module at {slime_train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train(args)

@dataclass
class Rollout:
    """Minimal model-independent representation of one sampled trajectory."""

    output: Any = None
    reward: float = 0.0
    old_log_prob: float = 0.0
    new_log_prob: float | None = None
    ref_log_prob: float | None = None
    entropy: float = 0.0
    tools_used: tuple[str, ...] = ()
    format_ok: bool = True
    valid_for_rl: bool = True
    information_gain: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "Rollout | Mapping[str, Any] | Sequence[Any]") -> "Rollout":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            tools = value.get("tools_used", value.get("tool_names", ()))
            if isinstance(tools, (str, bytes)):
                tools = (str(tools),)
            return cls(
                output=value.get("output", value.get("response")),
                reward=float(value.get("reward", value.get("total_reward", value.get("score", 0.0))) or 0.0),
                old_log_prob=float(value.get("old_log_prob", value.get("log_prob", 0.0)) or 0.0),
                new_log_prob=(
                    float(value["new_log_prob"])
                    if value.get("new_log_prob") is not None
                    else None
                ),
                ref_log_prob=(
                    float(value["ref_log_prob"])
                    if value.get("ref_log_prob") is not None
                    else None
                ),
                entropy=float(value.get("entropy", 0.0) or 0.0),
                tools_used=tuple(str(tool) for tool in (tools or ())),
                format_ok=bool(value.get("format_ok", value.get("format", True))),
                valid_for_rl=bool(value.get("valid_for_rl", not value.get("exclude_from_group_statistics", False))),
                information_gain=float(value.get("information_gain", 0.0) or 0.0),
                metadata=dict(value.get("metadata") or {}),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return cls(output=value[0], reward=float(value[1]))
        raise TypeError("rollout must be a Rollout, mapping, or (output, reward) sequence")


@dataclass
class TrainStepResult:
    """Auditable metrics from one group update."""

    metrics: dict[str, Any]

    @property
    def loss(self) -> float:
        return float(self.metrics.get("loss", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.metrics)


class AgenticRLTrainer:
    """Collect and score groups of agentic rollouts on the local machine."""

    def __init__(
        self,
        config: AgenticRLConfig | None = None,
        *,
        tool_names: Sequence[str] | None = None,
        selector: BayesToolSelector | None = None,
        replay_buffer: PriorityReplayBuffer | None = None,
        update_callback: Callable[[TrainStepResult, list[Rollout]], Any] | None = None,
    ) -> None:
        self.config = config or AgenticRLConfig()
        self.selector = selector
        if self.selector is None and self.config.bayes_tool_enabled and tool_names is not None:
            self.selector = BayesToolSelector(
                tool_names,
                prior_alpha=self.config.bayes_prior_alpha,
                prior_beta=self.config.bayes_prior_beta,
                confidence_threshold=self.config.tool_confidence_threshold,
                seed=self.config.seed,
            )
        self.replay_buffer = replay_buffer
        if self.replay_buffer is None and self.config.replay_enabled:
            self.replay_buffer = PriorityReplayBuffer(
                self.config.replay_capacity,
                alpha=self.config.replay_alpha,
                seed=self.config.seed,
            )
        self.reward_config = RewardConfig(
            kl_beta=self.config.kl_beta,
            information_gain_weight=self.config.information_gain_weight,
            multi_tool_bonus=self.config.multi_tool_bonus,
            tool_cost=self.config.tool_cost,
            format_error_penalty=self.config.format_error_penalty,
        )
        self.update_callback = update_callback
        self.step_count = 0
        self.history: list[dict[str, Any]] = []

    def select_tool(
        self,
        policy_probs: Mapping[str, float],
        *,
        call_probability: float = 1.0,
        threshold: float | None = None,
        sample: bool = True,
    ) -> ToolSelection | None:
        if self.selector is None:
            return None
        gate = float(call_probability)
        if not 0.0 <= gate <= 1.0:
            raise ValueError("call_probability must be in [0, 1]")
        gated_probs = {name: float(probability) * gate for name, probability in policy_probs.items()}
        selection = self.selector.select(gated_probs, threshold=threshold, sample=sample)
        from dataclasses import replace

        return replace(selection, call_probability=gate)

    def observe_tool(self, tool_name: str, success: bool, *, weight: float = 1.0) -> float:
        if self.selector is None:
            return 0.0
        return self.selector.update(tool_name, success, weight=weight)

    def score_rollout(
        self,
        *,
        output: Any,
        ground_truth: Any,
        format_ok: bool,
        tools_used: Iterable[Any] = (),
        information_gain: float = 0.0,
        kl_divergence: float = 0.0,
        task_score: float | None = None,
    ) -> dict[str, Any]:
        return compute_composite_reward(
            output,
            ground_truth,
            format_ok,
            tools_used,
            information_gain=information_gain,
            kl_divergence=kl_divergence,
            task_score=task_score,
            config=self.reward_config,
        )

    def rollout_temperature(self, entropy: float | None = None) -> float:
        """Return the configured temperature, optionally adapted by entropy."""

        temperature = float(self.config.temperature)
        if self.config.adaptive_temperature and entropy is not None:
            # Entropy is normalized to a bounded control signal for a model
            # callback; high uncertainty increases exploration smoothly.
            temperature += 0.2 * min(1.0, max(0.0, float(entropy)))
        return min(self.config.temperature_max, max(self.config.temperature_min, temperature))

    def run_episode(
        self,
        initial_observation: Any,
        policy_fn: Callable[[Any, "AgenticRLTrainer"], Mapping[str, Any]],
        tool_executor: Callable[[str, Any], Any],
        *,
        max_steps: int = 8,
        update_observation: Callable[[Any, Any], Any] | None = None,
    ) -> dict[str, Any]:
        """Run the local action/observation loop from the design document.

        ``policy_fn`` returns ``tool_probs`` and may return ``action="text"``
        to finish without a tool.  ``tool_executor`` returns either
        ``(result, success)`` or a mapping containing those fields.  This
        adapter is intentionally synchronous and dependency-free, making it a
        useful contract test for a later model/environment integration.
        """

        limit = int(max_steps)
        if limit <= 0:
            raise ValueError("max_steps must be greater than zero")
        observation = initial_observation
        steps: list[dict[str, Any]] = []
        for step_index in range(limit):
            decision = policy_fn(observation, self)
            if not isinstance(decision, Mapping):
                raise TypeError("policy_fn must return a mapping")
            if str(decision.get("action", "call_tool")).lower() in {"text", "final", "stop"}:
                return {
                    "observation": observation,
                    "steps": steps,
                    "terminated": True,
                    "stop_reason": "policy_text_action",
                }
            raw_probs = decision.get("tool_probs", decision.get("policy_probs"))
            if not isinstance(raw_probs, Mapping) or not raw_probs:
                raise ValueError("policy decision must contain a non-empty tool_probs mapping")
            selection = self.select_tool(
                raw_probs,
                call_probability=float(decision.get("call_probability", 1.0)),
                threshold=decision.get("threshold"),
                sample=bool(decision.get("sample", True)),
            )
            if selection is None or not selection.should_call or selection.tool_name is None:
                return {
                    "observation": observation,
                    "steps": steps,
                    "terminated": True,
                    "stop_reason": "bayes_threshold",
                }
            tool_name = selection.tool_name
            try:
                execution = tool_executor(tool_name, observation)
                if isinstance(execution, Mapping):
                    result = execution.get("result", execution.get("observation"))
                    success = bool(execution.get("success", False))
                elif isinstance(execution, Sequence) and not isinstance(execution, (str, bytes)) and len(execution) >= 2:
                    result, success = execution[0], bool(execution[1])
                else:
                    result, success = execution, bool(execution)
                error = None
            except Exception as exc:  # Keep the failed action observable for local training.
                result, success, error = {"error": str(exc)}, False, str(exc)
            information_gain = self.observe_tool(tool_name, success)
            steps.append(
                {
                    "step_index": step_index,
                    "tool": tool_name,
                    "selection": selection.to_dict(),
                    "result": result,
                    "success": success,
                    "error": error,
                    "information_gain": information_gain,
                }
            )
            observation = update_observation(observation, result) if update_observation else result
        return {
            "observation": observation,
            "steps": steps,
            "terminated": True,
            "stop_reason": "max_steps",
        }

    @staticmethod
    def _kl_value(rollout: Rollout) -> float:
        if rollout.new_log_prob is None or rollout.ref_log_prob is None:
            return 0.0
        # The full model KL is computed by the model backend.  This scalar
        # token-level proxy keeps the local harness useful without a model.
        return max(0.0, float(rollout.new_log_prob) - float(rollout.ref_log_prob))

    def _objective_metrics(
        self,
        rollouts: list[Rollout],
        advantages: list[float],
    ) -> dict[str, Any]:
        if not rollouts:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl": 0.0,
                "entropy": 0.0,
                "kl_penalty": 0.0,
                "entropy_bonus": 0.0,
            }
        old_log_probs = [float(item.old_log_prob) for item in rollouts]
        new_log_probs = [
            float(item.old_log_prob if item.new_log_prob is None else item.new_log_prob)
            for item in rollouts
        ]
        policy_loss = float(
            compute_grpo_loss(
                old_log_probs,
                new_log_probs,
                advantages,
                clip_epsilon=self.config.clip_epsilon,
            )
        )
        kl = sum(self._kl_value(item) for item in rollouts) / len(rollouts)
        entropy = sum(max(0.0, float(item.entropy)) for item in rollouts) / len(rollouts)
        kl_penalty = self.config.kl_beta * kl
        entropy_bonus = self.config.entropy_coef * entropy
        loss = policy_loss + kl_penalty - entropy_bonus
        return {
            "loss": float(loss),
            "policy_loss": policy_loss,
            "kl": kl,
            "entropy": entropy,
            "kl_penalty": kl_penalty,
            "entropy_bonus": entropy_bonus,
        }

    def update_group(self, raw_rollouts: Sequence[Rollout | Mapping[str, Any] | Sequence[Any]]) -> TrainStepResult:
        """Compute one GRPO/ARPO-style update from a rollout group."""

        if not raw_rollouts:
            raise ValueError("a rollout group must not be empty")
        rollouts = [Rollout.from_value(value) for value in raw_rollouts]
        rewards = [float(item.reward) for item in rollouts]
        valid_mask = [bool(item.valid_for_rl) for item in rollouts]
        advantages = compute_group_advantages(rewards, valid_mask=valid_mask)
        for rollout, advantage in zip(rollouts, advantages):
            rollout.metadata["group_advantage"] = float(advantage)

        current_valid = [item for item in rollouts if item.valid_for_rl]
        current_advantages = [advantage for item, advantage in zip(rollouts, advantages) if item.valid_for_rl]
        replay_indices: list[int] = []
        replay_used = False
        training_rollouts = current_valid
        training_advantages = current_advantages
        if self.replay_buffer is not None:
            for rollout, advantage in zip(rollouts, advantages):
                if rollout.valid_for_rl:
                    self.replay_buffer.add(
                        rollout,
                        rollout.reward,
                        priority=max(abs(float(advantage)), self.replay_buffer.epsilon),
                        metadata={"group_advantage": float(advantage)},
                    )
            if len(self.replay_buffer) >= self.config.replay_min_size:
                batch_size = min(self.config.replay_batch_size, len(self.replay_buffer))
                batch = self.replay_buffer.sample(batch_size, beta=self.config.replay_beta, replace=False)
                replay_rollouts = [item.trajectory for item in batch.items]
                if replay_rollouts:
                    training_rollouts = [Rollout.from_value(item) for item in replay_rollouts]
                    training_advantages = [
                        float(item.metadata.get("group_advantage", 0.0))
                        for item in batch.items
                    ]
                    replay_indices = list(batch.indices)
                    replay_used = True
                    updated_priorities = [
                        max(abs(float(rollout.reward)), self.replay_buffer.epsilon)
                        for rollout in training_rollouts
                    ]
                    self.replay_buffer.update_priorities(replay_indices, updated_priorities)

        objective = self._objective_metrics(training_rollouts, training_advantages)
        valid_rewards = [item.reward for item in current_valid]
        metrics = {
            **objective,
            "algorithm": self.config.algorithm,
            "group_size": len(rollouts),
            "valid_rollouts": len(current_valid),
            "invalid_rollouts": len(rollouts) - len(current_valid),
            "mean_reward": sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0,
            "reward_std": (
                math.sqrt(sum((value - (sum(valid_rewards) / len(valid_rewards))) ** 2 for value in valid_rewards) / len(valid_rewards))
                if valid_rewards
                else 0.0
            ),
            "mean_advantage": sum(current_advantages) / len(current_advantages) if current_advantages else 0.0,
            "replay_used": replay_used,
            "replay_size": len(self.replay_buffer) if self.replay_buffer is not None else 0,
            "replay_indices": replay_indices,
            "bayes_information_gain": self.selector.total_information_gain if self.selector is not None else 0.0,
        }
        self.step_count += 1
        self.history.append(dict(metrics))
        result = TrainStepResult(metrics)
        if self.update_callback is not None:
            self.update_callback(result, training_rollouts)
        return result

    def train_epoch(
        self,
        samples: Iterable[Any],
        rollout_fn: Callable[[Any, int, "AgenticRLTrainer"], Rollout | Mapping[str, Any] | Sequence[Any]],
    ) -> list[TrainStepResult]:
        """Collect ``group_size`` rollouts per sample and update locally."""

        results = []
        for sample in samples:
            group = [rollout_fn(sample, index, self) for index in range(self.config.group_size)]
            results.append(self.update_group(group))
        return results

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.step_count,
            "last": dict(self.history[-1]) if self.history else None,
            "replay": self.replay_buffer.stats() if self.replay_buffer is not None else None,
            "bayes": self.selector.state_dict() if self.selector is not None else None,
        }


def _config_from_namespace(namespace: argparse.Namespace) -> AgenticRLConfig:
    values = vars(namespace).copy()
    config_path = values.pop("config", None)
    for key in ("smoke", "steps"):
        values.pop(key, None)
    config_values = AgenticRLConfig.from_file(config_path).to_dict() if config_path else {}
    config_values.update({key: value for key, value in values.items() if value is not None})
    return AgenticRLConfig.from_mapping(config_values)


def run_smoke(config: AgenticRLConfig, steps: int = 4) -> dict[str, Any]:
    """Run a deterministic CPU-only smoke training loop."""

    trainer = AgenticRLTrainer(
        config,
        tool_names=("parse_document", "render_page", "ocr_region"),
    )
    tasks = [{"id": index, "answer": "yes" if index % 2 == 0 else "no"} for index in range(max(1, int(steps)))]

    def rollout_fn(task: Mapping[str, Any], group_index: int, owner: AgenticRLTrainer) -> Rollout:
        correct = (group_index + int(task["id"])) % 3 != 0
        policy_probs = {
            "parse_document": 0.65 if group_index % 2 == 0 else 0.25,
            "render_page": 0.25 if group_index % 2 == 0 else 0.65,
            "ocr_region": 0.10,
        }
        selection = owner.select_tool(
            policy_probs,
            call_probability=0.95,
            threshold=0.0,
            sample=False,
        )
        tool = selection.tool_name if selection is not None and selection.should_call else None
        if tool is not None:
            owner.observe_tool(tool, correct)
        tools_used = (tool,) if tool is not None else ()
        reward_parts = owner.score_rollout(
            output="yes" if correct else "unknown",
            ground_truth=task["answer"],
            format_ok=True,
            tools_used=tools_used,
            information_gain=0.05 if tool is not None else 0.0,
        )
        return Rollout(
            output="yes" if correct else "unknown",
            reward=reward_parts["total_reward"],
            old_log_prob=-0.20 - 0.01 * group_index,
            new_log_prob=-0.19 - 0.01 * group_index,
            ref_log_prob=-0.21 - 0.01 * group_index,
            entropy=0.5 + 0.05 * (group_index % 2),
            tools_used=tools_used,
            metadata={
                "task_id": task["id"],
                "reward_components": reward_parts,
                "bayes_selection": selection.to_dict() if selection is not None else None,
            },
        )

    trainer.train_epoch(tasks, rollout_fn)
    return trainer.summary()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--smoke", action="store_true", help="run the local deterministic smoke loop")
    parser.add_argument("--steps", type=int, default=4, help="number of smoke tasks")
    namespace = parser.parse_args(argv)
    if not namespace.smoke:
        parser.print_help()
        return 0
    config = _config_from_namespace(namespace)
    print(json.dumps(run_smoke(config, namespace.steps), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AgenticRLTrainer", "Rollout", "TrainStepResult", "run_smoke"]
