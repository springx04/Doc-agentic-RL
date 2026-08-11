"""Configuration and CLI arguments for the local agentic-RL harness."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass
class AgenticRLConfig:
    algorithm: str = "grpo"
    group_size: int = 4
    clip_epsilon: float = 0.2
    kl_beta: float = 0.1
    entropy_coef: float = 0.01
    learning_rate: float = 1e-5
    gradient_clip_norm: float = 5.0
    bayes_tool_enabled: bool = True
    tool_confidence_threshold: float = 0.8
    bayes_prior_alpha: float = 1.0
    bayes_prior_beta: float = 1.0
    replay_enabled: bool = True
    replay_capacity: int = 10_000
    replay_alpha: float = 0.6
    replay_beta: float = 0.4
    replay_batch_size: int = 32
    replay_min_size: int = 4
    information_gain_weight: float = 0.05
    multi_tool_bonus: float = 0.1
    tool_cost: float = 0.01
    format_error_penalty: float = -1.0
    process_reward_weight: float = 0.1
    tool_cost_weight: float = 0.05
    temperature: float = 1.0
    adaptive_temperature: bool = False
    temperature_min: float = 0.7
    temperature_max: float = 1.5
    seed: int = 42

    def __post_init__(self) -> None:
        self.algorithm = str(self.algorithm).lower()
        if self.algorithm not in {"grpo", "arpo"}:
            raise ValueError("algorithm must be 'grpo' or 'arpo'")
        if int(self.group_size) <= 0:
            raise ValueError("group_size must be greater than zero")
        if not 0.0 <= float(self.clip_epsilon) < 1.0:
            raise ValueError("clip_epsilon must be in [0, 1)")
        for name in (
            "kl_beta",
            "entropy_coef",
            "learning_rate",
            "gradient_clip_norm",
            "bayes_prior_alpha",
            "bayes_prior_beta",
            "replay_alpha",
            "replay_beta",
            "information_gain_weight",
            "multi_tool_bonus",
            "tool_cost",
            "process_reward_weight",
            "tool_cost_weight",
            "temperature",
            "temperature_min",
            "temperature_max",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.learning_rate) == 0.0 or float(self.gradient_clip_norm) == 0.0:
            raise ValueError("learning_rate and gradient_clip_norm must be greater than zero")
        if not 0.0 <= float(self.tool_confidence_threshold) <= 1.0:
            raise ValueError("tool_confidence_threshold must be in [0, 1]")
        if int(self.replay_capacity) <= 0:
            raise ValueError("replay_capacity must be greater than zero")
        if int(self.replay_batch_size) <= 0 or int(self.replay_min_size) <= 0:
            raise ValueError("replay batch sizes must be greater than zero")
        if float(self.temperature_min) > float(self.temperature_max):
            raise ValueError("temperature_min must not exceed temperature_max")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AgenticRLConfig":
        aliases = {
            "beta": "kl_beta",
            "alpha_ent": "entropy_coef",
            "G": "group_size",
            "epsilon": "clip_epsilon",
            "confidence_threshold": "tool_confidence_threshold",
        }
        normalized: dict[str, Any] = {}
        known = {field.name for field in fields(cls)}
        for key, value in values.items():
            target = aliases.get(str(key), str(key))
            if target in known:
                normalized[target] = value
        # Accept the natural nested form used by experiment configs.
        for section, prefix in (("bayes", "bayes_"), ("replay", "replay_")):
            nested = values.get(section)
            if isinstance(nested, Mapping):
                for key, value in nested.items():
                    target = f"{prefix}{key}"
                    if target in known:
                        normalized[target] = value
        return cls(**normalized)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgenticRLConfig":
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # Keep YAML optional for the lightweight local tests.
                raise ImportError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
            payload = yaml.safe_load(text)
        else:
            raise ValueError("config files must use .json, .yaml, or .yml")
        if not isinstance(payload, Mapping):
            raise ValueError("config file must contain an object/mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_agentic_rl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add feature flags that can also be merged into an existing launcher."""

    parser.add_argument("--config", type=Path, default=None, help="JSON/YAML agentic-RL config")
    parser.add_argument("--algorithm", choices=("grpo", "arpo"), default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--bayes-tool-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tool-confidence-threshold", type=float, default=None)
    parser.add_argument("--bayes-prior-alpha", type=float, default=None)
    parser.add_argument("--bayes-prior-beta", type=float, default=None)
    parser.add_argument("--replay-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--replay-alpha", type=float, default=None)
    parser.add_argument("--replay-beta", type=float, default=None)
    parser.add_argument("--replay-batch-size", type=int, default=None)
    parser.add_argument("--replay-min-size", type=int, default=None)
    parser.add_argument("--information-gain-weight", type=float, default=None)
    parser.add_argument("--multi-tool-bonus", type=float, default=None)
    parser.add_argument("--tool-cost", type=float, default=None)
    parser.add_argument("--format-error-penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--adaptive-temperature", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature-min", type=float, default=None)
    parser.add_argument("--temperature-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local BayesTool-RL/GRPO harness")
    return add_agentic_rl_args(parser)


def parse_args(argv: list[str] | None = None) -> AgenticRLConfig:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    values = vars(namespace).copy()
    config_path = values.pop("config", None)
    config_values: dict[str, Any] = {}
    if config_path is not None:
        config_values = AgenticRLConfig.from_file(config_path).to_dict()
    config_values.update({key: value for key, value in values.items() if value is not None})
    return AgenticRLConfig.from_mapping(config_values)


__all__ = ["AgenticRLConfig", "add_agentic_rl_args", "build_parser", "parse_args"]
