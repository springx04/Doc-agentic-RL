"""Environment-neutral records used by the Code Agent runtime.

The records intentionally keep hidden world labels out of the policy-facing
``visible`` methods.  Trainer-only metadata is carried separately by rollout
code and never interpolated into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping


FailureOrigin = Literal["none", "world_injected", "real_infrastructure", "model_action"]
ToolStatus = Literal["ok", "partial", "error", "timeout", "invalid", "empty"]


@dataclass(frozen=True)
class CodeToolResult:
    tool_name: str
    status: ToolStatus | str
    output: str = ""
    latency_ms: float = 0.0
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_origin: FailureOrigin = "none"
    partial: bool = False
    corruption_type: str | None = None
    observation_delivered: bool = True

    @property
    def valid_for_rl(self) -> bool:
        return self.failure_origin != "real_infrastructure"

    def visible_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "status": self.status,
            "output": self.output,
            "latency_ms": round(float(self.latency_ms), 3),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "partial": bool(self.partial),
            "observation_delivered": bool(self.observation_delivered),
            "metadata": dict(self.metadata),
        }

    def to_dict(self, *, trainer_only: bool = False) -> dict[str, Any]:
        data = self.visible_dict()
        if trainer_only:
            data.update(
                {
                    "failure_origin": self.failure_origin,
                    "corruption_type": self.corruption_type,
                    "valid_for_rl": self.valid_for_rl,
                }
            )
        return data


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    output: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    transport_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_origin(self) -> FailureOrigin:
        return "real_infrastructure" if self.transport_error else "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output": self.output,
            "duration_ms": float(self.duration_ms),
            "timed_out": bool(self.timed_out),
            "transport_error": self.transport_error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CodeLease:
    lease_id: str
    instance_id: str
    image_name: str
    cwd: str
    base_revision: str | None = None
    backend: str = "remote"
    state: str = "active"
    created_at: float = 0.0
    last_heartbeat: float = 0.0

    @property
    def active(self) -> bool:
        return self.state == "active"


@dataclass(frozen=True)
class CodeEvalResult:
    ok: bool
    resolved: bool
    returncode: int | None = None
    output: str = ""
    error: str | None = None
    lease_id: str | None = None
    failure_origin: FailureOrigin = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_for_rl(self) -> bool:
        return self.failure_origin != "real_infrastructure"


@dataclass(frozen=True)
class ModelGeneration:
    """One model response with the exact accounting returned by the model client."""

    text: str
    token_ids: tuple[int, ...] = ()
    token_mask: tuple[int, ...] = ()
    token_logprobs: tuple[float, ...] = ()
    raw: Any = None

    def __post_init__(self) -> None:
        if self.token_mask and len(self.token_mask) != len(self.token_ids):
            raise ValueError("token_mask must have the same length as token_ids")
        if self.token_logprobs and len(self.token_logprobs) != len(self.token_ids):
            raise ValueError("token_logprobs must have the same length as token_ids")


@dataclass(frozen=True)
class CodeMessage:
    role: str
    content: str
    token_ids: tuple[int, ...] = ()
    token_mask: tuple[int, ...] = ()
    token_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True)
class CodeReward:
    task_quality: float
    utility: float
    cost: float
    inefficiency: float
    failure_penalty: float
    resolved: bool
    valid_for_rl: bool
    failure_origin: FailureOrigin = "none"

    @property
    def scalar(self) -> float:
        return float(self.utility)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_quality": float(self.task_quality),
            "utility": float(self.utility),
            "cost": float(self.cost),
            "inefficiency": float(self.inefficiency),
            "failure_penalty": float(self.failure_penalty),
            "resolved": bool(self.resolved),
            "valid_for_rl": bool(self.valid_for_rl),
            "failure_origin": self.failure_origin,
        }


@dataclass(frozen=True)
class CodeTrajectory:
    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    reward: float
    messages: tuple[CodeMessage, ...]
    metadata: dict[str, Any]
    trainer_only_metadata: dict[str, Any] = field(default_factory=dict)

    def to_sample(self) -> dict[str, Any]:
        """Return only fields safe for the training framework's policy sample."""

        return {
            "tokens": list(self.tokens),
            "loss_mask": list(self.loss_mask),
            "rollout_log_probs": list(self.rollout_log_probs),
            "reward": float(self.reward),
            "metadata": dict(self.metadata),
        }


def with_metadata(record: Any, **values: Any) -> Any:
    """Small helper used by stage code while retaining frozen dataclasses."""

    if hasattr(record, "metadata"):
        return replace(record, metadata={**getattr(record, "metadata"), **values})
    raise TypeError("record has no metadata field")


__all__ = [
    "CodeEvalResult",
    "CodeLease",
    "CodeMessage",
    "CodeReward",
    "CodeToolResult",
    "CodeTrajectory",
    "ExecutionResult",
    "FailureOrigin",
    "ModelGeneration",
    "ToolStatus",
]
