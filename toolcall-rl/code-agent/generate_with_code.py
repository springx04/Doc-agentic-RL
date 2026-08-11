"""Slime-compatible Code rollout entry points; Doc entry points are untouched."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from .rollout import CodeRollout, generate_code_trajectory
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from rollout import CodeRollout, generate_code_trajectory


async def generate(sample: Mapping[str, Any], *, model_client: Any, code_env_client: Any, code_config: Any = None, **kwargs: Any) -> dict[str, Any]:
    return await generate_code_trajectory(sample, model_client, code_env_client, code_config, **kwargs)


async def generate_batch(samples: list[Mapping[str, Any]], *, model_client: Any, code_env_client: Any, code_config: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
    return [await generate(sample, model_client=model_client, code_env_client=code_env_client, code_config=code_config, **kwargs) for sample in samples]


__all__ = ["CodeRollout", "generate", "generate_batch", "generate_code_trajectory"]
