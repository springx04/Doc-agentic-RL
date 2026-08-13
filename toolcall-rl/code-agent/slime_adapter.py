"""Slime/SGLang adapter for the isolated text-only Code environment.

This module is intentionally separate from the document rollout entry point.
It uses only Code Agent modules and the generic Slime transport/types, so
loading it cannot pull document tools, state, checkpoints, or reward code into
a Code run.
"""

from __future__ import annotations

import inspect
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

try:
    from .env.client import CodeEnvClient
    from .rollout import generate_code_trajectory
    from .training.common import evaluator_patch_for_instance, evaluator_script_for_instance, load_instances
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from env.client import CodeEnvClient
    from rollout import generate_code_trajectory
    from training.common import evaluator_patch_for_instance, evaluator_script_for_instance, load_instances


class SGLangCodeModelClient:
    """Generate one Code action and retain only engine-issued token evidence."""

    def __init__(self, args: Any, sampling_params: Mapping[str, Any]) -> None:
        self.args = args
        self.state = GenerateState(args)
        self.sampling_params = dict(sampling_params)

    async def generate(self, messages: list[dict[str, str]], *, max_tokens: int, **_: Any) -> dict[str, Any]:
        tokenizer = self.state.tokenizer
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        params = dict(self.sampling_params)
        params["max_new_tokens"] = min(int(params.get("max_new_tokens", max_tokens)), int(max_tokens))
        response = await post(
            f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate",
            {"input_ids": prompt_ids, "sampling_params": params, "return_logprob": True},
        )
        metadata = response.get("meta_info") if isinstance(response, Mapping) else None
        token_logprobs = metadata.get("output_token_logprobs") if isinstance(metadata, Mapping) else None
        if not isinstance(token_logprobs, list):
            raise RuntimeError("SGLang Code generation omitted output token log-probabilities")
        try:
            logprobs = [float(item[0]) for item in token_logprobs]
            token_ids = [int(item[1]) for item in token_logprobs]
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("SGLang Code generation returned malformed token log-probabilities") from exc
        return {
            "text": str(response.get("text") or ""),
            "token_ids": token_ids,
            "token_mask": [1] * len(token_ids),
            "token_logprobs": logprobs,
        }


def _runtime_sample(sample: Sample) -> dict[str, Any]:
    metadata = dict(sample.metadata or {})
    public_instance = metadata.get("public_instance")
    if not isinstance(public_instance, Mapping):
        raise ValueError("Code Slime samples require metadata.public_instance from a Code SWE manifest")
    return {
        "text": str(sample.prompt),
        "environment": "code",
        "metadata": {"environment": "code", "public_instance": dict(public_instance)},
    }


@lru_cache(maxsize=2)
def _evaluator_catalog(manifest_path: str) -> dict[str, Any]:
    """Load private test material in the evaluator adapter only.

    Returned instances are never copied to the policy Sample, prompt, tool
    runtime, or trajectory metadata. The sole callers extract a command and
    test patch immediately before ``CleanEvaluator`` is invoked.
    """

    catalog = {instance.public.instance_id: instance for instance in load_instances(manifest_path)}
    if not catalog:
        raise ValueError("Code evaluator manifest is empty")
    return catalog


def _evaluator_for_instance(instance_id: str) -> tuple[str | None, str]:
    manifest_path = os.getenv("CODE_EVALUATOR_MANIFEST", "").strip()
    if not manifest_path:
        raise RuntimeError("CODE_EVALUATOR_MANIFEST is required for clean Code evaluation")
    instance = _evaluator_catalog(str(Path(manifest_path).resolve())).get(instance_id)
    if instance is None:
        raise KeyError(f"instance {instance_id!r} is absent from the evaluator manifest")
    return evaluator_script_for_instance(instance), evaluator_patch_for_instance(instance)


async def generate(args: Any, sample: Sample, sampling_params: Mapping[str, Any], evaluation: bool = False) -> Sample:
    """Slime custom-generation hook for one Code Stage-B trajectory.

    The hidden evaluator block is deliberately absent from ``Sample`` and the
    runtime row.  It is supplied by the pool-side evaluator configuration, not
    the policy callback.  This prevents private SWE patches/tests from entering
    Slime's prompt, rollout transport, or policy sample.
    """

    if getattr(args, "partial_rollout", False):
        raise ValueError("Code Agent does not support partial rollouts")
    if not isinstance(sample.metadata, Mapping):
        raise ValueError("Code Slime samples require mapping metadata")
    runtime_row = _runtime_sample(sample)
    instance_id = str(runtime_row["metadata"]["public_instance"].get("instance_id") or "")
    eval_script, evaluator_patch = _evaluator_for_instance(instance_id)
    if not eval_script:
        raise RuntimeError(f"Code evaluator manifest has no official test command for {instance_id!r}")
    model_client = SGLangCodeModelClient(args, sampling_params)
    async with CodeEnvClient() as environment:
        trajectory = await generate_code_trajectory(
            runtime_row,
            model_client=model_client,
            code_env_client=environment,
            seed=int(getattr(args, "rollout_seed", 0) or 0) + int(sample.index or 0),
            data_source=str(runtime_row["metadata"]["public_instance"].get("data_source") or ""),
            eval_script=eval_script,
            evaluator_patch=evaluator_patch,
        )
    if not trajectory["metadata"].get("policy_gradient_eligible", False):
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.metadata = {**dict(sample.metadata), "code_trajectory": trajectory["metadata"]}
        return sample
    sample.tokens = list(trajectory["tokens"])
    sample.loss_mask = list(trajectory["loss_mask"])
    sample.rollout_log_probs = list(trajectory["rollout_log_probs"])
    sample.response = "".join(message["content"] for message in trajectory["messages"])
    sample.response_length = len(sample.tokens)
    sample.reward = float(trajectory["reward"])
    sample.status = Sample.Status.COMPLETED
    # Do not attach trainer_only_metadata: it contains latent world diagnostics.
    sample.metadata = {**dict(sample.metadata), "code_trajectory": trajectory["metadata"]}
    return sample


async def reward_func(args: Any, sample: Sample, **_: Any) -> float:
    """Guard against accidental replacement of the environment-issued reward."""

    if sample.reward is None:
        raise RuntimeError("Code reward must be set by the clean Code evaluator during generation")
    return float(sample.reward)


__all__ = ["SGLangCodeModelClient", "generate", "reward_func"]
