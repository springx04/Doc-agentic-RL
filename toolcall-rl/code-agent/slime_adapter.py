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
from typing import Any, Mapping

try:  # Keep Code-only unit tests independent of optional Slime dependencies.
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample
except ImportError:  # pragma: no cover - production launcher requires Slime
    GenerateState = None  # type: ignore[assignment]
    post = None  # type: ignore[assignment]

    class Sample:  # type: ignore[no-redef]
        pass

try:
    from .env.client import CodeEnvClient
    from .rollout import capture_initial_branch_checkpoint, generate_code_trajectory
    from .bayestool.branching import close_sibling_leases, create_sibling_leases
    from .bayestool.grouping import decision_group_id
    from .bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from .config import DEFAULT_CODE_CONFIG, stable_hash
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from env.client import CodeEnvClient
    from rollout import capture_initial_branch_checkpoint, generate_code_trajectory
    from bayestool.branching import close_sibling_leases, create_sibling_leases
    from bayestool.grouping import decision_group_id
    from bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from config import DEFAULT_CODE_CONFIG, stable_hash


class SGLangCodeModelClient:
    """Generate one Code action and retain only engine-issued token evidence."""

    def __init__(self, args: Any, sampling_params: Mapping[str, Any]) -> None:
        if GenerateState is None:
            raise ImportError("Code Slime adapter requires the Slime rollout dependencies")
        self.args = args
        self.state = GenerateState(args)
        self.sampling_params = dict(sampling_params)
        self._initial_prompt_ids: list[int] | None = None
        self._latest_full_ids: list[int] | None = None
        self._action_spans: list[tuple[int, int, list[float]]] = []

    async def generate(self, messages: list[dict[str, str]], *, max_tokens: int, **_: Any) -> dict[str, Any]:
        tokenizer = self.state.tokenizer
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(token) for token in prompt_ids]
        if self._initial_prompt_ids is None:
            self._initial_prompt_ids = list(prompt_ids)
        elif prompt_ids[: len(self._initial_prompt_ids)] != self._initial_prompt_ids:
            raise RuntimeError("Code multi-turn prompt no longer preserves its initial token prefix")
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
        response_ids = [int(item) for item in token_ids]
        response_logprobs = [float(item) for item in logprobs]
        action_start = len(prompt_ids) - len(self._initial_prompt_ids)
        self._action_spans.append((action_start, len(response_ids), response_logprobs))
        self._latest_full_ids = prompt_ids + response_ids
        return {
            "text": str(response.get("text") or ""),
            "token_ids": response_ids,
            "token_mask": [1] * len(response_ids),
            "token_logprobs": response_logprobs,
        }

    def training_sequence(self) -> tuple[list[int], list[int], list[float]]:
        """Return full tokens plus a trainable suffix with observations off.

        The trainer must replay the same observation-conditioned context used
        during generation.  Only action spans receive loss weight one; all
        system/user/tool-observation framing tokens stay as context.
        """

        if self._initial_prompt_ids is None or self._latest_full_ids is None:
            raise RuntimeError("Code generation produced no tokenized trajectory")
        prefix = len(self._initial_prompt_ids)
        # FSDP packs the full input but expects response_length/loss_mask and
        # rollout logprobs to describe only its trailing response segment.
        # Retaining the initial task prompt here is essential: otherwise the
        # actor recomputes later action probabilities without the condition
        # that the rollout server used.
        tokens = list(self._latest_full_ids)
        suffix_length = len(tokens) - prefix
        mask = [0] * suffix_length
        logprobs = [0.0] * suffix_length
        for start, length, action_logprobs in self._action_spans:
            end = start + length
            if start < 0 or end > suffix_length or len(action_logprobs) != length:
                raise RuntimeError("Code action token span is inconsistent with the final multi-turn sequence")
            mask[start:end] = [1] * length
            logprobs[start:end] = action_logprobs
        if not any(mask):
            raise RuntimeError("Code trajectory has no trainable action tokens")
        return tokens, mask, logprobs


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


async def generate(args: Any, sample: Sample, sampling_params: Mapping[str, Any], evaluation: bool = False) -> Sample | list[Sample]:
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
    if _stage_c_enabled(args, sample):
        if evaluation:
            raise ValueError("Code Stage C branching is a training-only path; use the final checkpoint evaluator")
        return await generate_stage_c(args, sample, sampling_params)
    runtime_row = _runtime_sample(sample)
    instance_id = str(runtime_row["metadata"]["public_instance"].get("instance_id") or "")
    model_client = SGLangCodeModelClient(args, sampling_params)
    async with CodeEnvClient() as environment:
        trajectory = await generate_code_trajectory(
            runtime_row,
            model_client=model_client,
            code_env_client=environment,
            seed=int(getattr(args, "rollout_seed", 0) or 0) + int(sample.index or 0),
            data_source=str(runtime_row["metadata"]["public_instance"].get("data_source") or ""),
        )
    tokens, loss_mask, rollout_log_probs = model_client.training_sequence()
    if not trajectory["metadata"].get("policy_gradient_eligible", False):
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.metadata = {**dict(sample.metadata), "code_trajectory": trajectory["metadata"]}
        return sample
    sample.tokens = tokens
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.response = "".join(message["content"] for message in trajectory["messages"])
    sample.response_length = len(sample.loss_mask)
    sample.reward = float(trajectory["reward"])
    sample.status = Sample.Status.COMPLETED
    # Do not attach trainer_only_metadata: it contains latent world diagnostics.
    sample.metadata = {**dict(sample.metadata), "code_trajectory": trajectory["metadata"]}
    return sample


def _stage_c_enabled(args: Any, sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    return bool(metadata.get("code_stage_c") or os.getenv("CODE_STAGE", "").upper() == "C")


def _stage_c_group_size(args: Any) -> int:
    value = int(os.getenv("CODE_STAGE_C_GROUP_SIZE", "4"))
    if value not in {4, 8}:
        raise ValueError("Code Stage C sibling group size must be 4 or 8")
    return value


def _context_for_stage_c(public_instance: Mapping[str, Any]) -> Any:
    # The actual repository context is discovered inside each clean SWE lease.
    # No private verifier content participates in world sampling.
    return public_sampling_context(
        tool_budget=DEFAULT_CODE_CONFIG.tool_budget,
        repo_file_count=int(public_instance.get("repo_file_count", 0) or 0),
        tracked_extensions=tuple(public_instance.get("tracked_extensions") or ()),
        top_level_dirs=tuple(public_instance.get("top_level_dirs") or ()),
        languages=tuple(public_instance.get("languages") or ()),
        detected_frameworks=tuple(public_instance.get("detected_frameworks") or ()),
    )


async def generate_stage_c(args: Any, sample: Sample, sampling_params: Mapping[str, Any]) -> list[Sample]:
    """Expand one task into four worlds, each with a K-sibling decision group."""

    runtime_row = _runtime_sample(sample)
    public_instance = runtime_row["metadata"]["public_instance"]
    instance_id = str(public_instance.get("instance_id") or "")
    image_name = str(public_instance.get("image_name") or "")
    base_revision = public_instance.get("base_revision")
    group_size = _stage_c_group_size(args)
    seed = int(getattr(args, "rollout_seed", 0) or 0) + int(sample.index or 0)
    worlds = sample_required_worlds(
        instance_id=instance_id,
        image_name=image_name,
        base_revision=base_revision,
        context=_context_for_stage_c(public_instance),
        rollout_seed=seed,
    )
    output: list[Sample] = []
    async with CodeEnvClient() as environment:
        for world_index, world in enumerate(worlds):
            parent = await capture_initial_branch_checkpoint(
                runtime_row, code_env_client=environment, world=world
            )
            siblings = await create_sibling_leases(
                environment,
                parent,
                image_name=image_name,
                instance_id=instance_id,
                count=group_size,
                cwd="/testbed",
            )
            group_id = decision_group_id(
                instance_id=instance_id,
                latent_world_id=world.latent_world_id,
                decision_event_id="root",
                decision_prefix_hash_value=parent.decision_prefix_hash,
                repo_state_digest=parent.repo.repo_state_digest,
                belief_state_digest=stable_hash(parent.belief_state, prefix="code-belief-state-v1"),
                world_runtime_digest=parent.runtime_state_digest,
                coupling_id=world.coupling_id,
                world_slot_role=world.world_slot_role,
            )
            try:
                for sibling_index, sibling in enumerate(siblings):
                    child = Sample(
                        group_index=sample.group_index,
                        index=(int(sample.index or 0) * 100) + (world_index * group_size) + sibling_index,
                        prompt=sample.prompt,
                        label=sample.label,
                        metadata=dict(sample.metadata or {}),
                    )
                    model_client = SGLangCodeModelClient(args, sampling_params)
                    trajectory = await generate_code_trajectory(
                        runtime_row,
                        model_client=model_client,
                        code_env_client=environment,
                        world=world,
                        interaction_lease=sibling.lease,
                        branch_checkpoint=sibling.checkpoint,
                        seed=seed,
                        data_source=str(public_instance.get("data_source") or ""),
                    )
                    metadata = trajectory["metadata"]
                    child.tokens, child.loss_mask, child.rollout_log_probs = model_client.training_sequence()
                    child.response = "".join(message["content"] for message in trajectory["messages"])
                    child.response_length = len(child.loss_mask)
                    child.reward = float(trajectory["reward"])
                    child.status = Sample.Status.COMPLETED
                    child.remove_sample = not bool(metadata.get("policy_gradient_eligible", False))
                    child.metadata = {
                        **dict(sample.metadata or {}),
                        "environment": "code",
                        "instance_id": instance_id,
                        "valid_for_rl": bool(metadata.get("valid_for_rl", False)),
                        "coupling_id": world.coupling_id,
                        "latent_world_id": world.latent_world_id,
                        "world_slot_role": world.world_slot_role,
                        "decision_group_id": group_id,
                        "decision_group_size": group_size,
                        "selected_decision_event": "root",
                        "decision_prefix_hash": parent.decision_prefix_hash,
                        "repo_state_digest": parent.repo.repo_state_digest,
                        "initial_input_hash": parent.repo.repo_state_digest,
                        "runtime_state_digest": parent.runtime_state_digest,
                        "variant_id": sibling.variant_id,
                        "policy_version": "code-policy-v1",
                        "code_trajectory": metadata,
                    }
                    output.append(child)
            finally:
                await close_sibling_leases(environment, siblings)
    if len(output) != 4 * group_size:
        raise RuntimeError("Code Stage C did not produce four complete sibling groups")
    return output


async def reward_func(args: Any, sample: Sample, **_: Any) -> float:
    """Guard against accidental replacement of the environment-issued reward."""

    if sample.reward is None:
        raise RuntimeError("Code reward must be set by the clean Code evaluator during generation")
    return float(sample.reward)


__all__ = ["SGLangCodeModelClient", "generate", "reward_func"]
