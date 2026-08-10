import asyncio
import copy
import inspect
import json
import logging
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor:
        processor_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    output = await post(url, payload)

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        _apply_bayestool_meta_suffix_returns(args, samples)
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


def _apply_bayestool_meta_suffix_returns(args: Namespace, samples: list[Sample]) -> None:
    """Replace each meta-question reward with its discounted suffix return."""

    if not samples or not any(
        isinstance(sample.metadata, dict) and sample.metadata.get("meta_episode_id")
        for sample in samples
    ):
        return
    try:
        from bayestool.training import suffix_meta_returns
    except ImportError:
        return

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        episode_id = metadata.get("meta_episode_id")
        if episode_id:
            grouped.setdefault(str(episode_id), []).append(sample)
    discount = float(getattr(args, "bayestool_meta_discount", 0.95) or 0.95)
    for episode_samples in grouped.values():
        # A real shared-prefix branch can expand one meta question into a
        # primary sample plus several branch children.  Collapse those rows
        # to one utility per question before applying the cross-question
        # suffix return; every branch of that question must receive the same
        # suffix target.
        by_question: dict[int, list[Sample]] = {}
        for sample in episode_samples:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            try:
                question_index = int(metadata.get("meta_question_index", 0) or 0)
            except (TypeError, ValueError):
                question_index = 0
            by_question.setdefault(question_index, []).append(sample)

        question_rows: list[tuple[int, Sample]] = []
        for question_index, question_samples in by_question.items():
            primary = next(
                (
                    item
                    for item in question_samples
                    if not bool((item.metadata or {}).get("bayestool_branch_child"))
                ),
                question_samples[0],
            )
            question_rows.append((question_index, primary))
        question_rows.sort(key=lambda item: item[0])

        utilities: list[float] = []
        for _, sample in question_rows:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            bayes_utility = metadata.get("bayestool_utility", {})
            if isinstance(bayes_utility, dict) and bayes_utility.get("utility") is not None:
                utilities.append(float(bayes_utility["utility"]))
            else:
                try:
                    utilities.append(float(sample.get_reward_value(args)))
                except (TypeError, KeyError, ValueError):
                    utilities.append(0.0)
        suffixes = suffix_meta_returns(utilities, discount=discount)
        for (question_index, _), utility, suffix in zip(question_rows, utilities, suffixes, strict=True):
            for sample in by_question[question_index]:
                metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
                metadata["meta_utility"] = float(utility)
                metadata["meta_suffix_return"] = float(suffix)
                if getattr(args, "advantage_estimator", "") != "bayes_grpo":
                    continue
                if isinstance(sample.reward, dict):
                    reward = dict(sample.reward)
                    reward_key = args.reward_key or "score"
                    reward[reward_key] = float(suffix)
                    reward["meta_utility"] = float(utility)
                    sample.reward = reward
                else:
                    sample.reward = float(suffix)


def _bayestool_aux_state(sample: Sample) -> dict[str, Any] | None:
    """Expose the policy-side state needed to build a sibling pair.

    The state is derived from the decision snapshot captured before a tool
    result.  Hidden world labels and clean outputs are never copied into the
    tokenized prompt; they remain in the ordinary supervision metadata.
    """

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    nested = metadata.get("bayestool") if isinstance(metadata.get("bayestool"), dict) else {}
    if not nested.get("enabled"):
        return None
    state = metadata.get("bayes_aux_state")
    if not isinstance(state, dict):
        state = {}
    state = copy.deepcopy(state)
    state.setdefault("coupling_id", metadata.get("coupling_id") or nested.get("coupling_id"))
    state.setdefault("content_signature", metadata.get("bayes_content_signature") or nested.get("content_signature"))
    state.setdefault("ood_score", nested.get("ood_score", 1.0))
    state.setdefault("belief_snapshot", nested.get("belief_snapshot", {}))
    state.setdefault("world_id", metadata.get("world_id") or nested.get("world_id"))
    state.setdefault("latent_world_id", metadata.get("latent_world_id") or nested.get("latent_world_id"))
    state.setdefault("replica_id", metadata.get("replica_id", nested.get("replica_id", 0)))
    state.setdefault("prompt", metadata.get("bayes_aux_prompt") or str(sample.prompt))
    state.setdefault("sibling_group_id", metadata.get("sibling_group_id") or nested.get("sibling_group_id"))
    if not state.get("best_action"):
        history = nested.get("decision_history", [])
        if isinstance(history, list):
            for item in reversed(history):
                if isinstance(item, dict) and item.get("bayes_action"):
                    state["best_action"] = item["bayes_action"]
                    state["best_action_text"] = item.get("best_action_text", "")
                    state["best_action_margin"] = item.get("best_action_margin", 0.0)
                    state["candidate_actions"] = item.get("candidate_action_keys", [])
                    state["candidate_action_texts"] = {
                        str(candidate.get("key")): str(candidate.get("text") or "")
                        for candidate in item.get("candidate_actions", [])
                        if isinstance(candidate, dict) and candidate.get("key")
                    }
                    state["observed_prefix_js"] = item.get("observed_prefix_js", 1.0)
                    state["first_distinguishing_event_step"] = item.get("observed_prefix_event_count")
                    break
    return state


def _bayestool_aux_states(sample: Sample) -> list[dict[str, Any]]:
    """Return one auxiliary state per policy decision prefix.

    Pairing only the final decision loses the common prefix before two
    coupled worlds first diverge.  The rollout records each decision's public
    state; reconstructing these records here lets switch/pre-invariance
    bundles use the exact matching prefix instead of a post-divergence suffix.
    """

    base = _bayestool_aux_state(sample)
    if base is None:
        return []
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    nested = metadata.get("bayestool") if isinstance(metadata.get("bayestool"), dict) else {}
    history = metadata.get("bayes_decisions") or nested.get("decision_history") or []
    if not isinstance(history, list):
        return [base]
    states: list[dict[str, Any]] = []
    for decision in history:
        if not isinstance(decision, dict):
            continue
        state = copy.deepcopy(base)
        state.update(
            {
                "best_action": decision.get("bayes_action") or state.get("best_action"),
                "best_action_text": decision.get("best_action_text", state.get("best_action_text", "")),
                "best_action_margin": decision.get("best_action_margin", state.get("best_action_margin", 0.0)),
                "candidate_actions": decision.get("candidate_action_keys", state.get("candidate_actions", [])),
                "candidate_action_texts": {
                    str(item.get("key")): str(item.get("text") or "")
                    for item in decision.get("candidate_actions", [])
                    if isinstance(item, dict) and item.get("key")
                },
                "belief_snapshot": decision.get("belief_snapshot", state.get("belief_snapshot", {})),
                "observed_prefix_js": decision.get("observed_prefix_js", state.get("observed_prefix_js", 1.0)),
                "first_distinguishing_event_step": decision.get(
                    "observed_prefix_event_count", state.get("first_distinguishing_event_step")
                ),
                "prefix_observation_signature": decision.get(
                    "prefix_observation_signature", state.get("prefix_observation_signature", "")
                ),
                "content_signature": decision.get("content_signature", state.get("content_signature", "")),
                "prompt": decision.get("prompt", state.get("prompt", str(sample.prompt))),
            }
        )
        states.append(state)
    return states or [base]


def _attach_bayestool_auxiliary_records(args: Namespace, samples: list[Sample], tokenizer: Any) -> None:
    """Construct real tokenized switch/pre-invariance bundles from one rollout group."""

    if getattr(args, "advantage_estimator", "") != "bayes_grpo":
        return
    try:
        from bayestool.config import config_from_args
        from bayestool.training import (
            build_preinv_bundle,
            build_switch_bundle,
            build_switch_pair,
        )
    except ImportError:
        return
    config = config_from_args(args, enabled=True)
    states: list[tuple[Sample, dict[str, Any]]] = []
    for sample in samples:
        sample_states = _bayestool_aux_states(sample)
        if sample_states:
            if not isinstance(sample.metadata, dict):
                sample.metadata = {}
            sample.metadata.setdefault("bayes_aux_records", [])
            states.extend((sample, state) for state in sample_states)
    if len(states) < 2:
        return

    seen_by_sample: dict[int, set[str]] = {id(sample): set() for sample, _ in states}
    for left_index, (left_sample, left_state) in enumerate(states):
        for right_sample, right_state in states[left_index + 1 :]:
            # Replicas of the same world are useful for the sibling baseline,
            # but they are not a support-valid belief-switch pair.
            if left_sample is right_sample or (
                left_state.get("latent_world_id")
                and left_state.get("latent_world_id") == right_state.get("latent_world_id")
            ):
                continue
            switch_pair = build_switch_pair(left_state, right_state, config=config)
            if switch_pair is not None:
                switch_bundle = build_switch_bundle(
                    switch_pair,
                    prompt_u=str(left_state.get("prompt") or left_sample.prompt),
                    prompt_v=str(right_state.get("prompt") or right_sample.prompt),
                    tokenizer=tokenizer,
                )
                switch_bundle["loss_weight"] = float(config.auxiliary.switch_loss_weight)
                bundle_id = str(switch_bundle.get("switch_bundle_id"))
                switch_bundle["content_signature"] = switch_pair.content_signature
                switch_bundle["world_ids"] = [left_state.get("world_id"), right_state.get("world_id")]
                switch_bundle["latent_world_ids"] = [left_state.get("latent_world_id"), right_state.get("latent_world_id")]
                for sample in (left_sample, right_sample):
                    if bundle_id not in seen_by_sample[id(sample)]:
                        sample.metadata["bayes_aux_records"].append(copy.deepcopy(switch_bundle))
                        seen_by_sample[id(sample)].add(bundle_id)

            left_candidates = list(left_state.get("candidate_actions", []))
            right_candidates = list(right_state.get("candidate_actions", []))
            if left_candidates and left_candidates == right_candidates:
                left_texts = left_state.get("candidate_action_texts", {})
                right_texts = right_state.get("candidate_action_texts", {})
                actions = [
                    str(left_texts.get(str(key)) or right_texts.get(str(key)) or key)
                    for key in left_candidates
                ]
                preinv_bundle = build_preinv_bundle(
                    left_state,
                    right_state,
                    prompts=[str(left_state.get("prompt") or left_sample.prompt), str(right_state.get("prompt") or right_sample.prompt)],
                    actions=actions,
                    config=config,
                    tokenizer=tokenizer,
                )
                if preinv_bundle is not None:
                    bundle_id = str(preinv_bundle.get("preinv_bundle_id"))
                    preinv_bundle["world_ids"] = [left_state.get("world_id"), right_state.get("world_id")]
                    preinv_bundle["latent_world_ids"] = [left_state.get("latent_world_id"), right_state.get("latent_world_id")]
                    for sample in (left_sample, right_sample):
                        if bundle_id not in seen_by_sample[id(sample)]:
                            sample.metadata["bayes_aux_records"].append(copy.deepcopy(preinv_bundle))
                            seen_by_sample[id(sample)].add(bundle_id)


def _attach_bayestool_branch_records(args: Namespace, samples: list[Sample]) -> None:
    """Materialize finite branch utilities for Q-head replay and regret metrics."""

    if getattr(args, "advantage_estimator", "") != "bayes_grpo":
        return
    records_by_world: dict[tuple[str, str, str], list[tuple[Sample, dict[str, Any]]]] = {}
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        nested = metadata.get("bayestool") if isinstance(metadata.get("bayestool"), dict) else {}
        prefix_hash = metadata.get("bayestool_branch_prefix_hash") or nested.get("branch_prefix_hash")
        if not prefix_hash:
            continue
        world_id = str(metadata.get("world_id") or nested.get("world_id") or "")
        latent_world_id = str(metadata.get("latent_world_id") or nested.get("latent_world_id") or "")
        coupling_id = str(metadata.get("coupling_id") or nested.get("coupling_id") or "")
        action_key = metadata.get("bayestool_branch_action_key") or nested.get("branch_action_key")
        if not world_id or not coupling_id or not action_key:
            continue
        utility_meta = metadata.get("bayestool_utility", {})
        utility = utility_meta.get("utility") if isinstance(utility_meta, dict) else None
        if utility is None:
            continue
        record = {
            "coupling_id": coupling_id,
            "sibling_group_id": str(
                metadata.get("bayestool_branch_sibling_group_id")
                or metadata.get("sibling_group_id")
                or nested.get("sibling_group_id")
                or ""
            ),
            "prefix_hash": str(prefix_hash),
            "world_id": world_id,
            "latent_world_id": latent_world_id,
            "belief_version": int(
                (nested.get("belief_snapshot") or {}).get("version", 0)
                if isinstance(nested.get("belief_snapshot"), dict)
                else 0
            ),
            "action_key": str(action_key),
            "utility": float(utility),
            "horizon": int(nested.get("branch_horizon", 0) or metadata.get("bayestool_branch_horizon", 0) or 0),
            "is_policy_parent": not bool(metadata.get("bayestool_branch_child")),
        }
        q_features = metadata.get("bayestool_branch_q_features") or nested.get("branch_q_features")
        if isinstance(q_features, dict):
            for key in ("task_features", "particle_features", "action_features", "budget_features"):
                if isinstance(q_features.get(key), (list, tuple)):
                    record[key] = [float(value) for value in q_features[key]]
        records_by_world.setdefault((str(prefix_hash), world_id, latent_world_id), []).append((sample, record))

    for world_records in records_by_world.values():
        oracle = max(world_records, key=lambda item: (item[1]["utility"], item[1]["action_key"]))[1]
        policy_records = [record for _, record in world_records if record.get("is_policy_parent")]
        policy = policy_records[0] if policy_records else world_records[0][1]
        for sample, record in world_records:
            record["oracle_action"] = oracle["action_key"]
            record["oracle_utility"] = float(oracle["utility"])
            record["policy_utility"] = float(policy["utility"])
            record["relative_regret"] = float(oracle["utility"] - policy["utility"])
            record.pop("is_policy_parent", None)
            sample.metadata.setdefault("bayes_branch_records", []).append(record)


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    results = await asyncio.gather(*tasks)
    # A custom generator may expand one input into a meta-episode or a real
    # shared-prefix branch group.  Keep the rollout group flat so reward
    # calculation and the trainer see Sample objects, never nested lists.
    group: list[Sample] = []
    for result in results:
        if isinstance(result, list):
            group.extend(result)
        else:
            group.append(result)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    if not state.aborted and getattr(args, "advantage_estimator", "") == "bayes_grpo":
        _attach_bayestool_branch_records(args, group)
        _attach_bayestool_auxiliary_records(args, group, state.tokenizer)
        _apply_bayestool_meta_suffix_returns(args, group)

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    # In the explicit BayesTool plan, n_samples_per_prompt is the number of
    # primary realization trajectories (R); each primary is completed to K
    # records by shared-prefix continuations.  Keep the progress indicator
    # aligned with the actual expanded output while preserving the legacy
    # worlds x replicas sampler's old total.
    nominal_expansion = 1
    if (
        bool(getattr(args, "bayestool_enable", False))
        and getattr(args, "advantage_estimator", "") == "bayes_grpo"
        and int(getattr(args, "n_samples_per_prompt", 0) or 0)
        == int(getattr(args, "bayestool_worlds_per_prompt", 0) or 0)
    ):
        nominal_expansion = int(getattr(args, "bayestool_default_group_size", 4) or 4)
    pbar = tqdm(
        total=target_data_size * args.n_samples_per_prompt * nominal_expansion,
        desc="Rollout generation",
    )
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            for group in samples:
                for sample in group:
                    if not isinstance(sample.metadata, dict):
                        sample.metadata = {}
                    # The rollout id is allocated by the coordinator and is
                    # stable across retries/resume; it must not depend on
                    # local sample ordering or Ray worker rank.
                    sample.metadata["rollout_id"] = int(rollout_id)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            if not group:
                state.remaining_batch_size -= 1
                logger.warning("A rollout group expanded to zero samples; dropping it.")
                continue
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(len(group))

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            if not isinstance(sample.metadata, dict):
                sample.metadata = {}
            sample.metadata["rollout_id"] = int(rollout_id)
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        samples = sample if isinstance(sample, list) else [sample]
        if do_print:
            example = samples[0] if samples else None
            if example is None:
                logger.info("eval_rollout_single_dataset returned no samples")
                do_print = False
                pbar.update(1)
                continue
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(example.prompt) + example.response]} "
                f"reward={example.reward}"
            )
            do_print = False
        data.extend(samples)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
