"""Data packing utilities for FSDP backend to reduce padding overhead."""

import math

import torch

from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions


def _get_multimodal_balanced_partitions(
    seq_lengths: list[int],
    multimodal_train_inputs: list[dict | None] | None,
    k_partitions: int,
) -> list[list[int]]:
    """Create modality-homogeneous packs when a batch mixes VLM/text inputs."""

    def _default_partitions() -> list[list[int]]:
        return get_seqlen_balanced_partitions(
            seq_lengths,
            k_partitions=k_partitions,
            equal_size=False,
        )

    if not multimodal_train_inputs:
        return _default_partitions()

    multimodal_indices = [
        index for index, value in enumerate(multimodal_train_inputs) if value
    ]
    text_indices = [
        index for index, value in enumerate(multimodal_train_inputs) if not value
    ]
    if not multimodal_indices or not text_indices:
        return _default_partitions()
    if k_partitions < 2:
        raise RuntimeError("mixed FSDP batches require at least two modality-homogeneous packs")

    lower_multimodal_packs = max(1, k_partitions - len(text_indices))
    upper_multimodal_packs = min(len(multimodal_indices), k_partitions - 1)
    if lower_multimodal_packs > upper_multimodal_packs:
        raise RuntimeError(
            "cannot create non-empty modality-homogeneous FSDP packs from the local batch"
        )
    ideal_multimodal_packs = round(
        k_partitions * len(multimodal_indices) / len(seq_lengths)
    )
    multimodal_pack_count = min(
        max(ideal_multimodal_packs, lower_multimodal_packs),
        upper_multimodal_packs,
    )
    text_pack_count = k_partitions - multimodal_pack_count

    def _partition_subset(indices: list[int], count: int) -> list[list[int]]:
        relative_parts = get_seqlen_balanced_partitions(
            [seq_lengths[index] for index in indices],
            k_partitions=count,
            equal_size=False,
        )
        return [[indices[index] for index in part] for part in relative_parts]

    return _partition_subset(multimodal_indices, multimodal_pack_count) + _partition_subset(
        text_indices,
        text_pack_count,
    )


def pack_sequences(
    tokens: list[list[int]],
    loss_masks: list[list[int]],
    rewards: list[float],
    raw_rewards: list,
    response_lengths: list[int],
    advantages: list[float],
    returns: list[float],
    rollout_log_probs: list[list[float]] | None = None,
    multimodal_train_inputs: list[dict] | None = None,
    max_tokens_per_gpu: int | None = None,
    num_packs: int | None = None,
) -> list[dict]:
    """
    Pack sequences into dense batches with cumulative sequence lengths.

    Args:
        tokens: List of token sequences
        loss_masks: List of loss masks
        rewards: List of rewards per sequence
        raw_rewards: List of raw rewards per sequence
        response_lengths: List of response lengths per sequence
        advantages: List of advantages per sequence
        returns: List of returns per sequence
        rollout_log_probs: List of rollout log probabilities per sequence
        multimodal_train_inputs: List of dict of multimodal tensors for training per sequence
        max_tokens_per_gpu: Maximum tokens per GPU pack
        num_packs: Explicit number of packs to create

    Returns:
        List of packed batches with tokens, masks, cu_seqlens, rewards, raw_rewards, response_lengths, advantages, returns
    """
    if not tokens:
        return []

    seq_lengths = [len(t) for t in tokens]

    # Determine number of packs and use balanced partitioning
    if num_packs:
        k_partitions = num_packs
    elif max_tokens_per_gpu:
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_gpu))
    else:
        k_partitions = 1

    # Use balanced partitioning for optimal load distribution
    partitions = _get_multimodal_balanced_partitions(
        seq_lengths,
        multimodal_train_inputs,
        k_partitions,
    )

    # Pack each partition
    result = []
    for indices in partitions:
        # Build cumulative sequence lengths
        cu_seqlens = [0]
        flat_tokens = []
        flat_masks = []
        flat_positionids = []
        flat_advantages = []
        flat_returns = []
        flat_rollout_log_probs = []

        for i in indices:
            seq_tokens = tokens[i]
            seq_mask = loss_masks[i]
            seq_positionids = list(range(len(seq_tokens)))

            flat_tokens.extend(seq_tokens)
            flat_positionids.extend(seq_positionids)
            flat_masks.extend(seq_mask)
            flat_advantages.extend(advantages[i])
            flat_returns.extend(returns[i])
            if rollout_log_probs:
                flat_rollout_log_probs.extend(rollout_log_probs[i])
            cu_seqlens.append(cu_seqlens[-1] + len(seq_tokens))

        packed_batch = {
            "tokens": torch.tensor(flat_tokens, dtype=torch.long),
            "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
            "position_ids": torch.tensor(flat_positionids, dtype=torch.int),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "rewards": torch.tensor([rewards[i] for i in indices], dtype=torch.float32),
            "raw_reward": [raw_rewards[i] for i in indices],
            "response_lengths": [response_lengths[i] for i in indices],
            "advantages": torch.tensor(flat_advantages, dtype=torch.float32),
            "returns": torch.tensor(flat_returns, dtype=torch.float32),
            "rollout_log_probs": torch.tensor(
                flat_rollout_log_probs, dtype=torch.float32, device=torch.cuda.current_device()
            ),
        }

        # Collect and add multimodal training tensors for this partition
        if multimodal_train_inputs:
            multimodal_data = {}  # key -> concatenated tensor
            multimodal_num_items = {}  # key -> list of item counts per sequence
            for sequence_offset, i in enumerate(indices):
                # Keep one per-sequence count even when a zero-loss padding
                # sample carries no visual payload.  This lets unpacking and
                # FSDP model inputs remain aligned without calling .items() on
                # None.
                mm_inputs = multimodal_train_inputs[i] or {}
                for item_counts in multimodal_num_items.values():
                    item_counts.append(0)
                for key, mm_tensor in mm_inputs.items():
                    if not isinstance(mm_tensor, torch.Tensor):
                        mm_tensor = torch.tensor(mm_tensor)
                    item_count = mm_tensor.size(0)
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [0] * sequence_offset + [item_count]
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                        multimodal_num_items[key][-1] = item_count
            if multimodal_data:
                packed_batch["multimodal_train_inputs"] = multimodal_data
                packed_batch["multimodal_num_items"] = multimodal_num_items

        result.append(packed_batch)

    return result


def unpack_sequences(packed_batch: dict) -> list[dict]:
    """
    Unpack sequences from a packed batch.

    Args:
        packed_batch: Packed batch

    Returns:
        List of unpacked batches
    """

    cu_seqlens = packed_batch["cu_seqlens"]
    num_sequences = len(cu_seqlens) - 1
    response_lengths = packed_batch["response_lengths"]
    multimodal_num_items = packed_batch.get("multimodal_num_items", {})

    instances = []

    # Calculate pad_length by counting trailing zeros
    tokens = packed_batch["tokens"]
    nonzero_indices = (tokens != 0).nonzero(as_tuple=True)[0]
    if len(nonzero_indices) > 0:
        # Last non-zero index, pad_length is everything after it
        pad_length = len(tokens) - nonzero_indices[-1].item() - 1
    else:
        pad_length = 0  # No padding if no non-zero tokens (or all zeros)
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        instance = {}

        # Copy any additional attributes that might exist in the packed batch
        for key, value in packed_batch.items():
            if key not in instance:
                # Skip multimodal_num_items - it's metadata
                if key == "multimodal_num_items":
                    continue
                # Handle multimodal_train_inputs dict: split each tensor using multimodal_num_items
                elif key == "multimodal_train_inputs" and isinstance(value, dict):
                    instance[key] = {}
                    for mm_key, mm_tensor in value.items():
                        if mm_key in multimodal_num_items:
                            num_items_list = multimodal_num_items[mm_key]
                            start_mm_idx = sum(num_items_list[:i])
                            end_mm_idx = start_mm_idx + num_items_list[i]
                            if num_items_list[i] > 0:
                                instance[key][mm_key] = mm_tensor[start_mm_idx:end_mm_idx]
                # For tensor attributes, we need to slice them appropriately
                elif isinstance(value, torch.Tensor):
                    if key in ["log_probs", "ref_log_probs", "cur_log_probs", "entropy"]:
                        # These are computed from logits[:-1] so they have length seq_len-1
                        instance[key] = value[
                            end_idx - 1 - response_lengths[i] - pad_length : end_idx - 1 - pad_length
                        ]
                    elif key == "rollout_log_probs":
                        # rollout_log_probs is packed based on response_lengths, so slice differently
                        instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
                    elif key in ["tokens", "position_ids"]:
                        # For other tensor attributes, try to slice them
                        if len(value) > start_idx:
                            instance[key] = value[start_idx:end_idx]
                        else:
                            raise ValueError(f"Attribute {key} is not found in the packed batch")
                    elif key in ["loss_masks", "advantages", "returns"]:
                        instance[key] = value[sum(response_lengths[:i]) : sum(response_lengths[: i + 1])]
                elif isinstance(value, list):
                    instance[key] = value[i]
                else:
                    raise ValueError(f"Attribute {key} is not found in the packed batch")

        instances.append(instance)

    return instances
