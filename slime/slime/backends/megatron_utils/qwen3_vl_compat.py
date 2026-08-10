"""Runtime compatibility for the official Qwen3-VL Megatron provider.

Megatron's packed ``thd`` layout stores several logical samples in one
physical ``input_ids`` row and describes their boundaries in
``PackedSeqParams.cu_seqlens_q``.  The Qwen3-VL bridge implementation computes
multimodal RoPE indices from a dense ``[batch, sequence]`` layout and therefore
must be called once per packed segment before the resulting positions are
concatenated back to the ``thd`` layout.

This adapter keeps the official provider, vision tower, and MRoPE algorithm;
it only supplies the layout conversion at the boundary between Slime's packed
batcher and the provider's dense RoPE helper.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)


def _vision_counts(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[int, int]:
    """Count image/video entries represented by vision-start markers."""

    flat_ids = input_ids.reshape(-1)
    starts = torch.nonzero(flat_ids == vision_start_token_id, as_tuple=False).flatten()
    if starts.numel() == 0:
        return 0, 0
    starts = starts[starts + 1 < flat_ids.numel()]
    if starts.numel() == 0:
        return 0, 0
    next_ids = flat_ids[starts + 1]
    return int((next_ids == image_token_id).sum().item()), int((next_ids == video_token_id).sum().item())


def _slice_grid(grid: Any, offset: int, count: int, name: str) -> Any:
    """Take the vision metadata belonging to one packed segment."""

    if grid is None:
        if count:
            raise RuntimeError(f"Qwen3-VL packed MRoPE found {count} {name} tokens but no {name}_grid_thw")
        return None
    available = int(grid.shape[0])
    if offset + count > available:
        raise RuntimeError(
            f"Qwen3-VL packed MRoPE metadata underflow for {name}: "
            f"need [{offset}:{offset + count}], available={available}"
        )
    return grid[offset : offset + count]


def install_qwen3_vl_packed_mrope_compat(provider: Any = None) -> bool:
    """Patch the official Qwen3-VL model alias once in the current process.

    Returns ``True`` when the Qwen3-VL model module was patched and ``False``
    for non-Qwen bridge providers or environments without that provider.
    """

    provider_name = type(provider).__name__ if provider is not None else ""
    provider_module = type(provider).__module__ if provider is not None else ""
    if provider is not None and "Qwen3VL" not in provider_name and "qwen3_vl" not in provider_module:
        return False

    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen3_vl_model
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import utils as qwen3_vl_utils
    except ImportError:
        return False

    original = getattr(qwen3_vl_model, "get_rope_index", None)
    if original is None:
        return False
    if getattr(original, "_openclaw_packed_mrope_compat", False):
        return True

    @functools.wraps(original)
    def packed_mrope_compat(*args: Any, **kwargs: Any):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) >= 5:
            input_ids = args[4]
        packed_seq_params = kwargs.get("packed_seq_params")
        attention_mask = kwargs.get("attention_mask")

        # Dense inference/training batches already match the bridge contract.
        # The conversion is only needed for Slime's packed THD representation.
        if (
            packed_seq_params is None
            or input_ids is None
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            return original(*args, **kwargs)

        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu_seqlens is None or cu_seqlens.numel() < 3:
            return original(*args, **kwargs)

        cu = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
        total_tokens = int(input_ids.shape[1])
        if not cu or cu[0] != 0 or cu[-1] != total_tokens:
            raise RuntimeError(
                "Qwen3-VL packed MRoPE received invalid cu_seqlens_q: "
                f"cu={cu}, input_tokens={total_tokens}"
            )

        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        image_token_id = int(kwargs.get("image_token_id", args[1] if len(args) > 1 else 151655))
        video_token_id = int(kwargs.get("video_token_id", args[2] if len(args) > 2 else 151656))
        vision_start_token_id = int(
            kwargs.get("vision_start_token_id", args[3] if len(args) > 3 else 151652)
        )

        position_chunks = []
        delta_chunks = []
        image_offset = 0
        video_offset = 0
        for start, end in zip(cu[:-1], cu[1:], strict=True):
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            segment_ids = input_ids[:, start:end]
            image_count, video_count = _vision_counts(
                segment_ids,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
            )
            segment_kwargs = dict(kwargs)
            segment_kwargs["packed_seq_params"] = None
            segment_args = list(args)
            if len(segment_args) >= 5:
                segment_args[4] = segment_ids
            else:
                segment_kwargs["input_ids"] = segment_ids
            if attention_mask is None:
                segment_kwargs["attention_mask"] = torch.ones_like(segment_ids)
            elif attention_mask.ndim >= 2 and attention_mask.shape[-1] >= end:
                segment_kwargs["attention_mask"] = attention_mask[..., start:end]
            segment_kwargs["image_grid_thw"] = _slice_grid(image_grid_thw, image_offset, image_count, "image")
            segment_kwargs["video_grid_thw"] = _slice_grid(video_grid_thw, video_offset, video_count, "video")
            image_offset += image_count
            video_offset += video_count
            position_ids, mrope_delta = original(*segment_args, **segment_kwargs)
            position_chunks.append(position_ids)
            delta_chunks.append(mrope_delta)

        if not position_chunks:
            return original(*args, **kwargs)
        if image_grid_thw is not None and image_offset != int(image_grid_thw.shape[0]):
            raise RuntimeError(
                f"Qwen3-VL packed MRoPE consumed {image_offset} image grids, "
                f"but received {int(image_grid_thw.shape[0])}"
            )
        if video_grid_thw is not None and video_offset != int(video_grid_thw.shape[0]):
            raise RuntimeError(
                f"Qwen3-VL packed MRoPE consumed {video_offset} video grids, "
                f"but received {int(video_grid_thw.shape[0])}"
            )
        return torch.cat(position_chunks, dim=-1), torch.cat(delta_chunks, dim=0)

    packed_mrope_compat._openclaw_packed_mrope_compat = True
    qwen3_vl_model.get_rope_index = packed_mrope_compat
    qwen3_vl_utils.get_rope_index = packed_mrope_compat
    logger.info("Installed Qwen3-VL packed THD MRoPE compatibility adapter")
    return True
