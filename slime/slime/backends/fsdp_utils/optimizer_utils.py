"""Optimizer construction helpers for the FSDP backend."""

import torch


def build_fsdp_adamw(
    params,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
):
    """Build AdamW without CUDA foreach/fused temporary buffers.

    FSDP2 shards parameters, gradients, and optimizer state, but CUDA AdamW's
    foreach implementation still materializes temporary tensors for a large
    parameter list.  On a nearly full 80 GB rank that temporary allocation can
    fail at the first optimizer step even when forward/backward fit.  The
    single-tensor path keeps the same AdamW update semantics without that
    batch-sized peak.
    """

    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
