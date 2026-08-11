import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = ROOT / "slime"
_OPTIMIZER_PATH = SLIME_ROOT / "slime" / "backends" / "fsdp_utils" / "optimizer_utils.py"
_SPEC = importlib.util.spec_from_file_location("openclaw_fsdp_optimizer_utils", _OPTIMIZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_OPTIMIZER_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OPTIMIZER_UTILS)
build_fsdp_adamw = _OPTIMIZER_UTILS.build_fsdp_adamw


def test_fsdp_adamw_disables_batch_temporary_kernels_and_steps():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = build_fsdp_adamw(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False

    (parameter.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(parameter).all()
