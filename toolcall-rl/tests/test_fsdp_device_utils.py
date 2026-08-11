import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = ROOT / "slime"
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

_DEVICE_UTILS_PATH = SLIME_ROOT / "slime" / "backends" / "fsdp_utils" / "device_utils.py"
_SPEC = importlib.util.spec_from_file_location("openclaw_fsdp_device_utils", _DEVICE_UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DEVICE_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEVICE_UTILS)
scatter_selected_values = _DEVICE_UTILS.scatter_selected_values


def test_scatter_selected_values_preserves_selected_values_and_gradients():
    positions = torch.tensor([1, 4], dtype=torch.long)
    values = torch.tensor([0.25, -0.75], dtype=torch.float64, requires_grad=True)

    result = scatter_selected_values(positions, values, packed_length=6)

    assert result.device == values.device
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([0.0, 0.25, 0.0, 0.0, -0.75, 0.0]))
    result.sum().backward()
    assert torch.equal(values.grad, torch.tensor([1.0, 1.0], dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the CPU-index/CUDA-value regression")
def test_scatter_selected_values_moves_cpu_indices_to_cuda_values():
    positions = torch.tensor([0, 3], dtype=torch.long)
    values = torch.tensor([1.5, -2.0], device="cuda", requires_grad=True)

    result = scatter_selected_values(positions, values, packed_length=5)

    assert result.device == values.device
    assert torch.equal(
        result.cpu(),
        torch.tensor([1.5, 0.0, 0.0, -2.0, 0.0]),
    )
    result.sum().backward()
    assert torch.equal(values.grad.cpu(), torch.tensor([1.0, 1.0]))
