"""Small tensor helpers used by the FSDP training path."""

import torch


def scatter_selected_values(
    positions: torch.Tensor,
    values: torch.Tensor,
    packed_length: int,
) -> torch.Tensor:
    """Scatter selected values using the forward result's device.

    FSDP CPU offload can leave parameter storage on CPU while the forward
    result is produced on CUDA.  The selected positions may therefore be on a
    different device from the values being scattered.
    """

    output_device = values.device
    output = torch.zeros(packed_length, dtype=torch.float32, device=output_device)
    output.index_copy_(
        0,
        positions.to(device=output_device),
        values.to(device=output_device, dtype=torch.float32),
    )
    return output
