import sys
from pathlib import Path

import torch


SLIME_ROOT = Path(__file__).resolve().parents[2] / "slime"
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from slime.utils.action_training import (  # noqa: E402
    apply_action_training_overrides,
    action_reward_loss,
    build_action_training_overrides,
)


def test_rejected_action_reward_is_consumed_by_a_nonzero_gradient():
    signal = build_action_training_overrides(
        response_length=4,
        assistant_token_masks=[{"token_start": 1, "token_end": 3, "mask": 0}],
        action_rewards=[-0.5],
    )
    assert signal["action_reward_consumed"] is True
    assert signal["consumed_action_indices"] == [0]
    assert signal["action_token_mask"] == [0, 1, 1, 0]
    assert signal["action_advantages"] == [0.0, -0.5, -0.5, 0.0]

    log_prob_parameter = torch.nn.Parameter(torch.tensor(1.0))
    log_probs = log_prob_parameter * torch.ones(4)
    loss = action_reward_loss(log_probs, signal["action_advantages"], signal["action_token_mask"])
    loss.backward()
    gradient_norm = float(log_prob_parameter.grad.abs().item())
    assert gradient_norm > 0.0


def test_rejected_action_updates_the_effective_training_mask():
    result = apply_action_training_overrides(
        response_length=4,
        advantages=torch.zeros(4),
        returns=torch.zeros(4),
        loss_mask=torch.zeros(4, dtype=torch.int),
        assistant_token_masks=[{"token_start": 1, "token_end": 3, "mask": 0}],
        action_rewards=[-0.5],
    )

    assert result["action_reward_consumed"] is True
    assert result["consumed_action_indices"] == [0]
    assert result["applied_action_token_count"] == 2
    assert result["advantages"].tolist() == [0.0, -0.5, -0.5, 0.0]
    assert result["returns"].tolist() == [0.0, -0.5, -0.5, 0.0]
    assert result["loss_mask"].tolist() == [0, 1, 1, 0]
