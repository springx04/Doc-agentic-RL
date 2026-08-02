from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "slime"))

from slime.ray.reward_statistics import normalize_group_rewards
from slime_plugins.rollout_buffer.buffer import default_get_group_data_meta_info


def test_infrastructure_rewards_are_excluded_from_group_mean_and_std():
    rewards = normalize_group_rewards(
        [1.0, -1.0, 100.0, -100.0],
        [7, 7, 7, 7],
        [False, False, True, True],
        std_normalization=True,
    )
    valid_only = normalize_group_rewards([1.0, -1.0], [7, 7], [False, False], std_normalization=True)
    assert rewards[:2] == valid_only
    assert rewards[2:] == [0.0, 0.0]


def test_rollout_buffer_group_metadata_excludes_infrastructure_samples():
    info = default_get_group_data_meta_info(
        {
            "question-1": [
                {"reward": 1.0, "metadata": {"valid_for_rl": True}},
                {"reward": -1.0, "metadata": {"valid_for_rl": True}},
                {
                    "reward": 0.0,
                    "metadata": {"valid_for_rl": False, "exclude_from_group_statistics": True, "rollout_status": "infra_error"},
                },
                {
                    "reward": 0.0,
                    "metadata": {"valid_for_rl": False, "exclude_from_group_statistics": True, "rollout_status": "generation_empty"},
                },
            ]
        }
    )
    assert info["total_samples"] == 2
    assert info["num_groups"] == 1
    assert info["avg_reward"] == 0.0
