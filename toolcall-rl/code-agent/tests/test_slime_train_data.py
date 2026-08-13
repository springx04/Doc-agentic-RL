from types import SimpleNamespace

from slime_train_data import convert_samples_to_train_data


def _sample(role, variant, reward=1.0):
    return SimpleNamespace(
        tokens=[1, 2],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-1.0, -1.0],
        group_index=0,
        index=int(variant),
        status="completed",
        remove_sample=False,
        metadata={
            "environment": "code", "instance_id": "task", "coupling_id": "coupling",
            "latent_world_id": f"world-{role}", "world_slot_role": role,
            "decision_group_id": f"group-{role}", "decision_group_size": 4,
            "decision_prefix_hash": "prefix", "repo_state_digest": "repo",
            "variant_id": str(variant), "valid_for_rl": True,
        },
        get_reward_value=lambda args: reward,
    )


def test_code_converter_accepts_four_complete_world_groups():
    samples = [_sample(role, variant, reward=float(variant)) for role in ("healthy", "local_degradation", "shared_family_fault", "change") for variant in range(4)]
    result = convert_samples_to_train_data(SimpleNamespace(), samples)
    assert len(result["tokens"]) == 16
    assert result["bayes_grouping_report"]["question_count"] == 1
    assert set(result["bayes_group_ids"]) == {"group-healthy", "group-local_degradation", "group-shared_family_fault", "group-change"}
    assert sum(result["bayes_loss_weights"]) == 1.0
    assert result["bayes_advantages"][:4] == [-1.5, -0.5, 0.5, 1.5]
    assert result["advantages"] == result["bayes_advantages"]
    assert result["returns"][:4] == [0.0, 1.0, 2.0, 3.0]


def test_code_converter_rejects_entire_question_for_incomplete_sibling_group():
    samples = [_sample(role, variant) for role in ("healthy", "local_degradation", "shared_family_fault", "change") for variant in range(4)]
    samples.pop()
    result = convert_samples_to_train_data(SimpleNamespace(), samples)
    assert result["tokens"] == []
    assert result["advantages"] == []
    assert result["bayes_grouping_report"]["no_ready_questions"] is True
