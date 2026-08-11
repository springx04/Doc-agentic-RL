from bayestool.grouping import coupling_id, decision_group_id, validate_decision_group


def _row(**updates):
    value = {"instance_id": "i", "latent_world_id": "w", "repo_state_digest": "r", "decision_prefix_hash": "p", "failure_origin": "none", "valid_for_rl": True, "variant_id": "0"}
    value.update(updates)
    return value


def test_grouping_requires_same_decision_state():
    rows = [_row(variant_id=str(index)) for index in range(4)]
    assert validate_decision_group(rows, expected_k=4).valid_for_rl
    assert not validate_decision_group([_row(variant_id="0"), _row(variant_id="1", repo_state_digest="other"), _row(variant_id="2"), _row(variant_id="3")], expected_k=4).valid_for_rl
    assert not validate_decision_group([_row(variant_id="0"), _row(variant_id="1", latent_world_id="other"), _row(variant_id="2"), _row(variant_id="3")], expected_k=4).valid_for_rl
    assert validate_decision_group([_row(variant_id=str(index)) for index in range(8)], expected_k=8).valid_for_rl


def test_group_id_and_coupling_are_code_scoped():
    assert coupling_id(instance_id="i", image_name="img", base_revision=None, rollout_seed=1)
    group = decision_group_id(instance_id="i", latent_world_id="w", decision_event_id="0", decision_prefix_hash_value="p", repo_state_digest="r", belief_state_digest="b", world_runtime_digest="d", coupling_id="c", world_slot_role="healthy")
    assert group
