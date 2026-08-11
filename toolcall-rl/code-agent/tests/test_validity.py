from bayestool.validity import fail_closed_advantage, validate_group_records


def _records(k=4, **extra):
    return [{"instance_id": "i", "latent_world_id": "w", "repo_state_digest": "r", "decision_prefix_hash": "p", "variant_id": str(index), **extra} for index in range(k)]


def test_incomplete_or_real_infra_group_fails_closed():
    assert not validate_group_records(_records(3), expected_k=4).valid_for_rl
    rows = _records(4, failure_origin="real_infrastructure")
    assert not validate_group_records(rows, expected_k=4).valid_for_rl
    assert fail_closed_advantage([1, 2, 3, 4], rows, expected_k=4) == (0.0, 0.0, 0.0, 0.0)
