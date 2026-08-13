from slime_adapter import SGLangCodeModelClient


def test_multiturn_training_sequence_keeps_observations_as_zero_loss_context():
    client = object.__new__(SGLangCodeModelClient)
    client._initial_prompt_ids = [10, 11]
    # The second prompt contains the first action and a tool observation;
    # only model-emitted action spans are trainable.
    client._latest_full_ids = [10, 11, 21, 22, 30, 31, 41]
    client._action_spans = [(0, 2, [-1.0, -1.1]), (4, 1, [-1.2])]
    tokens, mask, logprobs = client.training_sequence()
    assert tokens == [10, 11, 21, 22, 30, 31, 41]
    assert mask == [1, 1, 0, 0, 1]
    assert logprobs == [-1.0, -1.1, 0.0, 0.0, -1.2]


def test_stage_c_source_declares_four_worlds_and_k_sibling_records():
    import inspect
    import slime_adapter

    source = inspect.getsource(slime_adapter.generate_stage_c)
    assert "sample_required_worlds" in source
    assert "create_sibling_leases" in source
    assert "decision_group_id" in source
    assert "len(output) != 4 * group_size" in source
