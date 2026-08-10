import sys
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = ROOT / "slime"
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

_DATA_PACKING_PATH = SLIME_ROOT / "slime" / "backends" / "fsdp_utils" / "data_packing.py"
_SPEC = importlib.util.spec_from_file_location("openclaw_fsdp_data_packing", _DATA_PACKING_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DATA_PACKING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DATA_PACKING)
_get_multimodal_balanced_partitions = _DATA_PACKING._get_multimodal_balanced_partitions
pack_sequences = _DATA_PACKING.pack_sequences

_SEQ_BALANCING_PATH = SLIME_ROOT / "slime" / "utils" / "seqlen_balancing.py"
_SEQ_SPEC = importlib.util.spec_from_file_location("openclaw_seqlen_balancing", _SEQ_BALANCING_PATH)
assert _SEQ_SPEC is not None and _SEQ_SPEC.loader is not None
_SEQ_BALANCING = importlib.util.module_from_spec(_SEQ_SPEC)
_SEQ_SPEC.loader.exec_module(_SEQ_BALANCING)
build_fsdp_modality_aligned_order = _SEQ_BALANCING.build_fsdp_modality_aligned_order
build_fsdp_modality_aligned_order_for_batches = _SEQ_BALANCING.build_fsdp_modality_aligned_order_for_batches
get_fsdp_modality_aligned_partitions = _SEQ_BALANCING.get_fsdp_modality_aligned_partitions


def test_mixed_modalities_are_partitioned_into_homogeneous_packs():
    partitions = _get_multimodal_balanced_partitions(
        [10, 9, 8, 7],
        [{"pixel_values": object()}, None, {"pixel_values": object()}, None],
        2,
    )

    assert len(partitions) == 2
    assert all(partitions)
    is_visual = lambda index: index in (0, 2)
    assert all(is_visual(index) for index in partitions[0])
    assert all(not is_visual(index) for index in partitions[1])


def test_all_visual_inputs_keep_the_default_balanced_schedule():
    partitions = _get_multimodal_balanced_partitions(
        [10, 9, 8, 7],
        [{"pixel_values": object()}] * 4,
        2,
    )

    assert len(partitions) == 2
    assert sorted(index for part in partitions for index in part) == [0, 1, 2, 3]


def test_pack_sequences_keeps_text_only_samples_in_mixed_batch():
    packed = pack_sequences(
        tokens=[[1, 2, 3], [4, 5]],
        loss_masks=[[1, 1, 1], [1, 1]],
        rewards=[1.0, -1.0],
        raw_rewards=[1.0, -1.0],
        response_lengths=[2, 1],
        advantages=[[0.1, 0.1, 0.0], [-0.1, 0.0]],
        returns=[[0.1, 0.1, 0.0], [-0.1, 0.0]],
        multimodal_train_inputs=[
            {"pixel_values": torch.ones((1, 2), dtype=torch.float32)},
            None,
        ],
        num_packs=2,
    )

    assert len(packed) == 2
    assert sum(int(batch["tokens"].numel()) for batch in packed) == 5
    assert any("multimodal_train_inputs" in batch for batch in packed)


def test_pack_sequences_carries_question_loss_weights():
    packed = pack_sequences(
        tokens=[[1, 2], [3, 4]],
        loss_masks=[[1, 1], [1, 1]],
        rewards=[1.0, -1.0],
        raw_rewards=[1.0, -1.0],
        response_lengths=[1, 1],
        advantages=[[0.1, 0.0], [-0.1, 0.0]],
        returns=[[0.1, 0.0], [-0.1, 0.0]],
        num_packs=1,
        bayes_loss_weights=[0.75, 0.25],
    )
    assert packed[0]["bayes_loss_weights"] == [0.75, 0.25]


def test_mixed_modalities_are_aligned_across_fsdp_ranks_without_dropping_real_samples():
    flags = [True, False, True, False, False, False]
    order = build_fsdp_modality_aligned_order(flags, dp_size=4, global_batch_size=8)

    assert len(order) == 8
    assert sum(index >= 0 and flags[index] for index in order) == 2
    assert order.count(-1) == 2
    assert order.count(-2) == 0

    partitions = get_fsdp_modality_aligned_partitions(len(order), dp_size=4, global_batch_size=8)
    assert [len(partition) for partition in partitions] == [2, 2, 2, 2]
    ordered_modalities = [index == -1 or (index >= 0 and flags[index]) for index in order]
    rank_modalities = [
        [ordered_modalities[index] for index in partition]
        for partition in partitions
    ]
    assert rank_modalities == [[True, False]] * 4


def test_modality_alignment_keeps_all_real_samples_when_multiple_batches_are_needed():
    flags = [True, False, False, False, False, False, False, False, False, False]
    order = build_fsdp_modality_aligned_order(flags, dp_size=4, global_batch_size=8)

    assert len(order) == 16
    assert sum(index >= 0 for index in order) == len(flags)
    assert order.count(-1) == 3
    assert order.count(-2) == 3


def test_modality_alignment_does_not_cross_question_batch_boundaries():
    flags = [True, False, False, True, False, False, False, False]
    order, sizes = build_fsdp_modality_aligned_order_for_batches(flags, [4, 4], dp_size=4)
    assert sizes == [8, 4]
    assert len(order) == sum(sizes)
    # The first four records are aligned independently; the second question
    # starts only after its own padded first batch.
    assert order[:4] == [0, 3, -1, -1]
    assert order[4:8] == [1, 2, -2, -2]
    assert order[8:] == [4, 5, 6, 7]
