import sys
import importlib.util
from pathlib import Path


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
