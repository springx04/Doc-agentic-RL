import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = ROOT / "slime"
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

_CHECKPOINT_PATH = SLIME_ROOT / "slime" / "backends" / "fsdp_utils" / "checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("openclaw_fsdp_checkpoint", _CHECKPOINT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKPOINT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKPOINT)


def test_empty_optional_dcp_directory_is_not_treated_as_checkpoint(tmp_path):
    empty_dir = tmp_path / "optimizer"
    empty_dir.mkdir()

    complete_dir = tmp_path / "lr_scheduler"
    complete_dir.mkdir()
    (complete_dir / ".metadata").write_bytes(b"dcp metadata")

    assert not _CHECKPOINT._is_dcp_checkpoint(empty_dir)
    assert _CHECKPOINT._is_dcp_checkpoint(complete_dir)
