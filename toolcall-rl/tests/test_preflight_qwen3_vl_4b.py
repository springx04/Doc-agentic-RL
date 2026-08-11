import json
from pathlib import Path
import sys


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from preflight_qwen3_vl_4b import (
    check_dataset,
    check_model,
    check_output_path,
    check_target_environment,
)


def _write_complete_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "test template"}),
        encoding="utf-8",
    )
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    return model


def test_complete_qwen3_vl_model_passes(tmp_path):
    result = check_model(_write_complete_model(tmp_path))
    assert result.ok
    assert result.details["model_type"] == "qwen3_vl"
    assert result.details["weight_file_count"] == 1


def test_missing_model_shard_fails(tmp_path):
    model = _write_complete_model(tmp_path)
    (model / "model-00001-of-00001.safetensors").unlink()
    result = check_model(model)
    assert not result.ok
    assert any("shard" in message for message in result.errors)


def test_document_dataset_checks_referenced_files(tmp_path):
    document = tmp_path / "fixture.md"
    document.write_text("# Revenue\n\n2025 revenue was $1.2 million.", encoding="utf-8")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "prompt": "Inspect the document.",
                "label": json.dumps({"answers": ["$1.2 million"], "metric": "exact_match"}),
                "metadata": {"document_path": str(document)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = check_dataset(dataset, "train", "train")
    assert result.ok
    assert result.details["rows"] == 1


def test_setup_refuses_to_merge_existing_environment(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "envs" / "openclaw-rl-qwen3vl"
    target.mkdir(parents=True)
    result = check_target_environment(target, workspace, "setup")
    assert not result.ok
    assert any("must not overwrite" in message for message in result.errors)


def test_train_refuses_existing_output_directory(tmp_path):
    workspace = tmp_path / "workspace"
    output = workspace / "outputs" / "existing"
    output.mkdir(parents=True)
    result = check_output_path(output, workspace, "train")
    assert not result.ok
    assert any("already exists" in message for message in result.errors)
