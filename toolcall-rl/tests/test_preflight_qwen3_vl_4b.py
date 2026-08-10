import argparse
import json
from pathlib import Path
import sys


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from preflight_qwen3_vl_4b import (  # noqa: E402
    _jsonl_report,
    _path_report,
    build_report,
)


def _path_args(tmp_path, **overrides):
    values = {
        "train_data": None,
        "eval_data": None,
        "expected_train_rows": None,
        "expected_eval_rows": None,
        "model_path": None,
        "output_dir": tmp_path / "outputs" / "run",
        "allow_existing_output": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_jsonl_report_rejects_invalid_rows(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"prompt":"valid"}\n{"question":"also valid"}\n{"label":"missing prompt"}\n',
        encoding="utf-8",
    )

    report = _jsonl_report(dataset, label="train")

    assert report["rows"] == 3
    assert report["errors"] == ["line_3:missing_prompt"]


def test_path_report_checks_expected_rows_and_model(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"valid"}\n', encoding="utf-8")
    args = _path_args(
        tmp_path,
        train_data=dataset,
        expected_train_rows=2,
        model_path=tmp_path / "missing-model",
    )

    report, errors = _path_report(args)

    assert report["train"]["rows"] == 1
    assert "train_row_count=1 expected=2" in errors
    assert "model_path_missing" in errors


def test_path_report_refuses_nonempty_output_without_override(tmp_path):
    output = tmp_path / "outputs" / "existing"
    output.mkdir(parents=True)
    (output / "sentinel").write_text("keep", encoding="utf-8")

    report, errors = _path_report(_path_args(tmp_path, output_dir=output))

    assert report["output_exists"] is True
    assert report["output_entries"] == 1
    assert errors == ["output_dir_not_empty"]


def test_path_report_allows_explicit_existing_output_override(tmp_path):
    output = tmp_path / "outputs" / "existing"
    output.mkdir(parents=True)
    (output / "sentinel").write_text("keep", encoding="utf-8")

    _, errors = _path_report(
        _path_args(tmp_path, output_dir=output, allow_existing_output=True)
    )

    assert errors == []


def test_build_report_is_offline_and_does_not_start_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "preflight_qwen3_vl_4b._module_report",
        lambda: {"required": {"torch": True}, "optional": {}, "python": "3.11"},
    )
    monkeypatch.setattr(
        "preflight_qwen3_vl_4b._gpu_report",
        lambda: {"available": True, "count": 1, "devices": ["gpu0"]},
    )

    report = build_report(
        argparse.Namespace(
            mode="smoke",
            train_data=None,
            eval_data=None,
            expected_train_rows=None,
            expected_eval_rows=None,
            model_path=None,
            output_dir=tmp_path / "outputs" / "run",
            min_gpus=1,
            allow_existing_output=False,
        )
    )

    assert report["ok"] is True
    assert report["offline"] is True
    assert report["no_download"] is True
    assert report["no_process_start"] is True
