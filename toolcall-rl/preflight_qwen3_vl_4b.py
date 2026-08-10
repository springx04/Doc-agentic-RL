#!/usr/bin/env python3
"""Offline preflight for Qwen3-VL DocVQA rollout/training runs.

The preflight is deliberately dependency-light and side-effect free.  It
checks the requested data/model/output paths, validates the JSONL contract,
reports visible GPU/runtime availability, and refuses to reuse a non-empty
output directory unless the caller explicitly allows it.  It never downloads,
installs, starts Ray/SGLang, or creates the output directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REQUIRED_MODULES = ("torch", "transformers", "ray", "sglang")
OPTIONAL_MODULES = ("PIL", "fitz", "rapidocr")


def _jsonl_report(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"label": label, "path": str(path), "rows": 0, "errors": ["missing_file"]}
    rows = 0
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line_{line_number}:invalid_json:{exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line_{line_number}:row_not_object")
                continue
            if not str(value.get("prompt") or value.get("question") or "").strip():
                errors.append(f"line_{line_number}:missing_prompt")
            rows += 1
    return {"label": label, "path": str(path), "rows": rows, "errors": errors[:20]}


def _module_report() -> dict[str, Any]:
    return {
        "required": {name: bool(importlib.util.find_spec(name)) for name in REQUIRED_MODULES},
        "optional": {name: bool(importlib.util.find_spec(name)) for name in OPTIONAL_MODULES},
        "python": sys.version.split()[0],
    }


def _gpu_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "count": 0, "error": "nvidia-smi_not_found"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {"available": bool(devices), "count": len(devices), "devices": devices}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "count": 0, "error": str(exc)}


def _path_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    report: dict[str, Any] = {}
    errors: list[str] = []
    for attr, label in (("train_data", "train"), ("eval_data", "eval")):
        value = getattr(args, attr)
        if value is None:
            continue
        item = _jsonl_report(Path(value).expanduser().resolve(), label=label)
        report[label] = item
        if item["errors"] or item["rows"] == 0:
            errors.append(f"{label}_data_invalid")
        expected = getattr(args, f"expected_{label}_rows")
        if expected is not None and item["rows"] != expected:
            errors.append(f"{label}_row_count={item['rows']} expected={expected}")

    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
        report["model_path"] = str(model_path)
        if not model_path.exists():
            errors.append("model_path_missing")

    output_dir = Path(args.output_dir).expanduser().resolve()
    report["output_dir"] = str(output_dir)
    if output_dir.exists():
        entries = list(output_dir.iterdir()) if output_dir.is_dir() else [output_dir]
        report["output_exists"] = True
        report["output_entries"] = len(entries)
        if entries and not args.allow_existing_output:
            errors.append("output_dir_not_empty")
    else:
        report["output_exists"] = False
        report["output_entries"] = 0
    return report, errors


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    path_report, path_errors = _path_report(args)
    modules = _module_report()
    gpu = _gpu_report()
    errors = list(path_errors)
    missing_required = [name for name, present in modules["required"].items() if not present]
    if missing_required:
        errors.append(f"missing_required_modules:{','.join(missing_required)}")
    if args.mode in {"train", "eval"} and args.min_gpus > int(gpu.get("count", 0)):
        errors.append(f"gpu_count={gpu.get('count', 0)} expected>={args.min_gpus}")
    return {
        "ok": not errors,
        "mode": args.mode,
        "offline": True,
        "no_download": True,
        "no_process_start": True,
        "errors": errors,
        "paths": path_report,
        "modules": modules,
        "gpu": gpu,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OPENCLAW_OCR_AUTO_BACKENDS": os.environ.get("OPENCLAW_OCR_AUTO_BACKENDS"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "smoke"), default="train")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument("--expected-eval-rows", type=int)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
