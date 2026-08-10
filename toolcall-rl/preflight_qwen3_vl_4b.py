#!/usr/bin/env python3
"""Read-only preflight checks for OpenClaw-RL with Qwen3-VL-4B.

The script deliberately uses only the Python standard library. It does not
create directories, install packages, download models, initialize CUDA, or
write reports. Results are emitted as JSON on stdout and failures use exit
code 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from email.parser import Parser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


GIB = 1024 ** 3
EXPECTED_MODEL_TYPES = {"qwen3_vl"}
REQUIRED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "transformers",
    "ray",
    "sglang",
    "sglang-router",
    "qwen-vl-utils",
    "slime",
    "jinja2",
    "pillow",
    "pymupdf",
    "docling",
    "rapidocr",
    "onnxruntime",
    "paddleocr",
    "paddlepaddle",
    "easyocr",
    "pdfplumber",
    "camelot-py",
    "python-docx",
    "python-pptx",
)


@dataclass
class CheckResult:
    name: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["ok"] = self.ok
        value["status"] = "fail" if self.errors else ("warning" if self.warnings else "pass")
        return value


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _gib(value: int) -> float:
    return round(value / GIB, 2)


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def _nearest_existing(path: Path) -> Optional[Path]:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def _directory_size(path: Path) -> Tuple[int, int, List[str]]:
    total = 0
    files = 0
    errors: List[str] = []
    for root, dirnames, filenames in os.walk(str(path), followlinks=False):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            item = Path(root) / filename
            try:
                if item.is_file():
                    total += item.stat().st_size
                    files += 1
            except OSError as exc:
                if len(errors) < 20:
                    errors.append("{}: {}".format(item, exc))
    return total, files, errors


def check_model(model_path: Path) -> CheckResult:
    result = CheckResult("model")
    result.details["path"] = str(model_path)
    if not model_path.is_dir():
        result.errors.append("model directory does not exist or is not readable")
        return result

    required_files = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    missing_required = [name for name in required_files if not (model_path / name).is_file()]
    result.details["required_files"] = {name: (model_path / name).is_file() for name in required_files}
    if missing_required:
        result.errors.append("missing model metadata: {}".format(", ".join(missing_required)))

    config: Dict[str, Any] = {}
    try:
        config = _load_json(model_path / "config.json")
    except (OSError, ValueError) as exc:
        result.errors.append("cannot parse config.json: {}".format(exc))

    model_type = str(config.get("model_type") or "")
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [architectures]
    result.details["model_type"] = model_type
    result.details["architectures"] = [str(item) for item in architectures]
    if model_type not in EXPECTED_MODEL_TYPES:
        result.errors.append("expected model_type qwen3_vl, found {!r}".format(model_type or None))
    if architectures and not any("Qwen3VL" in str(item) for item in architectures):
        result.errors.append("config architectures do not identify a Qwen3-VL model")

    tokenizer_config: Dict[str, Any] = {}
    tokenizer_path = model_path / "tokenizer_config.json"
    if tokenizer_path.is_file():
        try:
            tokenizer_config = _load_json(tokenizer_path)
        except (OSError, ValueError) as exc:
            result.errors.append("cannot parse tokenizer_config.json: {}".format(exc))
    has_chat_template = bool(tokenizer_config.get("chat_template")) or (
        model_path / "chat_template.jinja"
    ).is_file()
    result.details["chat_template"] = has_chat_template
    if not has_chat_template:
        result.errors.append("no chat template was found; tool-call rollout requires one")

    index_candidates = (
        model_path / "model.safetensors.index.json",
        model_path / "pytorch_model.bin.index.json",
    )
    index_path = next((path for path in index_candidates if path.is_file()), None)
    weight_files: List[Path] = []
    if index_path is not None:
        try:
            index = _load_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                result.errors.append("{} has no non-empty weight_map".format(index_path.name))
            else:
                weight_files = sorted(
                    {model_path / str(filename) for filename in weight_map.values()},
                    key=lambda item: str(item),
                )
        except (OSError, ValueError) as exc:
            result.errors.append("cannot parse {}: {}".format(index_path.name, exc))
    else:
        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = sorted(model_path.glob("pytorch_model*.bin"))

    missing_shards = [str(path) for path in weight_files if not path.is_file()]
    result.details["weight_index"] = str(index_path) if index_path else None
    result.details["weight_file_count"] = len(weight_files)
    result.details["missing_weight_files"] = missing_shards[:20]
    if not weight_files:
        result.errors.append("no model weight files or weight index were found")
    if missing_shards:
        result.errors.append("{} model weight shard(s) are missing".format(len(missing_shards)))

    total_size, file_count, stat_errors = _directory_size(model_path)
    result.details["directory_file_count"] = file_count
    result.details["directory_size_gib"] = _gib(total_size)
    if stat_errors:
        result.warnings.append(
            "could not stat {} model file(s); first errors: {}".format(
                len(stat_errors), stat_errors[:3]
            )
        )
    return result


def _parse_nvidia_rows(stdout: str, fields: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_row in csv.reader(stdout.splitlines()):
        if not raw_row:
            continue
        values = [item.strip() for item in raw_row]
        if len(values) != len(fields):
            continue
        row: Dict[str, Any] = dict(zip(fields, values))
        for memory_key in ("memory.total", "memory.free"):
            try:
                row[memory_key + "_gib"] = round(float(row[memory_key]) / 1024.0, 2)
            except (KeyError, TypeError, ValueError):
                row[memory_key + "_gib"] = None
        rows.append(row)
    return rows


def check_gpus(min_gpus: int, min_gpu_memory_gib: float) -> CheckResult:
    result = CheckResult("gpus")
    result.details["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    fields = (
        "index",
        "name",
        "memory.total",
        "memory.free",
        "compute_cap",
        "driver_version",
    )
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        result.errors.append("nvidia-smi query failed: {}".format(exc))
        return result

    if completed.returncode != 0:
        result.errors.append(
            "nvidia-smi returned {}: {}".format(
                completed.returncode, completed.stderr.strip()[-1000:]
            )
        )
        return result

    rows = _parse_nvidia_rows(completed.stdout, fields)
    result.details["count"] = len(rows)
    result.details["devices"] = rows
    if len(rows) < min_gpus:
        result.errors.append(
            "found {} GPU(s), but at least {} were requested".format(len(rows), min_gpus)
        )
    low_memory = [
        row
        for row in rows
        if isinstance(row.get("memory.total_gib"), float)
        and row["memory.total_gib"] < min_gpu_memory_gib
    ]
    if low_memory:
        result.warnings.append(
            "{} GPU(s) have less than {:.1f} GiB total memory; choose GPU count and "
            "batch settings from the reported hardware before training".format(
                len(low_memory), min_gpu_memory_gib
            )
        )
    busy = [
        row
        for row in rows
        if isinstance(row.get("memory.free_gib"), float)
        and isinstance(row.get("memory.total_gib"), float)
        and row["memory.free_gib"] < 0.8 * row["memory.total_gib"]
    ]
    if busy:
        result.warnings.append(
            "{} GPU(s) currently have more than 20 percent memory in use".format(len(busy))
        )
    return result


def check_disk(workspace: Path, model_path: Path, min_free_gib: float) -> CheckResult:
    result = CheckResult("disk")
    result.details["minimum_workspace_free_gib"] = min_free_gib
    if not workspace.is_dir():
        result.errors.append("workspace does not exist: {}".format(workspace))
        return result

    try:
        usage = shutil.disk_usage(str(workspace))
        result.details["workspace"] = {
            "path": str(workspace),
            "total_gib": _gib(usage.total),
            "used_gib": _gib(usage.used),
            "free_gib": _gib(usage.free),
        }
        if usage.free < min_free_gib * GIB:
            result.errors.append(
                "workspace has {:.2f} GiB free, below the requested {:.2f} GiB".format(
                    usage.free / GIB, min_free_gib
                )
            )
    except OSError as exc:
        result.errors.append("cannot inspect workspace disk: {}".format(exc))

    if model_path.exists():
        try:
            model_usage = shutil.disk_usage(str(model_path))
            result.details["model_mount"] = {
                "path": str(model_path),
                "total_gib": _gib(model_usage.total),
                "used_gib": _gib(model_usage.used),
                "free_gib": _gib(model_usage.free),
            }
        except OSError as exc:
            result.warnings.append("cannot inspect model mount disk: {}".format(exc))
    return result


def _read_distributions(env_path: Path) -> Tuple[Dict[str, str], List[str]]:
    distributions: Dict[str, str] = {}
    errors: List[str] = []
    site_packages = sorted((env_path / "lib").glob("python*/site-packages"))
    for site_path in site_packages:
        for metadata_path in sorted(site_path.glob("*.dist-info/METADATA")):
            try:
                with metadata_path.open("r", encoding="utf-8", errors="replace") as handle:
                    metadata = Parser().parsestr(handle.read())
                name = metadata.get("Name")
                version = metadata.get("Version")
                if name:
                    distributions[_normalise_distribution(name)] = str(version or "")
            except OSError as exc:
                if len(errors) < 20:
                    errors.append("{}: {}".format(metadata_path, exc))
    return distributions, errors


def check_target_environment(env_path: Path, workspace: Path, mode: str) -> CheckResult:
    result = CheckResult("target_environment")
    result.details["path"] = str(env_path)
    result.details["mode"] = mode
    if not _path_within(env_path, workspace):
        result.errors.append("target environment must stay within {}".format(workspace))
        return result

    if mode == "setup":
        result.details["exists"] = env_path.exists()
        if env_path.exists():
            result.errors.append(
                "target environment already exists; setup must not overwrite or merge into it"
            )
        return result

    if not env_path.is_dir():
        result.errors.append("target environment does not exist")
        return result
    python_path = env_path / "bin" / "python"
    if not python_path.is_file():
        python_path = env_path / "bin" / "python3"
    result.details["python"] = str(python_path)
    if not python_path.is_file():
        result.errors.append("target environment has no bin/python or bin/python3")

    distributions, metadata_errors = _read_distributions(env_path)
    result.details["distributions"] = {
        name: distributions.get(_normalise_distribution(name))
        for name in REQUIRED_DISTRIBUTIONS
    }
    missing = [
        name
        for name in REQUIRED_DISTRIBUTIONS
        if _normalise_distribution(name) not in distributions
    ]
    if missing:
        result.errors.append("missing required distributions: {}".format(", ".join(missing)))
    if metadata_errors:
        result.warnings.append(
            "could not read {} distribution metadata file(s)".format(len(metadata_errors))
        )

    cudnn_version = distributions.get("nvidia-cudnn-cu12")
    result.details["nvidia-cudnn-cu12"] = cudnn_version
    if cudnn_version is None:
        result.errors.append("nvidia-cudnn-cu12 is not installed")
    elif _version_tuple(cudnn_version) < _version_tuple("9.16.0.29"):
        result.errors.append(
            "nvidia-cudnn-cu12 {} is older than required 9.16.0.29".format(cudnn_version)
        )
    return result


def check_sglang_source(workspace: Path, env_path: Path, mode: str) -> CheckResult:
    result = CheckResult("sglang_source")
    candidates = [
        Path("/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"),
        workspace
        / ".openclaw-build"
        / "sglang"
        / "python"
        / "sglang"
        / "srt"
        / "entrypoints"
        / "http_server.py",
    ]
    for site_path in sorted((env_path / "lib").glob("python*/site-packages")):
        candidates.append(site_path / "sglang" / "srt" / "entrypoints" / "http_server.py")

    inspected: List[Dict[str, Any]] = []
    patched = False
    for candidate in candidates:
        if not candidate.is_file():
            inspected.append({"path": str(candidate), "exists": False})
            continue
        marker = False
        qwen3_vl_path = candidate.parents[1] / "models" / "qwen3_vl.py"
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
            marker = "post_process_weights" in text
        except OSError as exc:
            result.warnings.append("cannot inspect {}: {}".format(candidate, exc))
        inspected.append(
            {
                "path": str(candidate),
                "exists": True,
                "rl_weight_update_marker": marker,
                "qwen3_vl_model_file": qwen3_vl_path.is_file(),
            }
        )
        patched = patched or marker
    result.details["candidates"] = inspected
    result.details["rl_weight_update_marker_found"] = patched
    if mode == "train" and not patched:
        result.errors.append(
            "no SGLang installation with the RL weight-update endpoint was found"
        )
    elif not patched:
        result.warnings.append(
            "no local patched SGLang source was found; environment setup must resolve this "
            "without modifying source outside the workspace"
        )
    return result


def check_dataset(
    path: Path,
    role: str,
    mode: str,
    expected_rows: Optional[int] = None,
) -> CheckResult:
    result = CheckResult("{}_dataset".format(role))
    result.details["path"] = str(path)
    if not path.is_file():
        message = "{} dataset is missing".format(role)
        if mode == "train":
            result.errors.append(message)
        else:
            result.warnings.append(message)
        return result

    rows = 0
    malformed = 0
    missing_documents: List[str] = []
    missing_document_fields = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    item = json.loads(line)
                except ValueError:
                    malformed += 1
                    if malformed <= 20:
                        result.errors.append("{}:{} is not valid JSON".format(path, line_number))
                    continue
                if not isinstance(item, dict):
                    malformed += 1
                    if malformed <= 20:
                        result.errors.append("{}:{} is not an object".format(path, line_number))
                    continue
                if "prompt" not in item or "label" not in item:
                    malformed += 1
                    if malformed <= 20:
                        result.errors.append(
                            "{}:{} must contain prompt and label".format(path, line_number)
                        )
                metadata = item.get("metadata")
                document_value = metadata.get("document_path") if isinstance(metadata, dict) else None
                if not document_value:
                    missing_document_fields += 1
                    continue
                document_path = Path(str(document_value)).expanduser()
                if not document_path.is_absolute():
                    document_path = path.parent / document_path
                if not document_path.is_file() and len(missing_documents) < 50:
                    missing_documents.append(str(document_path))
    except OSError as exc:
        result.errors.append("cannot read {} dataset: {}".format(role, exc))
        return result

    result.details["rows"] = rows
    result.details["malformed_rows"] = malformed
    result.details["rows_without_metadata_document_path"] = missing_document_fields
    result.details["missing_documents"] = missing_documents
    if rows == 0:
        result.errors.append("{} dataset has no records".format(role))
    if expected_rows is not None and rows != expected_rows:
        result.errors.append(
            "{} dataset has {} row(s), expected {}".format(role, rows, expected_rows)
        )
    if missing_document_fields:
        result.errors.append(
            "{} row(s) have no metadata.document_path".format(missing_document_fields)
        )
    if missing_documents:
        result.errors.append(
            "at least {} referenced document(s) are missing".format(len(missing_documents))
        )
    return result


def check_output_path(
    output_dir: Path,
    workspace: Path,
    mode: str,
    allow_existing: bool = False,
) -> CheckResult:
    result = CheckResult("output")
    result.details["path"] = str(output_dir)
    result.details["exists"] = output_dir.exists()
    if not _path_within(output_dir, workspace):
        result.errors.append("output directory must stay within {}".format(workspace))
        return result
    if mode == "train" and output_dir.exists() and not allow_existing:
        result.errors.append(
            "output directory already exists; refusing a path that could overwrite a prior run"
        )
    parent = _nearest_existing(output_dir.parent)
    result.details["nearest_existing_parent"] = str(parent) if parent else None
    if parent is None:
        result.errors.append("no existing parent is available for the output directory")
    elif not os.access(str(parent), os.W_OK):
        result.errors.append("output parent is not writable: {}".format(parent))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    workspace = Path("/workspace/data")
    project = workspace / "OpenClaw-RL"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("setup", "train", "eval", "smoke"), default="setup"
    )
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        type=Path,
        default=Path("/models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument(
        "--target-env",
        type=Path,
        default=workspace / "envs" / "openclaw-rl-qwen3vl",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=project / "data" / "document-qa" / "train.jsonl",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=project / "data" / "document-qa" / "eval.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "outputs" / "qwen3-vl-4b-rl-smoke-PENDING",
    )
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=40.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=80.0)
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument("--expected-eval-rows", type=int)
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args(argv)


def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    workspace = args.workspace.resolve(strict=False)
    checks = [
        check_model(args.model),
        check_gpus(args.min_gpus, args.min_gpu_memory_gib),
        check_disk(workspace, args.model, args.min_free_disk_gib),
        check_target_environment(args.target_env, workspace, args.mode),
        check_sglang_source(workspace, args.target_env, args.mode),
        check_dataset(
            args.train_data,
            "train",
            args.mode,
            expected_rows=args.expected_train_rows,
        ),
        check_dataset(
            args.eval_data,
            "eval",
            args.mode,
            expected_rows=args.expected_eval_rows,
        ),
        check_output_path(
            args.output_dir,
            workspace,
            args.mode,
            allow_existing=args.allow_existing_output,
        ),
    ]
    return {
        "ok": all(check.ok for check in checks),
        "mode": args.mode,
        "read_only": True,
        "checks": [check.to_dict() for check in checks],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_preflight(args)
    except Exception as exc:
        report = {
            "ok": False,
            "mode": getattr(args, "mode", None),
            "read_only": True,
            "internal_error": "{}: {}".format(type(exc).__name__, exc),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
