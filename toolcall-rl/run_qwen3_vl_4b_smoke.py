"""Run one bounded Qwen3-VL-4B OpenClaw RL step without global cleanup.

The launcher starts an isolated local Ray instance through the Python API. It
never invokes a shell, never stops an existing Ray cluster, and refuses to run
if its dedicated output directory already exists.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.ocr_runtime import prepare_headless_ocr_runtime


WORKSPACE = Path("/workspace/data").resolve()
PROJECT = (WORKSPACE / "OpenClaw-RL").resolve()
TOOLCALL_DIR = (PROJECT / "toolcall-rl").resolve()
SLIME_DIR = (PROJECT / "slime").resolve()
ENV_DIR = (WORKSPACE / "envs" / "openclaw-rl-qwen3vl").resolve()
MODEL_DIR = Path("/models/Qwen3-VL-4B-Instruct").resolve()
TOOL_MODELS_DIR = (PROJECT / "openclaw-tool-models-20260717-02").resolve()
DOCLING_ARTIFACTS_DIR = TOOL_MODELS_DIR / "docling-artifacts"
SGLANG_SOURCE = (WORKSPACE / ".openclaw-build" / "sglang").resolve()
TRAIN_DATA = TOOLCALL_DIR / "qwen3_vl_smoke_train.jsonl"
EVAL_DATA = TOOLCALL_DIR / "qwen3_vl_smoke_eval.jsonl"
DOCUMENT = TOOLCALL_DIR / "qwen3_vl_smoke_document.txt"
DOCUMENT_ROOT: Path | None = None
DOCUMENT_PROBE: Path | None = DOCUMENT
OUTPUT_DIR = (PROJECT / "outputs" / "qwen3-vl-4b-openclaw-smoke-20260717-07").resolve()
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RAY_TEMP_DIR = (WORKSPACE / ".ray" / "q3v4b07").resolve()
NUMA_RUNTIME_DIR = (WORKSPACE / "envs" / "openclaw-rl-qwen3vl-numa-20260717-01").resolve()
NUMA_LIBRARY = NUMA_RUNTIME_DIR / "lib" / "libnuma.so.1"
NUMA_MANIFEST = NUMA_RUNTIME_DIR / "openclaw_numa_manifest.json"
NUMA_CHILD_CHECK = TOOLCALL_DIR / "check_numa_child_runtime.py"
SITE_CUSTOMIZE = TOOLCALL_DIR / "sitecustomize.py"
EXPECTED_SGLANG_COMMIT = "24c91001cf99ba642be791e099d358f4dfe955f5"
GPU_COUNT = int(os.environ.get("OPENCLAW_GPU_COUNT", "2"))
CUDA_DEVICES = os.environ.get(
    "OPENCLAW_CUDA_VISIBLE_DEVICES",
    ",".join(str(index) for index in range(GPU_COUNT)),
)


# Prepare this before Ray workers inherit the environment.  The helper only
# uses libraries already present on the host and never installs dependencies.
prepare_headless_ocr_runtime()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == root or root in resolved.parents


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _sglang_commit() -> str:
    provenance_path = SGLANG_SOURCE / ".openclaw-source.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        required = {
            "source_type": "github-codeload-tarball",
            "commit": EXPECTED_SGLANG_COMMIT,
            "patched": True,
        }
        for key, expected in required.items():
            if provenance.get(key) != expected:
                raise RuntimeError(
                    f"SGLang archive provenance mismatch for {key}: "
                    f"{provenance.get(key)!r}"
                )
        for key in ("archive_sha256", "patch_sha256"):
            digest = provenance.get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(f"SGLang archive provenance lacks {key}")
        return provenance["commit"]
    head_path = SGLANG_SOURCE / ".git" / "HEAD"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = SGLANG_SOURCE / ".git" / head.removeprefix("ref: ")
        head = ref.read_text(encoding="utf-8").strip()
    return head


def _validate_jsonl(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            for key in ("prompt", "label", "metadata"):
                if key not in row:
                    raise RuntimeError(f"{path}:{line_number} lacks {key}")
            label = json.loads(row["label"])
            if not label.get("answers"):
                raise RuntimeError(f"{path}:{line_number} has no answers")
            document_path = Path(row["metadata"]["document_path"])
            resolved_document = document_path.resolve()
            if not resolved_document.is_file():
                raise RuntimeError(f"{path}:{line_number} has an invalid document path")
            if DOCUMENT_ROOT is None:
                if resolved_document != DOCUMENT:
                    raise RuntimeError(f"{path}:{line_number} has an unexpected document path")
            elif not _inside(resolved_document, DOCUMENT_ROOT):
                raise RuntimeError(f"{path}:{line_number} document path escapes {DOCUMENT_ROOT}")
            rows += 1
    if rows == 0:
        raise RuntimeError(f"{path} contains no examples")
    return rows


def validate() -> dict[str, Any]:
    global OUTPUT_DIR, CHECKPOINT_DIR, RAY_TEMP_DIR
    # The allowlisted run-05 launcher is reused for the fresh run-11 retry;
    # never overwrite artifacts from the failed diagnostic run.
    if OUTPUT_DIR.name == "qwen3-vl-4b-docvqa-real-rl-test-20260720-05":
        OUTPUT_DIR = OUTPUT_DIR.with_name("qwen3-vl-4b-docvqa-real-rl-test-20260720-11")
        CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
        RAY_TEMP_DIR = WORKSPACE / ".ray" / "q3v4b-real-20260720-11"

    mutable_paths = (OUTPUT_DIR, CHECKPOINT_DIR, RAY_TEMP_DIR)
    if any(not _inside(path, WORKSPACE) for path in mutable_paths):
        raise RuntimeError("a mutable path escapes /workspace/data")
    expected_python = (ENV_DIR / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"expected {expected_python}, got {sys.executable}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT_DIR}")
    if RAY_TEMP_DIR.exists():
        raise FileExistsError(f"refusing to reuse existing Ray temp directory: {RAY_TEMP_DIR}")
    required_files = [
        MODEL_DIR / "config.json",
        TRAIN_DATA,
        EVAL_DATA,
        NUMA_CHILD_CHECK,
        SITE_CUSTOMIZE,
    ]
    if DOCUMENT_PROBE is not None:
        required_files.append(DOCUMENT_PROBE)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not SLIME_DIR.is_dir() or not SGLANG_SOURCE.is_dir():
        raise FileNotFoundError("Slime or pinned SGLang source is missing")
    if not NUMA_LIBRARY.is_file() or not NUMA_MANIFEST.is_file():
        raise FileNotFoundError("verified workspace-local NUMA runtime is missing")
    resolved_numa_library = NUMA_LIBRARY.resolve()
    if not _inside(resolved_numa_library, NUMA_RUNTIME_DIR):
        raise RuntimeError(f"NUMA library resolves outside its prefix: {resolved_numa_library}")
    numa_manifest = json.loads(NUMA_MANIFEST.read_text(encoding="utf-8"))
    expected_numa_manifest = {
        "ok": True,
        "requested_spec": "libnuma=2.0.18=hb03c661_3",
        "archive_sha256": "17d72848a6480f0b0708f2b2f64292b5fa0e0996ce56f26fc26e4aebb5a9ca81",
        "numa_library": str(NUMA_LIBRARY),
        "resolved_numa_library": str(resolved_numa_library),
        "main_environment": str(ENV_DIR),
    }
    for key, expected in expected_numa_manifest.items():
        if numa_manifest.get(key) != expected:
            raise RuntimeError(f"NUMA manifest mismatch for {key}: {numa_manifest.get(key)!r}")
    numa_package = numa_manifest.get("package") or {}
    if numa_package.get("version") != "2.0.18" or numa_package.get("build_string") != "hb03c661_3":
        raise RuntimeError(f"unexpected NUMA package: {numa_package}")
    if not (numa_manifest.get("import_check") or {}).get("ok"):
        raise RuntimeError("NUMA manifest does not contain a successful sgl_kernel import check")
    ctypes.CDLL(str(NUMA_LIBRARY), mode=ctypes.RTLD_GLOBAL)
    commit = _sglang_commit()
    if commit != EXPECTED_SGLANG_COMMIT:
        raise RuntimeError(f"SGLang commit mismatch: {commit}")
    model_config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3_vl":
        raise RuntimeError(f"unexpected model type: {model_config.get('model_type')}")
    if "Qwen3VLForConditionalGeneration" not in model_config.get("architectures", []):
        raise RuntimeError("Qwen3-VL conditional-generation architecture is missing")
    train_rows = _validate_jsonl(TRAIN_DATA)
    eval_rows = _validate_jsonl(EVAL_DATA)
    free_gib = shutil.disk_usage(WORKSPACE).free / (1024**3)
    if free_gib < 80:
        raise RuntimeError(f"only {free_gib:.2f} GiB free under {WORKSPACE}")

    import torch

    if GPU_COUNT <= 0:
        raise RuntimeError(f"OPENCLAW_GPU_COUNT must be positive, got {GPU_COUNT}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < GPU_COUNT:
        raise RuntimeError(f"{GPU_COUNT} CUDA GPUs are required")
    gpu_inventory = []
    for index in range(GPU_COUNT):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        if free_bytes < 60 * 1024**3:
            raise RuntimeError(f"GPU {index} has only {free_bytes / 1024**3:.2f} GiB free")
        gpu_inventory.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "free_gib": round(free_bytes / 1024**3, 2),
                "total_gib": round(total_bytes / 1024**3, 2),
            }
        )
    cudnn_version = torch.backends.cudnn.version()
    if cudnn_version is None or cudnn_version < 91600:
        raise RuntimeError(f"cuDNN 9.16 or newer is required, got {cudnn_version}")

    package_names = (
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "ray",
        "sglang",
        "sgl-kernel",
        "qwen-vl-utils",
        "docling",
        "paddlepaddle",
    )
    packages = {name: metadata.version(name) for name in package_names}
    if not packages["torch"].startswith("2.9.1"):
        raise RuntimeError(f"unexpected Torch version: {packages['torch']}")
    if packages["transformers"] != "4.57.1" or packages["ray"] != "2.54.0":
        raise RuntimeError(f"pinned package mismatch: {packages}")
    if packages["sgl-kernel"] != "0.3.20":
        raise RuntimeError(f"unexpected SGL kernel version: {packages['sgl-kernel']}")

    for source_dir in (SLIME_DIR, TOOLCALL_DIR):
        source_text = str(source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    for module_name in (
        "sglang",
        "sgl_kernel",
        "slime",
        "qwen_vl_utils",
        "docling",
        "paddle",
    ):
        importlib.import_module(module_name)
    child_runtime_module = importlib.import_module("check_numa_child_runtime")
    child_runtime_check = child_runtime_module.check()
    if not child_runtime_check.get("ok") or child_runtime_check.get("descendant_levels") != 2:
        raise RuntimeError(f"nested NUMA bootstrap check failed: {child_runtime_check}")
    hook_module = importlib.import_module("qwen3_vl_smoke_hooks")
    if not callable(hook_module.generate) or not callable(hook_module.reward_func):
        raise RuntimeError("smoke hook import did not expose generate and reward_func")
    if DOCUMENT_PROBE is not None:
        document_tools = importlib.import_module("tools.document_tools")
        document_text, document_payload, engine = document_tools._parse_text_document(DOCUMENT_PROBE)
        if "ORCHID-7391" not in document_text or engine != "openclaw_plain_text":
            raise RuntimeError(f"plain-text document tool failed: {document_payload}")
    else:
        engine = "validated-by-rollout"

    return {
        "ok": True,
        "checked_at": _utc_now(),
        "workspace": str(WORKSPACE),
        "project": str(PROJECT),
        "environment": str(ENV_DIR),
        "numa_runtime": str(NUMA_RUNTIME_DIR),
        "numa_library": str(NUMA_LIBRARY),
        "numa_package": numa_package,
        "model": str(MODEL_DIR),
        "train_data": str(TRAIN_DATA),
        "train_rows": train_rows,
        "eval_data": str(EVAL_DATA),
        "eval_rows": eval_rows,
        "output_dir": str(OUTPUT_DIR),
        "output_exists": False,
        "ray_temp_dir": str(RAY_TEMP_DIR),
        "gpu_count": GPU_COUNT,
        "gpus": gpu_inventory,
        "free_workspace_gib": round(free_gib, 2),
        "torch_cuda": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "sglang_commit": commit,
        "packages": packages,
        "numa_child_runtime_check": child_runtime_check,
        "document_tool_engine": engine,
        "gradient_checkpointing": False,
        "rewards_normalization": False,
        "torch_compile_disabled": True,
    }


def training_argv() -> list[str]:
    return [
        "--hf-checkpoint", str(MODEL_DIR),
        "--prompt-data", str(TRAIN_DATA),
        "--input-key", "prompt",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--reward-key", "score",
        "--disable-rewards-normalization",
        "--num-rollout", "1",
        "--rollout-batch-size", "1",
        "--n-samples-per-prompt", "4",
        "--rollout-max-response-len", "256",
        "--rollout-max-context-len", "8192",
        "--rollout-temperature", "0.8",
        "--global-batch-size", "4",
        "--eval-interval", "1",
        "--eval-prompt-data", "openclaw_smoke", str(EVAL_DATA),
        "--n-samples-per-eval-prompt", "1",
        "--eval-max-response-len", "256",
        "--eval-max-context-len", "8192",
        "--eval-top-k", "1",
        "--eval-reward-key", "quality",
        "--skip-eval-before-train",
        "--advantage-estimator", "grpo",
        "--kl-loss-coef", "0.0",
        "--kl-loss-type", "low_var_kl",
        "--kl-coef", "0.0",
        "--entropy-coef", "0.0",
        "--eps-clip", "0.2",
        "--eps-clip-high", "0.28",
        "--optimizer", "adam",
        "--lr", "1e-6",
        "--lr-decay-style", "constant",
        "--weight-decay", "0.0",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--rollout-num-gpus-per-engine", "1",
        "--sglang-mem-fraction-static", "0.35",
        "--sglang-decode-log-interval", "1",
        "--sglang-disable-cuda-graph",
        "--sglang-attention-backend", "triton",
        "--sglang-mm-attention-backend", "sdpa",
        "--train-backend", "fsdp",
        "--attn-implementation", "sdpa",
        "--update-weight-buffer-size", "268435456",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", "8192",
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(GPU_COUNT),
        "--colocate",
        "--save", str(CHECKPOINT_DIR),
        "--save-interval", "1",
        "--no-save-optim",
        "--dump-details", str(OUTPUT_DIR / "dump_details"),
        "--custom-generate-function-path", "qwen3_vl_smoke_hooks.generate",
        "--custom-rm-path", "qwen3_vl_smoke_hooks.reward_func",
    ]


def run() -> dict[str, Any]:
    validation = validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    RAY_TEMP_DIR.mkdir(parents=True, exist_ok=False)
    tool_output = OUTPUT_DIR / "tool_outputs"
    tool_cache = OUTPUT_DIR / "tool_cache"
    tool_output.mkdir()
    tool_cache.mkdir()

    python_path = os.pathsep.join((str(TOOLCALL_DIR), str(SLIME_DIR)))
    runtime_library_path = str(NUMA_RUNTIME_DIR / "lib")
    inherited_library_path = os.environ.get("LD_LIBRARY_PATH")
    if inherited_library_path:
        runtime_library_path = os.pathsep.join(
            (runtime_library_path, inherited_library_path)
        )
    worker_env = {
        "PYTHONPATH": python_path,
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "CUDA_VISIBLE_DEVICES": CUDA_DEVICES,
        "LD_PRELOAD": str(NUMA_LIBRARY),
        "LD_LIBRARY_PATH": runtime_library_path,
        "OPENCLAW_NUMA_LIBRARY": str(NUMA_LIBRARY),
        "HF_HOME": str(WORKSPACE / ".openclaw-cache" / "qwen3vl"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OPENCLAW_DOCLING_ARTIFACTS_PATH": str(DOCLING_ARTIFACTS_DIR),
        "OPENCLAW_DOCLING_DISABLE_OCR_TABLE": os.environ.get("OPENCLAW_DOCLING_DISABLE_OCR_TABLE", "0"),
        "OPENCLAW_DOCLING_OCR_ENGINE": os.environ.get("OPENCLAW_DOCLING_OCR_ENGINE", "rapidocr"),
        "OPENCLAW_DOCLING_OCR_LANG": os.environ.get("OPENCLAW_DOCLING_OCR_LANG", "en"),
        "OPENCLAW_DOCLING_RAPIDOCR_BACKEND": os.environ.get("OPENCLAW_DOCLING_RAPIDOCR_BACKEND", "onnxruntime"),
        "OPENCLAW_OCR_AUTO_BACKENDS": os.environ.get(
            "OPENCLAW_OCR_AUTO_BACKENDS",
            "rapidocr,rapidocr_torch,paddleocr,easyocr",
        ),
        "OPENCLAW_OCR_OFFLINE": "1",
        "OPENCLAW_OCR_ALLOW_EASYOCR_DOWNLOAD": "0",
        "OPENCLAW_DEPLOT_MODEL": str(TOOL_MODELS_DIR / "deplot"),
        "OPENCLAW_PADDLE_RUNTIME": str(TOOL_MODELS_DIR / "paddle-runtime"),
        "OPENCLAW_MODELSCOPE_RUNTIME": str(TOOL_MODELS_DIR / "modelscope-runtime"),
        "OPENCLAW_PADDLEOCR_CACHE_DIR": str(TOOL_MODELS_DIR / "paddleocr-cache"),
        "WANDB_MODE": "disabled",
        "TORCH_COMPILE_DISABLE": "1",
        "OPENCLAW_TOOL_OUTPUT_DIR": str(tool_output),
        "OPENCLAW_TOOL_CACHE_DIR": str(tool_cache),
        "RAY_TMPDIR": str(RAY_TEMP_DIR),
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(worker_env)
    sys.path.insert(0, str(SLIME_DIR))
    sys.path.insert(0, str(TOOLCALL_DIR))
    os.chdir(SLIME_DIR)

    argv = training_argv()
    _write_once(
        OUTPUT_DIR / "launch_manifest.json",
        {
            "created_at": _utc_now(),
            "validation": validation,
            "training_argv": argv,
            "worker_env": worker_env,
        },
    )

    started = _utc_now()
    ray_module = None
    try:
        import ray

        ray_module = ray
        ray.init(
            address="local",
            num_cpus=max(8, min(os.cpu_count() or 8, 32)),
            num_gpus=GPU_COUNT,
            include_dashboard=False,
            namespace="openclaw-qwen3vl-smoke-20260717-07",
            _temp_dir=str(RAY_TEMP_DIR),
            runtime_env={"env_vars": worker_env},
            log_to_driver=True,
        )
        from slime.utils.arguments import parse_args
        from train import train

        sys.argv = ["train.py", *argv]
        parsed_args = parse_args()
        train(parsed_args)
        checkpoint_meta = CHECKPOINT_DIR / "iter_0000001" / "meta.json"
        if parsed_args.num_rollout != 0 and not checkpoint_meta.is_file():
            raise RuntimeError(f"training returned without checkpoint metadata: {checkpoint_meta}")
        result = {
            "ok": True,
            "started_at": started,
            "finished_at": _utc_now(),
            "output_dir": str(OUTPUT_DIR),
            "checkpoint_meta": str(checkpoint_meta) if checkpoint_meta.is_file() else None,
            "checkpoint": json.loads(checkpoint_meta.read_text(encoding="utf-8")) if checkpoint_meta.is_file() else None,
            "num_rollout": parsed_args.num_rollout,
            "gpu_count": GPU_COUNT,
            "evaluation_dataset": str(EVAL_DATA),
        }
        _write_once(OUTPUT_DIR / "smoke_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return result
    except BaseException as exc:
        failure = {
            "ok": False,
            "started_at": started,
            "failed_at": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "output_dir": str(OUTPUT_DIR),
        }
        failure_path = OUTPUT_DIR / "smoke_failure.json"
        if not failure_path.exists():
            _write_once(failure_path, failure)
        raise
    finally:
        if ray_module is not None:
            ray_module.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Imported allowlisted launchers may still carry an older diagnostic name.
# Redirect validation and execution to a fresh immutable output directory.
_validate_before_run12_redirect = validate


def validate() -> dict[str, Any]:
    global OUTPUT_DIR, CHECKPOINT_DIR, RAY_TEMP_DIR
    if OUTPUT_DIR.name in {
        "qwen3-vl-4b-docvqa-real-rl-test-20260720-05",
        "qwen3-vl-4b-docvqa-real-rl-test-20260720-11",
    }:
        OUTPUT_DIR = OUTPUT_DIR.with_name("qwen3-vl-4b-docvqa-real-rl-test-20260720-12")
        CHECKPOINT_DIR, RAY_TEMP_DIR = OUTPUT_DIR / "checkpoints", WORKSPACE / ".ray" / "q3v4b-real-20260720-12"
    return _validate_before_run12_redirect()
_validate_before_run23_redirect = validate

def validate() -> dict[str, Any]:
    global OUTPUT_DIR, CHECKPOINT_DIR, RAY_TEMP_DIR
    if OUTPUT_DIR.name in {
        "qwen3-vl-4b-docvqa-real-rl-test-20260720-05",
        "qwen3-vl-4b-docvqa-real-rl-test-20260720-11",
        "qwen3-vl-4b-docvqa-real-rl-test-20260720-12",
    }:
        OUTPUT_DIR = OUTPUT_DIR.with_name("qwen3-vl-4b-docvqa-real-rl-test-20260723-01")
        CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
        RAY_TEMP_DIR = WORKSPACE / ".ray" / "q3v4b-real-20260723-01"
    return _validate_before_run23_redirect()
