#!/usr/bin/env bash
set -euo pipefail

# This script deliberately resolves Apex relative to the project instead of
# relying on a temporary directory outside the workspace.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APEX_DIR="${APEX_DIR:-${SCRIPT_DIR}/apex}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${APEX_DIR}/setup.py" ]]; then
    echo "Apex source is missing: ${APEX_DIR}" >&2
    echo "Clone NVIDIA Apex at third_party/apex or set APEX_DIR explicitly." >&2
    exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}"

# Apex's PEP 517 build reads these environment variables. Do not use pip's
# deprecated --global-option flags: PEP 517 may ignore them and produce a
# pure-Python wheel without the CUDA extensions required by Megatron.
export APEX_CPP_EXT=1
export APEX_CUDA_EXT=1

"${PYTHON}" -m pip install \
    --verbose \
    --no-build-isolation \
    --no-cache-dir \
    --force-reinstall \
    "${APEX_DIR}"

"${PYTHON}" - <<'PY'
import importlib.util

module_name = "fused_weight_gradient_mlp_cuda"
if importlib.util.find_spec(module_name) is None:
    raise SystemExit(
        f"{module_name} was not installed; Apex CUDA extension build is incomplete"
    )

module = __import__(module_name)
print(f"{module_name}: {module.__file__}")
PY
