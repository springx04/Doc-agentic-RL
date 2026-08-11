#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Transformer Engine 2.12.0 accepts FlashAttention through 2.8.3 inclusive;
# 2.8.3.post1 is rejected by its version gate even though it is otherwise
# source-compatible.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ "${PYTHON_BIN}" != */* ]]; then
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not available: ${PYTHON_BIN}" >&2
    exit 1
fi

if "${PYTHON_BIN}" - <<'PY'
from flash_attn.flash_attn_interface import flash_attn_varlen_func
print(f"flash-attn is already importable: {flash_attn_varlen_func.__name__}")
PY
then
    exit 0
fi

echo "Installing flash-attn==${FLASH_ATTN_VERSION} for CUDA architectures ${FLASH_ATTN_CUDA_ARCHS}."
echo "The source build intentionally uses --no-deps so it cannot replace the pinned CUDA/cuDNN runtime."
env \
    MAX_JOBS="${MAX_JOBS}" \
    FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    "${PYTHON_BIN}" -m pip install \
    --no-deps \
    --no-build-isolation \
    --no-cache-dir \
    "flash-attn==${FLASH_ATTN_VERSION}"

"${PYTHON_BIN}" - <<'PY'
from flash_attn.flash_attn_interface import flash_attn_varlen_func
print(f"flash-attn CUDA extension verified: {flash_attn_varlen_func.__name__}")
PY
