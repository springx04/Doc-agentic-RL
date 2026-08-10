# Vendored third-party sources

This directory contains source trees that are required to reproduce the
project's runtime installation and must remain inside the workspace.

## NVIDIA Apex

`apex/` is NVIDIA Apex at commit
`9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`. The Qwen3-VL Megatron path uses
the compiled `fused_weight_gradient_mlp_cuda` extension for gradient
accumulation fusion; the unrelated package named `apex` on PyPI is not a
replacement.

From the project root, install all Apex C++/CUDA extensions into the active
Python environment with:

```bash
PYTHON=/workspace/data/envs/openclaw-rl-qwen3vl/bin/python \
  third_party/install_apex_cuda.sh
```

The script derives the source path from its own location, uses
`APEX_CPP_EXT=1` and `APEX_CUDA_EXT=1`, and verifies the target extension by
importing it after installation. Set `CUDA_HOME`, `MAX_JOBS`,
`TORCH_CUDA_ARCH_LIST`, or `APEX_DIR` when the local CUDA/runtime layout
differs.

## SGLang

The server deployment keeps the SGLang source tree at
`third_party/sglang/`; it was migrated from the temporary
`/workspace/data/.openclaw-build/sglang` directory so the runtime does not
depend on a cleanable external path. The BayesTool launcher automatically
prepends `third_party/sglang/python` to the Ray worker `PYTHONPATH` when this
tree is present.

After moving or restoring this source tree, recreate the editable installation
from the project path with the source revision recorded by the tree:

```bash
PYTHON=/workspace/data/envs/openclaw-rl-qwen3vl/bin/python
"$PYTHON" -m pip install --no-deps --no-build-isolation --editable \
  third_party/sglang/python
```

Because the migrated source tree has no upstream `.git` metadata, its
`python/pyproject.toml` pins the matching version instead of allowing an
editable install to report `0.0.0`.

The SGLang package requires `sgl-kernel==0.3.20`; the A100 runtime must also
have `libnuma.so.1`, which is provided by the system package `libnuma1`.

## Packed attention and FlashAttention

The Qwen3-VL trainer uses packed THD inputs (`thd_thd_thd` with
padding-causal masking). On the target A100 environment, Transformer Engine
2.12.0 does not expose its cuDNN fused sub-backend for this layout, while its
exact `UnfusedDotProductAttention` backend is complete and is the launcher
default. This is a performance choice, not an attention or training-path
substitute; it preserves the packed attention semantics.

To install the optional A100 FlashAttention CUDA extension without allowing
pip to replace the pinned PyTorch/CUDA/cuDNN stack, run from the project root:

```bash
PYTHON_BIN=/workspace/data/envs/openclaw-rl-qwen3vl/bin/python \
  third_party/install_flash_attn_cuda.sh
```

The script builds only SM80 by default, uses `--no-deps`, and verifies
`flash_attn_varlen_func` after installing the exact TE-compatible
`flash-attn==2.8.3` release. `2.8.3.post1` is rejected by Transformer Engine
2.12.0's inclusive upper version gate. After that verification, set
`ATTENTION_BACKEND=flash` for the launcher and rerun the packed GPU smoke;
without that verification, keep the default `unfused` backend.
