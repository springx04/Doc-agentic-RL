"""Verify the opt-in NUMA bootstrap through two fresh Python descendants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path("/workspace/data").resolve()
PROJECT = (WORKSPACE / "OpenClaw-RL").resolve()
TOOLCALL_DIR = (PROJECT / "toolcall-rl").resolve()
SLIME_DIR = (PROJECT / "slime").resolve()
EXPECTED_PYTHON = (
    WORKSPACE / "envs" / "openclaw-rl-qwen3vl" / "bin" / "python"
).resolve()
NUMA_LIBRARY = (
    WORKSPACE
    / "envs"
    / "openclaw-rl-qwen3vl-numa-20260717-01"
    / "lib"
    / "libnuma.so.1"
)
EXPECTED_NUMA_LIBRARY = str(NUMA_LIBRARY.resolve())


INNER_CODE = f"""import importlib.metadata as metadata
import json
import os
import sys
import torch
import sgl_kernel
expected = {EXPECTED_NUMA_LIBRARY!r}
actual = os.environ.get("OPENCLAW_NUMA_PRELOADED")
if actual != expected:
    raise RuntimeError(f"NUMA bootstrap mismatch: {{actual!r}} != {{expected!r}}")
print(json.dumps({{
    "ok": True,
    "python": sys.executable,
    "base_prefix": sys.base_prefix,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "sgl_kernel": metadata.version("sgl-kernel"),
    "numa_preloaded": actual,
}}, sort_keys=True))
"""


def check() -> dict[str, object]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON:
        raise RuntimeError(f"expected {EXPECTED_PYTHON}, got {sys.executable}")
    if not NUMA_LIBRARY.is_file():
        raise FileNotFoundError(NUMA_LIBRARY)

    python_path = os.pathsep.join((str(TOOLCALL_DIR), str(SLIME_DIR)))
    clean_env = os.environ.copy()
    clean_env.update(
        {
            "PYTHONPATH": python_path,
            "OPENCLAW_NUMA_LIBRARY": str(NUMA_LIBRARY),
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    clean_env.pop("LD_PRELOAD", None)
    clean_env.pop("LD_LIBRARY_PATH", None)

    outer_code = (
        "import os, subprocess, sys\n"
        f"inner_code = {INNER_CODE!r}\n"
        "completed = subprocess.run("
        "[sys.executable, '-B', '-c', inner_code], "
        "env=os.environ.copy(), text=True, capture_output=True, timeout=300"
        ")\n"
        "sys.stdout.write(completed.stdout)\n"
        "sys.stderr.write(completed.stderr)\n"
        "raise SystemExit(completed.returncode)\n"
    )
    completed = subprocess.run(
        [str(EXPECTED_PYTHON), "-B", "-c", outer_code],
        env=clean_env,
        text=True,
        capture_output=True,
        timeout=360,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"two-level NUMA bootstrap check failed with exit code {completed.returncode}\n"
            f"stdout: {completed.stdout[-4000:]}\n"
            f"stderr: {completed.stderr[-4000:]}"
        )
    inner_result = json.loads(completed.stdout.strip().splitlines()[-1])
    result = {
        "ok": True,
        "descendant_levels": 2,
        "ld_preload_removed": True,
        "ld_library_path_removed": True,
        "pythonpath": python_path,
        "inner": inner_result,
    }
    return result


def main() -> int:
    print(json.dumps(check(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
