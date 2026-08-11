"""Preload the verified OpenClaw NUMA runtime in explicitly opted-in children."""

from __future__ import annotations

import ctypes
import os


_EXPECTED_NUMA_LIBRARY = (
    "/workspace/data/envs/openclaw-rl-qwen3vl-numa-20260717-01/lib/libnuma.so.1"
)
_CONFIGURED_NUMA_LIBRARY = os.environ.get("OPENCLAW_NUMA_LIBRARY")

if _CONFIGURED_NUMA_LIBRARY:
    configured = os.path.realpath(_CONFIGURED_NUMA_LIBRARY)
    expected = os.path.realpath(_EXPECTED_NUMA_LIBRARY)
    if configured != expected:
        raise RuntimeError(
            f"OPENCLAW_NUMA_LIBRARY must resolve to {expected}, got {configured}"
        )
    ctypes.CDLL(expected, mode=ctypes.RTLD_GLOBAL)
    os.environ["OPENCLAW_NUMA_PRELOADED"] = expected
