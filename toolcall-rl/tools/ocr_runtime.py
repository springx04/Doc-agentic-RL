"""Headless native-library setup for the offline OCR backends."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any


_LOADED_LIBRARIES: dict[str, Any] = {}


def _library_directories() -> list[Path]:
    configured = os.environ.get("OPENCLAW_OCR_LIBRARY_DIR", "")
    values = [item for item in configured.split(os.pathsep) if item]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        values.append(str(Path(conda_prefix) / "lib"))
    values.extend(
        (
            str(Path(sys.prefix) / "lib"),
            "/opt/conda/lib",
            "/opt/conda/pkgs/libgl-1.7.0-ha4b6fd6_2/lib",
            "/opt/conda/pkgs/libglib-2.86.0-h1fed272_0/lib",
        )
    )
    directories: list[Path] = []
    seen: set[str] = set()
    for value in values:
        directory = Path(value).expanduser()
        key = str(directory)
        if key in seen or not directory.is_dir():
            continue
        seen.add(key)
        directories.append(directory)
    return directories


def prepare_headless_ocr_runtime() -> dict[str, list[str]]:
    """Make already-installed GL dependencies visible before importing cv2.

    The server's OpenCV build is a GUI-enabled wheel and asks the dynamic
    loader for ``libGL.so.1`` and ``libgthread-2.0.so.0``.  The libraries are
    already present in the image, but the Ray environment does not expose
    their directory.  Preloading the absolute files works even after Python
    has started; adding the directory to ``LD_LIBRARY_PATH`` also propagates
    the fix to subprocesses.  No package or model download is performed.
    """

    if os.name == "nt":
        return {"directories": [], "loaded": []}

    directories = _library_directories()
    usable = [
        directory
        for directory in directories
        if (directory / "libGL.so.1").is_file()
        or (directory / "libgthread-2.0.so.0").is_file()
    ]
    if usable:
        current = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
        additions = [str(directory) for directory in usable if str(directory) not in current]
        if additions:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join((*additions, *current))

    loaded: list[str] = []
    # libgthread is loaded first because cv2 may report it after libGL has
    # already been resolved.  RTLD_GLOBAL makes the SONAME visible to cv2.
    for library_name in ("libgthread-2.0.so.0", "libGL.so.1"):
        if library_name in _LOADED_LIBRARIES:
            loaded.append(library_name)
            continue
        for directory in directories:
            path = directory / library_name
            if not path.is_file():
                continue
            try:
                handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            _LOADED_LIBRARIES[library_name] = handle
            loaded.append(library_name)
            break
    return {"directories": [str(directory) for directory in usable], "loaded": loaded}
