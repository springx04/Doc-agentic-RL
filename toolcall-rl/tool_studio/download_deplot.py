"""Download the DePlot checkpoint into Tool Studio's temporary model cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    local_app_data = Path(os.environ["LOCALAPPDATA"])
    store_site = next((local_app_data / "Packages").glob("PythonSoftwareFoundation.Python.3.11_*/LocalCache/local-packages/Python311/site-packages"))
    sys.path.append(str(store_site))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(name, None)

    from huggingface_hub import snapshot_download

    target = Path(os.environ.get("TEMP", ".")) / "openclaw_deplot_model"
    print(snapshot_download("google/deplot", local_dir=target, max_workers=1), flush=True)


if __name__ == "__main__":
    main()
