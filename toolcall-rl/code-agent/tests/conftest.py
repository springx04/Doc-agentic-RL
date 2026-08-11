from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CODE_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_AGENT_ROOT))


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "test_app.py").write_text("from app import VALUE\n\ndef test_value():\n    assert VALUE == 2\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@localhost"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Code Tests"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True, capture_output=True)
    return root


@pytest.fixture
def patch_value():
    def _patch_value(old: int = 1, new: int = 2) -> str:
        return f"""diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-VALUE = {old}
+VALUE = {new}
"""

    return _patch_value
