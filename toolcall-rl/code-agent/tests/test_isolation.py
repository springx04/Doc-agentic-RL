from __future__ import annotations

import subprocess
from pathlib import Path


CODE_AGENT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = (
    "toolcall-rl/generate_with_retool.py",
    "toolcall-rl/bayestool",
    "toolcall-rl/tools.py",
    "toolcall-rl/train_grpo.py",
)


def test_code_changes_are_scoped_away_from_doc_agent():
    root = CODE_AGENT_ROOT.parents[1]
    tracked_doc_changes = subprocess.run(
        ["git", "diff", "--name-only", "--", *DOC_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked_doc_changes == []

    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert all(line[3:].replace("\\", "/").startswith("toolcall-rl/code-agent/") for line in status if line.strip())


def test_code_runtime_imports_resolve_inside_code_agent():
    import bayestool
    import data
    import env
    import tools

    for module in (bayestool, data, env, tools):
        module_path = Path(module.__file__).resolve()
        assert module_path.is_relative_to(CODE_AGENT_ROOT)


def test_code_source_has_no_doc_runtime_imports():
    forbidden = ("generate_with_retool", "doc_bayestool", "DOC_ENV_SERVER_URL")
    for path in CODE_AGENT_ROOT.rglob("*.py"):
        if path == Path(__file__):
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path
