"""Shared offline helpers for Code Stage A-D entry points."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from ..config import DEFAULT_CODE_CONFIG, CodeConfig
    from ..data.preprocess_swe import read_rows
    from ..data.schema import SWEInstance
    from ..env.client import LocalCodeEnvClient
except ImportError:  # pragma: no cover
    from config import DEFAULT_CODE_CONFIG, CodeConfig
    from data.preprocess_swe import read_rows
    from data.schema import SWEInstance
    from env.client import LocalCodeEnvClient


class ScriptedModelClient:
    """Deterministic model client for local tests/smoke, not a training policy."""

    def __init__(self, responses: Iterable[Any]):
        self.responses = list(responses)
        self.index = 0

    async def generate(self, messages: list[dict[str, str]], *, max_tokens: int, **kwargs: Any) -> Any:
        if not self.responses:
            return "<abstain>no scripted response</abstain>"
        value = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return value


def load_instances(path: str | Path) -> list[SWEInstance]:
    return [SWEInstance.from_raw(row) for row in read_rows(path)]


def repository_public_context(repository_root: str | Path, *, tool_budget: int) -> dict[str, Any]:
    root = Path(repository_root)
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts]
    extensions = sorted({path.suffix for path in files if path.suffix})
    top_dirs = sorted({path.relative_to(root).parts[0] for path in files if len(path.relative_to(root).parts) > 1})
    languages = sorted({"python" if extension == ".py" else "javascript" if extension in {".js", ".ts", ".tsx"} else "go" if extension == ".go" else "rust" if extension == ".rs" else "" for extension in extensions} - {""})
    frameworks = []
    if any(path.name == "pytest.ini" or path.name == "pyproject.toml" for path in files):
        frameworks.append("python")
    if any(path.name == "package.json" for path in files):
        frameworks.append("node")
    return {"tool_budget": tool_budget, "repo_file_count": len(files), "tracked_extensions": extensions, "top_level_dirs": top_dirs, "languages": languages, "detected_frameworks": sorted(set(frameworks))}


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def make_local_client(repository_root: str | Path) -> LocalCodeEnvClient:
    return LocalCodeEnvClient(repository_root)


def default_sample(instance: SWEInstance) -> dict[str, Any]:
    return instance.to_runtime_row()


def evaluator_script_for_instance(instance: SWEInstance, override: str | None = None) -> str | None:
    """Resolve evaluator-only commands without exposing them to the policy."""

    if override and str(override).strip():
        return str(override)
    private = instance.evaluator_private.values
    for key in ("eval_script", "test_command"):
        value = private.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def choose_seed(seed: int, index: int) -> int:
    return random.Random(seed + index * 1009).randrange(0, 2**31 - 1)


__all__ = ["ScriptedModelClient", "choose_seed", "default_sample", "evaluator_script_for_instance", "load_instances", "make_local_client", "repository_public_context", "write_json"]
