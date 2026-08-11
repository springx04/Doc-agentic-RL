"""Single registry and dispatch point for all Code tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

try:
    from ..config import CODE_TOOL_FAMILIES, CODE_TOOL_NAMES
    from ..protocol import validate_tool_arguments
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from config import CODE_TOOL_FAMILIES, CODE_TOOL_NAMES
    from protocol import validate_tool_arguments
    from schemas import CodeToolResult

from ._common import ToolExecutionContext, clean_failure, invalid_result
from .apply_patch import execute as execute_apply_patch
from .git_diff import execute as execute_git_diff
from .list_tree import execute as execute_list_tree
from .read_file import execute as execute_read_file
from .run_checks import execute as execute_run_checks
from .run_command import execute as execute_run_command
from .run_tests import execute as execute_run_tests
from .search_code import execute as execute_search_code


Handler = Callable[[dict[str, Any], ToolExecutionContext], Awaitable[CodeToolResult]]


@dataclass(frozen=True)
class CodeToolSpec:
    name: str
    family: str
    handler: Handler


class CodeToolRegistry:
    def __init__(self, specs: Mapping[str, CodeToolSpec] | None = None):
        if specs is None:
            specs = {
                "list_tree": CodeToolSpec("list_tree", "inspection_core", execute_list_tree),
                "search_code": CodeToolSpec("search_code", "inspection_core", execute_search_code),
                "read_file": CodeToolSpec("read_file", "inspection_core", execute_read_file),
                "apply_patch": CodeToolSpec("apply_patch", "mutation_core", execute_apply_patch),
                "git_diff": CodeToolSpec("git_diff", "inspection_core", execute_git_diff),
                "run_tests": CodeToolSpec("run_tests", "validation_core", execute_run_tests),
                "run_checks": CodeToolSpec("run_checks", "validation_core", execute_run_checks),
                "run_command": CodeToolSpec("run_command", "execution_core", execute_run_command),
            }
        missing = set(CODE_TOOL_NAMES) - set(specs)
        if missing:
            raise ValueError(f"Code tool registry missing tools: {sorted(missing)}")
        self.specs = dict(specs)

    def get(self, tool_name: str) -> CodeToolSpec:
        try:
            return self.specs[tool_name]
        except KeyError as exc:
            raise KeyError(f"unknown Code tool: {tool_name}") from exc

    async def execute(self, tool_name: str, arguments: Mapping[str, Any], context: ToolExecutionContext) -> CodeToolResult:
        if tool_name not in self.specs:
            return invalid_result(tool_name, f"unknown Code tool: {tool_name}")
        normalized, error = validate_tool_arguments(tool_name, arguments)
        if error or normalized is None:
            return invalid_result(tool_name, error or "invalid arguments")
        try:
            return await self.specs[tool_name].handler(normalized, context)
        except Exception as exc:
            return clean_failure(tool_name, exc)

    async def dispatch(self, action: Any, context: ToolExecutionContext) -> CodeToolResult:
        tool_name = getattr(action, "tool_name", None) or action.get("name")
        arguments = getattr(action, "arguments", None) or action.get("arguments", {})
        return await self.execute(str(tool_name), arguments, context)

    def tool_families(self) -> dict[str, tuple[str, ...]]:
        return {family: tuple(names) for family, names in CODE_TOOL_FAMILIES.items()}


DEFAULT_REGISTRY = CodeToolRegistry()


__all__ = ["CodeToolRegistry", "CodeToolSpec", "DEFAULT_REGISTRY", "ToolExecutionContext"]
