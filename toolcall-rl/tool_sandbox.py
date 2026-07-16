"""Registry and executor for document-understanding tools."""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from tools.document_tools import DOC_TOOL_SPECS, execute_document_tool
except Exception as exc:  # pragma: no cover - keep rollout startup resilient
    DOC_TOOL_SPECS = {}
    DOCUMENT_TOOL_IMPORT_ERROR = str(exc)
else:
    DOCUMENT_TOOL_IMPORT_ERROR = None


TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "max_obs_chars": 8192,
    "tool_concurrency": 32,
}

SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


class ToolRegistry:
    """Register and execute the document tools used by RL rollouts."""

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        for name, spec in DOC_TOOL_SPECS.items():
            self.register_tool(name, spec)

    def register_tool(self, name: str, tool_spec: dict[str, Any]) -> None:
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        if DOCUMENT_TOOL_IMPORT_ERROR:
            return f"Error: document tools failed to import: {DOCUMENT_TOOL_IMPORT_ERROR}"
        async with SEMAPHORE:
            return await asyncio.to_thread(execute_document_tool, tool_name, arguments)


tool_registry = ToolRegistry()
