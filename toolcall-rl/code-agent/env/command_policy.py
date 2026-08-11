"""Guardrails preventing ``run_command`` from bypassing logical tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

try:
    from ..schemas import CodeToolResult, ExecutionResult
except ImportError:  # pragma: no cover
    from schemas import CodeToolResult, ExecutionResult


class CommandClass(str, Enum):
    SEARCH = "SEARCH"
    READ = "READ"
    EDIT = "EDIT"
    TEST = "TEST"
    CHECK = "CHECK"
    VCS = "VCS"
    RUNTIME = "RUNTIME"
    PACKAGE_MUTATION = "PACKAGE_MUTATION"
    BACKGROUND_PROCESS = "BACKGROUND_PROCESS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    command_class: CommandClass
    reason: str = ""


class CodeCommandPolicy:
    """Classify and validate commands before they reach the executor."""

    _search = re.compile(r"(?:^|\s)(?:rg|grep|find|fd)(?:\s|$)")
    _read = re.compile(r"(?:^|\s)(?:cat|head|tail|less|more|sed\s+-n|awk)(?:\s|$)")
    _test = re.compile(r"(?:^|\s)(?:pytest|tox|nose2|unittest)(?:\s|$)|python\s+-m\s+(?:pytest|unittest)")
    _check = re.compile(r"(?:^|\s)(?:ruff|mypy|pyright|flake8|eslint|tsc|pylint)(?:\s|$)|python\s+-m\s+(?:compileall|py_compile)")
    # ``python -c`` is allowed as a runtime diagnostic and then protected by
    # the tracked-state guard below.  This is important: a mutation attempt is
    # observed and restored rather than silently becoming a persistent edit.
    _edit = re.compile(r"(?:sed\s+-i|perl\s+-pi|python\s+-\s*<<?|tee\s|\becho\s+.*>)", re.I)
    _vcs = re.compile(r"(?:^|\s)git(?:\s|$)")
    _package = re.compile(r"(?:^|\s)(?:pip|pip3|uv|poetry|npm|yarn|pnpm|apt|apt-get|conda)\s+(?:install|add|remove|update)")
    _background = re.compile(r"(?:&\s*$|\bnohup\b|\bsystemctl\b|\bdocker\b|\bscreen\b|\btmux\b)")

    @classmethod
    def classify(cls, command: str) -> CommandClass:
        text = str(command or "").strip()
        if not text:
            return CommandClass.UNKNOWN
        if cls._background.search(text):
            return CommandClass.BACKGROUND_PROCESS
        if cls._package.search(text):
            return CommandClass.PACKAGE_MUTATION
        if cls._edit.search(text):
            return CommandClass.EDIT
        if cls._search.search(text):
            return CommandClass.SEARCH
        if cls._read.search(text):
            return CommandClass.READ
        if cls._test.search(text):
            return CommandClass.TEST
        if cls._check.search(text):
            return CommandClass.CHECK
        if cls._vcs.search(text):
            return CommandClass.VCS
        if any(token in text for token in ("python ", "python3 ", "node ", "ruby ", "java ", "./", "bash ")):
            return CommandClass.RUNTIME
        return CommandClass.UNKNOWN

    @classmethod
    def validate(cls, command: str) -> PolicyDecision:
        command_class = cls.classify(command)
        if command_class == CommandClass.SEARCH:
            return PolicyDecision(False, command_class, "use search_code or list_tree instead")
        if command_class == CommandClass.READ:
            return PolicyDecision(False, command_class, "use read_file instead")
        if command_class == CommandClass.EDIT:
            return PolicyDecision(False, command_class, "persistent edits must use apply_patch")
        if command_class == CommandClass.TEST:
            return PolicyDecision(False, command_class, "use run_tests instead")
        if command_class == CommandClass.CHECK:
            return PolicyDecision(False, command_class, "use run_checks instead")
        if command_class == CommandClass.PACKAGE_MUTATION:
            return PolicyDecision(False, command_class, "package installation or mutation is forbidden")
        if command_class == CommandClass.BACKGROUND_PROCESS:
            return PolicyDecision(False, command_class, "background processes and service control are forbidden")
        if command_class == CommandClass.VCS:
            # Repository state and VCS/network operations have dedicated
            # ownership in git_diff/apply_patch/checkpoint handling.  Allowing
            # an unlisted git subcommand here would leave bypasses such as
            # push, remote, config, worktree, or tag creation.
            return PolicyDecision(False, command_class, "Git commands are unavailable in run_command; use git_diff or apply_patch")
        return PolicyDecision(True, command_class)

    @classmethod
    async def execute_guarded(
        cls,
        command: str,
        *,
        diff: Callable[[], Awaitable[str]],
        execute: Callable[[str], Awaitable[ExecutionResult]],
        reset_to_patch: Callable[[str], Awaitable[ExecutionResult]],
    ) -> CodeToolResult:
        """Run a permitted command and restore any tracked-file mutation."""

        decision = cls.validate(command)
        if not decision.allowed:
            return CodeToolResult(
                tool_name="run_command",
                status="invalid",
                output=decision.reason,
                metadata={"command_class": decision.command_class.value},
                failure_origin="model_action",
            )
        before = await diff()
        result = await execute(command)
        after = await diff()
        if after != before:
            restored = await reset_to_patch(before)
            return CodeToolResult(
                tool_name="run_command",
                status="invalid" if restored.ok else "error",
                output="run_command attempted a persistent tracked-file mutation; state was restored",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                metadata={"command_class": decision.command_class.value, "restored": restored.ok},
                failure_origin="model_action" if restored.ok else "real_infrastructure",
            )
        status = "ok" if result.ok else ("timeout" if result.timed_out else "error")
        return CodeToolResult(
            tool_name="run_command",
            status=status,
            output=result.output,
            latency_ms=result.duration_ms,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            metadata={"command_class": decision.command_class.value},
            failure_origin=result.failure_origin,
        )


__all__ = ["CodeCommandPolicy", "CommandClass", "PolicyDecision"]
