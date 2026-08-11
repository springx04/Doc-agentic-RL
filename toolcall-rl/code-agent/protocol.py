"""Strict one-action-per-turn protocol for Code Agent rollouts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from .config import CODE_TOOL_NAMES
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from config import CODE_TOOL_NAMES


_TOOL_RE = re.compile(r"\A\s*<tool_call>(?P<body>.*?)</tool_call>\s*\Z", re.I | re.S)
_FINAL_RE = re.compile(r"\A\s*<final>(?P<body>.*?)</final>\s*\Z", re.I | re.S)
_ABSTAIN_RE = re.compile(r"\A\s*<abstain>(?P<body>.*?)</abstain>\s*\Z", re.I | re.S)
_OPEN_TAG_RE = re.compile(r"<\s*(tool_call|final|abstain)\b", re.I)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    final_text: str | None = None
    protocol_error: str | None = None
    raw: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.kind in {"final", "abstain"}

    @property
    def is_valid(self) -> bool:
        return self.kind in {"tool_call", "final", "abstain"} and self.protocol_error is None


def _invalid(raw: str, reason: str) -> ParsedAction:
    return ParsedAction(kind="invalid", protocol_error=reason, raw=raw)


def parse_action(text: str | None) -> ParsedAction:
    """Parse exactly one complete action and reject all surrounding prose."""

    raw = "" if text is None else str(text)
    if not raw.strip():
        return _invalid(raw, "empty assistant turn")
    if len(_OPEN_TAG_RE.findall(raw)) != 1:
        return _invalid(raw, "assistant turn must contain exactly one action")

    match = _TOOL_RE.fullmatch(raw)
    if match:
        body = match.group("body").strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            return _invalid(raw, f"invalid tool-call JSON: {exc.msg}")
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            return _invalid(raw, "tool_call JSON keys must be exactly name and arguments")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip():
            return _invalid(raw, "tool_call name must be a non-empty string")
        if not isinstance(arguments, dict):
            return _invalid(raw, "tool_call arguments must be an object")
        name = name.strip()
        if name not in CODE_TOOL_NAMES:
            return _invalid(raw, f"unknown Code tool: {name}")
        normalized, error = validate_tool_arguments(name, arguments)
        if error:
            return _invalid(raw, error)
        return ParsedAction(kind="tool_call", tool_name=name, arguments=normalized, raw=raw)

    match = _FINAL_RE.fullmatch(raw)
    if match:
        body = match.group("body").strip()
        if not body:
            return _invalid(raw, "final content must not be empty")
        return ParsedAction(kind="final", final_text=body, raw=raw)

    match = _ABSTAIN_RE.fullmatch(raw)
    if match:
        body = match.group("body").strip()
        if not body:
            return _invalid(raw, "abstain reason must not be empty")
        return ParsedAction(kind="abstain", final_text=body, raw=raw)

    return _invalid(raw, "action must be a complete tool_call, final, or abstain tag")


def _expect_keys(arguments: Mapping[str, Any], required: set[str], optional: set[str]) -> str | None:
    keys = set(arguments)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        return f"missing tool arguments: {', '.join(sorted(missing))}"
    if unknown:
        return f"unknown tool arguments: {', '.join(sorted(unknown))}"
    return None


def _string(arguments: Mapping[str, Any], name: str, *, nonempty: bool = True) -> str | None:
    value = arguments.get(name)
    if not isinstance(value, str) or (nonempty and not value.strip()):
        return f"{name} must be a{' non-empty' if nonempty else ''} string"
    return None


def _integer(arguments: Mapping[str, Any], name: str, low: int, high: int) -> str | None:
    value = arguments.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        return f"{name} must be an integer in [{low}, {high}]"
    return None


def validate_tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and normalize the public schema for one Code tool."""

    if not isinstance(arguments, Mapping):
        return None, "tool arguments must be an object"
    args = dict(arguments)
    if tool_name == "list_tree":
        error = _expect_keys(args, set(), {"path", "max_depth"})
        if error:
            return None, error
        args.setdefault("path", ".")
        args.setdefault("max_depth", 2)
        error = _string(args, "path") or _integer(args, "max_depth", 0, 8)
    elif tool_name == "search_code":
        error = _expect_keys(args, {"query"}, {"path", "glob", "max_results"})
        if error:
            return None, error
        args.setdefault("path", ".")
        args.setdefault("glob", "*")
        args.setdefault("max_results", 50)
        error = _string(args, "query") or _string(args, "path") or _string(args, "glob", nonempty=False)
        error = error or _integer(args, "max_results", 1, 1000)
    elif tool_name == "read_file":
        error = _expect_keys(args, {"path"}, {"start_line", "end_line"})
        if error:
            return None, error
        args.setdefault("start_line", 1)
        args.setdefault("end_line", 240)
        error = _string(args, "path") or _integer(args, "start_line", 1, 10_000_000)
        error = error or _integer(args, "end_line", 1, 10_000_000)
        if error is None and args["end_line"] < args["start_line"]:
            error = "end_line must be greater than or equal to start_line"
        if error is None and args["end_line"] - args["start_line"] + 1 > 240:
            error = "read_file range exceeds the 240-line per-call limit"
    elif tool_name == "apply_patch":
        error = _expect_keys(args, {"patch"}, set()) or _string(args, "patch")
        if error is None and "diff --git " not in str(args["patch"]):
            error = "patch must be a unified git diff containing diff --git headers"
    elif tool_name == "git_diff":
        error = _expect_keys(args, set(), set())
    elif tool_name == "run_tests":
        error = _expect_keys(args, set(), {"target", "args", "timeout"})
        if error:
            return None, error
        args.setdefault("target", "")
        args.setdefault("args", "")
        args.setdefault("timeout", 180)
        error = _string(args, "target", nonempty=False) or _string(args, "args", nonempty=False)
        error = error or _integer(args, "timeout", 1, 3600)
    elif tool_name == "run_checks":
        error = _expect_keys(args, set(), {"check", "path", "timeout"})
        if error:
            return None, error
        args.setdefault("check", "compile")
        args.setdefault("path", ".")
        args.setdefault("timeout", 180)
        error = _string(args, "check") or _string(args, "path") or _integer(args, "timeout", 1, 3600)
    elif tool_name == "run_command":
        error = _expect_keys(args, {"command"}, {"timeout"})
        if error:
            return None, error
        args.setdefault("timeout", 180)
        error = _string(args, "command") or _integer(args, "timeout", 1, 3600)
    else:
        return None, f"unknown Code tool: {tool_name}"
    return (args, None) if error is None else (None, error)


def render_tool_call(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Render the canonical exact protocol emitted in prompts/tests."""

    normalized, error = validate_tool_arguments(tool_name, arguments)
    if error or normalized is None:
        raise ValueError(error or "invalid tool arguments")
    return f"<tool_call>{json.dumps({'name': tool_name, 'arguments': normalized}, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


__all__ = ["ParsedAction", "parse_action", "render_tool_call", "validate_tool_arguments"]
