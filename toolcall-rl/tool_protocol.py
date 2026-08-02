"""Strict parsing for assistant tool-call and final-answer turns.

The rollout protocol is deliberately small: one assistant turn contains one
complete ``<tool_call>...</tool_call>`` or ``<final>...</final>`` action.  Both
actions must occupy the whole turn; reward evaluation may separately recover a
single final span to distinguish semantic correctness from protocol validity.
This module never searches for a usable tool call inside arbitrary prose.  That
is important because tool instructions and previous observations can
themselves contain the protocol tags as examples.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_ACTION_TAG_RE = re.compile(r"<\s*(tool_call|final)\b[^>]*>", re.IGNORECASE)
_ACTION_MARKER_RE = re.compile(r"</?\s*(tool_call|final)\b", re.IGNORECASE)
_FINAL_RE = re.compile(r"\s*<final>(?P<body>.*?)</final>\s*\Z", re.IGNORECASE | re.DOTALL)
_TOOL_RE = re.compile(r"\s*<tool_call>(?P<body>.*?)</tool_call>\s*\Z", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(
    r"\s*<function=(?P<name>[A-Za-z_][\w.-]*)>"
    r"(?P<body>.*?)</function>\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][\w.-]*)>(?P<value>.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAction:
    """Result of parsing exactly one assistant turn."""

    kind: str
    value: Any = None
    candidate_action_count: int = 0
    reason: str | None = None
    raw: str = ""
    parsed_tool_name: str | None = None

    @property
    def is_action(self) -> bool:
        return self.kind in {"tool_call", "final"}


def count_candidate_actions(text: str | None) -> int:
    """Count action openings without treating closing tags as actions."""

    return len(_ACTION_TAG_RE.findall(text or ""))


def _maybe_json_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _parse_json_tool_body(body: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON tool payload: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "tool payload must be a JSON object"
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None, "tool payload is missing a non-empty name"
    if not isinstance(arguments, dict):
        return None, "tool arguments must be a JSON object"
    return {"name": name.strip(), "arguments": arguments}, None


def _parse_xml_tool_body(body: str) -> tuple[dict[str, Any] | None, str | None]:
    function_match = _FUNCTION_RE.fullmatch(body)
    if function_match is None:
        return None, "malformed XML tool payload"

    params_body = function_match.group("body")
    arguments: dict[str, Any] = {}
    cursor = 0
    for match in _PARAM_RE.finditer(params_body):
        if params_body[cursor : match.start()].strip():
            return None, "unexpected text outside XML parameter tags"
        name = match.group("name")
        if name in arguments:
            return None, f"duplicate XML parameter: {name}"
        arguments[name] = _maybe_json_value(match.group("value"))
        cursor = match.end()
    if params_body[cursor:].strip():
        return None, "unclosed or malformed XML parameter tag"
    return {"name": function_match.group("name"), "arguments": arguments}, None


def parse_assistant_action(text: str | None) -> ParsedAction:
    """Parse one assistant action while preserving malformed raw text.

    Tool calls and final answers must occupy the whole turn.  A completely
    tag-free turn is reported as ``no_action`` so callers can distinguish an
    empty model response from an action syntax error.  A final span embedded
    in prose is deliberately a protocol error; reward code can still score
    its answer content independently.
    """

    raw = text or ""
    candidate_count = count_candidate_actions(raw)
    has_marker = bool(_ACTION_MARKER_RE.search(raw))

    if candidate_count == 0 and not has_marker:
        if not raw.strip():
            return ParsedAction("no_action", candidate_action_count=0, raw=raw)
        return ParsedAction("no_action", candidate_action_count=0, reason="no complete action tag", raw=raw)

    if candidate_count != 1:
        return ParsedAction(
            "protocol_error",
            candidate_action_count=candidate_count,
            reason="assistant turn must contain exactly one action",
            raw=raw,
        )

    final_match = _FINAL_RE.fullmatch(raw)
    if final_match is not None:
        value = final_match.group("body").strip()
        if not value:
            return ParsedAction("protocol_error", candidate_action_count=1, reason="final answer is empty", raw=raw)
        return ParsedAction("final", value=value, candidate_action_count=1, raw=raw)

    tool_match = _TOOL_RE.fullmatch(raw)
    if tool_match is not None:
        body = tool_match.group("body").strip()
        if body.startswith("{"):
            value, reason = _parse_json_tool_body(body)
        else:
            value, reason = _parse_xml_tool_body(body)
        if value is None:
            return ParsedAction("protocol_error", candidate_action_count=1, reason=reason, raw=raw)
        return ParsedAction(
            "tool_call",
            value=value,
            candidate_action_count=1,
            raw=raw,
            parsed_tool_name=str(value.get("name")),
        )

    return ParsedAction(
        "protocol_error",
        candidate_action_count=1,
        reason="action tags are not a complete whole-turn match",
        raw=raw,
    )


# Short alias for callers that prefer the protocol vocabulary.
parse_action = parse_assistant_action


__all__ = ["ParsedAction", "count_candidate_actions", "parse_action", "parse_assistant_action"]
