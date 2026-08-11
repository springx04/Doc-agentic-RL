"""Strict parsing for assistant tool-call and final-answer turns.

The rollout protocol is deliberately small: one assistant turn contains one
complete ``<tool_call>...</tool_call>``, ``<final>...</final>`` or
``<abstain>...</abstain>`` action.  All three
actions must occupy the whole turn; reward evaluation may separately recover a
single final span to distinguish semantic correctness from protocol validity.
This module never searches for a usable tool call inside arbitrary prose.  That
is important because tool instructions and previous observations can
themselves contain the protocol tags as examples.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


_ACTION_TAG_RE = re.compile(r"<\s*(tool_call|final|abstain)\b[^>]*>", re.IGNORECASE)
_ACTION_MARKER_RE = re.compile(r"</?\s*(tool_call|final|abstain)\b", re.IGNORECASE)
_FINAL_RE = re.compile(r"\s*<final>(?P<body>.*?)</final>\s*\Z", re.IGNORECASE | re.DOTALL)
_ABSTAIN_RE = re.compile(r"\s*<abstain>(?P<body>.*?)</abstain>\s*\Z", re.IGNORECASE | re.DOTALL)
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
_FUNCTION_CALL_RE = re.compile(
    r"\s*(?P<name>[A-Za-z_][\w.-]*)\s*\((?P<body>.*?)\)\s*\Z",
    re.DOTALL,
)
_FUNCTION_JSON_RE = re.compile(
    r"\s*(?P<name>[A-Za-z_][\w.-]*)\s*(?P<body>\{.*\})\s*\Z",
    re.DOTALL,
)

_DOCUMENT_TOOL_NAMES = frozenset(
    {
        "render_page",
        "crop_region",
        "zoom_region",
        "parse_document",
        "detect_layout",
        "ocr_region",
        "extract_table",
        "chart_to_table",
    }
)
_DOCUMENT_PATH_ALIASES = ("path", "file_path", "pdf_path", "document", "filename", "name")


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
        return self.kind in {"tool_call", "final", "abstain"}


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


def _literal_ast_value(node: ast.AST, source: str | None = None) -> tuple[Any, str | None]:
    """Evaluate only Python literal syntax used by function-style calls."""

    if source is not None:
        try:
            return json.loads(source), None
        except (TypeError, json.JSONDecodeError):
            pass
    try:
        return ast.literal_eval(node), None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        return None, f"function-style argument must be a literal: {exc}"


def _parse_function_call_body(body: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a single structured call such as ``parse_document(path=...)``.

    The input is parsed as an AST and literal-evaluated; it is never executed.
    A single positional object is supported for model formats that emit
    ``tool({"key": "value"})`` in place of the JSON envelope.
    """

    try:
        expression = ast.parse(body, mode="eval").body
    except SyntaxError as exc:
        return None, f"invalid function-style tool payload: {exc.msg}"
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return None, "function-style tool payload must be one direct function call"
    if not re.fullmatch(r"[A-Za-z_][\w.-]*", expression.func.id):
        return None, "function-style tool name is malformed"
    if len(expression.args) > 1:
        return None, "function-style tool call accepts at most one positional object"

    arguments: dict[str, Any] = {}
    if expression.args:
        if expression.args[0].__class__ is not ast.Dict:
            return None, "function-style positional argument must be an object"
        value, reason = _literal_ast_value(
            expression.args[0],
            ast.get_source_segment(body, expression.args[0]),
        )
        if reason is not None:
            return None, reason
        if not isinstance(value, dict):
            return None, "function-style positional argument must be an object"
        arguments.update(value)

    for keyword in expression.keywords:
        if keyword.arg is None:
            return None, "function-style tool call does not allow **arguments"
        if keyword.arg in arguments:
            return None, f"duplicate function-style argument: {keyword.arg}"
        value, reason = _literal_ast_value(
            keyword.value,
            ast.get_source_segment(body, keyword.value),
        )
        if reason is not None:
            return None, reason
        arguments[keyword.arg] = value

    return {"name": expression.func.id, "arguments": arguments}, None


def _parse_function_style_tool_body(body: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the bounded function-call variants emitted by some Qwen models."""

    function_match = _FUNCTION_CALL_RE.fullmatch(body)
    if function_match is not None:
        return _parse_function_call_body(body.strip())

    # Some model templates omit the parentheses around a JSON object:
    # ``parse_document{"page": 0}``.  Keep this branch JSON-only so arbitrary
    # prose or Python expressions cannot become executable tool arguments.
    json_match = _FUNCTION_JSON_RE.fullmatch(body)
    if json_match is None:
        return None, "malformed function-style tool payload"
    try:
        arguments = json.loads(json_match.group("body"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON function-style arguments: {exc.msg}"
    if not isinstance(arguments, dict):
        return None, "function-style JSON arguments must be an object"
    return {"name": json_match.group("name"), "arguments": arguments}, None


def _move_argument_alias(
    arguments: dict[str, Any],
    canonical_name: str,
    aliases: tuple[str, ...],
) -> str | None:
    """Move one explicitly supported alias and reject ambiguous duplicates."""

    present = [alias for alias in aliases if alias in arguments]
    if canonical_name in arguments and present:
        return f"both {canonical_name} and its alias {present[0]} were provided"
    if len(present) > 1:
        return f"multiple aliases for {canonical_name} were provided: {', '.join(present)}"
    if present:
        arguments[canonical_name] = arguments.pop(present[0])
    return None


def _normalize_page_alias(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if "page" not in arguments:
        return None
    canonical_name = "page_numbers" if tool_name == "parse_document" else "page_number"
    if canonical_name in arguments:
        return f"both {canonical_name} and its alias page were provided"
    page = arguments.pop("page")
    if tool_name == "parse_document":
        arguments[canonical_name] = page if isinstance(page, list) else [page]
    else:
        arguments[canonical_name] = page
    return None


def normalize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize only documented compatibility aliases into tool-schema fields."""

    if not isinstance(arguments, dict):
        return None, "tool arguments must be an object"
    normalized = dict(arguments)
    tool_name = str(tool_name).strip()
    if tool_name in _DOCUMENT_TOOL_NAMES:
        reason = _move_argument_alias(normalized, "document_path", _DOCUMENT_PATH_ALIASES)
        if reason is not None:
            return None, reason
        reason = _normalize_page_alias(tool_name, normalized)
        if reason is not None:
            return None, reason
        if "box" in normalized:
            reason = _move_argument_alias(normalized, "bbox", ("box",))
            if reason is not None:
                return None, reason
    return normalized, None


def parse_assistant_action(text: str | None) -> ParsedAction:
    """Parse one assistant action while preserving malformed raw text.

    Tool calls, final answers, and abstentions must occupy the whole turn.  A completely
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

    abstain_match = _ABSTAIN_RE.fullmatch(raw)
    if abstain_match is not None:
        value = abstain_match.group("body").strip()
        if not value:
            return ParsedAction(
                "protocol_error",
                candidate_action_count=1,
                reason="abstention reason is empty",
                raw=raw,
            )
        return ParsedAction("abstain", value=value, candidate_action_count=1, raw=raw)

    tool_match = _TOOL_RE.fullmatch(raw)
    if tool_match is not None:
        body = tool_match.group("body").strip()
        if body.startswith("{"):
            value, reason = _parse_json_tool_body(body)
        elif body.casefold().startswith("<function="):
            value, reason = _parse_xml_tool_body(body)
        else:
            value, reason = _parse_function_style_tool_body(body)
        if value is None:
            return ParsedAction("protocol_error", candidate_action_count=1, reason=reason, raw=raw)
        normalized_arguments, normalization_reason = normalize_tool_arguments(
            str(value.get("name") or ""),
            dict(value.get("arguments") or {}),
        )
        if normalized_arguments is None:
            return ParsedAction(
                "protocol_error",
                candidate_action_count=1,
                reason=normalization_reason,
                raw=raw,
            )
        value = {"name": str(value.get("name") or "").strip(), "arguments": normalized_arguments}
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


__all__ = [
    "ParsedAction",
    "count_candidate_actions",
    "normalize_tool_arguments",
    "parse_action",
    "parse_assistant_action",
]
