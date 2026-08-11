"""Exact 96-dimensional Code observation feature contract."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Mapping

try:
    from ..config import CODE_BELIEF_FEATURE_DIM, CODE_ERROR_FAMILIES, CODE_STATUS_NAMES, CODE_TOOL_NAMES, stable_hash
except ImportError:  # pragma: no cover
    from config import CODE_BELIEF_FEATURE_DIM, CODE_ERROR_FAMILIES, CODE_STATUS_NAMES, CODE_TOOL_NAMES, stable_hash


GENERAL_NAMES = (
    "latency", "char_count", "token_count", "line_count", "result_bytes", "truncation_ratio", "returncode", "has_stderr",
    "has_error_keyword", "path_count", "file_count", "match_count", "test_count", "information_gain", "relative_cost", "output_nonempty",
)
STRUCTURE_NAMES = (
    "path_valid", "command_valid", "patch_valid", "diff_hunk_count", "changed_file_count", "added_line_count", "deleted_line_count",
    "search_match_density", "read_line_coverage", "test_summary_parseable", "test_pass_ratio", "syntax_error_present", "schema_valid",
    "duplicate_ratio", "structure_score", "missing_flag",
)
SEMANTIC_NAMES = (
    "query_term_overlap", "requested_path_match", "command_output_consistency", "diff_action_consistency", "git_status_consistency",
    "changed_trace_overlap", "repeated_output_ratio", "stderr_ratio", "success_signal", "failure_signal", "numeric_signal_count",
    "stacktrace_present", "compile_error_present", "observation_consistency", "semantic_score", "missing_flag",
)
TASK_NAMES = (
    "kind_bugfix", "kind_feature", "kind_refactor", "kind_other", "inspected_file_count", "touched_file_count", "tests_run_count",
    "failing_tests_seen", "remaining_budget", "patch_nonempty", "patch_size", "evidence_sufficient", "phase_localize", "phase_modify",
    "phase_validate", "phase_submit",
)
HISTORY_NAMES = (
    "consecutive_failures", "recent_surprise", "last_tool_index", "last_family_index", "call_index", "family_failure_count",
    "tool_failure_count", "recent_information_gain", "recent_status_error", "patch_churn_ratio",
)

CODE_BELIEF_FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"tool_{name}" for name in CODE_TOOL_NAMES]
    + [f"status_{name}" for name in CODE_STATUS_NAMES]
    + [f"error_{name}" for name in CODE_ERROR_FAMILIES]
    + list(GENERAL_NAMES)
    + list(STRUCTURE_NAMES)
    + list(SEMANTIC_NAMES)
    + list(TASK_NAMES)
    + list(HISTORY_NAMES)
)
assert len(CODE_BELIEF_FEATURE_NAMES) == CODE_BELIEF_FEATURE_DIM
CODE_BELIEF_FEATURE_SCHEMA_HASH = hashlib.sha256("\n".join(CODE_BELIEF_FEATURE_NAMES).encode("utf-8")).hexdigest()


def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def extract_code_belief_features(
    *,
    tool_name: str,
    result: Any,
    task_state: Any,
    history: Iterable[Mapping[str, Any]] = (),
    public_context: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    """Create features from public observation/history only."""

    public = dict(public_context or {})
    rows = list(history)
    output = str(getattr(result, "output", "") or "")
    status = str(getattr(result, "status", "empty") or "empty")
    error_family = str(getattr(result, "metadata", {}).get("error_family", "none") or "none")
    if error_family not in CODE_ERROR_FAMILIES:
        error_family = "none"
    values: list[float] = []
    values.extend(1.0 if tool_name == name else 0.0 for name in CODE_TOOL_NAMES)
    values.extend(1.0 if status == name else 0.0 for name in CODE_STATUS_NAMES)
    values.extend(1.0 if error_family == name else 0.0 for name in CODE_ERROR_FAMILIES)
    metadata = dict(getattr(result, "metadata", {}) or {})
    lines = output.splitlines()
    tokens = re.findall(r"\w+", output)
    latency = _clip(float(getattr(result, "latency_ms", 0.0) or 0.0) / 5000.0)
    truncation = 1.0 if bool(getattr(result, "partial", False) or metadata.get("partial")) else 0.0
    returncode = getattr(result, "returncode", None)
    general = [
        latency, _clip(len(output) / 16384.0), _clip(len(tokens) / 4096.0), _clip(len(lines) / 240.0), _clip(len(output.encode("utf-8")) / 65536.0),
        truncation, _clip((abs(int(returncode)) if isinstance(returncode, int) else 0) / 8.0), _clip(bool(getattr(result, "stderr", ""))),
        _clip(bool(re.search(r"error|fail|exception|traceback", output, re.I))), _clip(len(re.findall(r"[A-Za-z0-9_.-]+/", output)) / 50.0),
        _clip(metadata.get("file_count", 0) / 100.0), _clip(metadata.get("match_count", 0) / 50.0), _clip(len(re.findall(r"pass|fail|error", output, re.I)) / 30.0),
        _clip(metadata.get("information_gain", 0.0)), _clip(metadata.get("relative_cost", 1.0) / 4.0), _clip(bool(output)),
    ]
    diff_hunks = output.count("@@")
    changed_files = len(re.findall(r"(?:^|\n)diff --git ", output))
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    pass_count = len(re.findall(r"PASSED|passed|PASS", output))
    fail_count = len(re.findall(r"FAILED|failed|FAIL|ERROR|error", output))
    structure = [
        _clip(public.get("path_valid", 1.0)), _clip(public.get("command_valid", 1.0)), _clip(public.get("patch_valid", 1.0)),
        _clip(diff_hunks / 20.0), _clip(changed_files / 20.0), _clip(added / 500.0), _clip(deleted / 500.0),
        _clip(metadata.get("match_count", 0) / max(1, len(lines))), _clip(metadata.get("read_line_coverage", 0.0)),
        _clip(bool(metadata.get("test_summary_parseable", status in {"ok", "error"}))), _clip(pass_count / max(1, pass_count + fail_count)),
        _clip(bool(re.search(r"syntaxerror|syntax error", output, re.I))), _clip(public.get("schema_valid", 1.0)),
        _clip(1.0 - len(set(lines)) / max(1, len(lines))), _clip(sum(general[:4]) / 4.0), 1.0 if output == "" else 0.0,
    ]
    query_terms = set(re.findall(r"\w+", str(public.get("query", "")).lower()))
    output_terms = set(re.findall(r"\w+", output.lower()))
    overlap = len(query_terms & output_terms) / max(1, len(query_terms))
    requested = str(public.get("path", ""))
    semantic = [
        _clip(overlap), _clip(bool(requested and requested in output)), _clip(public.get("command_output_consistency", 0.5)),
        _clip(public.get("diff_action_consistency", 0.5)), _clip(public.get("git_status_consistency", 0.5)),
        _clip(public.get("changed_trace_overlap", 0.0)), _clip(sum(1 for row in rows if row.get("output") == output) / max(1, len(rows))),
        _clip(len(str(getattr(result, "stderr", "") or "")) / max(1, len(output))), _clip(status == "ok"), _clip(status in {"error", "timeout", "invalid"}),
        _clip(len(re.findall(r"\b\d+(?:\.\d+)?\b", output)) / 20.0), _clip(bool(re.search(r"traceback|stack trace", output, re.I))),
        _clip(bool(re.search(r"compile|syntax", output, re.I))), _clip(public.get("observation_consistency", 0.5)),
        _clip(sum(general[7:]) / max(1, len(general[7:]))), 0.0,
    ]
    task_kind = str(getattr(task_state, "task_kind", "other"))
    phase = str(getattr(task_state, "phase", "LOCALIZE"))
    task = [
        float(task_kind == "bugfix"), float(task_kind == "feature"), float(task_kind == "refactor"), float(task_kind not in {"bugfix", "feature", "refactor"}),
        _clip(len(getattr(task_state, "inspected_files", ())) / 100.0), _clip(len(getattr(task_state, "touched_files", ())) / 50.0),
        _clip(getattr(task_state, "tests_run_count", 0) / 30.0), _clip(getattr(task_state, "failing_tests_seen", 0) / 30.0),
        _clip(getattr(task_state, "remaining_tool_budget", 0) / 30.0), float(bool(getattr(task_state, "patch_nonempty", False))),
        _clip(getattr(task_state, "patch_size", 0) / 65536.0), float(bool(getattr(task_state, "evidence_sufficient", False))),
        float(phase == "LOCALIZE"), float(phase == "MODIFY"), float(phase == "VALIDATE"), float(phase == "SUBMIT"),
    ]
    failures = [row for row in rows if str(row.get("status", "")) in {"error", "timeout", "invalid"}]
    family_failures = [row for row in failures if row.get("family") == public.get("family")]
    history_features = [
        _clip(len(failures[-5:]) / 5.0), _clip(public.get("recent_surprise", 0.0)), _clip(CODE_TOOL_NAMES.index(tool_name) / 7.0 if tool_name in CODE_TOOL_NAMES else 0.0),
        _clip(public.get("family_index", 0) / 3.0), _clip(getattr(task_state, "call_index", len(rows)) / 30.0), _clip(len(family_failures) / 5.0),
        _clip(sum(1 for row in failures if row.get("tool") == tool_name) / 5.0), _clip(sum(float(row.get("information_gain", 0.0) or 0.0) for row in rows[-5:]) / 5.0),
        float(status in {"error", "timeout"}), _clip(public.get("patch_churn_ratio", 0.0)),
    ]
    values.extend(general + structure + semantic + task + history_features)
    assert len(values) == CODE_BELIEF_FEATURE_DIM, len(values)
    return tuple(_clip(value, -1.0, 1.0) for value in values)


__all__ = ["CODE_BELIEF_FEATURE_NAMES", "CODE_BELIEF_FEATURE_SCHEMA_HASH", "extract_code_belief_features"]
