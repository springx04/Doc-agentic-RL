"""Public task state for a Code episode."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


PHASES = ("LOCALIZE", "MODIFY", "VALIDATE", "SUBMIT")


@dataclass
class CodeTaskStateView:
    instance_id: str
    problem_statement: str
    task_kind: str = "bugfix"
    phase: str = "LOCALIZE"
    inspected_files: set[str] = field(default_factory=set)
    touched_files: set[str] = field(default_factory=set)
    search_queries: list[str] = field(default_factory=list)
    tests_run_count: int = 0
    failing_tests_seen: int = 0
    patch_nonempty: bool = False
    patch_size: int = 0
    validation_since_last_edit: bool = False
    last_validation_returncode: int | None = None
    remaining_tool_budget: int = 24
    last_tool: str | None = None
    last_result_status: str | None = None
    evidence_sufficient: bool = False
    call_index: int = 0
    validation_runtime_ms: float = 0.0

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "problem_statement": self.problem_statement,
            "task_kind": self.task_kind,
            "phase": self.phase,
            "inspected_files": sorted(self.inspected_files),
            "touched_files": sorted(self.touched_files),
            "search_queries": list(self.search_queries[-20:]),
            "tests_run_count": self.tests_run_count,
            "failing_tests_seen": self.failing_tests_seen,
            "patch_nonempty": self.patch_nonempty,
            "patch_size": self.patch_size,
            "validation_since_last_edit": self.validation_since_last_edit,
            "last_validation_returncode": self.last_validation_returncode,
            "remaining_tool_budget": self.remaining_tool_budget,
            "last_tool": self.last_tool,
            "last_result_status": self.last_result_status,
            "evidence_sufficient": self.evidence_sufficient,
            "call_index": self.call_index,
        }

    def update(self, tool_name: str, result: Any, *, patch: str | None = None, elapsed_ms: float = 0.0) -> None:
        self.call_index += 1
        self.remaining_tool_budget = max(0, self.remaining_tool_budget - 1)
        self.last_tool = tool_name
        self.last_result_status = str(getattr(result, "status", "error"))
        metadata = dict(getattr(result, "metadata", {}) or {})
        if tool_name == "read_file" and result.status in {"ok", "partial"}:
            path = metadata.get("path")
            if path:
                self.inspected_files.add(str(path))
        if tool_name == "search_code":
            query = metadata.get("query")
            if query:
                self.search_queries.append(str(query))
        if tool_name in {"run_tests", "run_checks"}:
            self.tests_run_count += 1 if tool_name == "run_tests" else 0
            returncode = getattr(result, "returncode", None)
            self.last_validation_returncode = returncode
            self.validation_runtime_ms += float(getattr(result, "latency_ms", elapsed_ms) or 0.0)
            if returncode not in (None, 0):
                self.failing_tests_seen += _failure_count(getattr(result, "output", ""))
            self.validation_since_last_edit = True
        if tool_name == "apply_patch" and result.status == "ok":
            self.validation_since_last_edit = False
            self.last_validation_returncode = None
        if patch is not None:
            self.patch_nonempty = bool(patch.strip())
            self.patch_size = len(patch)
            touched = metadata.get("touched_files")
            if isinstance(touched, (list, tuple, set)):
                self.touched_files.update(str(item) for item in touched)
            elif tool_name == "apply_patch":
                self.touched_files.update(_patch_paths(patch))
        self.evidence_sufficient = bool(
            self.patch_nonempty and self.validation_since_last_edit and self.last_validation_returncode == 0
        )
        self.phase = self.compute_phase()

    def compute_phase(self) -> str:
        if not self.patch_nonempty:
            return "LOCALIZE"
        if not self.validation_since_last_edit:
            return "MODIFY"
        if self.last_validation_returncode != 0:
            return "VALIDATE"
        return "SUBMIT"

    def reset_for_new_task(self, *, instance_id: str, problem_statement: str, task_kind: str = "bugfix", tool_budget: int | None = None) -> None:
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.task_kind = task_kind
        self.phase = "LOCALIZE"
        self.inspected_files.clear()
        self.touched_files.clear()
        self.search_queries.clear()
        self.tests_run_count = 0
        self.failing_tests_seen = 0
        self.patch_nonempty = False
        self.patch_size = 0
        self.validation_since_last_edit = False
        self.last_validation_returncode = None
        if tool_budget is not None:
            self.remaining_tool_budget = int(tool_budget)
        self.last_tool = None
        self.last_result_status = None
        self.evidence_sufficient = False
        self.call_index = 0
        self.validation_runtime_ms = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodeTaskStateView":
        """Restore the public task state captured at a decision boundary."""

        state = cls(
            instance_id=str(value.get("instance_id") or "unknown-instance"),
            problem_statement=str(value.get("problem_statement") or ""),
            task_kind=str(value.get("task_kind") or "bugfix"),
            remaining_tool_budget=max(0, int(value.get("remaining_tool_budget", 0))),
        )
        state.phase = str(value.get("phase") or "LOCALIZE")
        if state.phase not in PHASES:
            raise ValueError(f"invalid Code task phase in checkpoint: {state.phase}")
        state.inspected_files = {str(item) for item in value.get("inspected_files", ())}
        state.touched_files = {str(item) for item in value.get("touched_files", ())}
        state.search_queries = [str(item) for item in value.get("search_queries", ())]
        for name in ("tests_run_count", "failing_tests_seen", "patch_size", "call_index"):
            setattr(state, name, max(0, int(value.get(name, 0))))
        state.patch_nonempty = bool(value.get("patch_nonempty", False))
        state.validation_since_last_edit = bool(value.get("validation_since_last_edit", False))
        raw_returncode = value.get("last_validation_returncode")
        state.last_validation_returncode = int(raw_returncode) if isinstance(raw_returncode, int) else None
        state.last_tool = str(value["last_tool"]) if value.get("last_tool") is not None else None
        state.last_result_status = str(value["last_result_status"]) if value.get("last_result_status") is not None else None
        state.evidence_sufficient = bool(value.get("evidence_sufficient", False))
        state.validation_runtime_ms = max(0.0, float(value.get("validation_runtime_ms", 0.0) or 0.0))
        if state.evidence_sufficient != (state.patch_nonempty and state.validation_since_last_edit and state.last_validation_returncode == 0):
            raise ValueError("Code task checkpoint has inconsistent evidence_sufficient")
        return state

    def to_dict(self) -> dict[str, Any]:
        value = self.to_prompt_dict()
        value.update({"inspected_files": sorted(self.inspected_files), "touched_files": sorted(self.touched_files)})
        return value


def _failure_count(output: str) -> int:
    matches = re.findall(r"(?:FAILED|ERROR|FAIL(?:URE)?)", str(output), re.I)
    return max(1, len(matches))


def _patch_paths(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in str(patch).splitlines():
        if line.startswith("+++ b/"):
            paths.add(line[6:].split("\t", 1)[0])
        elif line.startswith("+++ ") and line[4:].strip() != "/dev/null":
            paths.add(line[4:].split("\t", 1)[0])
    return paths


__all__ = ["CodeTaskStateView", "PHASES"]
