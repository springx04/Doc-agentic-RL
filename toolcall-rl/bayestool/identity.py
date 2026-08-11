"""Shared identity helpers for BayesTool data builders and rollouts."""

from __future__ import annotations

import hashlib
import re
from typing import Any


_QUESTION_MARKER = re.compile(
    r"(?:^|\n)Question:\s*(.*?)(?:\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def canonical_task_question(task_prompt: Any) -> str:
    """Return the question portion used by the runtime coupling identity."""

    if isinstance(task_prompt, list):
        users = [
            str(message.get("content", ""))
            for message in task_prompt
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        text = users[-1] if users else str(task_prompt)
    else:
        text = str(task_prompt or "")
    match = _QUESTION_MARKER.search(text)
    return match.group(1).strip() if match else text


def make_coupling_id(document_digest: str, task_prompt: Any, task_id: Any = "") -> str:
    """Build the canonical question-scoped coupling identity."""

    payload = "|".join(
        (
            str(document_digest),
            canonical_task_question(task_prompt),
            str(task_id or ""),
        )
    )
    return "coupling-" + hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:24]
