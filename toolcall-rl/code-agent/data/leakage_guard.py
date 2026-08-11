"""Keep evaluator-only SWE labels out of Code rollout decision modules."""

from __future__ import annotations

import re
from typing import Any, Mapping


HIDDEN_CODE_INSTANCE_KEYS = frozenset(
    {
        "patch",
        "gold_patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "fail_to_pass",
        "pass_to_pass",
        "resolved",
        "official_resolved",
        "verifier_result",
        "evaluation_result",
        "eval_script",
        "test_command",
    }
)
_HIDDEN_KEY_RE = re.compile(r"gold|fail.?to.?pass|pass.?to.?pass|verifier|official|resolved|eval_script|test_patch", re.I)


def is_hidden_key(key: str) -> bool:
    text = str(key)
    return text in HIDDEN_CODE_INSTANCE_KEYS or bool(_HIDDEN_KEY_RE.search(text))


def split_public_private(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    for key, value in raw.items():
        if is_hidden_key(str(key)):
            private[str(key)] = value
        elif isinstance(value, Mapping):
            child_public, child_private = split_public_private(value)
            if child_public:
                public[str(key)] = child_public
            if child_private:
                private[str(key)] = child_private
        elif isinstance(value, list) and any(isinstance(item, Mapping) and split_public_private(item)[1] for item in value):
            public_items = []
            private_items = []
            for item in value:
                if isinstance(item, Mapping):
                    item_public, item_private = split_public_private(item)
                    public_items.append(item_public)
                    private_items.append(item_private)
                else:
                    public_items.append(item)
            public[str(key)] = public_items
            private[str(key)] = private_items
        else:
            public[str(key)] = value
    return public, private


def assert_public_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_hidden_key(str(key)):
                raise ValueError(f"hidden Code instance key leaked into public view: {key}")
            assert_public_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_public_safe(item)


def public_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    public, _ = split_public_private(raw)
    assert_public_safe(public)
    return public


__all__ = ["HIDDEN_CODE_INSTANCE_KEYS", "assert_public_safe", "is_hidden_key", "public_view", "split_public_private"]
