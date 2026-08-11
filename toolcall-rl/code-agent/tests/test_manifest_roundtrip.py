from __future__ import annotations

import json

from data.preprocess_swe import preprocess
from data.schema import SWEInstance


def test_preprocessed_code_manifest_roundtrips_public_and_private_fields() -> None:
    row = preprocess(
        [
            {
                "instance_id": "demo-1",
                "problem_statement": "Fix the failing implementation.",
                "image_name": "local/demo:latest",
                "eval_script": "python -m pytest -q",
                "patch": "private gold patch",
            }
        ]
    )[0]

    restored = SWEInstance.from_raw(row)

    assert restored.public.instance_id == "demo-1"
    assert restored.public.problem_statement == "Fix the failing implementation."
    assert restored.evaluator_private.values["eval_script"] == "python -m pytest -q"
    assert restored.evaluator_private.values["patch"] == "private gold patch"

    policy_visible = json.dumps(restored.to_runtime_row(), ensure_ascii=False)
    assert "eval_script" not in policy_visible
    assert "private gold patch" not in policy_visible
