import json

from data.runtime_manifest import materialize_public_manifest


def test_runtime_manifest_excludes_evaluator_private_fields(tmp_path):
    source = tmp_path / "source.jsonl"
    target = tmp_path / "runtime.jsonl"
    source.write_text(
        json.dumps(
            {
                "text": "fix the public bug",
                "environment": "code",
                "metadata": {
                    "public_instance": {"instance_id": "case-1", "problem_statement": "fix the public bug", "image_name": "local"},
                    "evaluator_private": {"test_patch": "secret test", "FAIL_TO_PASS": ["secret"]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert materialize_public_manifest(source, target) == 1
    runtime = json.loads(target.read_text(encoding="utf-8"))
    payload = json.dumps(runtime)
    assert "evaluator_private" not in payload
    assert "secret" not in payload
    assert runtime["metadata"]["public_instance"]["instance_id"] == "case-1"
