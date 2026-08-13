import json
from types import SimpleNamespace

from eval_adapter import log_eval_rollout_data


def test_eval_adapter_writes_only_public_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_OUTPUT_DIR", str(tmp_path))
    samples = [
        SimpleNamespace(metadata={"public_instance": {"instance_id": f"eval-{index}"}, "code_trajectory": {"resolved": index == 0, "valid_for_rl": True, "termination_reason": "final", "tool_calls_used": 3}}, reward=1.0)
        for index in range(50)
    ]
    assert log_eval_rollout_data(0, None, {"code_swe_eval": {"samples": samples}}) is False
    path = tmp_path / "eval" / "final_checkpoint_eval_0.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 50
    assert rows[0]["instance_id"] == "eval-0"
    assert "evaluator_private" not in path.read_text(encoding="utf-8")
