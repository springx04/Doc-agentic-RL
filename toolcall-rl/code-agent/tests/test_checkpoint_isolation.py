import json

import pytest

from bayestool.belief import CodeBeliefFilter, load_belief_checkpoint, save_belief_checkpoint
from bayestool.q_model import CodeQModel, load_q_checkpoint, save_q_checkpoint
from bayestool.risk_model import CodeRiskModel, load_risk_checkpoint, save_risk_checkpoint


def test_code_belief_rejects_doc_checkpoint(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"metadata": {"environment": "doc", "checkpoint_type": "belief", "feature_dim": 96, "feature_schema_hash": "x"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_belief_checkpoint(path)


def test_code_q_and_risk_checkpoints_have_independent_gates(tmp_path):
    q_path, risk_path = tmp_path / "q.json", tmp_path / "risk.json"
    save_q_checkpoint(q_path, CodeQModel())
    save_risk_checkpoint(risk_path, CodeRiskModel())
    assert load_q_checkpoint(q_path)["metadata"]["environment"] == "code"
    assert load_risk_checkpoint(risk_path)["metadata"]["environment"] == "code"
