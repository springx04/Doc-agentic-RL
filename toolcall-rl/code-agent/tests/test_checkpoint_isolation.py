import json

import pytest

from bayestool.belief import CodeBeliefFilter, load_belief_checkpoint, save_belief_checkpoint
from bayestool.q_model import CodeQModel, load_q_checkpoint, save_q_checkpoint
from bayestool.risk_model import CodeRiskModel, load_risk_checkpoint, save_risk_checkpoint
from data.manifests import CodeManifest, load_and_validate_capabilities


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


def test_capability_manifest_loads_and_validates_real_code_checkpoints(tmp_path):
    belief_path = tmp_path / "belief.json"
    q_path = tmp_path / "q.json"
    risk_path = tmp_path / "risk.json"
    save_belief_checkpoint(belief_path, belief=CodeBeliefFilter())
    save_q_checkpoint(q_path, CodeQModel())
    save_risk_checkpoint(risk_path, CodeRiskModel())
    manifest_path = tmp_path / "capabilities.json"
    CodeManifest(
        stage="C",
        base_model="local-smoke",
        belief_checkpoint=belief_path.name,
        q_checkpoint=q_path.name,
        risk_checkpoint=risk_path.name,
    ).save(manifest_path)

    loaded = load_and_validate_capabilities(manifest_path, stage="C")

    assert loaded.belief_checkpoint == str(belief_path)
    assert loaded.q_checkpoint == str(q_path)
    assert loaded.risk_checkpoint == str(risk_path)
