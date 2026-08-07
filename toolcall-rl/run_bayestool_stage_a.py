"""Run the executable BayesTool Stage-A replay/training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_bayestool_replay import export_replay
from train_bayestool_belief import main as train_belief


def run_stage_a(
    artifact: Path,
    output_dir: Path,
    *,
    epochs: int,
    device: str,
    fit_q: bool,
    fit_risk: bool,
    risk_label_key: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "belief_replay.jsonl"
    replay_manifest = export_replay(artifact, replay_path, split="train")
    belief_path = output_dir / "belief_filter.pt"
    q_path = output_dir / "bayes_q_head.pt"
    risk_path = output_dir / "answer_risk.json"
    argv = [
        "--input",
        str(replay_path),
        "--output",
        str(belief_path),
        "--epochs",
        str(max(1, epochs)),
        "--device",
        device,
    ]
    if fit_q:
        argv.extend(["--q-replay", str(replay_path), "--q-output", str(q_path)])
    if fit_risk:
        argv.extend(
            [
                "--risk-validation",
                str(replay_path),
                "--risk-output",
                str(risk_path),
                "--risk-label-key",
                risk_label_key,
            ]
        )
    train_belief(argv)
    manifest = {
        "schema_version": "bayestool-capability-manifest-v1",
        "stage": "a",
        "artifact": str(artifact),
        "replay": replay_manifest,
        "replay_path": str(replay_path),
        "checkpoints": {
            "belief": str(belief_path),
            "q": str(q_path) if fit_q else None,
            "risk": str(risk_path) if fit_risk else None,
        },
        "capabilities": {
            "canonical_replay": True,
            "belief": belief_path.exists(),
            "q": q_path.exists() if fit_q else False,
            "risk": risk_path.exists() if fit_risk else False,
        },
        "fit_options": {
            "epochs": max(1, epochs),
            "device": device,
            "fit_q": fit_q,
            "fit_risk": fit_risk,
            "risk_label_key": risk_label_key,
        },
    }
    target = manifest_path or output_dir / "bayestool_capability_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="rollout_interactions.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit-q", action="store_true")
    parser.add_argument("--fit-risk", action="store_true")
    parser.add_argument("--risk-label-key", default="answer_correct")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    run_stage_a(
        args.artifact,
        args.output_dir,
        epochs=args.epochs,
        device=args.device,
        fit_q=args.fit_q,
        fit_risk=args.fit_risk,
        risk_label_key=args.risk_label_key,
        manifest_path=args.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
