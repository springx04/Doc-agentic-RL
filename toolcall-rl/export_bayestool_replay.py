"""Export real rollout artifacts into strict BayesTool trainer JSONL.

The RL worker writes one ``rollout_interactions.json`` artifact containing
serialized train/eval payloads.  Stage-A trainers consume JSONL, so this
adapter keeps the split explicit, preserves the canonical replay and the
branch/Q records, and fails closed on malformed replay instead of silently
training on a partial trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from bayestool.replay import validate_canonical_replay


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _payloads(artifact: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(artifact, Mapping) and isinstance(artifact.get("records"), list):
        records = artifact["records"]
    elif isinstance(artifact, Mapping):
        records = [artifact]
    elif isinstance(artifact, list):
        records = artifact
    else:
        raise ValueError("rollout artifact must be a JSON object or list")

    for record in records:
        if not isinstance(record, Mapping):
            continue
        payload = record.get("payload", record)
        if not isinstance(payload, Mapping):
            continue
        source = str(record.get("source") or payload.get("source") or "train")
        yield source, payload


def _sample_reward(sample: Mapping[str, Any]) -> dict[str, Any]:
    value = sample.get("reward")
    return dict(value) if isinstance(value, Mapping) else {}


def _sample_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    value = sample.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _is_eval_source(source: str) -> bool:
    value = source.casefold()
    return "eval" in value or "test" in value or value.endswith(".pt") and "/eval" in value


def _trainer_record(
    sample: Mapping[str, Any],
    *,
    source: str,
    sample_index: int,
) -> dict[str, Any] | None:
    metadata = _sample_metadata(sample)
    replay = sample.get("belief_replay")
    if not isinstance(replay, Mapping):
        replay = metadata.get("belief_replay")
    if not isinstance(replay, Mapping):
        return None
    replay = dict(replay)
    errors = validate_canonical_replay(replay)
    if errors:
        raise ValueError(
            f"invalid canonical replay for source={source!r} sample={sample_index}: "
            + "; ".join(errors)
        )

    reward = _sample_reward(sample)
    answer_correct = None
    for key in ("answer_correct", "exact_acc", "acc", "quality"):
        if key in reward:
            answer_correct = reward[key]
            break
    # Keep only JSON-serializable runtime metadata needed by the trainers and
    # risk/Q calibrators.  The full response/token arrays remain in the RL
    # artifact and are intentionally not duplicated into the replay JSONL.
    selected_metadata: dict[str, Any] = {
        key: metadata[key]
        for key in (
            "rollout_id",
            "world_id",
            "latent_world_id",
            "latent_seed",
            "world_slot",
            "replica_id",
            "coupling_id",
            "sibling_group_id",
            "navigation_state",
            "bayestool",
            "bayes_supervision",
            "bayes_branch_events",
            "branch_events",
            "q_records",
            "branch_records",
            "bayes_branch_records",
            "task_features",
            "risk_label",
            "answer_correct",
            "answer_error",
            "valid_for_rl",
            "rollout_status",
        )
        if key in metadata
    }
    for key in ("q_records", "branch_records", "bayes_branch_records", "task_features"):
        if key not in selected_metadata and key in sample:
            selected_metadata[key] = sample[key]
    if answer_correct is not None:
        selected_metadata["answer_correct"] = answer_correct
    record: dict[str, Any] = {
        "trajectory_id": str(
            replay.get("trajectory_id")
            or metadata.get("rollout_id")
            or f"sample-{sample_index}"
        ),
        "rollout_id": metadata.get("rollout_id"),
        "source": source,
        "belief_replay": replay,
        "metadata": selected_metadata,
        "reward": reward,
        "valid_for_rl": sample.get("valid_for_rl", metadata.get("valid_for_rl", True)),
        "rollout_status": sample.get("rollout_status", metadata.get("rollout_status")),
    }
    if answer_correct is not None:
        record["answer_correct"] = answer_correct
    return record


def export_replay(
    input_path: Path,
    output_path: Path,
    *,
    split: str = "train",
) -> dict[str, Any]:
    artifact = _read_json(input_path)
    records: list[dict[str, Any]] = []
    payload_count = 0
    skipped_without_replay = 0
    for source, payload in _payloads(artifact):
        is_eval = _is_eval_source(source)
        if split == "train" and is_eval:
            continue
        if split == "eval" and not is_eval:
            continue
        payload_count += 1
        samples = payload.get("samples", [])
        if not isinstance(samples, list):
            continue
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                continue
            record = _trainer_record(sample, source=source, sample_index=sample_index)
            if record is None:
                skipped_without_replay += 1
                continue
            records.append(record)
    if not records:
        raise ValueError(
            f"no canonical replay records found in {input_path} for split={split!r}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "bayestool-stage-a-replay-v1",
        "input": str(input_path),
        "output": str(output_path),
        "split": split,
        "payload_count": payload_count,
        "record_count": len(records),
        "skipped_without_replay": skipped_without_replay,
        "event_count": sum(
            int(record["belief_replay"].get("event_count", 0) or 0)
            for record in records
        ),
        "hidden_label_count": sum(
            sum(
                bool(event.get("hidden_label"))
                for event in record["belief_replay"].get("events", [])
                if isinstance(event, Mapping)
            )
            for record in records
        ),
        "q_source_count": sum(
            any(
                key in record or key in record.get("metadata", {})
                for key in ("q_records", "branch_records", "bayes_branch_records", "task_features")
            )
            for record in records
        ),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval", "all"), default="train")
    args = parser.parse_args(argv)
    print(json.dumps(export_replay(args.input, args.output, split=args.split), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
