"""Build coupled-world and meta-episode manifests without copying documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from bayestool.config import default_config, stage_schedule
from bayestool.meta_episode import build_meta_episode
from bayestool.world import document_hash, sample_tool_world, sample_world_type


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("data") or value.get("records") or value.get("examples")
    if not isinstance(value, list):
        raise ValueError("input must be a JSON list or JSONL")
    return [dict(item) for item in value]


def coupling_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    document = str(
        record.get("document_path")
        or record.get("pdf_path")
        or metadata.get("document_path")
        or record.get("file_path")
        or ""
    )
    question = str(record.get("question") or record.get("query") or record.get("prompt") or "")
    task_id = str(record.get("id") or record.get("task_id") or "")
    payload = f"{document_hash(document)}|{question}|{task_id}"
    return "coupling-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def build_manifest(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    world_type_probabilities: Any = None,
    session_state_probabilities: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = default_config(enabled=True)
    def _probability_pairs(value: Any, label: str) -> tuple[tuple[str, float], ...]:
        raw = value
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            raw = tuple((str(key), float(value)) for key, value in raw.items())
        elif isinstance(raw, (list, tuple)):
            raw = tuple((str(item[0]), float(item[1])) for item in raw)
        else:
            raise ValueError(f"{label} probabilities must be a JSON object or pair list")
        if not raw:
            raise ValueError(f"{label} probabilities cannot be empty")
        return raw
    if world_type_probabilities is not None:
        config = replace(
            config,
            world_type_probabilities=_probability_pairs(world_type_probabilities, "world type"),
        )
    if session_state_probabilities is not None:
        config = replace(
            config,
            session_state_probabilities=_probability_pairs(session_state_probabilities, "session state"),
        )
    coupled: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        cid = coupling_id(record)
        metadata = dict(record.get("metadata") or {})
        document = (
            record.get("document_path")
            or record.get("pdf_path")
            or record.get("file_path")
            or metadata.get("document_path")
            or ""
        )
        sampling_context = {
            "page_count": record.get("page_count", metadata.get("page_count")),
            "tool_argument_capabilities": record.get(
                "tool_argument_capabilities", metadata.get("tool_argument_capabilities", {})
            ),
            "tool_budget": record.get("tool_budget", metadata.get("tool_budget", 8)),
        }
        worlds = []
        for slot in range(config.worlds_per_prompt):
            for replica_id in range(config.replicas_per_world):
                spec = sample_tool_world(
                    cid,
                    world_slot=slot,
                    replica_id=replica_id,
                    rollout_id=seed + record_index,
                    config=config,
                    sampling_context=sampling_context,
                )
                worlds.append(spec.to_dict())
        metadata.update({
            "coupling_id": cid,
            "document_hash": document_hash(document),
            "worlds_per_prompt": config.worlds_per_prompt,
            "replicas_per_world": config.replicas_per_world,
            "world_sampling_context": sampling_context,
            "fixed_world_specs": worlds,
            "world_type_probabilities": dict(config.world_type_probabilities),
            "session_state_probabilities": dict(config.session_state_probabilities),
            "bayestool_stage_schedule": [
                {
                    "name": item.name,
                    "objective": item.objective,
                    "runtime_mode": item.runtime_mode,
                    "branch_probability": item.branch_probability,
                    "use_meta_episode": item.use_meta_episode,
                    "use_persistent_session_belief": item.use_persistent_session_belief,
                    "expected_update_fraction": item.expected_update_fraction,
                }
                for item in stage_schedule()
            ],
            "sampled_training_world_types": [
                sample_world_type(
                    cid,
                    rollout_id=seed + record_index,
                    world_slot=config.worlds_per_prompt + index,
                    replica_id=0,
                    config=config,
                )
                for index in range(max(8, config.worlds_per_prompt * config.replicas_per_world))
            ],
        })
        output = dict(record)
        output["metadata"] = metadata
        coupled.append(output)

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        document = str(
            record.get("document_path")
            or record.get("pdf_path")
            or record.get("file_path")
            or record.get("metadata", {}).get("document_path")
            or ""
        )
        by_document[document].append(record)
    meta_episodes = []
    for document, group in sorted(by_document.items()):
        if len(group) < config.meta.questions_per_episode_min:
            continue
        episode = build_meta_episode(group, config=config, seed=seed)
        meta_episodes.append({
            "meta_episode_id": episode.meta_episode_id,
            "episode_content_id": episode.episode_content_id,
            "document_path": document,
            "meta_trajectory_ids": [
                f"{episode.meta_episode_id}:q{index}" for index in range(len(episode.questions))
            ],
            "questions": [
                {
                    "prompt": question.prompt,
                    "label": question.label,
                    "metadata": question.metadata,
                    "meta_episode_id": episode.meta_episode_id,
                    "episode_content_id": episode.episode_content_id,
                    "meta_trajectory_id": question.metadata.get(
                        "meta_trajectory_id", f"{episode.meta_episode_id}:q{index}"
                    ),
                    "question_index": index,
                }
                for index, question in enumerate(episode.questions)
            ],
        })
    return coupled, meta_episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--world-type-probabilities",
        default=None,
        help="JSON object or pair list controlling extra training-world sampling, e.g. '{\"healthy\":0.2,...}'",
    )
    parser.add_argument(
        "--session-state-probabilities",
        default=None,
        help="JSON object or pair list controlling session-state sampling",
    )
    args = parser.parse_args(argv)
    records = _read_records(args.input)
    coupled, meta = build_manifest(
        records,
        seed=args.seed,
        world_type_probabilities=args.world_type_probabilities,
        session_state_probabilities=args.session_state_probabilities,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in coupled:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.meta_output:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"coupled_records": len(coupled), "meta_episodes": len(meta)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
