"""Build coupled-world and meta-episode manifests without copying documents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from bayestool.config import default_config, stage_schedule
from bayestool.grouping import (
    K8RollingScheduler,
    QuestionRolloutPlan,
    make_question_rollout_plan,
    select_extra_variants,
)
from bayestool.identity import make_coupling_id
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


def _document_for_hash(document: str, document_root: str | Path | None = None) -> str:
    """Resolve deployment paths to local bytes without rewriting metadata paths."""

    if Path(document).is_file():
        return document
    if document_root:
        candidate = Path(document_root) / Path(document).name
        if candidate.is_file():
            return str(candidate)
    return document


def coupling_id(record: dict[str, Any], *, document_root: str | Path | None = None) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    document = str(
        record.get("document_path")
        or record.get("pdf_path")
        or metadata.get("document_path")
        or record.get("file_path")
        or ""
    )
    task_prompt = record.get("prompt") or record.get("question") or record.get("query") or ""
    task_id = str(record.get("id") or record.get("task_id") or "")
    return make_coupling_id(document_hash(_document_for_hash(document, document_root)), task_prompt, task_id)


def build_manifest(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    document_root: str | Path | None = None,
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
    # A rolling K8 policy is stateful across questions.  Keep independent
    # histories only when a manifest explicitly requests different policy
    # bounds; constructing a scheduler inside the question loop would reset
    # the rolling floor/ceiling on every question.
    rolling_schedulers: dict[tuple[float, float, float, int, int], K8RollingScheduler] = {}
    coupled: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        cid = coupling_id(record, document_root=document_root)
        metadata = dict(record.get("metadata") or {})
        document = (
            record.get("document_path")
            or record.get("pdf_path")
            or record.get("file_path")
            or metadata.get("document_path")
            or ""
        )
        document_digest = document_hash(_document_for_hash(str(document), document_root))
        sampling_context = {
            "page_count": record.get("page_count", metadata.get("page_count")),
            "tool_argument_capabilities": record.get(
                "tool_argument_capabilities", metadata.get("tool_argument_capabilities", {})
            ),
            "tool_budget": record.get("tool_budget", metadata.get("tool_budget", 8)),
        }
        question_id = str(
            record.get("question_id")
            or metadata.get("question_id")
            or record.get("id")
            or record.get("task_id")
            or cid
        )
        raw_plan = metadata.get("question_rollout_plan")
        extra_variants = metadata.get("bayestool_extra_variants", ())
        if not isinstance(extra_variants, (list, tuple)):
            extra_variants = ()
        if len(extra_variants) > 2:
            raise ValueError(
                f"{question_id}: bayestool_extra_variants already contains {len(extra_variants)} entries; "
                "select candidates explicitly before building the manifest"
            )
        variant_candidates = metadata.get("bayestool_variant_candidates", ())
        variant_selection: dict[str, Any] = {}
        if not extra_variants and isinstance(variant_candidates, (list, tuple)):
            extra_variants, variant_selection = select_extra_variants(
                question_id,
                variant_candidates,
                seed=seed + record_index,
                max_extra=2,
                exploration_probability=float(metadata.get("bayestool_variant_exploration_probability", 0.10) or 0.10),
            )
        requested_group_size = metadata.get("bayestool_group_size", getattr(config, "default_group_size", 4))
        k8_selection = None
        if isinstance(raw_plan, Mapping):
            # An upstream builder may already have frozen a plan.  Preserve
            # it and fail closed on a question-id mismatch instead of
            # silently replacing the producer's world/variant contract.
            plan = QuestionRolloutPlan.from_mapping(raw_plan)
            if plan.question_id != question_id:
                raise ValueError(
                    f"{question_id}: question_rollout_plan question_id {plan.question_id!r} does not match"
                )
        else:
            if isinstance(requested_group_size, str) and requested_group_size.casefold() == "rolling":
                target_ratio = float(metadata.get("bayestool_k8_target_ratio", 0.25) or 0.25)
                floor = float(metadata.get("bayestool_k8_floor", 0.0) or 0.0)
                ceiling = float(metadata.get("bayestool_k8_ceiling", 1.0) or 1.0)
                window = int(metadata.get("bayestool_k8_window", 32) or 32)
                scheduler_key = (target_ratio, floor, ceiling, window, int(seed))
                scheduler = rolling_schedulers.setdefault(
                    scheduler_key,
                    K8RollingScheduler(
                        target_ratio=target_ratio,
                        floor=floor,
                        ceiling=ceiling,
                        window=window,
                        seed=seed,
                    ),
                )
                requested_group_size, k8_selection = scheduler.choose(question_id, requested_k=4)
            plan = make_question_rollout_plan(
                question_id,
                policy_version=str(getattr(config, "policy_version", "bayestool-policy-v1")),
                seed=seed + record_index,
                group_size=int(requested_group_size),
                extra_variants=extra_variants,
            )
        selection_items = (
            [dict(item) for item in variant_selection.get("candidates", []) if isinstance(item, dict)]
            if isinstance(variant_selection, dict)
            else []
        )
        if k8_selection is not None:
            selection_items.append(dict(k8_selection))
        if selection_items:
            plan = replace(
                plan,
                variant_selection=tuple(selection_items),
            )
        worlds = []
        finalized_realizations = []
        for slot, realization in enumerate(plan.realizations):
            # Persist one latent realization spec.  K independent
            # continuations are generated from this frozen world at rollout
            # time; storing K copies would reintroduce replica-defined
            # semantics and inflate the manifest.
            spec = sample_tool_world(
                cid,
                world_slot=slot,
                replica_id=0,
                rollout_id=seed + record_index,
                config=config,
                sampling_context=sampling_context,
                world_slot_role=realization.world_slot_role,
                variant_id=realization.variant_id,
            )
            worlds.append(spec.to_dict())
            finalized_realizations.append(
                replace(realization, latent_world_id=spec.latent_world_id)
            )
        plan = replace(plan, realizations=tuple(finalized_realizations), latent_ids_finalized=True)
        metadata.update({
            "coupling_id": cid,
            "document_hash": document_digest,
            "question_id": question_id,
            "question_rollout_plan": plan.to_dict(),
            "bayestool_variant_selection": dict(variant_selection),
            "bayestool_k8_selection": k8_selection,
            "realization_count": plan.group_count,
            "records_per_question": plan.record_count,
            # These are compatibility diagnostics only.  They are not used
            # to infer the plan, group ids, or loss weights.
            "worlds_per_prompt": plan.group_count,
            "replicas_per_world": config.replicas_per_world,
            "legacy_replica_fields_semantics": "continuation_only;plan_is_authoritative",
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
                    world_slot=plan.group_count + index,
                    replica_id=0,
                    config=config,
                )
                for index in range(max(8, plan.group_count))
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
    parser.add_argument(
        "--document-root",
        type=Path,
        default=None,
        help="Local PDF root used only to resolve deployment paths for content hashing.",
    )
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
        document_root=args.document_root,
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
