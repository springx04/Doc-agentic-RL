"""Code-only Bayes-ARPO batch conversion for Slime's generic FSDP actor.

Slime's default Bayes planner is intentionally Doc-specific.  This adapter
uses its public custom-conversion hook rather than changing that planner.  It
accepts only Code rollout metadata and fails closed on incomplete K-sibling or
four-world questions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


_REQUIRED_ROLES = frozenset({"healthy", "local_degradation", "shared_family_fault", "change"})


def _meta(sample: Any) -> dict[str, Any]:
    value = getattr(sample, "metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _valid_group(rows: list[Any]) -> tuple[bool, str]:
    if len(rows) not in {4, 8}:
        return False, "invalid_k"
    metadata = [_meta(row) for row in rows]
    expected = {int(value.get("decision_group_size", 0) or 0) for value in metadata}
    if expected != {len(rows)}:
        return False, "declared_k_mismatch"
    for key in ("instance_id", "latent_world_id", "repo_state_digest", "decision_prefix_hash", "coupling_id", "world_slot_role"):
        values = {str(value.get(key) or "") for value in metadata}
        if len(values) != 1 or "" in values:
            return False, f"group_{key}_mismatch"
    variants = {str(value.get("variant_id") or "") for value in metadata}
    if len(variants) != len(rows):
        return False, "duplicate_variant"
    if any(not bool(value.get("valid_for_rl", False)) or getattr(row, "remove_sample", False) for row, value in zip(rows, metadata, strict=True)):
        return False, "invalid_rollout"
    return True, "ok"


def convert_samples_to_train_data(args: Any, samples: list[Any]) -> dict[str, Any]:
    """Prepare full Code questions for generic Slime/FSDP training.

    Every accepted question contains exactly one valid K=4/K=8 decision group
    for each required hidden world. Invalid groups invalidate their entire
    question rather than receiving a synthetic baseline or cross-task repair.
    """

    groups: dict[str, list[Any]] = defaultdict(list)
    questions: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        metadata = _meta(sample)
        if metadata.get("environment") != "code":
            continue
        group_id = str(metadata.get("decision_group_id") or "")
        instance_id = str(metadata.get("instance_id") or "")
        if not group_id or not instance_id:
            continue
        groups[group_id].append(sample)
        questions[instance_id].append(sample)

    valid_groups: set[str] = set()
    violations: list[dict[str, Any]] = []
    for group_id, rows in groups.items():
        valid, reason = _valid_group(rows)
        if valid:
            valid_groups.add(group_id)
        else:
            violations.append({"group_id": group_id, "reason": reason, "size": len(rows)})

    accepted: list[Any] = []
    skipped_questions: list[dict[str, Any]] = []
    for instance_id, rows in questions.items():
        by_role: dict[str, set[str]] = defaultdict(set)
        coupling_ids: set[str] = set()
        for row in rows:
            metadata = _meta(row)
            group_id = str(metadata.get("decision_group_id") or "")
            by_role[str(metadata.get("world_slot_role") or "")].add(group_id)
            coupling_ids.add(str(metadata.get("coupling_id") or ""))
        all_groups = {group for values in by_role.values() for group in values}
        valid = (
            set(by_role) == _REQUIRED_ROLES
            and all(len(by_role[role]) == 1 for role in _REQUIRED_ROLES)
            and all_groups.issubset(valid_groups)
            and len(coupling_ids) == 1
            and "" not in coupling_ids
        )
        if not valid:
            skipped_questions.append({"instance_id": instance_id, "groups": sorted(all_groups)})
            continue
        accepted.extend(sorted(rows, key=lambda row: (str(_meta(row)["world_slot_role"]), str(_meta(row)["variant_id"]))))

    if not accepted:
        return {
            "tokens": [], "response_lengths": [], "rewards": [], "raw_reward": [], "truncated": [],
            "group_indices": [], "sample_indices": [], "loss_masks": [], "rollout_log_probs": [],
            "bayes_group_ids": [], "sibling_group_ids": [], "bayes_sibling_weights": [],
            "bayes_sibling_baselines": [], "bayes_advantages": [], "bayes_loss_weights": [],
            "bayes_batch_sizes": [], "bayes_question_ids": [],
            "bayes_grouping_report": {"environment": "code", "violations": violations, "skipped_questions": skipped_questions, "no_ready_questions": True},
            "advantages": [], "returns": [],
        }

    question_count = len({str(_meta(row)["instance_id"]) for row in accepted})
    group_rewards: dict[str, list[float]] = defaultdict(list)
    for row in accepted:
        group_rewards[str(_meta(row)["decision_group_id"])].append(float(row.get_reward_value(args)))
    baselines = {group: sum(values) / len(values) for group, values in group_rewards.items()}
    weights = [1.0 / (question_count * 4 * len(groups[str(_meta(row)["decision_group_id"])])) for row in accepted]
    advantages = [float(row.get_reward_value(args)) - baselines[str(_meta(row)["decision_group_id"])] for row in accepted]
    return {
        "tokens": [row.tokens for row in accepted],
        "response_lengths": [row.response_length for row in accepted],
        "rewards": [float(row.get_reward_value(args)) for row in accepted],
        "raw_reward": [float(row.get_reward_value(args)) for row in accepted],
        "truncated": [1 if str(getattr(row, "status", "")) == "Status.TRUNCATED" else 0 for row in accepted],
        "group_indices": [row.group_index for row in accepted],
        "sample_indices": [row.index for row in accepted],
        "loss_masks": [row.loss_mask for row in accepted],
        "rollout_log_probs": [row.rollout_log_probs for row in accepted],
        "bayes_group_ids": [str(_meta(row)["decision_group_id"]) for row in accepted],
        "sibling_group_ids": [str(_meta(row)["decision_group_id"]) for row in accepted],
        "bayes_sibling_weights": [1.0] * len(accepted),
        "bayes_sibling_baselines": [baselines[str(_meta(row)["decision_group_id"])] for row in accepted],
        "bayes_advantages": advantages,
        # FSDP's generic packer consumes these fields.  For Code Bayes-ARPO
        # the exact sibling-relative values are already computed globally
        # above, before any DP partitioning.
        "advantages": advantages,
        "returns": [float(row.get_reward_value(args)) for row in accepted],
        "bayes_loss_weights": weights,
        "bayes_batch_sizes": [len(accepted)],
        "bayes_question_ids": [str(_meta(row)["instance_id"]) for row in accepted],
        "bayes_grouping_report": {"environment": "code", "question_count": question_count, "valid_group_count": len(valid_groups), "violations": violations, "skipped_questions": skipped_questions},
    }


__all__ = ["convert_samples_to_train_data"]
