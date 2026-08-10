from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayestool.grouping import (
    WORLD_SLOT_ROLES,
    compute_hierarchical_loss_weights,
    make_question_rollout_plan,
    validate_bayestool_question_records,
)
from bayestool.training import make_bayestool_decision_group_id
from bayestool.world import WorldRuntime, sample_tool_world


def _records(
    question: str = "q",
    *,
    group_size: int = 4,
    extra_variants: tuple[tuple[str, str], ...] = (),
):
    plan = make_question_rollout_plan(
        question,
        group_size=group_size,
        extra_variants=[{"world_slot_role": role, "variant_id": variant} for role, variant in extra_variants],
    )
    records = []
    sample_index = 0
    for realization in plan.realizations:
        base = {
            "question_id": question,
            "latent_world_id": realization.latent_world_id or f"{question}:{realization.world_slot_role}:{realization.variant_id}",
            "initial_input_hash": f"input:{question}",
            "decision_event_id": realization.selected_decision_event,
            "selected_decision_event": realization.selected_decision_event,
            "decision_prefix_hash": realization.decision_prefix_hash or "root",
            "runtime_state_digest": realization.runtime_state_digest or "runtime:frozen",
            "world_slot_role": realization.world_slot_role,
            "variant_id": realization.variant_id,
            "policy_version": realization.policy_version,
            "decision_group_size": realization.k,
            "coupling_id": question,
        }
        group_id = make_bayestool_decision_group_id(base)
        for _ in range(realization.k):
            metadata = {**base, "decision_group_id": group_id, "sample_id": f"sample-{sample_index}"}
            records.append({"metadata": metadata})
            sample_index += 1
    return plan, records


def test_explicit_plan_sizes_cover_r4k4_and_r6k8():
    assert make_question_rollout_plan("q", group_size=4).record_count == 16
    assert make_question_rollout_plan(
        "q", group_size=8, extra_variants=["v1", "v2"]
    ).record_count == 48


def test_primary_realization_index_maps_four_required_world_roles():
    plan = make_question_rollout_plan("world-map", group_size=4)
    fixed_specs = [
        sample_tool_world(
            "world-map-coupling",
            world_slot=index,
            replica_id=0,
            rollout_id=7,
            world_slot_role=realization.world_slot_role,
            variant_id=realization.variant_id,
        ).to_dict()
        for index, realization in enumerate(plan.realizations)
    ]
    runtimes = [
        WorldRuntime.for_sample(
            coupling_id="world-map-coupling",
            sample_index=0,
            realization_index=index,
            rollout_id=999,
            question_rollout_plan=plan,
            fixed_world_specs=fixed_specs,
        )
        for index in range(plan.group_count)
    ]
    assert [runtime.spec.world_slot_role for runtime in runtimes] == list(WORLD_SLOT_ROLES)
    assert [runtime.spec.world_slot for runtime in runtimes] == list(range(4))
    assert all(runtime.spec.replica_id == 0 for runtime in runtimes)


def test_explicit_manifest_wins_over_legacy_world_type_hint():
    plan = make_question_rollout_plan("manifest-authority", group_size=4)
    realization = plan.realizations[1]
    fixed_specs = [
        sample_tool_world(
            "manifest-coupling",
            world_slot=index,
            replica_id=0,
            rollout_id=7,
            world_slot_role=item.world_slot_role,
            variant_id=item.variant_id,
        )
        for index, item in enumerate(plan.realizations)
    ]
    fixed_spec = fixed_specs[1]
    runtime = WorldRuntime.for_sample(
        coupling_id="manifest-coupling",
        sample_index=0,
        realization_index=1,
        rollout_id=999,
        question_rollout_plan=plan,
        fixed_world_specs=[item.to_dict() for item in fixed_specs],
        world_type="healthy",
    )
    assert runtime.spec.world_slot_role == realization.world_slot_role
    assert runtime.spec.latent_world_id == fixed_spec.latent_world_id


def test_expanded_record_offset_still_materializes_siblings_from_one_latent_spec():
    plan = make_question_rollout_plan("record-map", group_size=4)
    fixed_specs = [
        sample_tool_world(
            "record-map-coupling",
            world_slot=index,
            replica_id=0,
            rollout_id=11,
            world_slot_role=realization.world_slot_role,
            variant_id=realization.variant_id,
        ).to_dict()
        for index, realization in enumerate(plan.realizations)
    ]
    sibling = WorldRuntime.for_sample(
        coupling_id="record-map-coupling",
        sample_index=1,
        rollout_id=999,
        question_rollout_plan=plan,
        fixed_world_specs=fixed_specs,
    )
    assert sibling.spec.world_slot == 0
    assert sibling.spec.replica_id == 1
    assert sibling.spec.latent_world_id == fixed_specs[0]["latent_world_id"]


def test_hierarchical_weights_are_invariant_to_k_and_variant_count():
    _, records4 = _records("q4", group_size=4)
    weights4, report4 = compute_hierarchical_loss_weights(records4)
    assert sum(weights4) == pytest.approx(1.0)
    assert all(value == pytest.approx(0.25) for value in report4["group_weight_sums"].values())

    _, records8 = _records("q8", group_size=8)
    weights8, report8 = compute_hierarchical_loss_weights(records8)
    assert sum(weights8) == pytest.approx(1.0)
    assert all(value == pytest.approx(0.25) for value in report8["group_weight_sums"].values())

    _, records_variant = _records("qv", group_size=4, extra_variants=(("local_degradation", "v1"),))
    weights_variant, report_variant = compute_hierarchical_loss_weights(records_variant)
    assert sum(weights_variant) == pytest.approx(1.0)
    assert report_variant["slot_weight_sums"]["qv:local_degradation"] == pytest.approx(0.25)


def test_question_validator_rejects_cross_question_and_missing_role():
    _, records = _records("q", group_size=4)
    records[0]["metadata"]["question_id"] = "other"
    report = validate_bayestool_question_records(records)
    assert not report["valid"]
    assert report["violations"]

    _, records = _records("q2", group_size=4)
    for record in records:
        if record["metadata"]["world_slot_role"] == WORLD_SLOT_ROLES[-1]:
            record["metadata"]["world_slot_role"] = ""
    report = validate_bayestool_question_records(records)
    assert not report["valid"]
    assert any("missing_required_world_slot_role" in item.get("errors", []) for item in report["violations"])
