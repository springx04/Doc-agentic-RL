from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayestool.grouping import (
    K8RollingScheduler,
    WORLD_SLOT_ROLES,
    compute_hierarchical_loss_weights,
    make_question_rollout_plan,
    select_extra_variants,
    validate_bayestool_question_records,
    validate_question_rollout_plan_records,
)
from bayestool.batching import assign_equal_cardinality_lpt, pack_questions_by_cost
from bayestool.training import make_bayestool_decision_group_id
from bayestool.world import WorldRuntime, sample_tool_world
from build_bayestool_data import build_manifest


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
    assert make_question_rollout_plan("q5", group_size=4, realization_count=5).group_count == 5
    assert make_question_rollout_plan("q6", group_size=8, realization_count=6).record_count == 48


def test_variant_selection_is_seeded_and_fails_closed_after_selection():
    candidates = [
        {"world_slot_role": "local_degradation", "variant_id": "low", "coverage_deficit": 0.1},
        {"world_slot_role": "change", "variant_id": "high", "calibration_gap": 0.9},
        {"world_slot_role": "shared_family_fault", "variant_id": "mid", "diagnostic_value": 0.5},
    ]
    left, left_report = select_extra_variants("variant-q", candidates, seed=19)
    right, right_report = select_extra_variants("variant-q", candidates, seed=19)
    assert left == right
    assert left_report == right_report
    assert len(left) == 2
    assert all("score" in item and "sampling_probability" in item for item in left_report["candidates"])
    with pytest.raises(ValueError):
        make_question_rollout_plan("too-many", extra_variants=["a", "b", "c"])


def test_variant_exploration_skips_zero_probability_candidates():
    selected, report = select_extra_variants(
        "zero-probability-q",
        [
            {"world_slot_role": "local_degradation", "variant_id": "underflow", "score": -1e9},
            {"world_slot_role": "change", "variant_id": "high-value", "score": 0.0},
        ],
        seed=7,
        max_extra=1,
        exploration_probability=1.0,
    )

    assert [item["variant_id"] for item in selected] == ["high-value"]
    assert report["candidates"][0]["sampling_probability"] == 0.0
    assert report["candidates"][1]["sampling_probability"] == 1.0


def test_k8_rolling_scheduler_enforces_floor_and_ceiling():
    scheduler = K8RollingScheduler(target_ratio=0.5, floor=0.5, ceiling=0.5, window=2, seed=3)
    first, first_report = scheduler.choose("k8-q1")
    second, second_report = scheduler.choose("k8-q2")
    assert first == 8
    assert second == 4
    assert first_report["reason"] == "rolling_floor_recovery"
    assert second_report["reason"] == "rolling_ceiling_guard"


def test_manifest_rolling_scheduler_keeps_history_across_questions():
    records = [
        {
            "id": "rolling-q1",
            "question": "first",
            "metadata": {
                "bayestool_group_size": "rolling",
                "bayestool_k8_target_ratio": 0.5,
                "bayestool_k8_floor": 0.5,
                "bayestool_k8_ceiling": 0.5,
                "bayestool_k8_window": 2,
            },
        },
        {
            "id": "rolling-q2",
            "question": "second",
            "metadata": {
                "bayestool_group_size": "rolling",
                "bayestool_k8_target_ratio": 0.5,
                "bayestool_k8_floor": 0.5,
                "bayestool_k8_ceiling": 0.5,
                "bayestool_k8_window": 2,
            },
        },
    ]
    coupled, _ = build_manifest(records, seed=17)
    selections = [item["metadata"]["bayestool_k8_selection"] for item in coupled]
    assert [item["selected_k"] for item in selections] == [8, 4]
    assert selections[1]["ratio_before"] == pytest.approx(1.0)


def test_manifest_preserves_an_upstream_explicit_plan():
    plan = make_question_rollout_plan("explicit-manifest", group_size=8, realization_count=5)
    record = {
        "id": "explicit-manifest",
        "question": "explicit question",
        "metadata": {"question_rollout_plan": plan.to_dict()},
    }
    coupled, _ = build_manifest([record], seed=21)
    persisted = coupled[0]["metadata"]["question_rollout_plan"]
    assert persisted["question_id"] == plan.question_id
    assert persisted["group_count"] == 5
    assert [item["k"] for item in persisted["realizations"]] == [8] * 5


def test_cost_packing_keeps_questions_atomic_and_honors_target():
    ready = [("long", [0] * 16), ("short-a", [1] * 16), ("short-b", [2] * 16)]
    batches, target, source = pack_questions_by_cost(
        ready,
        {"long": 100.0, "short-a": 40.0, "short-b": 40.0},
        questions_per_step=2,
        max_questions_per_step=2,
        target_global_train_cost=140.0,
    )
    assert target == pytest.approx(140.0)
    assert source == "explicit_cost_target"
    assert [[question_id for question_id, _ in batch] for batch in batches] == [
        ["long", "short-a"],
        ["short-b"],
    ]
    assert all(indices for batch in batches for _, indices in batch)


def test_lpt_balances_cost_with_equal_local_cardinality():
    rank_indices, rank_costs = assign_equal_cardinality_lpt(
        list(range(8)),
        {index: float(9 - index) for index in range(8)},
        dp_size=4,
    )
    assert all(len(indices) == 2 for indices in rank_indices)
    assert sorted(index for indices in rank_indices for index in indices) == list(range(8))
    assert max(rank_costs) - min(rank_costs) <= 1.0


def test_manifest_validation_requires_exact_realization_and_k_contract():
    plan4, records4 = _records("manifest-r4", group_size=4)
    report4 = validate_question_rollout_plan_records(records4, plan4)
    assert report4["valid"]
    assert report4["expected_realizations"]

    plan6, records6 = _records(
        "manifest-r6",
        group_size=8,
        extra_variants=(("local_degradation", "v1"), ("change", "v2")),
    )
    assert plan6.group_count == 6
    assert plan6.record_count == 48
    assert validate_question_rollout_plan_records(records6, plan6)["valid"]

    # A record from another realization cannot silently fill a missing group.
    records6[-1]["metadata"]["variant_id"] = "wrong-variant"
    invalid = validate_question_rollout_plan_records(records6, plan6)
    assert not invalid["valid"]
    assert invalid["violations"] or invalid["plan_errors"]


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


def test_finalized_plan_rejects_a_different_frozen_latent_world():
    plan = make_question_rollout_plan("strict-world", group_size=4)
    fixed_specs = [
        sample_tool_world(
            "strict-world-coupling",
            world_slot=index,
            replica_id=0,
            rollout_id=7,
            world_slot_role=item.world_slot_role,
            variant_id=item.variant_id,
        )
        for index, item in enumerate(plan.realizations)
    ]
    finalized = replace(
        plan,
        realizations=tuple(
            replace(item, latent_world_id=spec.latent_world_id)
            for item, spec in zip(plan.realizations, fixed_specs, strict=True)
        ),
        latent_ids_finalized=True,
    )
    bad_specs = [item.to_dict() for item in fixed_specs]
    bad_specs[0]["latent_world_id"] = "wrong-latent"
    with pytest.raises(ValueError, match="latent_world_id"):
        WorldRuntime.for_sample(
            coupling_id="strict-world-coupling",
            sample_index=0,
            realization_index=0,
            question_rollout_plan=finalized,
            fixed_world_specs=bad_specs,
        )


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
    values = list(report_variant["group_weight_sums"].values())
    assert sum(value == pytest.approx(0.125) for value in values) == 2
    assert sum(value == pytest.approx(0.25) for value in values) == 3


def test_group_identity_rejects_mixed_runtime_or_prefix():
    _, records = _records("identity", group_size=4)
    records[1]["metadata"]["runtime_state_digest"] = "runtime:other"
    report = validate_bayestool_question_records(records)
    assert not report["valid"]
    assert any("runtime_state_not_frozen" in item.get("errors", []) for item in report["violations"])


def test_multiple_questions_share_a_manifest_step_but_keep_question_mass_separate():
    _, left = _records("step-q1", group_size=4)
    _, right = _records("step-q2", group_size=4)
    weights, report = compute_hierarchical_loss_weights(left + right, question_count=2)
    assert sum(weights) == pytest.approx(1.0)
    assert report["question_weight_sums"]["step-q1"] == pytest.approx(0.5)
    assert report["question_weight_sums"]["step-q2"] == pytest.approx(0.5)
    assert all(value == pytest.approx(0.125) for value in report["group_weight_sums"].values())


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
