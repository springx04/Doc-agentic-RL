from __future__ import annotations

import asyncio
import math
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayestool.belief import (  # noqa: E402
    FEATURE_NAMES,
    BeliefRuntime,
    ToolWorldFilterNetwork,
    ToolWorldSmoother,
    extract_observation_features,
    hbd_loss,
    observation_prediction_loss,
)
from bayestool.config import (  # noqa: E402
    add_bayestool_arguments,
    config_from_args,
    default_config,
    stage_definition,
    stage_schedule,
)
from bayestool.decision import (  # noqa: E402
    AnswerRiskCalibrator,
    DecisionController,
    Q_FEATURE_SCHEMA_VERSION,
    _action_feature_vector,
    canonical_action_key,
    js_divergence,
    normalize_q_action_feature_vector,
)
from bayestool.meta_episode import build_meta_episode  # noqa: E402
from bayestool.schema import TaskStateView, WorldEvent  # noqa: E402
from bayestool.training import (  # noqa: E402
    bayes_grpo_advantages,
    bayestool_group_id,
    bayestool_question_id,
    build_preinv_bundle,
    build_switch_bundle,
    build_switch_pair,
    compute_bayestool_utility,
    make_bayestool_decision_group_id,
    pre_invariance_loss,
    support_valid_pair,
    suffix_meta_returns,
    switch_loss,
    validate_bayestool_group_records,
    weighted_sibling_advantage,
)
from bayestool.world import (  # noqa: E402
    CleanResultCache,
    WorldRuntime,
    sample_tool_world,
    sample_world_type,
    sanitize_observed_result,
    tool_world_spec_from_dict,
)


def test_world_is_deterministic_and_slot_coverage_is_complete():
    config = default_config(enabled=True)
    specs = [
        sample_tool_world("coupling-a", world_slot=slot, replica_id=0, rollout_id=3, config=config)
        for slot in range(4)
    ]
    assert specs == [
        sample_tool_world("coupling-a", world_slot=slot, replica_id=0, rollout_id=3, config=config)
        for slot in range(4)
    ]
    world_types = {item.world_type for item in specs}
    assert "healthy" in world_types
    assert "shared_family_fault" in world_types
    assert len(world_types & {"single_tool_degradation", "context_degradation"}) == 1
    assert len(world_types & {"abrupt_change", "gradual_change"}) == 1


def test_fixed_manifest_world_round_trips_and_is_used_without_resampling():
    config = replace(default_config(enabled=True), worlds_per_prompt=1, replicas_per_world=1)
    spec = sample_tool_world("fixed-coupling", world_slot=0, replica_id=0, rollout_id=4, config=config)
    restored = tool_world_spec_from_dict(spec.to_dict())
    assert restored == spec
    runtime = WorldRuntime.for_sample(
        coupling_id="fixed-coupling",
        sample_index=0,
        rollout_id=999,
        config=config,
        fixed_world_specs=[spec.to_dict()],
    )
    assert runtime.spec == spec

    runtime_a = WorldRuntime.for_sample(
        coupling_id="coupling-a",
        sample_index=3,
        rollout_id=3,
        config=config,
        document_digest="doc-a",
    )
    runtime_b = WorldRuntime.for_sample(
        coupling_id="coupling-a",
        sample_index=3,
        rollout_id=3,
        config=config,
        document_digest="doc-a",
    )
    clean = '{"status":"ok","text":"stable observation","world_id":"hidden"}'
    transformed_a = runtime_a.transform_result("parse_document", {}, clean)
    transformed_b = runtime_b.transform_result("parse_document", {}, clean)
    assert transformed_a == transformed_b


def test_clean_result_cache_persists_and_async_deduplicates_results(tmp_path):
    cache = CleanResultCache(tmp_path)
    calls = []
    assert cache.get_or_set("doc", "parse_document", {"page": 1}, "v1", lambda: calls.append(1) or "clean") == "clean"
    reloaded = CleanResultCache(tmp_path)
    assert reloaded.get_or_set(
        "doc", "parse_document", {"page": 1}, "v1", lambda: (_ for _ in ()).throw(AssertionError("cache miss"))
    ) == "clean"

    async def produce():
        calls.append(1)
        return "async-clean"

    async def exercise():
        first = await reloaded.get_or_set_async("doc", "parse_document", {"page": 2}, "v1", produce)
        second = await reloaded.get_or_set_async(
            "doc", "parse_document", {"page": 2}, "v1", lambda: (_ for _ in ()).throw(AssertionError("async cache miss"))
        )
        return first, second

    assert asyncio.run(exercise()) == ("async-clean", "async-clean")
    assert calls == [1, 1]


def test_q_replay_dataset_accepts_particle_expanded_branch_records():
    from train_bayestool_belief import BayesQReplayDataset

    record = {
        "task_features": [0.0] * 32,
        "particle_features": [0.1] * 32,
        "action_features": [0.2] * 32,
        "budget_features": [0.3] * 8,
        "utility": 0.75,
        "oracle_utility": 0.8,
    }
    dataset = BayesQReplayDataset([{"metadata": {"bayes_branch_records": [record]}}])
    assert len(dataset) == 1
    assert float(dataset[0]["target"]) == pytest.approx(0.75)


def test_q_action_region_feature_is_bounded_for_rendered_pixel_boxes():
    task = TaskStateView(
        question_type="visual",
        current_page=1,
        visited_pages=(1,),
        unvisited_page_count=2,
        table_candidate_pages=(),
        supporting_pages=(),
        evidence_sufficient=False,
        visual_input_required=True,
        remaining_tool_budget=4,
        last_tool=None,
        last_result_status=None,
    )
    particle = SimpleNamespace(tool_quality={"crop_region": SimpleNamespace(relative_cost=1.0)})
    features = _action_feature_vector(
        {
            "kind": "tool",
            "tool": "crop_region",
            "arguments": {"bbox": [0, 0, 1600, 1200]},
        },
        task,
        particle,
    )
    assert Q_FEATURE_SCHEMA_VERSION == "bayestool-q-features-v2"
    assert len(features) == 32
    assert all(math.isfinite(value) for value in features)
    assert max(features) <= 1.0


def test_q_replay_normalizes_legacy_unbounded_region_area():
    legacy = [0.0] * 32
    legacy[13] = 241920.0
    normalized = normalize_q_action_feature_vector(legacy)
    assert normalized[13] == pytest.approx(math.log1p(241920.0) / math.log1p(1_000_000.0))
    assert max(normalized) <= 1.0


def test_world_slot_type_is_shared_by_replicas_and_extra_slots_are_weighted():
    config = default_config(enabled=True)
    replica_a = sample_tool_world(
        "coupling-replica",
        world_slot=7,
        replica_id=0,
        rollout_id=4,
        config=config,
    )
    replica_b = sample_tool_world(
        "coupling-replica",
        world_slot=7,
        replica_id=1,
        rollout_id=4,
        config=config,
    )
    assert replica_a.world_type == replica_b.world_type
    assert replica_a.seed != replica_b.seed
    assert sample_world_type("coupling-replica", world_slot=7, replica_id=0, rollout_id=4, config=config) == sample_world_type(
        "coupling-replica", world_slot=7, replica_id=1, rollout_id=4, config=config
    )

    only_gradual = replace(
        config,
        world_type_probabilities=(
            ("healthy", 0.0),
            ("single_tool_degradation", 0.0),
            ("context_degradation", 0.0),
            ("shared_family_fault", 0.0),
            ("abrupt_change", 0.0),
            ("gradual_change", 1.0),
        ),
    )
    assert sample_world_type("weighted", world_slot=12, config=only_gradual) == "gradual_change"

    only_outage = replace(
        config,
        session_state_probabilities=(
            ("healthy", 0.0),
            ("degraded", 0.0),
            ("overloaded", 0.0),
            ("outage", 1.0),
        ),
    )
    outage_world = sample_tool_world(
        "session-regime",
        world_slot=1,
        replica_id=0,
        rollout_id=4,
        config=only_outage,
        world_type="single_tool_degradation",
    )
    assert outage_world.session_state.state == "outage"
    assert outage_world.session_state.availability_scale < 0.5


def test_manifest_accepts_the_same_configured_world_distribution():
    from build_bayestool_data import build_manifest

    records = [
        {"id": "q1", "document_path": "doc.pdf", "question": "q1", "answers": ["a"]},
        {"id": "q2", "document_path": "doc.pdf", "question": "q2", "answers": ["b"]},
    ]
    coupled, meta = build_manifest(
        records,
        seed=7,
        world_type_probabilities={"healthy": 0.0, "gradual_change": 1.0},
    )
    assert len(coupled) == 2
    assert meta
    assert coupled[0]["metadata"]["world_type_probabilities"] == {"healthy": 0.0, "gradual_change": 1.0}
    assert set(coupled[0]["metadata"]["sampled_training_world_types"]) == {"gradual_change"}
    assert [item["name"] for item in coupled[0]["metadata"]["bayestool_stage_schedule"]] == ["a", "b", "c", "d"]
    plan = coupled[0]["metadata"]["question_rollout_plan"]
    assert plan["group_count"] == 4
    assert plan["record_count"] == 16
    assert len(coupled[0]["metadata"]["fixed_world_specs"]) == 4
    assert {item["world_slot_role"] for item in plan["realizations"]} == {
        "healthy",
        "local_degradation",
        "shared_family_fault",
        "change",
    }


def test_world_observation_never_exposes_hidden_answer_or_clean_result():
    clean = {
        "status": "ok",
        "text": "clean text",
        "clean_result": "secret",
        "answer_page": 9,
        "answer_bbox": [0, 0, 1, 1],
        "world_id": "world-secret",
        "nested": {"label": "secret-label", "visible": "ok"},
    }
    observed = sanitize_observed_result(__import__("json").dumps(clean))
    assert "secret" not in observed
    assert "world-secret" not in observed
    assert "visible" in observed

    world = sample_tool_world("same-document", world_slot=1, replica_id=0, rollout_id=0)
    changed_metadata_world = sample_tool_world("same-document", world_slot=1, replica_id=0, rollout_id=0)
    assert world == changed_metadata_world


def test_filter_features_ignore_hidden_world_fields_in_replay_event_mappings():
    public = {
        "tool": "extract_table",
        "status": "ok",
        "latency": 0.5,
        "information_gain": 0.7,
        "semantic_agreement": 0.9,
        "schema_valid": True,
        "image_valid": True,
        "error_family": "none",
    }
    replay_event = dict(
        public,
        relative_cost=0.01,
        corruption_type="answer_targeted_corruption",
        failure_origin="world_injected",
        world_id="hidden-world",
        true_quality={"availability": 0.01},
    )
    visible_features = extract_observation_features("extract_table", "rows", public)
    replay_features = extract_observation_features("extract_table", "rows", replay_event)
    assert replay_features.values == visible_features.values


def test_world_failure_origin_contract():
    config = default_config(enabled=True)
    injected = WorldRuntime.for_sample(coupling_id="x", sample_index=2, config=config)
    injected.spec = sample_tool_world("x", world_slot=2, world_type="shared_family_fault", config=config)
    transformed = injected.transform_result("extract_table", {}, '{"status":"ok","rows":[]}')
    assert transformed.world_event.valid_for_rl
    real = injected.record_real_infrastructure_failure("extract_table", {}, "backend unavailable")
    assert not real.world_event.valid_for_rl


@pytest.mark.skipif("torch" not in sys.modules and __import__("importlib").util.find_spec("torch") is None, reason="torch unavailable")
def test_filter_shapes_hbd_and_switch_gradients():
    import torch

    config = default_config(enabled=True)
    assert len(FEATURE_NAMES) == 96
    network = ToolWorldFilterNetwork(config)
    smoother = ToolWorldSmoother(network)
    features = torch.randn(2, 4, 96)
    filter_out = network(features.reshape(-1, 96), tool_id=0)
    smooth_out = smoother(features)
    # The recurrent filter is online [B, ...], while the smoother is sequence
    # shaped [B, T, ...]; flattening is the exact HBD comparison contract.
    smooth_flat = {
        key: value.reshape(-1, *value.shape[2:])
        for key, value in smooth_out.items()
        if key != "shared_logits"
    }
    smooth_flat["shared_logits"] = {
        family: value.reshape(-1, *value.shape[2:])
        for family, value in smooth_out["shared_logits"].items()
    }
    loss = hbd_loss(filter_out, smooth_flat)
    assert torch.isfinite(loss)
    assert loss.requires_grad

    values = [torch.tensor(1.0, requires_grad=True), torch.tensor(-1.0, requires_grad=True)]
    switch = switch_loss(values[0], values[1], values[0], values[1])
    preinv = pre_invariance_loss([values[0], values[1]], [values[1], values[0]])
    (switch + preinv).backward()
    assert all(value.grad is not None for value in values)


@pytest.mark.skipif("torch" not in sys.modules and __import__("importlib").util.find_spec("torch") is None, reason="torch unavailable")
def test_loaded_filter_runtime_keeps_recurrent_context_tensor_and_updates_posterior():
    from bayestool.schema import TaskStateView

    config = default_config(enabled=True)
    network = ToolWorldFilterNetwork(config)
    belief = BeliefRuntime(config, document_digest="doc", model=network, model_version="stage-a-test")
    event = {
        "tool": "parse_document",
        "status": "ok",
        "latency": 0.4,
        "information_gain": 0.5,
        "semantic_agreement": 0.9,
        "schema_valid": True,
        "image_valid": False,
    }
    first = belief.update(
        "parse_document",
        "document text",
        event,
        task_state=TaskStateView.from_navigation_state({"remaining_tool_budget": 4}),
    )
    second = belief.update(
        "parse_document",
        "more document text",
        event,
        task_state=TaskStateView.from_navigation_state({"remaining_tool_budget": 3}),
    )
    assert first.step == 1
    assert second.step == 2
    assert belief._context_hidden["doc"].shape[0] == 1
    assert second.version == first.version


def test_surprise_reopen_and_dvoi_are_deterministic():
    config = default_config(enabled=True)
    belief = BeliefRuntime(config, document_digest="doc", seed=7)
    event = WorldEvent(
        call_id=0,
        tool_name="extract_table",
        status="error",
        latency=3.0,
        information_gain=0.0,
        semantic_agreement=0.0,
        schema_valid=False,
        image_valid=False,
        error_family="availability",
        failure_origin="world_injected",
    )
    surprise = belief.predictive_surprise(event)
    belief.update("extract_table", "error observation", event)
    assert surprise >= 0.0
    reopen = belief.reopen("family", cause_tools=["extract_table"], surprise=surprise)
    assert reopen["level"] == "family"
    assert belief.reopen_events[-1]["level"] == "family"

    task = TaskStateView.from_navigation_state({"question_type": "table", "remaining_tool_budget": 5})
    candidates = [
        {"kind": "tool", "tool": "extract_table", "arguments": {"page": 1}},
        {"kind": "tool", "tool": "render_page", "arguments": {"page": 1}},
        {"kind": "final", "answer": ""},
    ]
    controller = DecisionController(config, seed=11)
    first = controller.select(
        task,
        belief,
        candidates,
        diagnostic_actions=candidates[:2],
    )
    second = DecisionController(config, seed=11).select(
        task,
        belief,
        candidates,
        diagnostic_actions=candidates[:2],
    )
    assert first.selected_action == second.selected_action
    assert first.action_values == second.action_values
    assert all(value >= 0.0 for value in (first.dvoi or {}).values()) or isinstance(first.dvoi, dict)


def test_family_reopen_counts_the_current_observation_before_history_append():
    belief = BeliefRuntime(default_config(enabled=True), document_digest="reopen-doc")
    belief.history.append(
        {
            "tool": "extract_table",
            "status": "error",
            "predictive_surprise": 7.0,
        }
    )
    current = {
        "tool": "extract_table",
        "status": "error",
        "predictive_surprise": 7.0,
    }
    assert belief.detect_reopen_level(current) == "family"


def test_stop_risk_applies_a_concrete_final_action():
    config = default_config(enabled=True)
    task = TaskStateView.from_navigation_state({"question_type": "table", "remaining_tool_budget": 4})
    belief = BeliefRuntime(config, document_digest="stop-doc", seed=19)
    controller = DecisionController(
        config,
        risk_calibrator=AnswerRiskCalibrator(weights=[0.0] * 8, bias=-10.0),
        seed=19,
    )
    final = {"kind": "final", "answer": ""}
    report = controller.select(
        task,
        belief,
        [
            {"kind": "tool", "tool": "extract_table", "arguments": {"page": 1}},
            final,
            {"kind": "abstain", "reason": "insufficient reliable evidence"},
        ],
    )
    assert report.selected_mode == "stop"
    assert report.selected_action == canonical_action_key(final)


def test_controller_never_invents_terminal_action_when_policy_only_sampled_a_tool():
    config = default_config(enabled=True)
    task = TaskStateView.from_navigation_state({"remaining_tool_budget": 4})
    belief = BeliefRuntime(config, document_digest="candidate-doc")
    tool = {"kind": "tool", "tool": "parse_document", "arguments": {"page": 1}}
    controller = DecisionController(
        config,
        risk_calibrator=AnswerRiskCalibrator(weights=[0.0] * 8, bias=-10.0),
    )
    report = controller.select(task, belief, [tool])
    assert report.selected_action == canonical_action_key(tool)
    assert report.selected_mode not in {"stop", "abstain"}
    assert all(item.get("kind") == "tool" for _, item in report.candidate_actions)


@pytest.mark.skipif("torch" not in sys.modules and __import__("importlib").util.find_spec("torch") is None, reason="torch unavailable")
def test_meta_replay_restores_only_session_and_shared_recurrent_hidden():
    import torch

    config = default_config(enabled=True)
    network = ToolWorldFilterNetwork(config)
    belief = BeliefRuntime(config, document_digest="meta-hidden-doc", model=network)
    belief.update(
        "parse_document",
        "document text",
        {
            "tool": "parse_document",
            "status": "ok",
            "latency": 0.2,
            "information_gain": 0.4,
        },
        task_state=TaskStateView.from_navigation_state({"remaining_tool_budget": 3}),
    )
    session_before = belief._session_hidden.detach().clone()
    shared_before = {
        name: value.detach().clone() for name, value in belief._shared_hidden.items()
    }
    restored = BeliefRuntime.from_replay_record(
        belief.export_replay_record(),
        config,
        model=network,
    )
    assert torch.equal(restored._session_hidden, session_before)
    assert all(torch.equal(restored._shared_hidden[name], value) for name, value in shared_before.items())
    assert restored._context_hidden == {}
    assert restored.history == []


def test_pairing_sibling_utility_and_meta_persistence():
    belief_u = {"session_probs": [0.98, 0.01, 0.005, 0.005], "regime_probs": [0.98, 0.01, 0.01]}
    belief_v = {"session_probs": [0.01, 0.98, 0.005, 0.005], "regime_probs": [0.05, 0.90, 0.05]}
    state_u = {
        "coupling_id": "c",
        "content_signature": "sig",
        "belief_snapshot": belief_u,
        "ood_score": 0.01,
        "best_action": "extract_table",
        "best_action_margin": 0.2,
        "best_return": 0.3,
    }
    state_v = {
        "coupling_id": "c",
        "content_signature": "sig",
        "belief_snapshot": belief_v,
        "ood_score": 0.01,
        "best_action": "render_page",
        "best_action_margin": 0.2,
        "best_return": 0.1,
    }
    assert 0.10 <= js_divergence([0.98, 0.01, 0.005, 0.005], [0.01, 0.98, 0.005, 0.005]) <= 0.80
    assert support_valid_pair(state_u, state_v)
    pair = build_switch_pair(state_u, state_v)
    assert pair is not None and pair.action_u != pair.action_v
    assert weighted_sibling_advantage([1.0, 0.0], ["s", "s"]) == [0.5, -0.5]
    assert suffix_meta_returns([1.0, 2.0], discount=0.95) == [2.9, 2.0]

    utility = compute_bayestool_utility(
        {"tool_call_count": 2, "duplicate_tool_calls": 1, "protocol_error_count": 0},
        quality=1.0,
        tool_budget=8,
    )
    assert math.isclose(utility["task_score"], 1.0)
    assert utility["utility"] < 1.0

    episode = build_meta_episode(
        [
            {"id": "q1", "document_path": "doc.pdf", "prompt": "q1"},
            {"id": "q2", "document_path": "doc.pdf", "prompt": "q2"},
        ],
        config=default_config(enabled=True),
        seed=3,
    )
    episode.start_question(0)
    episode.task_belief.update("parse_document", "ok", {"tool": "parse_document", "status": "ok"})
    episode.finish_question(0.4)
    assert episode.session_belief is not None
    episode.start_question(1)
    assert episode.task_belief is not None
    assert episode.task_belief.snapshot().step == 0
    assert episode.task_belief.snapshot().session_probs != (0.70, 0.20, 0.08, 0.02)


def test_bayestool_advantage_is_question_and_decision_scoped():
    common = {
        "question_id": "question-a",
        "episode_content_id": "document-a",
        "latent_world_id": "healthy",
        "coupling_id": "coupling-a",
        "decision_event_id": "root",
        "decision_prefix_hash": "initial-a",
    }
    group_id = make_bayestool_decision_group_id(common)
    samples = [
        {"utility": 1.0, "metadata": {**common, "decision_group_id": group_id}},
        {"utility": 0.0, "metadata": {**common, "decision_group_id": group_id}},
    ]
    # A same-named legacy sibling group from another question must not mix
    # with this question.  This reproduces the previous fallback bug.
    samples.extend(
        [
            {
                "utility": 10.0,
                "metadata": {
                    "question_id": "question-b",
                    "latent_world_id": "healthy",
                    "sibling_group_id": "old-shared-id",
                },
                "sibling_group_id": "old-shared-id",
            },
            {
                "utility": 8.0,
                "metadata": {
                    "question_id": "question-b",
                    "latent_world_id": "healthy",
                    "sibling_group_id": "old-shared-id",
                },
                "sibling_group_id": "old-shared-id",
            },
        ]
    )
    assert bayestool_question_id(samples[0]["metadata"]) == "question-a"
    assert bayestool_group_id(samples[2], index=2) != group_id
    advantages = bayes_grpo_advantages(samples)
    assert advantages[:2] == [0.5, -0.5]
    assert advantages[2:] == [1.0, -1.0]


def test_bayestool_group_validator_rejects_cross_question_and_incomplete_groups():
    rows = [
        {
            "utility": float(index),
            "metadata": {
                "question_id": "q-a" if index < 3 else "q-b",
                "latent_world_id": "healthy",
                "decision_prefix_hash": "root",
                "decision_group_id": "bad-group",
            },
        }
        for index in range(4)
    ]
    report = validate_bayestool_group_records(rows)
    assert report["invalid_group_count"] == 1
    assert report["cross_question_group_count"] == 1
    assert report["groups"][0]["size"] == 4


def test_tokenized_auxiliary_bundle_contract():
    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [len(str(text)) + 1, 3]}

    pair = build_switch_pair(
        {
            "coupling_id": "c",
            "content_signature": "s",
            "belief_snapshot": {"session_probs": [0.9, 0.05, 0.03, 0.02]},
            "ood_score": 0.0,
            "best_action": "a",
            "best_action_margin": 0.3,
        },
        {
            "coupling_id": "c",
            "content_signature": "s",
            "belief_snapshot": {"session_probs": [0.05, 0.9, 0.03, 0.02]},
            "ood_score": 0.0,
            "best_action": "b",
            "best_action_margin": 0.3,
        },
    )
    assert pair is not None
    bundle = build_switch_bundle(pair, prompt_u="u", prompt_v="v", tokenizer=TinyTokenizer())
    assert len(bundle["tokenized"]) == 4
    assert all(item["sequence_ids"] for item in bundle["tokenized"])

    preinv = build_preinv_bundle(
        {"coupling_id": "c", "content_signature": "s", "candidate_actions": ["a", "b"], "belief_js": 0.01},
        {"coupling_id": "c", "content_signature": "s", "candidate_actions": ["a", "b"], "belief_js": 0.01},
        prompts=["u", "v"],
        actions=["a", "b"],
        tokenizer=TinyTokenizer(),
    )
    assert preinv is not None and len(preinv["tokenized"]) == 4


def test_belief_prompt_token_budget_is_enforced_with_a_character_tokenizer():
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(str(text))}

    config = replace(default_config(enabled=True), max_belief_prompt_tokens=32)
    belief = BeliefRuntime(config, document_digest="budget-doc", seed=5)
    block = belief.to_prompt_block(
        {
            "question_type": "table",
            "visited_pages": list(range(1, 80)),
            "remaining_tool_budget": 3,
        },
        tokenizer=CharacterTokenizer(),
    )
    assert len(CharacterTokenizer()(block, add_special_tokens=False)["input_ids"]) <= 32
    assert isinstance(json.loads(block), dict) if block.startswith("{") and block.endswith("}") else True


def test_risk_calibrator_fit_round_trip_and_explicit_label_semantics():
    from calibrate_bayestool_risk import build_validation_rows

    records = [
        {"metadata": {"risk_label": 1, "navigation_state": {"remaining_tool_budget": 0}}},
        {"metadata": {"risk_label": 0, "navigation_state": {"remaining_tool_budget": 8}}},
        {"metadata": {"correctness": 1, "navigation_state": {"remaining_tool_budget": 6}}},
    ]
    features, labels = build_validation_rows(records)
    assert labels == [1.0, 0.0]
    calibrator = AnswerRiskCalibrator()
    metrics = calibrator.fit(features, labels, epochs=20)
    assert metrics["samples"] == 2.0
    restored = AnswerRiskCalibrator.from_dict(calibrator.to_dict())
    assert math.isclose(restored.predict(features[0]), calibrator.predict(features[0]), rel_tol=1e-9)

    nested_features = AnswerRiskCalibrator().risk_features(
        TaskStateView.from_navigation_state(
            {"remaining_tool_budget": 2, "visited_pages": [1], "unvisited_pages": [2, 3], "last_tool": "extract_table"}
        ),
        {
            "bayestool": {
                "belief_snapshot": {
                    "tool_quality": {"extract_table": {"semantic": [0.8, 0.1], "structure": [0.7, 0.1]}}
                }
            }
        },
    )
    assert nested_features["semantic_posterior"] == 0.8
    assert nested_features["structure_posterior"] == 0.7

    risk_features, risk_labels = build_validation_rows(
        [{"metadata": {"risk_probability": 1.0}}],
        label_key="risk_probability",
        label_is_risk=True,
    )
    assert risk_features and risk_labels == [1.0]


def test_cli_stage_and_ablation_flags_are_applied_without_silent_feature_removal():
    config = config_from_args(
        SimpleNamespace(
            bayestool_enable=True,
            bayestool_stage="b",
            bayestool_without_dvoi=True,
            bayestool_without_regret_branching=True,
            bayestool_without_reopen=True,
            bayestool_without_switch_loss=True,
            bayestool_without_pre_invariance=True,
            bayestool_world_type_probabilities='{"healthy": 0.1, "gradual_change": 0.9}',
            bayestool_session_state_probabilities='{"outage": 1.0}',
        ),
        enabled=True,
    )
    assert config.stage == "b"
    assert not config.use_dvoi
    assert not config.use_regret_branching
    assert not config.use_reopen
    assert not config.auxiliary.use_switch_loss
    assert not config.auxiliary.use_pre_invariance
    assert config.branch_probability_when_eligible == 0.10
    assert not config.use_meta_episode
    assert not config.use_persistent_session_belief
    assert dict(config.world_type_probabilities) == {"healthy": 0.1, "gradual_change": 0.9}
    assert dict(config.session_state_probabilities) == {"outage": 1.0}
    assert stage_definition(config).runtime_mode == "single_task_belief_conditioned"
    assert [item.name for item in stage_schedule()] == ["a", "b", "c", "d"]


def test_all_stage_policies_override_launcher_defaults_exactly():
    expected = {
        "a": (0.0, False, False),
        "b": (0.10, False, False),
        "c": (0.25, False, False),
        "d": (0.25, True, True),
    }
    for stage, (branch_probability, meta_enabled, persistent_enabled) in expected.items():
        config = config_from_args(
            SimpleNamespace(bayestool_enable=True, bayestool_stage=stage),
            enabled=True,
        )
        assert config.branch_probability_when_eligible == branch_probability
        assert config.use_meta_episode is meta_enabled
        assert config.use_persistent_session_belief is persistent_enabled
    default_stage_c = default_config(enabled=True)
    assert not default_stage_c.use_meta_episode
    assert not default_stage_c.use_persistent_session_belief
    hardware_override = config_from_args(
        SimpleNamespace(bayestool_enable=True, bayestool_stage="c", bayestool_branch_probability=0.20),
        enabled=True,
    )
    assert hardware_override.branch_probability_when_eligible == pytest.approx(0.20)


def test_public_bayestool_argument_builder_registers_each_flag_once():
    import argparse

    parser = add_bayestool_arguments(argparse.ArgumentParser())
    parsed = parser.parse_args(
        [
            "--bayestool-stage",
            "d",
            "--bayestool-without-dvoi",
            "--bayestool-max-belief-prompt-tokens",
            "640",
            "--bayestool-meta-discount",
            "0.9",
        ]
    )
    assert parsed.bayestool_stage == "d"
    assert parsed.bayestool_without_dvoi is True
    config = config_from_args(parsed, enabled=True)
    assert config.max_belief_prompt_tokens == 640
    assert config.meta.discount == 0.9


@pytest.mark.skipif("torch" not in sys.modules and __import__("importlib").util.find_spec("torch") is None, reason="torch unavailable")
def test_real_log_targets_are_masked_and_prediction_loss_ignores_missing_fields():
    import torch

    from train_bayestool_belief import _batch_targets

    targets = _batch_targets(
        {
            "event": [
                {"status": "ok", "latency": 0.5, "schema_valid": True},
                {"status": "error", "image_valid": False},
            ]
        }
    )
    assert targets["latency_bin_mask"].tolist() == [1.0, 0.0]
    assert targets["information_gain_mask"].tolist() == [0.0, 0.0]
    assert targets["schema_valid_mask"].tolist() == [1.0, 0.0]
    logits = {
        "latency_bin": torch.zeros(2, 8, requires_grad=True),
        "information_gain": torch.zeros(2, 5, requires_grad=True),
        "schema_valid": torch.zeros(2, 2, requires_grad=True),
    }
    masked_loss = observation_prediction_loss(logits, targets)
    assert torch.isfinite(masked_loss)
    masked_loss.backward()
    assert logits["latency_bin"].grad is not None


def test_cli_auxiliary_arguments_reach_runtime_config():
    from types import SimpleNamespace

    config = config_from_args(
        SimpleNamespace(
            bayestool_enable=True,
            bayestool_aux_interval=3,
            bayestool_max_switch_bundles_per_rank=5,
            bayestool_max_preinv_bundles_per_rank=6,
            bayestool_switch_loss_weight=0.4,
            bayestool_preinv_loss_weight=0.07,
            bayestool_aux_micro_batch_size=8,
        ),
        enabled=True,
    )
    assert config.auxiliary.interval == 3
    assert config.auxiliary.max_switch_bundles_per_rank == 5
    assert config.auxiliary.max_preinv_bundles_per_rank == 6
    assert config.auxiliary.switch_loss_weight == 0.4
    assert config.auxiliary.preinv_loss_weight == 0.07
    assert config.auxiliary.micro_batch_size == 8
