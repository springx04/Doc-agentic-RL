import json
import hashlib
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayestool.config import default_config, validate_stage_capabilities  # noqa: E402
from bayestool.decision import _particle_feature_vector, sample_posterior_particles  # noqa: E402
from bayestool.meta_episode import build_meta_episode  # noqa: E402
from bayestool.replay import export_canonical_replay, validate_canonical_replay  # noqa: E402
from bayestool.schema import ContextRule  # noqa: E402
from bayestool.world import (  # noqa: E402
    ObservationCorruptionAdapter,
    WorldSamplingContext,
    WorldRuntime,
    sample_tool_world,
)


def test_replicas_share_latent_world_but_have_distinct_observation_rng():
    context = WorldSamplingContext(page_count=4, tool_budget=6)
    left = sample_tool_world("coupling", world_slot=1, replica_id=0, rollout_id=9, sampling_context=context)
    right = sample_tool_world("coupling", world_slot=1, replica_id=1, rollout_id=9, sampling_context=context)
    assert left.latent_world_id == right.latent_world_id
    assert left.latent_seed == right.latent_seed
    assert left.seed != right.seed
    assert left.to_latent_dict() == right.to_latent_dict()

    left_runtime = WorldRuntime(left, sampling_context=context)
    right_runtime = WorldRuntime(right, sampling_context=context)
    _, left_event, _ = left_runtime.transform_result("parse_document", {}, json.dumps({"status": "ok", "text": "doc"}))
    _, right_event, _ = right_runtime.transform_result("parse_document", {}, json.dumps({"status": "ok", "text": "doc"}))
    assert left_event.call_id == right_event.call_id == 0
    assert left_event.latency != right_event.latency or left_event.semantic_agreement == right_event.semantic_agreement


def test_context_rules_match_multi_page_public_arguments():
    rule = ContextRule(
        scope="page",
        tool_names=("parse_document",),
        page_numbers=(3,),
    )
    assert rule.matches("parse_document", page_numbers=(2, 3))
    assert not rule.matches("parse_document", page_numbers=(1, 2))


def test_context_sampling_uses_public_page_count_and_reports_activation():
    context = WorldSamplingContext(page_count=4, tool_budget=6)
    spec = sample_tool_world(
        "context-coupling",
        world_slot=0,
        replica_id=0,
        rollout_id=3,
        world_type="context_degradation",
        sampling_context=context,
    )
    rule = spec.context_rules[0]
    assert rule.page_numbers and 1 <= rule.page_numbers[0] <= 4
    runtime = WorldRuntime(spec, sampling_context=context)
    page = rule.page_numbers[0]
    _, event, _ = runtime.transform_result(
        rule.tool_names[0],
        {"page_numbers": [page]},
        json.dumps({"status": "ok", "text": "public observation"}),
    )
    assert event.page_numbers == (page,)
    report = runtime.context_metadata()
    assert report["context_rule_match_count"] == 1
    assert report["first_context_match_call"] == 0
    assert report["affected_call_count"] == 1
    assert report["context_rule_not_exercised"] is False


def test_sampling_schedules_stay_inside_public_budget():
    context = WorldSamplingContext(page_count=2, tool_budget=3)
    for slot in range(4):
        spec = sample_tool_world("coupling", world_slot=slot, replica_id=0, rollout_id=11, sampling_context=context)
        for segment in spec.regime_schedule:
            assert 1 <= segment.start_call <= context.tool_budget
            if segment.end_call is not None:
                assert segment.start_call <= segment.end_call <= context.tool_budget


def test_schedule_metadata_tracks_target_effective_calls():
    context = WorldSamplingContext(page_count=2, tool_budget=6)
    spec = sample_tool_world(
        "schedule-coupling",
        world_slot=0,
        replica_id=0,
        rollout_id=13,
        world_type="abrupt_change",
        sampling_context=context,
    )
    segment = spec.regime_schedule[0]
    target = next(iter(segment.tool_overrides))
    runtime = WorldRuntime(spec, sampling_context=context)
    clean = json.dumps({"status": "ok", "text": "stable observation"})
    for _ in range(int(segment.start_call) + 2):
        runtime.transform_result(target, {}, clean)
    report = runtime.schedule_metadata()[0]
    assert report["first_effective_call"] == segment.start_call
    assert report["last_effective_call"] >= segment.start_call
    assert report["affected_call_count"] >= 1
    assert report["schedule_not_exercised"] is False


def test_for_sample_preserves_explicit_public_tool_budget():
    runtime = WorldRuntime.for_sample(
        coupling_id="budget-coupling",
        sample_index=0,
        rollout_id=1,
        tool_budget=3,
        sampling_context={"page_count": 4},
    )
    assert runtime.tool_budget == 3
    assert runtime.sampling_context.page_count == 4
    for segment in runtime.spec.regime_schedule:
        assert segment.start_call <= 3
        if segment.end_call is not None:
            assert segment.end_call <= 3


def test_bayestool_image_output_accepts_same_source_and_target(tmp_path):
    runtime = WorldRuntime.for_sample(
        coupling_id="same-image-coupling",
        sample_index=0,
        rollout_id=1,
        output_root=tmp_path,
    )
    coupling_key = hashlib.sha256(runtime.spec.coupling_id.encode("utf-8")).hexdigest()[:16]
    world_key = hashlib.sha256(runtime.spec.world_id.encode("utf-8")).hexdigest()[:16]
    target_dir = tmp_path / "tool_outputs" / "bayestool" / coupling_key / world_key / "0"
    target_dir.mkdir(parents=True)
    source = target_dir / "image_0.png"
    source.write_bytes(b"already-emitted")

    replacements = runtime._make_image_outputs(
        [str(source)],
        call_id=0,
        corruption=None,
        rng=random.Random(0),
    )

    assert replacements[str(source)] == str(source)


def test_schema_aware_corruption_keeps_json_valid():
    quality_spec = sample_tool_world("coupling", world_slot=1, replica_id=0, rollout_id=2).tool_states["parse_document"]
    # Force a material corruption probability while keeping the adapter
    # independent of the answer/document target.
    from bayestool.schema import ToolQualitySpec

    quality = ToolQualitySpec(
        availability=quality_spec.availability,
        semantic_accuracy=0.40,
        structure_fidelity=0.40,
        relative_cost=quality_spec.relative_cost,
    )
    observed, corruption, schema_valid, applied = ObservationCorruptionAdapter.corrupt(
        "parse_document",
        json.dumps({"status": "ok", "pages": [{"page_number": 1, "text": "amount 42"}]}),
        quality,
        __import__("random").Random(4),
    )
    assert corruption is not None and applied
    assert schema_valid
    assert isinstance(json.loads(observed), dict)


def test_observation_corruption_preserves_artifact_paths():
    from bayestool.schema import ToolQualitySpec

    observed, corruption, schema_valid, applied = ObservationCorruptionAdapter.corrupt(
        "ocr_region",
        json.dumps(
            {
                "status": "partial",
                "text": "amount 42 is shown here",
                "image_path": "/workspace/output/crop.png",
                "document_path": "/workspace/docs/example.pdf",
            }
        ),
        ToolQualitySpec(
            availability=1.0,
            semantic_accuracy=0.40,
            structure_fidelity=0.40,
            relative_cost=1.0,
        ),
        __import__("random").Random(7),
    )
    payload = json.loads(observed)
    assert corruption is not None and applied and schema_valid
    assert payload["image_path"] == "/workspace/output/crop.png"
    assert payload["document_path"] == "/workspace/docs/example.pdf"


def test_canonical_replay_is_ordered_and_contains_next_tool():
    metadata = {
        "rollout_id": 17,
        "coupling_id": "c",
        "latent_world_id": "latent",
        "replica_id": 1,
        "document_hash": "doc",
        "tool_execution": {
            "calls": [
                {
                    "kind": "tool_call",
                    "executed": True,
                    "tool": "parse_document",
                    "arguments": {},
                    "observed_result": "{\"status\":\"ok\"}",
                    "world_event": {"call_id": 0, "tool": "parse_document", "status": "ok"},
                    "bayes_supervision": {"tool_name": "parse_document"},
                    "task_state_before": {"question": "what", "remaining_tool_budget": 2},
                    "task_state_after": {"question": "what", "remaining_tool_budget": 1},
                },
                {
                    "kind": "tool_call",
                    "executed": True,
                    "tool": "render_page",
                    "arguments": {"page_number": 1},
                    "observed_result": "{\"status\":\"ok\"}",
                    "world_event": {"call_id": 1, "tool": "render_page", "status": "ok"},
                    "bayes_supervision": {"tool_name": "render_page"},
                    "task_state_before": {"question": "what", "remaining_tool_budget": 1},
                    "task_state_after": {"question": "what", "remaining_tool_budget": 0},
                },
            ]
        },
    }
    replay = export_canonical_replay(metadata)
    assert replay["events"][0]["next_tool_id"] == "render_page"
    assert validate_canonical_replay(replay) == []


def test_factorized_particle_features_include_late_tool_posteriors():
    snapshot = __import__("bayestool.belief", fromlist=["BeliefRuntime"]).BeliefRuntime(
        default_config(enabled=True)
    ).snapshot()
    particle = sample_posterior_particles(snapshot, count=1, seed=3)[0]
    baseline = _particle_feature_vector(particle)
    from dataclasses import replace

    qualities = dict(particle.tool_quality)
    qualities["chart_to_table"] = replace(qualities["chart_to_table"], semantic_accuracy=0.01)
    changed = _particle_feature_vector(replace(particle, tool_quality=qualities))
    assert len(baseline) == len(changed) == 32
    assert baseline != changed


def test_capability_gate_requires_explicit_heuristic_ablation(tmp_path):
    with pytest.raises(ValueError):
        validate_stage_capabilities("c")
    manifest = validate_stage_capabilities(
        "c",
        allow_heuristic_belief=True,
        allow_heuristic_q=True,
        allow_heuristic_risk=True,
    )
    assert manifest["heuristic_belief"] and manifest["heuristic_q"] and manifest["heuristic_risk"]


def test_meta_episode_has_explicit_ids_and_rejects_too_few_questions():
    with pytest.raises(ValueError):
        build_meta_episode([{"id": "only", "document_path": "doc"}])
    episode = build_meta_episode(
        [
            {"id": "q1", "document_path": "doc", "prompt": "one"},
            {"id": "q2", "document_path": "doc", "prompt": "two"},
        ]
    )
    assert episode.episode_content_id
    assert episode.questions[0].metadata["meta_trajectory_id"].startswith(episode.meta_episode_id)
