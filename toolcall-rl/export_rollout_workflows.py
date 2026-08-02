"""Export a readable, bounded subset of Slime rollout workflows.

The exporter deliberately excludes token arrays, masks, tensors and other
large vector fields.  It keeps the text timeline needed to inspect an agent:
prompt -> model tool call -> tool result -> next model context -> final answer.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("/workspace/data/OpenClaw-RL/outputs/qwen3-vl-4b-docvqa-baseline-20260723-07-full200")
DEFAULT_LIMIT = 20
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_INTERPRETER = re.compile(r"<interpreter>\s*(.*?)\s*</interpreter>", re.DOTALL | re.IGNORECASE)
_FINAL = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL | re.IGNORECASE)


def _mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, dict) else None


def _find_samples(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 8:
        return []
    mapping = _mapping(value)
    if mapping is not None:
        samples = mapping.get("samples")
        if isinstance(samples, (list, tuple)) and all(_mapping(item) is not None for item in samples):
            normalized = [_mapping(item) for item in samples]
            if any("prompt" in item or "response" in item for item in normalized):
                return [item for item in normalized if item is not None]
        for item in mapping.values():
            found = _find_samples(item, depth + 1)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_samples(item, depth + 1)
            if found:
                return found
    return []


def _json_or_text(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _workflow(prompt: str, response: str) -> tuple[list[dict[str, str]], str]:
    """Produce a readable textual interaction timeline from a rollout response."""
    events: list[tuple[int, str, str]] = [(0, "Initial model input (prompt)", prompt)]
    for match in _TOOL_CALL.finditer(response):
        events.append((match.start(), "Model response: tool call", match.group(1).strip()))
    for match in _INTERPRETER.finditer(response):
        events.append((match.start(), "Tool result returned to model", match.group(1).strip()))
    for match in _FINAL.finditer(response):
        events.append((match.start(), "Model final answer", match.group(1).strip()))
    events.sort(key=lambda item: item[0])
    timeline = [{"step": index + 1, "title": title, "text": text} for index, (_, title, text) in enumerate(events)]
    final_matches = _FINAL.findall(response)
    return timeline, final_matches[-1].strip() if final_matches else ""


def _media_summary(multimodal: dict[str, Any]) -> dict[str, Any]:
    """Keep media references/descriptions, never image pixels or tensors."""
    summary: dict[str, Any] = {}
    for key in ("images", "videos"):
        value = multimodal.get(key)
        if not value:
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        descriptions = []
        for item in items:
            if isinstance(item, (str, Path)):
                descriptions.append({"reference": str(item)})
                continue
            description = {"type": f"{type(item).__module__}.{type(item).__qualname__}"}
            size = getattr(item, "size", None)
            if isinstance(size, (tuple, list)) and len(size) <= 3:
                description["size"] = list(size)
            descriptions.append(description)
        summary[key] = {"count": len(items), "items": descriptions}
    return summary


def _sample_view(sample: dict[str, Any], position: int) -> dict[str, Any]:
    prompt = str(sample.get("prompt", ""))
    response = str(sample.get("response", ""))
    reward = _mapping(sample.get("reward")) or {}
    metadata = _mapping(sample.get("metadata")) or {}
    rollout_status = str(sample.get("rollout_status") or reward.get("rollout_status") or metadata.get("rollout_status") or "")
    valid_for_rl = bool(
        sample.get("valid_for_rl", reward.get("valid_for_rl", metadata.get("valid_for_rl", rollout_status not in {"generation_empty", "generation_error", "context_overflow", "infra_error", "tool_error"})))
    )
    exclude_from_group_statistics = bool(
        sample.get(
            "exclude_from_group_statistics",
            reward.get("exclude_from_group_statistics", metadata.get("exclude_from_group_statistics", not valid_for_rl)),
        )
    )
    action_statistics = _mapping(metadata.get("action_statistics")) or {}
    tool_execution = _mapping(metadata.get("tool_execution")) or {}
    tool_call_count = int(
        sample.get(
            "tool_call_count",
            reward.get("tool_call_count", metadata.get("tool_call_count", tool_execution.get("call_count", 0))),
        )
        or 0
    )
    valid_call_fallback = sum(
        1
        for call in (tool_execution.get("calls", []) if isinstance(tool_execution.get("calls", []), list) else [])
        if isinstance(call, dict) and call.get("executed", True) and call.get("success", True)
    )
    valid_tool_call_count = int(
        sample.get(
            "valid_tool_call_count",
            reward.get(
                "valid_tool_call_count",
                metadata.get(
                    "valid_tool_call_count",
                    tool_execution.get("valid_call_count", valid_call_fallback),
                ),
            ),
        )
        or 0
    )
    tool_error_count = int(
        sample.get(
            "tool_error_count",
            reward.get("tool_error_count", metadata.get("tool_error_count", tool_execution.get("error_count", 0))),
        )
        or 0
    )
    generation_steps = metadata.get("generation_steps") if isinstance(metadata.get("generation_steps"), list) else []
    generation_called = bool(metadata.get("generation_called", any(step.get("generation_called") for step in generation_steps if isinstance(step, dict))))
    generation_call_count = int(metadata.get("generation_call_count", sum(1 for step in generation_steps if isinstance(step, dict) and step.get("generation_called"))))
    label = _json_or_text(sample.get("label"))
    timeline, final_answer = _workflow(prompt, response)
    multimodal = _mapping(sample.get("multimodal_inputs")) or {}
    return {
        "sample_number": position,
        "sample_index": sample.get("index"),
        "group_index": sample.get("group_index", metadata.get("group_index")),
        "ground_truth": label,
        "final_answer": final_answer,
        "reward": {key: reward.get(key) for key in ("score", "total_reward", "answer_reward", "evidence_reward", "evidence_localization_reward", "evidence_alignment_reward", "page_visit_reward", "tool_selection_reward", "tool_cost", "process_reward", "acc", "exact_acc", "quality", "answer_correctness", "answer_conciseness", "format_validity", "raw_anls", "format", "metric", "tool_call_count", "valid_tool_call_count", "tool_error_count", "pred", "valid_for_rl", "exclude_from_group_statistics", "rollout_status", "premature_final", "final_supported_by_evidence", "evidence_sufficient", "prediction_found_in_document", "prediction_relation_matched", "reward_consistency_valid", "multi_call_rollout", "multi_tool_type_rollout", "completed_multi_tool_type_rollout") if key in reward},
        "diagnostics": {
            "rollout_status": rollout_status,
            "rollout_status_reason": metadata.get("rollout_status_reason"),
            "valid_for_rl": valid_for_rl,
            "exclude_from_group_statistics": exclude_from_group_statistics,
            "generation_called": generation_called,
            "generation_call_count": generation_call_count,
            "tool_call_count": tool_call_count,
            "valid_tool_call_count": valid_tool_call_count,
            "tool_error_count": tool_error_count,
            "generation_steps": generation_steps,
            "raw_generation_text": metadata.get("raw_generation_text", ""),
            "raw_generation_texts": metadata.get("raw_generation_texts", []),
            "image_input_count": int(metadata.get("image_input_count", 0) or 0),
            "image_token_count": int(metadata.get("image_token_count", 0) or 0),
            "vision_input_attached": bool(metadata.get("vision_input_attached", False)),
            "vision_placeholder_present": bool(metadata.get("vision_placeholder_present", False)),
            "image_tensor_count": int(metadata.get("image_tensor_count", 0) or 0),
            "consumed_image_paths": metadata.get("consumed_image_paths", []),
            "action_statistics": {key: int(action_statistics.get(key, 0) or 0) for key in ("candidate_action_count", "executed_action_count", "valid_action_count", "invalid_action_count", "ignored_action_count", "protocol_error_count")},
            "assistant_turns": metadata.get("assistant_turns", []),
            "tool_execution": tool_execution,
            "visited_pages": metadata.get("visited_pages", (metadata.get("navigation_state") or {}).get("visited_pages", [])),
            "parsed_pages": metadata.get("parsed_pages", (metadata.get("navigation_state") or {}).get("parsed_pages", [])),
            "rendered_pages": metadata.get("rendered_pages", (metadata.get("navigation_state") or {}).get("rendered_pages", [])),
            "cropped_pages": metadata.get("cropped_pages", (metadata.get("navigation_state") or {}).get("cropped_pages", [])),
            "ocr_pages": metadata.get("ocr_pages", (metadata.get("navigation_state") or {}).get("ocr_pages", [])),
            "cropped_regions": metadata.get("cropped_regions", (metadata.get("navigation_state") or {}).get("cropped_regions", [])),
            "zoomed_regions": metadata.get("zoomed_regions", (metadata.get("navigation_state") or {}).get("zoomed_regions", [])),
            "ocr_regions": metadata.get("ocr_regions", (metadata.get("navigation_state") or {}).get("ocr_regions", [])),
            "answer_page": metadata.get("answer_page", (metadata.get("navigation_state") or {}).get("answer_page")),
            "answer_bbox": metadata.get("answer_bbox", (metadata.get("navigation_state") or {}).get("answer_bbox")),
            "evidence_sufficient": bool(metadata.get("evidence_sufficient", (metadata.get("navigation_state") or {}).get("evidence_sufficient", False))),
            "evidence_reason": metadata.get("evidence_reason", (metadata.get("navigation_state") or {}).get("evidence_reason")),
            "evidence_candidates": metadata.get("evidence_candidates", (metadata.get("navigation_state") or {}).get("evidence_candidates", [])),
            "visual_evidence_sufficient": bool(metadata.get("visual_evidence_sufficient", (metadata.get("navigation_state") or {}).get("visual_evidence_sufficient", False))),
            "supporting_pages": metadata.get("supporting_pages", (metadata.get("navigation_state") or {}).get("supporting_pages", [])),
            "supporting_regions": metadata.get("supporting_regions", (metadata.get("navigation_state") or {}).get("supporting_regions", [])),
            "stop_reason": metadata.get("stop_reason", (metadata.get("navigation_state") or {}).get("stop_reason")),
            "answer_page_visited": bool(metadata.get("answer_page_visited", False)),
            "premature_final": bool(metadata.get("premature_final", False)),
            "duplicate_page_calls": int(metadata.get("duplicate_page_calls", 0) or 0),
            "duplicate_region_calls": int(metadata.get("duplicate_region_calls", 0) or 0),
            "unnecessary_tool_calls": int(metadata.get("unnecessary_tool_calls", 0) or 0),
            "no_information_gain_calls": int(metadata.get("no_information_gain_calls", 0) or 0),
            "final_supported_by_evidence": bool(metadata.get("final_supported_by_evidence", False)),
            "prediction_found_in_document": bool(metadata.get("prediction_found_in_document", False)),
            "prediction_relation_matched": bool(metadata.get("prediction_relation_matched", False)),
            "runtime_evidence_source": metadata.get("runtime_evidence_source", "tool_observation"),
            "ground_truth_metadata_used_in_prompt": bool(metadata.get("ground_truth_metadata_used_in_prompt", False)),
            "ground_truth_metadata_used_in_action_selection": bool(metadata.get("ground_truth_metadata_used_in_action_selection", False)),
            "had_evidence_guard_recovery": bool(metadata.get("had_evidence_guard_recovery", False)),
            "rejected_final_count": int(metadata.get("rejected_final_count", 0) or 0),
            "had_infra_error": bool(metadata.get("had_infra_error", False)),
            "infra_error_messages": metadata.get("infra_error_messages", []),
            "assistant_token_masks": metadata.get("assistant_token_masks", []),
            "action_rewards": metadata.get("action_rewards", []),
            "rejected_action_indices": metadata.get("rejected_action_indices", []),
            "reward_consistency_valid": bool(metadata.get("reward_consistency_valid", True)),
            "reward_consistency_errors": metadata.get("reward_consistency_errors", []),
            "action_reward_consumed": metadata.get("action_reward_consumed", []),
            "all_pages_checked": bool(metadata.get("all_pages_checked", False)),
            "generation_turn_count": int(metadata.get("generation_turn_count", generation_call_count) or 0),
            "unique_tool_count": int(metadata.get("unique_tool_count", len(tool_execution.get("unique_tools", []))) or 0),
            "multi_turn_rollout": bool(metadata.get("multi_turn_rollout", generation_call_count > 1)),
            "multi_tool_rollout": bool(metadata.get("multi_tool_rollout", int(tool_execution.get("call_count", 0) or 0) >= 2)),
            "multi_call_rollout": bool(metadata.get("multi_call_rollout", int(tool_execution.get("call_count", 0) or 0) >= 2)),
            "multi_tool_type_rollout": bool(metadata.get("multi_tool_type_rollout", len(tool_execution.get("unique_tools", [])) >= 2)),
            "completed_multi_tool_rollout": bool(metadata.get("completed_multi_tool_rollout", False)),
            "completed_multi_tool_type_rollout": bool(metadata.get("completed_multi_tool_type_rollout", False)),
        },
        "image_or_video_inputs": _media_summary(multimodal),
        "timeline": timeline,
        "raw_model_response": response,
    }


_EVIDENCE_RELATION_THRESHOLD = 0.75


def _valid_candidates(item: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = item.get("diagnostics", {}).get("evidence_candidates", [])
    if not isinstance(candidates, list):
        return []
    result = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("satisfies_question_constraints"):
            continue
        try:
            relation_score = float(candidate.get("relation_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            relation_score = 0.0
        if relation_score >= _EVIDENCE_RELATION_THRESHOLD:
            result.append(candidate)
    return result


def _group_reward_ordering(
    items: list[dict[str, Any]],
    left_predicate,
    right_predicate,
) -> tuple[float | None, int]:
    """Mean pairwise ``reward(left) > reward(right)`` within rollout groups."""
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        if not item["diagnostics"].get("valid_for_rl"):
            continue
        group = item.get("group_index")
        if group is None:
            continue
        grouped.setdefault(group, []).append(item)
    comparisons: list[bool] = []
    for group_items in grouped.values():
        left = [item for item in group_items if left_predicate(item)]
        right = [item for item in group_items if right_predicate(item)]
        for better in left:
            better_reward = better["reward"].get("total_reward", better["reward"].get("score", 0.0))
            for worse in right:
                worse_reward = worse["reward"].get("total_reward", worse["reward"].get("score", 0.0))
                if isinstance(better_reward, (int, float)) and isinstance(worse_reward, (int, float)):
                    comparisons.append(float(better_reward) > float(worse_reward))
    # ``None`` is intentional: a missing group id or a group without a
    # comparable pair is not a zero-accuracy result.  Reporting 0.0 here
    # made an unavailable ordering audit look like an observed failure.
    return (sum(comparisons) / len(comparisons) if comparisons else None, len(comparisons))


def _group_reward_ordering_metrics(items: list[dict[str, Any]]) -> dict[str, float | int | None]:
    def answer_correct(item):
        return float(item["reward"].get("answer_correctness", 0.0) or 0.0) > 0.0

    def answer_incorrect(item):
        return not answer_correct(item)

    def relevant_wrong(item):
        return answer_incorrect(item) and float(item["reward"].get("evidence_localization_reward", 0.0) or 0.0) > 0.0

    def irrelevant_wrong(item):
        return answer_incorrect(item) and float(item["reward"].get("evidence_localization_reward", 0.0) or 0.0) <= 0.0

    def valid_format(item):
        return float(item["reward"].get("format_validity", 0.0) or 0.0) >= 1.0

    def invalid_format(item):
        return not valid_format(item)

    def redundant(item):
        diagnostics = item["diagnostics"]
        return bool(
            int(diagnostics.get("duplicate_page_calls", 0) or 0)
            + int(diagnostics.get("duplicate_region_calls", 0) or 0)
            + int(diagnostics.get("unnecessary_tool_calls", 0) or 0)
            + int(diagnostics.get("no_information_gain_calls", 0) or 0)
        )

    comparisons = {
        "correct_vs_incorrect_ordering_accuracy": _group_reward_ordering(items, answer_correct, answer_incorrect),
        "relevant_wrong_vs_irrelevant_wrong_ordering_accuracy": _group_reward_ordering(items, relevant_wrong, irrelevant_wrong),
        "valid_format_vs_invalid_format_ordering_accuracy": _group_reward_ordering(items, valid_format, invalid_format),
        "efficient_vs_redundant_ordering_accuracy": _group_reward_ordering(items, lambda item: not redundant(item), redundant),
    }
    group_ids = {
        item.get("group_index")
        for item in items
        if item["diagnostics"].get("valid_for_rl") and item.get("group_index") is not None
    }
    return {
        **{name: accuracy for name, (accuracy, _count) in comparisons.items()},
        "group_count": len(group_ids),
        "unassigned_valid_rollout_count": sum(
            1
            for item in items
            if item["diagnostics"].get("valid_for_rl") and item.get("group_index") is None
        ),
        "comparable_pair_count": sum(count for _accuracy, count in comparisons.values()),
    }


def build_report(payload: Any, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    samples = _find_samples(payload)
    selected = [_sample_view(sample, index + 1) for index, sample in enumerate(samples[:limit])]
    rewards = [item["reward"] for item in selected]
    valid_items = [item for item in selected if item["diagnostics"].get("valid_for_rl") is True]

    def mean(field: str) -> float | None:
        values = [float(item["reward"][field]) for item in valid_items if isinstance(item["reward"].get(field), (int, float))]
        return sum(values) / len(values) if values else None

    infra_statuses = {"generation_empty", "generation_error", "context_overflow", "infra_error", "tool_error"}
    tool_usage: dict[str, int] = {}
    for item in selected:
        calls = item["diagnostics"].get("tool_execution", {}).get("calls", [])
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict) and call.get("tool") and call.get("executed", True):
                    tool_name = str(call["tool"])
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
    statuses = [str(item["diagnostics"].get("rollout_status", "")) for item in selected]
    protocol_error_count = sum(
        1
        for item in selected
        if str(item["diagnostics"].get("rollout_status", "")) == "model_protocol_error"
        or int(item["diagnostics"].get("action_statistics", {}).get("protocol_error_count", 0) or 0) > 0
    )
    valid_scores = [float(item["reward"]["score"]) for item in valid_items if isinstance(item["reward"].get("score"), (int, float))]
    completion_count = sum(status == "completed" for status in statuses)
    multi_step_count = sum(int(item["diagnostics"].get("generation_call_count", 0) or 0) > 1 for item in selected)
    multi_tool_count = sum(bool(item["diagnostics"].get("multi_tool_rollout")) for item in selected)
    completed_multi_tool_count = sum(bool(item["diagnostics"].get("completed_multi_tool_rollout")) for item in selected)
    multi_call_count = sum(bool(item["diagnostics"].get("multi_call_rollout")) for item in selected)
    multi_tool_type_count = sum(bool(item["diagnostics"].get("multi_tool_type_rollout")) for item in selected)
    completed_multi_tool_type_count = sum(bool(item["diagnostics"].get("completed_multi_tool_type_rollout")) for item in selected)
    diagnostics_summary = {
        "completion_rate": completion_count / len(selected) if selected else 0.0,
        "generation_error_count": sum(status == "generation_error" for status in statuses),
        "infra_error_count": sum(status in infra_statuses for status in statuses),
        "protocol_error_count": protocol_error_count,
        "valid_for_rl_count": len(valid_items),
        "mean_score_over_valid_rollouts": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "tool_usage_distribution": dict(sorted(tool_usage.items())),
        "multi_step_rollout_count": multi_step_count,
        "multi_turn_rollout_count": sum(bool(item["diagnostics"].get("multi_turn_rollout")) for item in selected),
        "multi_tool_rollout_count": multi_tool_count,
        "completed_multi_tool_rollout_count": completed_multi_tool_count,
        "multi_call_rollout_count": multi_call_count,
        "multi_tool_type_rollout_count": multi_tool_type_count,
        "completed_multi_tool_type_rollout_count": completed_multi_tool_type_count,
        "evidence_sufficient_count": sum(bool(item["diagnostics"].get("evidence_sufficient")) for item in selected),
        "evidence_insufficient_count": sum(not bool(item["diagnostics"].get("evidence_sufficient")) for item in selected),
        "reward_consistency_invalid_count": sum(not bool(item["diagnostics"].get("reward_consistency_valid", True)) for item in selected),
        "evidence_guard_recovery_count": sum(bool(item["diagnostics"].get("had_evidence_guard_recovery")) for item in selected),
        "mean_tool_calls_over_valid_rollouts": (
            sum(float(item["diagnostics"].get("tool_call_count", 0) or 0) for item in valid_items) / len(valid_items)
            if valid_items else 0.0
        ),
        "answer_page_visited_count": sum(bool(item["diagnostics"].get("answer_page_visited")) for item in selected),
        "premature_final_count": sum(bool(item["diagnostics"].get("premature_final")) for item in selected),
        "evidence_sufficient_without_valid_candidate_count": sum(
            bool(item["diagnostics"].get("evidence_sufficient")) and not _valid_candidates(item)
            for item in selected
        ),
        "final_supported_without_relation_count": sum(
            bool(item["diagnostics"].get("final_supported_by_evidence"))
            and not bool(item["diagnostics"].get("prediction_relation_matched"))
            for item in selected
        ),
        "final_supported_without_document_match_count": sum(
            bool(item["diagnostics"].get("final_supported_by_evidence"))
            and not bool(item["diagnostics"].get("prediction_found_in_document"))
            for item in selected
        ),
        "table_fallback_success_count": sum(
            any(
                isinstance(candidate, dict)
                and str(candidate.get("source_type", "")) in {"table", "extract_table"}
                and candidate.get("metric")
                and candidate.get("row_key")
                and candidate.get("column_key")
                for candidate in (item["diagnostics"].get("evidence_candidates") or [])
            )
            for item in selected
        ),
        "reward_consistency_error_counts": {
            error: sum(
                error in (item["diagnostics"].get("reward_consistency_errors") or [])
                for item in selected
            )
            for error in sorted(
                {
                    error
                    for item in selected
                    for error in (item["diagnostics"].get("reward_consistency_errors") or [])
                }
            )
        },
        "group_reward_ordering_metrics": _group_reward_ordering_metrics(selected),
        # These are deterministic pre-training canaries, reported by the test
        # harness rather than inferred from a random policy rollout.
        "action_reward_gradient_test_passed": None,
        "visual_canary_passed": None,
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sample_count": len(samples),
        "exported_sample_count": len(selected),
        "selection": {"strategy": "first_n", "limit": limit},
        "summary_for_exported_subset": {"mean_score": mean("score"), "mean_acc": mean("acc"), "mean_exact_acc": mean("exact_acc"), "mean_quality": mean("quality"), "total_tool_errors": sum(int(item["reward"].get("tool_error_count", 0) or 0) for item in valid_items), **diagnostics_summary},
        "samples": selected,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary_for_exported_subset"]
    lines = ["# Rollout workflow subset", "", f"- Source samples: {report['source_sample_count']}", f"- Exported samples: {report['exported_sample_count']}", f"- Subset mean ANLS quality: {summary['mean_quality']}", f"- Subset normalized accuracy: {summary['mean_acc']}", f"- Subset strict accuracy: {summary['mean_exact_acc']}", f"- Tool errors: {summary['total_tool_errors']}"]
    for sample in report["samples"]:
        lines.extend(["", f"## Sample {sample['sample_number']} (rollout index: {sample['sample_index']})", "", "### Ground truth", "```json", json.dumps(sample["ground_truth"], ensure_ascii=False, indent=2), "```", "", "### Reward", "```json", json.dumps(sample["reward"], ensure_ascii=False, indent=2), "```"])
        if sample["image_or_video_inputs"]:
            lines.extend(["", "### Model image/video references", "```json", json.dumps(sample["image_or_video_inputs"], ensure_ascii=False, indent=2), "```"])
        for event in sample["timeline"]:
            lines.extend(["", f"### {event['step']}. {event['title']}", "```text", event["text"], "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--prefix", default="workflow_subset")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    import torch

    source = args.output_dir / "dump_details" / "rollout_data" / "eval_0.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    report = build_report(payload, args.limit)
    json_path = args.output_dir / f"{args.prefix}_{args.limit}.json"
    markdown_path = args.output_dir / f"{args.prefix}_{args.limit}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(markdown_path), "exported": report["exported_sample_count"], "source_samples": report["source_sample_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
