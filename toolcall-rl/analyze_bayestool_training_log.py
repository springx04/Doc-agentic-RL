#!/usr/bin/env python3
"""Audit a BayesTool RL launcher log without confusing execution with learning.

The async trainer can successfully initialize models, collect rollouts, run
forward passes, and save a checkpoint while still producing no usable policy
gradient.  This script extracts the rollout/reward/optimizer evidence and
classifies a run as ``effective``, ``partial_signal``, ``chain_only``, or
``failed``.  It is intentionally conservative: a successful job is not
reported as effective unless reward/advantage and update evidence are all
present, and BayesTool runs must exercise at least one tool action.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAPPING_MARKER_RE = re.compile(r"\b(reward|rollout|step|perf)\s*(?:\d+)?\s*:\s*\{")
CHECKPOINT_FILE_NAMES = {
    "latest_checkpointed_iteration.txt",
    "latest_checkpointed_iteration.txt.tmp",
}
ROLLOUT_ARTIFACT_NAME = "rollout_interactions.json"


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _balanced_mapping(text: str, opening_index: int) -> str | None:
    """Return one Python mapping literal starting at ``opening_index``.

    ``ast.literal_eval`` is used only after this small scanner has isolated a
    complete literal.  The scanner understands nested lists/dicts/tuples and
    quoted strings, which are present in the reward metadata emitted by the
    rollout manager.
    """

    if opening_index >= len(text) or text[opening_index] != "{":
        return None

    closing_for = {"{": "}", "[": "]", "(": ")"}
    stack: list[str] = ["}"]
    quote: str | None = None
    escaped = False

    for index in range(opening_index + 1, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            quote = char
            continue
        if char in closing_for:
            stack.append(closing_for[char])
            continue
        if char in ("}", "]", ")"):
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[opening_index : index + 1]

    return None


def _parse_mapping(text: str, opening_index: int) -> dict[str, Any] | None:
    literal = _balanced_mapping(text, opening_index)
    if literal is None:
        return None
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _extract_reward_records(text: str) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    parse_failures = 0
    for marker in re.finditer(r"\breward\s*:\s*\{", text):
        record = _parse_mapping(text, marker.end() - 1)
        if record is None:
            parse_failures += 1
            continue
        if any(key in record for key in ("score", "total_reward", "valid_for_rl")):
            records.append(record)
    return records, parse_failures


def _extract_metric_records(text: str) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    parse_failures = 0
    for marker in MAPPING_MARKER_RE.finditer(text):
        # Reward mappings are reported separately and must not be counted as
        # aggregate metrics a second time.
        if marker.group(1) == "reward":
            continue
        record = _parse_mapping(text, marker.end() - 1)
        if record is None:
            parse_failures += 1
            continue
        records.append(record)
    return records, parse_failures


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _finite_number(value)) is not None]


def _summary(values: Iterable[Any]) -> dict[str, Any]:
    numbers = _numbers(values)
    if not numbers:
        return {
            "count": 0,
            "unique_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "population_std": None,
            "nonzero_count": 0,
        }
    return {
        "count": len(numbers),
        "unique_count": len({round(number, 12) for number in numbers}),
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "population_std": statistics.pstdev(numbers),
        "nonzero_count": sum(abs(number) > 1e-12 for number in numbers),
    }


def _metric_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values_by_key: dict[str, list[Any]] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(key, str):
                values_by_key.setdefault(key, []).append(value)
    return {
        key: _summary(values)
        for key, values in sorted(values_by_key.items())
        if _numbers(values)
    }


def _record_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _positive_record_count(records: list[dict[str, Any]], key: str) -> int:
    count = 0
    for record in records:
        values = _numbers([record.get(key)])
        if values and values[0] > 0:
            count += 1
    return count


def _checkpoint_present(output_dir: Path | None) -> bool:
    if output_dir is None or not output_dir.exists():
        return False
    for name in CHECKPOINT_FILE_NAMES:
        if (output_dir / name).exists():
            return True
    try:
        return any(
            path.is_file() and ("latest_checkpointed_iteration" in path.name or "iter_" in path.name)
            for path in output_dir.rglob("*")
        )
    except OSError:
        return False


def _smoke_result_status(output_dir: Path | None) -> tuple[bool, bool]:
    """Read the bounded launcher result when the Ray wrapper is quiet.

    The real launcher can finish successfully without printing Ray's
    ``Job '...' succeeded`` line (for example when the caller captures the
    driver log rather than the submitter log).  ``smoke_result.json`` is
    written by the same run directory and is the stronger success signal.
    """

    if output_dir is None:
        return False, False
    result_path = output_dir / "smoke_result.json"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False, False
    if not isinstance(payload, dict):
        return False, False
    if payload.get("ok") is True or str(payload.get("status", "")).casefold() in {
        "ok",
        "succeeded",
        "success",
        "completed",
    }:
        return True, False
    if payload.get("ok") is False or str(payload.get("status", "")).casefold() in {
        "failed",
        "error",
    }:
        return False, True
    return False, False


def _sample_reward_record(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one serialized rollout sample into the audit schema."""

    if not isinstance(sample, dict) or not isinstance(sample.get("reward"), dict):
        return None
    record = dict(sample["reward"])
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    for key in (
        "valid_for_rl",
        "tool_call_count",
        "valid_tool_call_count",
        "tool_error_count",
        "rollout_status",
        "exclude_from_group_statistics",
        "protocol_error_count",
    ):
        if key not in record:
            if key in sample:
                record[key] = sample[key]
            elif key in metadata:
                record[key] = metadata[key]
    return record


def _artifact_rollout_records(output_dir: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load training rollout rewards from the serialized run artifact.

    Driver logs often contain only one aggregate reward mapping per rollout
    batch.  The JSON exporter contains the actual per-sample reward and
    validity fields, so it is the authoritative source whenever available.
    Evaluation samples are deliberately reported separately and never mixed
    into the policy-update evidence.
    """

    empty = {
        "used": False,
        "source": None,
        "training_records": 0,
        "evaluation_records": 0,
        "parse_failures": 0,
    }
    if output_dir is None:
        return [], empty
    artifact_path = output_dir / ROLLOUT_ARTIFACT_NAME
    if not artifact_path.is_file():
        return [], empty
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        result = dict(empty)
        result["source"] = str(artifact_path)
        result["parse_failures"] = 1
        return [], result

    raw_records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        result = dict(empty)
        result["source"] = str(artifact_path)
        result["parse_failures"] = 1
        return [], result

    training: list[dict[str, Any]] = []
    evaluation_count = 0
    parse_failures = 0
    for record in raw_records:
        if not isinstance(record, dict):
            parse_failures += 1
            continue
        source = str(record.get("source") or "").casefold()
        payload_record = record.get("payload", record)
        samples = payload_record.get("samples", []) if isinstance(payload_record, dict) else []
        if not isinstance(samples, list):
            parse_failures += 1
            continue
        is_evaluation = "/eval" in source or "\\eval" in source or Path(source).name.casefold().startswith("eval_")
        target = [] if is_evaluation else training
        for sample in samples:
            reward_record = _sample_reward_record(sample)
            if reward_record is None:
                parse_failures += 1
                continue
            if is_evaluation:
                evaluation_count += 1
            else:
                target.append(reward_record)

    result = {
        "used": bool(training),
        "source": str(artifact_path),
        "training_records": len(training),
        "evaluation_records": evaluation_count,
        "parse_failures": parse_failures,
    }
    return training, result


def audit_training_log(text: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a conservative, JSON-serializable audit report for one log."""

    clean = _strip_ansi(text)
    reward_records, reward_parse_failures = _extract_reward_records(clean)
    metric_records, metric_parse_failures = _extract_metric_records(clean)
    metric_summaries = _metric_summaries(metric_records)
    output_path = Path(output_dir) if output_dir is not None else None
    artifact_records, artifact_info = _artifact_rollout_records(output_path)
    if artifact_records:
        reward_records = artifact_records
        reward_parse_failures = 0

    reward_values = [
        record.get("total_reward", record.get("score")) for record in reward_records
    ]
    reward_summary = _summary(reward_values)
    valid_for_rl = sum(record.get("valid_for_rl") is True for record in reward_records)
    excluded = sum(record.get("exclude_from_group_statistics") is True for record in reward_records)
    tool_call_count = _positive_record_count(reward_records, "tool_call_count")
    valid_tool_call_count = _positive_record_count(reward_records, "valid_tool_call_count")
    protocol_error_count = _positive_record_count(reward_records, "protocol_error_count")
    reward_summary.update(
        {
            "observed_records": len(reward_records),
            "valid_for_rl_records": valid_for_rl,
            "excluded_records": excluded,
            "tool_active_records": tool_call_count,
            "valid_tool_active_records": valid_tool_call_count,
            "protocol_error_records": protocol_error_count,
            "rollout_status_counts": _record_counts(reward_records, "rollout_status"),
        }
    )

    smoke_succeeded, smoke_failed = _smoke_result_status(output_path)
    ray_succeeded = bool(re.search(r"Job\s+['\"].*?['\"]\s+succeeded", clean))
    ray_failed = bool(re.search(r"Job\s+['\"].*?['\"]\s+failed", clean))
    job_succeeded = bool(ray_succeeded or smoke_succeeded)
    job_failed = bool(ray_failed or smoke_failed)
    has_ref_log_probs = bool(re.search(r"\bref_log_probs\b", clean))
    has_on_policy_log_probs = bool(
        re.search(r"timer\s+log_probs\s+(?:start|end)", clean, re.IGNORECASE)
        and re.search(r"(?:rollout(?:/|_)log_probs|\blog_probs\b)", clean)
    )
    has_log_probs = bool(has_ref_log_probs or has_on_policy_log_probs or re.search(r"\blog_probs\b", clean))
    has_train_metrics = any(
        key.startswith("train/") for key in metric_summaries
    ) or "actor_train_time" in clean
    has_optimizer_update = bool(
        re.search(
            r"timer\s+update_weights\s+(?:start|end)|successfully saved checkpoint",
            clean,
            re.IGNORECASE,
        )
    )
    checkpoint_saved = bool(
        re.search(r"successfully saved checkpoint|saving checkpoint at iteration", clean)
    ) or _checkpoint_present(output_path)
    framework_chain_passed = bool(
        job_succeeded
        and has_log_probs
        and has_train_metrics
        and has_optimizer_update
        and checkpoint_saved
    )

    advantage_summary = metric_summaries.get("rollout/bayes_sibling_advantages", _summary([]))
    generic_advantage_summary = metric_summaries.get("rollout/advantages", _summary([]))
    aux_summary = metric_summaries.get("rollout/bayes_aux_records", _summary([]))
    branch_summary = metric_summaries.get("rollout/bayes_branch_records", _summary([]))
    action_token_summary = metric_summaries.get("rollout/action_reward_token_count", _summary([]))
    action_penalty_summary = metric_summaries.get("rollout/action_reward_abs_sum", _summary([]))
    action_reward_signal = bool(
        action_token_summary["nonzero_count"] > 0
        and action_penalty_summary["nonzero_count"] > 0
    )
    loss_summary = metric_summaries.get("train/loss", _summary([]))
    pg_loss_summary = metric_summaries.get("train/pg_loss", _summary([]))
    grad_summary = metric_summaries.get("train/grad_norm", _summary([]))
    policy_update_signal = bool(
        reward_summary["population_std"] not in (None, 0.0)
        or advantage_summary["nonzero_count"] > 0
        or generic_advantage_summary["nonzero_count"] > 0
        or aux_summary["nonzero_count"] > 0
        or action_reward_signal
    )
    nonzero_update = bool(
        loss_summary["nonzero_count"] > 0
        or pg_loss_summary["nonzero_count"] > 0
        or grad_summary["nonzero_count"] > 0
    )
    bayestool_action_signal = bool(tool_call_count or valid_tool_call_count)
    usable_rollout_data = bool(reward_records and valid_for_rl > 0)

    diagnoses: list[str] = []
    if job_failed or not job_succeeded:
        diagnoses.append("job did not complete successfully")
    if not reward_records:
        diagnoses.append("no parseable rollout reward records were found")
    elif valid_for_rl == 0:
        diagnoses.append("no observed rollout was marked valid_for_rl")
    if reward_summary["unique_count"] == 1 and reward_records:
        diagnoses.append("all observed rewards are identical; GRPO/Bayes sibling contrast is unavailable")
    if reward_records and tool_call_count == 0:
        diagnoses.append("no observed rollout executed a tool action; the BayesTool path was not exercised")
    if reward_records and protocol_error_count == len(reward_records):
        diagnoses.append("all observed rollout records ended in protocol errors")
    if reward_records and not policy_update_signal:
        diagnoses.append("reward/advantage/auxiliary metrics contain no nonzero learning signal")
    if reward_records and not nonzero_update:
        diagnoses.append("loss, policy loss, and gradient norm are all zero or absent")
    if not has_ref_log_probs and not has_on_policy_log_probs:
        diagnoses.append("reference log-probability stage is absent")
    if not has_log_probs:
        diagnoses.append("actor log-probability stage is absent")
    if not has_train_metrics:
        diagnoses.append("actor optimizer training metrics are absent")
    if not checkpoint_saved:
        diagnoses.append("no checkpoint save evidence was found")

    if not framework_chain_passed:
        status = "failed"
    elif usable_rollout_data and policy_update_signal and nonzero_update and bayestool_action_signal:
        status = "effective"
    elif usable_rollout_data and policy_update_signal and nonzero_update:
        status = "partial_signal"
    else:
        status = "chain_only"

    zero_std_metrics = {
        key: value
        for key, value in metric_summaries.items()
        if "zero_std" in key
    }
    return {
        "status": status,
        "framework_chain_passed": framework_chain_passed,
        "rl_signal_effective": status == "effective",
        "policy_update_signal_present": policy_update_signal,
        "usable_rollout_data_present": usable_rollout_data,
        "bayestool_action_signal_present": bayestool_action_signal,
        "action_reward_signal_present": action_reward_signal,
        "nonzero_optimizer_update_present": nonzero_update,
        "job": {
            "succeeded": job_succeeded,
            "failed": job_failed,
            "ray_submitter_succeeded": ray_succeeded,
            "smoke_result_succeeded": smoke_succeeded,
            "reference_log_probs_seen": has_ref_log_probs,
            "on_policy_log_probs_seen": has_on_policy_log_probs,
            "log_probs_seen": has_log_probs,
            "train_metrics_seen": has_train_metrics,
            "optimizer_update_seen": has_optimizer_update,
            "checkpoint_saved": checkpoint_saved,
        },
        "rollouts": reward_summary,
        "metrics": {
            "records_observed": len(metric_records),
            "parse_failures": metric_parse_failures,
            "summaries": metric_summaries,
            "zero_std_metrics": zero_std_metrics,
        },
        "parser": {
            "reward_parse_failures": reward_parse_failures,
            "metric_parse_failures": metric_parse_failures,
            "rollout_artifact": artifact_info,
        },
        "diagnoses": diagnoses,
    }


def audit_log_file(log_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(log_path)
    return audit_training_log(path.read_text(encoding="utf-8", errors="replace"), output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="launcher.log or captured Ray job log")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="optional run directory used to verify checkpoint files",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="also write the report to this JSON file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 2 unless the report classifies the run as effective",
    )
    args = parser.parse_args(argv)
    report = audit_log_file(args.log, args.output_dir)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(serialized)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    if args.strict and not report["rl_signal_effective"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
