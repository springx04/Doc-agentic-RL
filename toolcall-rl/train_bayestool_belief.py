"""Stage-A Tool-World filter/smoother training entry point.

The loader accepts JSONL replay records produced by rollout metadata.  Hidden
state targets are consumed only when they are present in synthetic-world
records; real logs use observable status/agreement/calibration proxy targets
and never fabricate a latent label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from bayestool.belief import (
    FEATURE_NAMES,
    BeliefRuntime,
    ToolWorldFilterNetwork,
    ToolWorldSmoother,
    calibration_brier,
    extract_observation_features,
    hbd_loss,
    observation_prediction_loss,
)
from bayestool.config import TOOL_NAMES, BayesToolConfig, default_config
from bayestool.decision import (
    AnswerRiskCalibrator,
    BayesQHead,
    Q_FEATURE_SCHEMA_VERSION,
    normalize_q_action_feature_vector,
)
from bayestool.schema import TaskStateView
from bayestool.training import bayes_q_gaussian_nll
from bayestool.replay import validate_canonical_replay

try:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("train_bayestool_belief.py requires PyTorch") from exc


logger = logging.getLogger("bayestool-belief")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


class BeliefEventDataset(Dataset):
    """Group replay events into ordered causal chunks for filter/smoother training.

    A JSONL line may represent one event or a complete trajectory under
    ``events``/``trajectory``/``execution_trace``.  Explicit
    ``trajectory_id`` (or ``rollout_id``) is preferred; when it is absent each
    line is kept independent rather than accidentally mixing questions from
    the same document.  The latter is conservative for real logs and avoids
    leaking future observations across trajectories.
    """

    def __init__(self, records: Iterable[Mapping[str, Any]], sequence_length: int = 16) -> None:
        self.rows: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record_index, raw_record in enumerate(records):
            record = dict(raw_record)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            canonical = record.get("belief_replay") or record.get("canonical_replay")
            if canonical is None and isinstance(metadata, Mapping):
                canonical = metadata.get("belief_replay") or metadata.get("canonical_replay")
            if isinstance(canonical, Mapping):
                validation_errors = validate_canonical_replay(canonical)
                if validation_errors:
                    logger.warning(
                        "Skipping invalid canonical replay %s: %s",
                        record_index,
                        "; ".join(validation_errors[:4]),
                    )
                    continue
                record = {**record, "canonical_replay": dict(canonical)}
                metadata = {**metadata, "canonical_replay": dict(canonical)}
            trajectory_id = str(
                record.get("trajectory_id")
                or record.get("rollout_id")
                or (canonical.get("trajectory_id") if isinstance(canonical, Mapping) else None)
                or metadata.get("trajectory_id")
                or metadata.get("rollout_id")
                or f"record:{record_index}"
            )
            raw_events = (
                canonical.get("events")
                if isinstance(canonical, Mapping)
                else record.get("events") or record.get("trajectory") or record.get("execution_trace")
            )
            if not isinstance(raw_events, (list, tuple)):
                raw_events = metadata.get("execution_trace") if isinstance(metadata.get("execution_trace"), (list, tuple)) else None
            if not isinstance(raw_events, (list, tuple)):
                raw_events = [record]
            hidden_sequence = record.get("hidden_supervision") or record.get("tool_state_label")
            if not isinstance(hidden_sequence, (list, tuple)):
                hidden_sequence = metadata.get("bayes_supervision") if isinstance(metadata.get("bayes_supervision"), (list, tuple)) else None
            history: list[Mapping[str, Any]] = []
            for step_index, raw_event in enumerate(raw_events):
                event_record = dict(raw_event) if isinstance(raw_event, Mapping) else {"observed_result": str(raw_event)}
                event = event_record.get("world_event", event_record.get("event", event_record))
                event = dict(event) if isinstance(event, Mapping) else {}
                tool = str(
                    event_record.get("tool")
                    or event_record.get("tool_name")
                    or event.get("tool")
                    or event.get("tool_name")
                    or "parse_document"
                )
                result = str(
                    event_record.get("observed_result")
                    or event_record.get("result")
                    or event_record.get("output")
                    or ""
                )
                task_value = (
                    event_record.get("task_state_before")
                    or event_record.get("task_state")
                    or record.get("task_state")
                    or metadata.get("task_state", {})
                )
                task = TaskStateView.from_navigation_state(task_value if isinstance(task_value, Mapping) else {})
                hidden_value = (
                    event_record.get("hidden_label")
                    or event_record.get("hidden_supervision")
                    or event_record.get("tool_state_label")
                )
                if hidden_value is None and isinstance(hidden_sequence, (list, tuple)) and step_index < len(hidden_sequence):
                    hidden_value = hidden_sequence[step_index]
                hidden_value = hidden_value if isinstance(hidden_value, Mapping) else {}
                features = extract_observation_features(tool, result, event, task, history)
                raw_step_index = event_record.get("call_id", event.get("call_id", step_index))
                try:
                    step_value = int(float(raw_step_index))
                except (TypeError, ValueError):
                    step_value = step_index
                row: dict[str, Any] = {
                    "features": torch.tensor(features.values, dtype=torch.float32),
                    "task_projection": torch.tensor(BeliefRuntime._task_projection(task), dtype=torch.float32),
                    "event": event,
                    "tool_id": TOOL_NAMES.index(tool) if tool in TOOL_NAMES else 0,
                    "has_hidden_target": bool(hidden_value),
                    "hidden": dict(hidden_value),
                    "trajectory_id": trajectory_id,
                    "step_index": step_value,
                }
                self.rows.append(row)
                grouped[trajectory_id].append(row)
                history.append(event)

        chunk_length = max(1, int(sequence_length))
        self.sequences: list[list[dict[str, Any]]] = []
        for trajectory_id, rows in grouped.items():
            rows.sort(key=lambda row: int(row.get("step_index", 0)))
            for start in range(0, len(rows), chunk_length):
                self.sequences.append(rows[start : start + chunk_length])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return self.sequences[index]


def _fixed_vector(value: Any, width: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return (values + [0.0] * width)[:width]


class BayesQReplayDataset(Dataset):
    """Replay rows for the finite-branch BayesQHead Gaussian target."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.rows: list[dict[str, torch.Tensor]] = []
        for record in records:
            sources: list[Mapping[str, Any]] = [record]
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                sources.append(metadata)
                nested_bayes = metadata.get("bayestool")
                if isinstance(nested_bayes, Mapping):
                    sources.append(nested_bayes)
            raw_groups: list[Any] = []
            for source in sources:
                for field in ("q_records", "branch_records", "bayes_branch_records"):
                    if field in source and source.get(field) is not None:
                        raw_groups.append(source.get(field))
                if "task_features" in source:
                    raw_groups.append(source)

            def iter_rows(value: Any) -> Iterable[Mapping[str, Any]]:
                if isinstance(value, Mapping):
                    if "task_features" in value:
                        yield value
                        return
                    for child in value.values():
                        yield from iter_rows(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        yield from iter_rows(child)

            for raw in (row for group in raw_groups for row in iter_rows(group)):
                if not isinstance(raw, Mapping):
                    continue
                task = _fixed_vector(raw.get("task_features"), 32)
                particle = _fixed_vector(raw.get("particle_features"), 32)
                action = _fixed_vector(raw.get("action_features"), 32)
                budget = _fixed_vector(raw.get("budget_features"), 8)
                if task is None or particle is None or action is None or budget is None:
                    continue
                action = normalize_q_action_feature_vector(action)
                target = raw.get("utility", raw.get("policy_utility", raw.get("oracle_utility")))
                try:
                    target_value = float(target)
                except (TypeError, ValueError):
                    continue
                self.rows.append(
                    {
                        "task": torch.tensor(task, dtype=torch.float32),
                        "particle": torch.tensor(particle, dtype=torch.float32),
                        "action": torch.tensor(action, dtype=torch.float32),
                        "budget": torch.tensor(budget, dtype=torch.float32),
                        "target": torch.tensor(target_value, dtype=torch.float32),
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _categorical_target(event: Mapping[str, Any], key: str, classes: tuple[str, ...], default: str) -> int:
    value = str(event.get(key) or default).casefold()
    return classes.index(value) if value in classes else classes.index(default)


def _batch_targets(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    events = batch["event"] if isinstance(batch, Mapping) else batch
    if isinstance(events, Mapping):
        events = [events]
    status_classes = ("ok", "partial", "error", "timeout", "invalid", "empty")

    def present(event: Mapping[str, Any], key: str) -> bool:
        return key in event and event.get(key) is not None

    status = torch.tensor(
        [_categorical_target(event, "status", status_classes, "ok") for event in events],
        dtype=torch.long,
    )
    latency = torch.tensor(
        [max(0, min(7, int(float(event.get("latency", 0.0) or 0.0) * 2))) for event in events],
        dtype=torch.long,
    )
    information = torch.tensor(
        [max(0, min(4, int(float(event.get("information_gain", 0.0) or 0.0) * 5))) for event in events],
        dtype=torch.long,
    )
    semantic = torch.tensor(
        [max(0, min(4, int(float(event.get("semantic_agreement", 0.0) or 0.0) * 5))) for event in events],
        dtype=torch.long,
    )
    schema = torch.tensor(
        [float(bool(event.get("schema_valid", False))) for event in events],
        dtype=torch.float32,
    )
    image = torch.tensor(
        [float(bool(event.get("image_valid", False))) for event in events],
        dtype=torch.float32,
    )
    return {
        "status": status,
        "latency_bin": latency,
        "information_gain": information,
        "semantic_agreement": semantic,
        "schema_valid": schema,
        "image_valid": image,
        "status_mask": torch.tensor([float(present(event, "status")) for event in events], dtype=torch.float32),
        "latency_bin_mask": torch.tensor([float(present(event, "latency")) for event in events], dtype=torch.float32),
        "information_gain_mask": torch.tensor([float(present(event, "information_gain")) for event in events], dtype=torch.float32),
        "semantic_agreement_mask": torch.tensor([float(present(event, "semantic_agreement")) for event in events], dtype=torch.float32),
        "schema_valid_mask": torch.tensor([float(present(event, "schema_valid")) for event in events], dtype=torch.float32),
        "image_valid_mask": torch.tensor([float(present(event, "image_valid")) for event in events], dtype=torch.float32),
    }


def _collate(sequences: list[list[dict[str, Any]]]) -> dict[str, Any]:
    if not sequences:
        raise ValueError("cannot collate an empty replay batch")
    batch_size = len(sequences)
    max_length = max(len(sequence) for sequence in sequences)
    features = torch.zeros(batch_size, max_length, len(FEATURE_NAMES), dtype=torch.float32)
    task_projection = torch.zeros(batch_size, max_length, 16, dtype=torch.float32)
    tool_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    events: list[list[dict[str, Any]]] = [[] for _ in sequences]
    hidden: list[list[dict[str, Any]]] = [[] for _ in sequences]
    next_events: list[list[dict[str, Any] | None]] = [[] for _ in sequences]
    for batch_index, sequence in enumerate(sequences):
        for time_index, row in enumerate(sequence):
            features[batch_index, time_index] = row["features"]
            task_projection[batch_index, time_index] = row["task_projection"]
            tool_ids[batch_index, time_index] = int(row["tool_id"])
            mask[batch_index, time_index] = True
            events[batch_index].append(row["event"])
            hidden[batch_index].append(row["hidden"])
            next_events[batch_index].append(
                sequence[time_index + 1]["event"] if time_index + 1 < len(sequence) else None
            )
    return {
        "features": features,
        "task_projection": task_projection,
        "event": events,
        "next_event": next_events,
        "tool_id": tool_ids,
        "mask": mask,
        "has_hidden_target": any(row["has_hidden_target"] for sequence in sequences for row in sequence),
        "hidden": hidden,
    }


def _smoother_hidden_target_loss(smoother_output: Mapping[str, torch.Tensor], hidden: list[Any]) -> torch.Tensor:
    """Use only explicit synthetic hidden labels; real logs contribute zero."""

    session_names = ("healthy", "degraded", "overloaded", "outage")
    regime_names = ("stable", "abrupt_transition", "gradual_transition")
    device = smoother_output["session_logits"].device
    session_logits = smoother_output["session_logits"]
    regime_logits = smoother_output["regime_logits"]
    change_logits = smoother_output["change_logits"]
    session_targets: list[int] = []
    regime_targets: list[int] = []
    change_targets: list[float] = []
    positions: list[int] = []
    for index, value in enumerate(hidden):
        if not isinstance(value, Mapping):
            continue
        session = str(value.get("session_state") or "").casefold()
        regime = str(value.get("regime_state") or "").casefold()
        if session not in session_names and regime not in regime_names:
            continue
        positions.append(index)
        session_targets.append(session_names.index(session) if session in session_names else 0)
        regime_targets.append(regime_names.index(regime) if regime in regime_names else 0)
        change_targets.append(float(regime in {"abrupt_transition", "gradual_transition"}))
    if not positions:
        return session_logits.sum() * 0.0
    index_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    total = (
        torch.nn.functional.cross_entropy(session_logits.index_select(0, index_tensor), torch.tensor(session_targets, device=device))
        + torch.nn.functional.cross_entropy(regime_logits.index_select(0, index_tensor), torch.tensor(regime_targets, device=device))
        + torch.nn.functional.binary_cross_entropy_with_logits(
            change_logits.index_select(0, index_tensor), torch.tensor(change_targets, dtype=torch.float32, device=device)
        )
    )
    shared_logits = smoother_output.get("shared_logits", {})
    if isinstance(shared_logits, Mapping):
        for family, logits in shared_logits.items():
            family_positions: list[int] = []
            family_targets: list[int] = []
            family_names = ("healthy", "degraded", "down")
            for index, value in enumerate(hidden):
                if not isinstance(value, Mapping):
                    continue
                shared = value.get("shared_states", {})
                state = str(shared.get(family, "")).casefold() if isinstance(shared, Mapping) else ""
                if state in family_names:
                    family_positions.append(index)
                    family_targets.append(family_names.index(state))
            if family_positions:
                positions_tensor = torch.tensor(family_positions, dtype=torch.long, device=device)
                total = total + torch.nn.functional.cross_entropy(
                    logits.index_select(0, positions_tensor),
                    torch.tensor(family_targets, dtype=torch.long, device=device),
                )
    # The smoother also receives explicit hidden quality/cost labels from
    # synthetic worlds.  Train all tool dimensions with masks instead of only
    # session/regime labels; real trajectories without these labels contribute
    # zero to this branch.
    quality_raw = smoother_output.get("quality_raw")
    cost_raw = smoother_output.get("cost_raw")
    if isinstance(quality_raw, torch.Tensor):
        params = torch.nn.functional.softplus(quality_raw) + 1.0
        means = params[..., 0] / params.sum(dim=-1)
        quality_losses: list[torch.Tensor] = []
        cost_losses: list[torch.Tensor] = []
        for index, value in enumerate(hidden):
            if not isinstance(value, Mapping):
                continue
            tool_name = str(value.get("tool_name") or "")
            quality = value.get("quality")
            if tool_name not in TOOL_NAMES or not isinstance(quality, Mapping):
                continue
            tool_index = TOOL_NAMES.index(tool_name)
            targets = [
                float(quality.get("availability", 0.0) or 0.0),
                float(quality.get("semantic_accuracy", 0.0) or 0.0),
                float(quality.get("structure_fidelity", 0.0) or 0.0),
                1.0 / max(0.05, float(quality.get("calibration_temperature", 1.0) or 1.0)),
            ]
            quality_losses.append(
                torch.nn.functional.mse_loss(
                    means[index, tool_index, :4],
                    torch.tensor(targets, dtype=torch.float32, device=device).clamp(0.0, 1.0),
                )
            )
            if isinstance(cost_raw, torch.Tensor):
                predicted_cost = torch.exp(cost_raw[index, tool_index, 0]).clamp_min(0.05)
                target_cost = max(0.05, float(quality.get("relative_cost", 1.0) or 1.0))
                cost_losses.append(
                    torch.nn.functional.smooth_l1_loss(
                        torch.log(predicted_cost),
                        torch.tensor(math.log(target_cost), dtype=torch.float32, device=device),
                    )
                )
        if quality_losses:
            total = total + torch.stack(quality_losses).mean()
        if cost_losses:
            total = total + torch.stack(cost_losses).mean()
    return total


def _transition_target_loss(
    filter_output: Mapping[str, torch.Tensor],
    events: list[Any],
    hidden: list[Any],
) -> torch.Tensor:
    """Train the transition head only where a transition target is observable."""

    logits = filter_output["transition_logits"]
    targets: list[int] = []
    positions: list[int] = []
    regime_names = {"stable": 0, "abrupt_transition": 1, "gradual_transition": 2}
    for index, (event, hidden_value) in enumerate(zip(events, hidden, strict=False)):
        event = event if isinstance(event, Mapping) else {}
        hidden_value = hidden_value if isinstance(hidden_value, Mapping) else {}
        regime = str(hidden_value.get("regime_state") or event.get("regime_state") or "").casefold()
        if regime in regime_names:
            positions.append(index)
            targets.append(regime_names[regime])
            continue
        if "change_detected" in event:
            positions.append(index)
            targets.append(1 if bool(event.get("change_detected")) else 0)
    if not positions:
        return logits.sum() * 0.0
    position_tensor = torch.tensor(positions, dtype=torch.long, device=logits.device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits.index_select(0, position_tensor), target_tensor)


def _calibration_loss(
    filter_output: Mapping[str, torch.Tensor],
    events: list[Any],
    tool_ids: torch.Tensor,
) -> torch.Tensor:
    """Brier loss for observable quality proxies and explicit change labels."""

    quality_raw = filter_output["quality_raw"]
    quality_params = torch.nn.functional.softplus(quality_raw) + 1.0
    quality_means = quality_params[..., 0] / quality_params.sum(dim=-1)
    row_ids = torch.arange(quality_means.shape[0], device=quality_means.device)
    selected_quality = quality_means[row_ids, tool_ids.to(device=quality_means.device)]
    quality_targets: dict[int, list[float]] = {0: [], 1: [], 2: []}
    quality_masks: dict[int, list[bool]] = {0: [], 1: [], 2: []}
    change_targets: list[float] = []
    change_masks: list[bool] = []
    for event in events:
        event = event if isinstance(event, Mapping) else {}
        status = str(event.get("status") or "").casefold()
        availability = {"ok": 1.0, "partial": 0.5, "error": 0.0, "timeout": 0.0, "invalid": 0.0, "empty": 0.0}.get(status)
        quality_targets[0].append(float(availability or 0.0))
        quality_masks[0].append(availability is not None)
        semantic = event.get("semantic_agreement")
        structure = event.get("schema_valid")
        quality_targets[1].append(float(semantic) if semantic is not None else 0.0)
        quality_masks[1].append(semantic is not None)
        quality_targets[2].append(float(bool(structure)) if structure is not None else 0.0)
        quality_masks[2].append(structure is not None)
        if "change_detected" in event:
            change_targets.append(float(bool(event.get("change_detected"))))
            change_masks.append(True)
        else:
            change_targets.append(0.0)
            change_masks.append(False)

    losses: list[torch.Tensor] = []
    for dimension in range(3):
        mask = torch.tensor(quality_masks[dimension], dtype=torch.bool, device=quality_means.device)
        if bool(mask.any()):
            target = torch.tensor(quality_targets[dimension], dtype=torch.float32, device=quality_means.device)
            losses.append(calibration_brier(selected_quality[:, dimension][mask], target[mask]))
    change_logits = filter_output["change_logits"]
    change_mask = torch.tensor(change_masks, dtype=torch.bool, device=change_logits.device)
    if bool(change_mask.any()):
        change_target = torch.tensor(change_targets, dtype=torch.float32, device=change_logits.device)
        losses.append(calibration_brier(torch.sigmoid(change_logits[change_mask]), change_target[change_mask]))
    return sum(losses, quality_raw.sum() * 0.0) / max(1, len(losses))


def _flatten_sequence_output(output: Mapping[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Flatten [B,T,...] outputs and retain only real replay positions."""

    flat_mask = mask.reshape(-1)
    result: dict[str, Any] = {}
    for key, value in output.items():
        if key in {"shared_logits", "shared_hidden", "observation_logits"} or not isinstance(value, torch.Tensor):
            continue
        result[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])[flat_mask]
    shared = output.get("shared_logits", {})
    if isinstance(shared, Mapping):
        result["shared_logits"] = {
            family: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])[flat_mask]
            for family, value in shared.items()
        }
    observations = output.get("observation_logits", {})
    if isinstance(observations, Mapping):
        result["observation_logits"] = {
            name: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])[flat_mask]
            for name, value in observations.items()
        }
    return result


def _flatten_nested_events(values: list[list[Any]], mask: torch.Tensor) -> list[Any]:
    output: list[Any] = []
    for batch_index, row in enumerate(values):
        for time_index, value in enumerate(row):
            if bool(mask[batch_index, time_index]):
                output.append(value)
    return output


def _next_event_targets(values: list[list[Any]], mask: torch.Tensor) -> tuple[list[Any], torch.Tensor]:
    targets: list[Any] = []
    positions: list[int] = []
    flat_index = 0
    for batch_index, row in enumerate(values):
        for time_index, value in enumerate(row):
            if bool(mask[batch_index, time_index]):
                if value is not None:
                    targets.append(value)
                    positions.append(flat_index)
                flat_index += 1
    return targets, torch.tensor(positions, dtype=torch.long)


def _belief_batch_losses(
    network: ToolWorldFilterNetwork,
    smoother: ToolWorldSmoother,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    use_hbd: bool = True,
) -> dict[str, torch.Tensor]:
    features = batch["features"].to(device)
    task_projection = batch["task_projection"].to(device)
    tool_ids = batch["tool_id"].to(device)
    mask = batch["mask"].to(device)
    if not bool(mask.any()):
        zero = next(network.parameters()).sum() * 0.0
        return {name: zero for name in ("loss", "hbd", "transition", "obs", "calibration", "smoother")}
    filter_sequence = network.forward_sequence(features, tool_ids, task_projection=task_projection, mask=mask)
    smooth_sequence = smoother(features)
    filter_output = _flatten_sequence_output(filter_sequence, mask)
    smooth_single = _flatten_sequence_output(smooth_sequence, mask)
    events = _flatten_nested_events(batch["event"], mask)
    hidden = _flatten_nested_events(batch["hidden"], mask)
    loss_hbd = hbd_loss(filter_output, smooth_single) if use_hbd else filter_output["session_logits"].sum() * 0.0
    next_events, next_positions = _next_event_targets(batch["next_event"], mask)
    if next_events:
        next_positions = next_positions.to(device)
        observation_logits = {
            name: logits.index_select(0, next_positions)
            for name, logits in filter_output["observation_logits"].items()
        }
        loss_obs = observation_prediction_loss(observation_logits, _batch_targets(next_events))
    else:
        loss_obs = filter_output["session_logits"].sum() * 0.0
    loss_smoother = _smoother_hidden_target_loss(smooth_single, hidden)
    loss_transition = _transition_target_loss(filter_output, events, hidden)
    tool_ids_flat = tool_ids.reshape(-1)[mask.reshape(-1)]
    loss_calibration = _calibration_loss(filter_output, events, tool_ids_flat)
    loss = loss_hbd + 0.5 * loss_transition + loss_obs + 0.2 * loss_calibration + loss_smoother
    return {
        "loss": loss,
        "hbd": loss_hbd,
        "transition": loss_transition,
        "obs": loss_obs,
        "calibration": loss_calibration,
        "smoother": loss_smoother,
    }


def train_epoch(
    network: ToolWorldFilterNetwork,
    smoother: ToolWorldSmoother,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_hbd: bool = True,
) -> dict[str, float]:
    network.train()
    smoother.train()
    totals = {
        "loss": 0.0,
        "hbd": 0.0,
        "transition": 0.0,
        "obs": 0.0,
        "calibration": 0.0,
        "smoother": 0.0,
    }
    batches = 0
    for batch in loader:
        losses = _belief_batch_losses(network, smoother, batch, device=device, use_hbd=use_hbd)
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group.get("params", [])
            if parameter.grad is not None
        ]
        clip_grad_norm_(optimizer_parameters, 1.0)
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def evaluate_belief_epoch(
    network: ToolWorldFilterNetwork,
    smoother: ToolWorldSmoother,
    loader: DataLoader,
    *,
    device: torch.device,
    use_hbd: bool = True,
) -> dict[str, float]:
    network.eval()
    smoother.eval()
    totals = {name: 0.0 for name in ("loss", "hbd", "transition", "obs", "calibration", "smoother")}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            losses = _belief_batch_losses(network, smoother, batch, device=device, use_hbd=use_hbd)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def train_q_head(
    q_head: BayesQHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, float]:
    q_head.train()
    total = 0.0
    batches = 0
    for batch in loader:
        prediction = q_head(
            batch["task"].to(device),
            batch["particle"].to(device),
            batch["action"].to(device),
            batch["budget"].to(device),
        )
        loss = bayes_q_gaussian_nll(prediction, batch["target"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(q_head.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
    return {"q_loss": total / max(1, batches), "q_rows": float(len(loader.dataset))}


def _document_split_key(record: Mapping[str, Any], index: int) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    value = (
        record.get("document_hash")
        or metadata.get("document_hash")
        or record.get("document_path")
        or metadata.get("document_path")
        or record.get("trajectory_id")
        or record.get("rollout_id")
        or f"record:{index}"
    )
    return str(value)


def split_records_by_document(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> dict[str, list[dict[str, Any]]]:
    """Split replay by document identity so questions from one PDF cannot cross splits."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[_document_split_key(record, index)].append(dict(record))
    keys = sorted(grouped)
    decorated = sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{seed}|{key}".encode("utf-8", "surrogatepass")).digest(),
    )
    if len(decorated) < 3:
        return {"train": [item for key in decorated for item in grouped[key]], "validation": [], "test": []}
    def _requested_count(fraction: float) -> int:
        fraction = max(0.0, float(fraction))
        if fraction <= 0.0:
            return 0
        return max(1, int(round(len(decorated) * fraction)))

    validation_count = _requested_count(validation_fraction)
    test_count = _requested_count(test_fraction)
    while validation_count + test_count >= len(decorated):
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    test_keys = decorated[:test_count]
    validation_keys = decorated[test_count : test_count + validation_count]
    train_keys = decorated[test_count + validation_count :]
    return {
        "train": [item for key in train_keys for item in grouped[key]],
        "validation": [item for key in validation_keys for item in grouped[key]],
        "test": [item for key in test_keys for item in grouped[key]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL belief replay")
    parser.add_argument("--output", type=Path, required=True, help="filter checkpoint path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--without-hbd",
        action="store_true",
        help="Ablation switch: do not distil the offline smoother into the online filter.",
    )
    parser.add_argument(
        "--q-replay",
        type=Path,
        default=None,
        help="Optional JSONL replay with task/particle/action/budget features and utility targets",
    )
    parser.add_argument("--q-output", type=Path, default=None, help="Optional BayesQHead checkpoint")
    parser.add_argument("--q-epochs", type=int, default=1)
    parser.add_argument("--q-batch-size", type=int, default=256)
    parser.add_argument("--q-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--risk-validation", type=Path, default=None, help="Optional validation rollout JSONL for risk calibration")
    parser.add_argument("--risk-output", type=Path, default=None, help="Optional AnswerRiskCalibrator JSON checkpoint")
    parser.add_argument("--risk-label-key", default=None, help="Optional dotted risk/correctness label field")
    parser.add_argument(
        "--risk-label-is-risk",
        action="store_true",
        help="Interpret --risk-label-key as a risk probability/flag instead of a correctness field.",
    )
    parser.add_argument("--risk-epochs", type=int, default=400)
    parser.add_argument("--risk-learning-rate", type=float, default=0.05)
    parser.add_argument("--risk-l2", type=float, default=1.0e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    torch.manual_seed(args.seed)
    records = _read_jsonl(args.input)
    config = default_config(enabled=True)
    split = split_records_by_document(
        records,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    sequence_length = max(1, int(args.sequence_length or config.belief.sequence_length))
    train_dataset = BeliefEventDataset(split["train"], sequence_length=sequence_length)
    validation_dataset = BeliefEventDataset(split["validation"], sequence_length=sequence_length)
    test_dataset = BeliefEventDataset(split["test"], sequence_length=sequence_length)
    if not train_dataset:
        raise ValueError("belief replay has no training trajectories after document-level splitting")
    network = ToolWorldFilterNetwork(config).to(args.device)
    smoother = ToolWorldSmoother(network).to(args.device)
    network_parameters = list(network.parameters())
    network_parameter_ids = {id(parameter) for parameter in network_parameters}
    smoother_parameters = [
        parameter for parameter in smoother.parameters() if id(parameter) not in network_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        network_parameters + smoother_parameters,
        lr=args.learning_rate,
        weight_decay=config.belief.weight_decay,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
        if validation_dataset
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
        if test_dataset
        else None
    )
    metrics: dict[str, float] = {}
    best_validation_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    best_network_state: dict[str, Any] | None = None
    best_smoother_state: dict[str, Any] | None = None

    def _cpu_state(module: torch.nn.Module) -> dict[str, Any]:
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            network,
            smoother,
            train_loader,
            optimizer,
            device=torch.device(args.device),
            use_hbd=not args.without_hbd,
        )
        validation_metrics = (
            evaluate_belief_epoch(
                network,
                smoother,
                validation_loader,
                device=torch.device(args.device),
                use_hbd=not args.without_hbd,
            )
            if validation_loader is not None
            else {}
        )
        monitor_loss = float(validation_metrics.get("loss", train_metrics["loss"]))
        metrics = dict(train_metrics)
        metrics.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        logger.info("epoch=%d train=%s validation=%s", epoch, train_metrics, validation_metrics)
        if validation_loader is None or monitor_loss < best_validation_loss - float(args.early_stop_min_delta):
            best_validation_loss = monitor_loss
            best_epoch = epoch
            stale_epochs = 0
            best_network_state = _cpu_state(network)
            best_smoother_state = _cpu_state(smoother)
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, int(args.early_stop_patience)):
                logger.info("early stopping at epoch=%d after %d stale epochs", epoch, stale_epochs)
                break

    if best_network_state is not None:
        network.load_state_dict(best_network_state)
    if best_smoother_state is not None:
        smoother.load_state_dict(best_smoother_state)
    if test_loader is not None:
        test_metrics = evaluate_belief_epoch(
            network,
            smoother,
            test_loader,
            device=torch.device(args.device),
            use_hbd=not args.without_hbd,
        )
        metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
    metrics.update(
        {
            "best_epoch": float(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "train_sequences": float(len(train_dataset)),
            "validation_sequences": float(len(validation_dataset)),
            "test_sequences": float(len(test_dataset)),
        }
    )
    q_metrics: dict[str, float] = {}
    if args.q_replay is not None:
        q_dataset = BayesQReplayDataset(_read_jsonl(args.q_replay))
        if not q_dataset:
            raise ValueError("BayesQ replay is empty or has no fixed-width feature rows")
        q_head = BayesQHead().to(args.device)
        q_optimizer = torch.optim.AdamW(q_head.parameters(), lr=args.q_learning_rate, weight_decay=1.0e-4)
        q_loader = DataLoader(q_dataset, batch_size=args.q_batch_size, shuffle=True)
        for epoch in range(args.q_epochs):
            q_metrics = train_q_head(q_head, q_loader, q_optimizer, device=torch.device(args.device))
            logger.info("q_epoch=%d metrics=%s", epoch, q_metrics)
        q_output = args.q_output or args.output.with_suffix(".qhead.pt")
        q_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": q_head.state_dict(),
                "metrics": q_metrics,
                "feature_dims": {"task": 32, "particle": 32, "action": 32, "budget": 8},
                "q_feature_schema_version": Q_FEATURE_SCHEMA_VERSION,
            },
            q_output,
        )
        metrics.update(q_metrics)
    if args.risk_validation is not None:
        from calibrate_bayestool_risk import build_validation_rows

        validation_records = _read_jsonl(args.risk_validation)
        risk_features, risk_labels = build_validation_rows(
            validation_records,
            label_key=args.risk_label_key,
            label_is_risk=args.risk_label_is_risk,
        )
        if not risk_features:
            raise ValueError("risk validation replay has no usable correctness/error labels")
        calibrator = AnswerRiskCalibrator()
        risk_metrics = calibrator.fit(
            risk_features,
            risk_labels,
            epochs=args.risk_epochs,
            learning_rate=args.risk_learning_rate,
            l2=args.risk_l2,
        )
        risk_output = args.risk_output or args.output.with_suffix(".risk.json")
        risk_output.parent.mkdir(parents=True, exist_ok=True)
        risk_output.write_text(
            json.dumps(
                {
                    "version": "bayestool-risk-calibrator-v1",
                    "calibrator": calibrator.to_dict(),
                    "metrics": risk_metrics,
                    "source_rows": len(validation_records),
                    "fitted_rows": len(risk_features),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metrics.update({f"risk_{key}": value for key, value in risk_metrics.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": network.state_dict(),
            "smoother_state": smoother.state_dict(),
            "config": config.to_dict(),
            "feature_names": list(FEATURE_NAMES),
            "metrics": metrics,
            "belief_model_version": f"stage-a-{args.seed}-{args.epochs}",
            "document_split": {
                "train_records": len(split["train"]),
                "validation_records": len(split["validation"]),
                "test_records": len(split["test"]),
            },
        },
        args.output,
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
