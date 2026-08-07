"""Online Tool-World filter, offline smoother, and observation predictor."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import CONTENT_TYPES, ERROR_FAMILIES, STATUS_NAMES, TOOL_FAMILIES, TOOL_NAMES, BayesToolConfig, default_config
from .schema import (
    BeliefSnapshot,
    ObservationFeatures as SchemaObservationFeatures,
    ObservationPrediction,
    TaskStateView,
    ToolQualityPosterior,
    WorldEvent,
)

try:  # PyTorch is optional for local parser/world tests.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _group_names(prefix: str, names: Sequence[str], count: int | None = None) -> list[str]:
    values = [f"{prefix}:{name}" for name in names]
    if count is not None:
        values.extend(f"{prefix}:summary_{index}" for index in range(max(0, count - len(values))))
    return values


FEATURE_NAMES: tuple[str, ...] = tuple(
    _group_names("tool", TOOL_NAMES)
    + _group_names("status", STATUS_NAMES)
    + _group_names("error", ERROR_FAMILIES)
    + [
        "general:latency",
        "general:char_count",
        "general:token_count",
        "general:image_count",
        "general:page_count",
        "general:line_count",
        "general:table_marker_count",
        "general:figure_marker_count",
        "general:truncation_ratio",
        "general:result_bytes",
        "general:has_error",
        "general:has_status",
        "general:has_image",
        "general:has_bbox",
        "general:information_gain",
        "general:cost",
        "structure:bbox_valid",
        "structure:row_count",
        "structure:column_count",
        "structure:schema_valid",
        "structure:duplicate_ratio",
        "structure:region_count",
        "structure:table_count",
        "structure:layout_count",
        "structure:has_header",
        "structure:has_cells",
        "structure:has_coordinates",
        "structure:has_markdown_table",
        "structure:has_json_schema",
        "structure:structure_score",
        "structure:missing_flag",
        "structure:reserved",
        "semantic:ocr_parse_agreement",
        "semantic:table_text_agreement",
        "semantic:numeric_agreement",
        "semantic:question_term_overlap",
        "semantic:repeated_text_ratio",
        "semantic:confidence_mean",
        "semantic:confidence_std",
        "semantic:language_consistency",
        "semantic:value_count",
        "semantic:nonempty_ratio",
        "semantic:semantic_score",
        "semantic:has_confidence",
        "semantic:has_text",
        "semantic:has_numeric",
        "semantic:missing_flag",
        "semantic:reserved",
        "task:question_type_text",
        "task:question_type_table",
        "task:question_type_chart",
        "task:question_type_visual",
        "task:current_page",
        "task:visited_page_count",
        "task:unvisited_page_count",
        "task:remaining_budget",
        "task:evidence_sufficient",
        "task:visual_input_required",
        "task:repeat_call",
        "task:information_gain",
        "task:phase_search",
        "task:phase_commit",
        "task:phase_probe",
        "task:missing_flag",
        "history:consecutive_failures",
        "history:recent_surprise",
        "history:last_tool_index",
        "history:last_family_index",
        "history:call_index",
        "history:family_failure_count",
        "history:tool_failure_count",
        "history:recent_information_gain",
        "history:recent_status_error",
        "history:reserved",
    ]
)

if len(FEATURE_NAMES) != 96:  # Fail loudly if the named feature contract changes.
    raise RuntimeError(f"BayesTool feature contract must remain 96-dimensional, got {len(FEATURE_NAMES)}")

FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_PUBLIC_EVENT_FIELDS = frozenset(
    {
        "tool",
        "tool_name",
        "status",
        "latency",
        "information_gain",
        "semantic_agreement",
        "schema_valid",
        "image_valid",
        "error_family",
        "page_number",
        "page_numbers",
        "region",
        "content_type",
        "observation_status",
        "corruption_applied",
        "execution_succeeded",
        "observation_delivered",
    }
)


def _clip(value: Any, low: float = -5.0, high: float = 5.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        value = 0.0
    return max(low, min(high, value))


def _log1p(value: Any) -> float:
    try:
        value = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
    return _clip(math.log1p(value))


def _get(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    mapping = mapping or {}
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _text_stats(text: str) -> dict[str, float]:
    text = str(text or "")
    lines = text.splitlines()
    numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", text)
    return {
        "char_count": _log1p(len(text)),
        "token_count": _log1p(len(text.split())),
        "image_count": _log1p(len(re.findall(r"(?:png|jpg|jpeg|webp)", text, re.IGNORECASE))),
        "page_count": _log1p(len(re.findall(r"page|页", text, re.IGNORECASE))),
        "line_count": _log1p(len(lines)),
        "table_marker_count": _log1p(text.count("|")),
        "figure_marker_count": _log1p(len(re.findall(r"figure|图", text, re.IGNORECASE))),
        "truncation_ratio": 1.0 if "truncated" in text.casefold() or "截断" in text else 0.0,
        "result_bytes": _log1p(len(text.encode("utf-8", "ignore"))),
        "has_error": float(bool(re.search(r"error|failed|timeout|不可用|失败", text, re.IGNORECASE))),
        "has_status": float(bool(re.search(r"status|状态", text, re.IGNORECASE))),
        "has_image": float(bool(re.search(r"png|jpg|jpeg|webp|image", text, re.IGNORECASE))),
        "has_bbox": float(bool(re.search(r"bbox|bounding|坐标", text, re.IGNORECASE))),
        "information_gain": _log1p(len(numbers) + len(text) / 100.0),
        "cost": _log1p(len(text) / 200.0),
    }


def _normalise_event(event: WorldEvent | Mapping[str, Any] | None, *, tool_name: str, result: str) -> dict[str, Any]:
    if isinstance(event, WorldEvent):
        value = event.visible_dict()
    elif isinstance(event, Mapping):
        # Replay rows may carry ``WorldEvent.to_dict()`` beside the event as
        # hidden supervision.  Reconstruct the exact public observation view
        # instead of allowing relative_cost, corruption_type, world_id, or
        # any other label-bearing field to enter filter features.
        value = {str(key): event[key] for key in _PUBLIC_EVENT_FIELDS if key in event}
    else:
        value = {}
    value.setdefault("tool", tool_name)
    value.setdefault("status", "ok" if result.strip() else "empty")
    value.setdefault("latency", 0.0)
    value.setdefault("information_gain", 0.0)
    value.setdefault("semantic_agreement", 0.5)
    value.setdefault("schema_valid", True)
    value.setdefault("image_valid", True)
    value.setdefault("error_family", "none")
    return value


def extract_observation_features(
    tool_name: str,
    observed_result: str,
    event: WorldEvent | Mapping[str, Any] | None = None,
    task_state: TaskStateView | Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> SchemaObservationFeatures:
    """Extract the fixed named 96-D observation contract.

    This function accepts only policy-visible result/event information.  The
    hidden world label is deliberately not part of its signature.
    """

    task = task_state if isinstance(task_state, TaskStateView) else TaskStateView.from_navigation_state(task_state or {})
    event_data = _normalise_event(event, tool_name=tool_name, result=observed_result)
    values = [0.0] * len(FEATURE_NAMES)
    missing: list[str] = []

    def set_value(name: str, value: Any, *, missing_if_none: bool = False) -> None:
        if name not in FEATURE_INDEX:
            return
        if value is None and missing_if_none:
            missing.append(name)
            value = 0.0
        values[FEATURE_INDEX[name]] = _clip(value)

    for name in TOOL_NAMES:
        set_value(f"tool:{name}", float(name == tool_name))
    status = str(event_data.get("status") or "empty").casefold()
    if status not in STATUS_NAMES:
        status = "invalid"
    for name in STATUS_NAMES:
        set_value(f"status:{name}", float(name == status))
    error_family = str(event_data.get("error_family") or "none").casefold()
    if error_family not in ERROR_FAMILIES:
        error_family = "none"
    for name in ERROR_FAMILIES:
        set_value(f"error:{name}", float(name == error_family))

    stats = _text_stats(observed_result)
    for name, value in stats.items():
        set_value(f"general:{name}", value)
    set_value("general:latency", _log1p(event_data.get("latency")), missing_if_none=event_data.get("latency") is None)
    set_value("general:information_gain", event_data.get("information_gain"), missing_if_none=event_data.get("information_gain") is None)
    set_value("general:cost", event_data.get("relative_cost", stats["cost"]))

    rows = len(re.findall(r"(?:^|\n)\s*\|", observed_result))
    columns = max((line.count("|") for line in observed_result.splitlines()), default=0)
    structure_values = {
        "bbox_valid": float(bool(event_data.get("region") is not None or "bbox" in observed_result)),
        "row_count": _log1p(rows),
        "column_count": _log1p(columns),
        "schema_valid": float(bool(event_data.get("schema_valid", True))),
        "duplicate_ratio": float(len(observed_result) > 0 and observed_result.count("\n") != len(set(observed_result.splitlines()))),
        "region_count": _log1p(len(re.findall(r"region|bbox|区域", observed_result, re.IGNORECASE))),
        "table_count": _log1p(len(re.findall(r"table|表格", observed_result, re.IGNORECASE))),
        "layout_count": _log1p(len(re.findall(r"layout|paragraph|title|figure", observed_result, re.IGNORECASE))),
        "has_header": float(bool(re.search(r"header|标题", observed_result, re.IGNORECASE))),
        "has_cells": float("|" in observed_result),
        "has_coordinates": float(bool(re.search(r"\b\d+(?:\.\d+)?[, ]+\d", observed_result))),
        "has_markdown_table": float("|" in observed_result and "---" in observed_result),
        "has_json_schema": float(observed_result.lstrip().startswith(("{", "["))),
        "structure_score": float(event_data.get("schema_valid", True)) * float(event_data.get("semantic_agreement", 0.5)),
        "missing_flag": float(not observed_result.strip()),
        "reserved": 0.0,
    }
    for name, value in structure_values.items():
        set_value(f"structure:{name}", value)

    numeric_values = re.findall(r"[-+]?\d+(?:[.,]\d+)?", observed_result)
    confidence_values = [float(value) for value in re.findall(r"confidence\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)", observed_result, re.IGNORECASE)]
    question_tokens = {
        token.casefold()
        for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(task.question or ""))
        if len(token) > 1
    }
    observed_tokens = {
        token.casefold()
        for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(observed_result or ""))
        if len(token) > 1
    }
    question_overlap = (
        len(question_tokens & observed_tokens) / max(1, len(question_tokens))
        if question_tokens
        else 0.0
    )
    semantic_values = {
        "ocr_parse_agreement": event_data.get("semantic_agreement", 0.5),
        "table_text_agreement": float("|" in observed_result and "table" in observed_result.casefold()),
        "numeric_agreement": float(bool(numeric_values)),
        "question_term_overlap": question_overlap,
        "repeated_text_ratio": float(len(observed_result) > 0 and len(set(observed_result.split())) < max(1, len(observed_result.split())) * 0.65),
        "confidence_mean": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "confidence_std": (
            (sum((value - sum(confidence_values) / len(confidence_values)) ** 2 for value in confidence_values) / len(confidence_values)) ** 0.5
            if confidence_values
            else 0.0
        ),
        "language_consistency": float(not bool(re.search(r"�", observed_result))),
        "value_count": _log1p(len(numeric_values)),
        "nonempty_ratio": float(bool(observed_result.strip())),
        "semantic_score": event_data.get("semantic_agreement", 0.5),
        "has_confidence": float(bool(confidence_values)),
        "has_text": float(bool(observed_result.strip())),
        "has_numeric": float(bool(numeric_values)),
        "missing_flag": float(not observed_result.strip()),
        "reserved": 0.0,
    }
    for name, value in semantic_values.items():
        set_value(f"semantic:{name}", value)

    question_type = task.question_type.casefold()
    task_values = {
        "question_type_text": float(question_type == "text"),
        "question_type_table": float(question_type == "table"),
        "question_type_chart": float(question_type == "chart"),
        "question_type_visual": float(question_type in {"visual", "figure"}),
        "current_page": _log1p(task.current_page or 0),
        "visited_page_count": _log1p(len(task.visited_pages)),
        "unvisited_page_count": _log1p(task.unvisited_page_count or 0),
        "remaining_budget": _log1p(task.remaining_tool_budget),
        "evidence_sufficient": float(task.evidence_sufficient),
        "visual_input_required": float(task.visual_input_required),
        "repeat_call": float(bool(history and history[-1].get("tool") == tool_name)),
        "information_gain": event_data.get("information_gain", 0.0),
        "phase_search": float(task.phase == "search"),
        "phase_commit": float(task.phase == "commit"),
        "phase_probe": float(task.phase == "probe"),
        "missing_flag": 0.0,
    }
    for name, value in task_values.items():
        set_value(f"task:{name}", value)

    history = history or ()
    recent_failures = 0
    for item in reversed(history):
        if str(item.get("status", "ok")) in {"error", "timeout", "invalid", "empty"}:
            recent_failures += 1
        else:
            break
    family = next((name for name, tools in TOOL_FAMILIES.items() if tool_name in tools), "")
    family_failures = sum(
        str(item.get("status", "ok")) in {"error", "timeout", "invalid", "empty"}
        and str(item.get("tool")) in TOOL_FAMILIES.get(family, ())
        for item in history
    )
    tool_failures = sum(
        str(item.get("status", "ok")) in {"error", "timeout", "invalid", "empty"}
        and str(item.get("tool")) == tool_name
        for item in history
    )
    last_tool = str(history[-1].get("tool")) if history else ""
    last_family = next((name for name, tools in TOOL_FAMILIES.items() if last_tool in tools), "")
    recent_surprise = float(history[-1].get("predictive_surprise", 0.0)) if history else 0.0
    recent_gain = float(history[-1].get("information_gain", 0.0)) if history else 0.0
    history_values = {
        "consecutive_failures": _log1p(recent_failures),
        "recent_surprise": _log1p(recent_surprise),
        "last_tool_index": float(TOOL_NAMES.index(last_tool) / max(1, len(TOOL_NAMES) - 1)) if last_tool in TOOL_NAMES else 0.0,
        "last_family_index": float(list(TOOL_FAMILIES).index(last_family) / max(1, len(TOOL_FAMILIES) - 1)) if last_family in TOOL_FAMILIES else 0.0,
        "call_index": _log1p(len(history)),
        "family_failure_count": _log1p(family_failures),
        "tool_failure_count": _log1p(tool_failures),
        "recent_information_gain": _log1p(recent_gain),
        "recent_status_error": float(status in {"error", "timeout", "invalid", "empty"}),
        "reserved": 0.0,
    }
    for name, value in history_values.items():
        set_value(f"history:{name}", value)

    return SchemaObservationFeatures(tuple(_clip(value) for value in values), FEATURE_NAMES, tuple(missing))


# Public alias keeps the name in the final implementation plan while the
# schema module remains the canonical data-record definition.
ObservationFeatures = SchemaObservationFeatures


if torch is not None:

    class ToolWorldFilterNetwork(nn.Module):
        """Small recurrent filter specified by the method, not a second LLM."""

        def __init__(self, config: BayesToolConfig | None = None) -> None:
            super().__init__()
            config = config or default_config(enabled=True)
            belief = config.belief
            self.config = config
            self.observation_encoder = nn.Sequential(
                nn.Linear(belief.feature_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
            )
            self.session_filter = nn.GRUCell(256, belief.session_hidden)
            self.context_filter = nn.GRUCell(256, belief.context_hidden)
            self.shared_filters = nn.ModuleDict({name: nn.GRUCell(256, belief.shared_hidden) for name in TOOL_FAMILIES})
            self.tool_embedding = nn.Embedding(len(TOOL_NAMES), belief.tool_embedding_dim)
            self.session_head = nn.Linear(belief.session_hidden, 4)
            self.regime_head = nn.Linear(belief.session_hidden, 3)
            self.shared_heads = nn.ModuleDict({name: nn.Linear(belief.shared_hidden, 3) for name in TOOL_FAMILIES})
            quality_dim = len(TOOL_NAMES) * 4 * 2
            self.quality_head = nn.Linear(belief.session_hidden, quality_dim)
            self.cost_head = nn.Linear(belief.session_hidden, len(TOOL_NAMES) * 2)
            self.change_head = nn.Linear(belief.session_hidden, 1)
            # The transition target contract is the same three-way regime
            # contract used by ``regime_head``: stable, abrupt, gradual.
            observation_input_dim = belief.session_hidden + belief.context_hidden + belief.tool_embedding_dim + 16
            self.transition_head = nn.Linear(observation_input_dim, 3)
            self.observation_head = nn.Sequential(
                nn.Linear(observation_input_dim, 128),
                nn.GELU(),
                nn.Linear(128, 6 + 8 + 5 + 5 + 2 + 2),
            )

        def encode(self, features: Tensor) -> Tensor:
            return self.observation_encoder(features)

        def forward(
            self,
            features: Tensor,
            *,
            session_hidden: Tensor | None = None,
            context_hidden: Tensor | None = None,
            shared_hidden: Mapping[str, Tensor] | None = None,
            tool_id: int | Tensor = 0,
            task_projection: Tensor | None = None,
        ) -> dict[str, Any]:
            encoded = self.encode(features)
            batch = encoded.shape[0]
            device = encoded.device
            session_hidden = session_hidden if session_hidden is not None else torch.zeros(batch, self.session_filter.hidden_size, device=device)
            context_hidden = context_hidden if context_hidden is not None else torch.zeros(batch, self.context_filter.hidden_size, device=device)
            shared_hidden = shared_hidden or {}
            session_next = self.session_filter(encoded, session_hidden)
            context_next = self.context_filter(encoded, context_hidden)
            shared_next = {}
            shared_logits = {}
            for family, cell in self.shared_filters.items():
                hidden = shared_hidden.get(family)
                if hidden is None:
                    hidden = torch.zeros(batch, cell.hidden_size, device=device)
                next_hidden = cell(encoded, hidden)
                shared_next[family] = next_hidden
                shared_logits[family] = self.shared_heads[family](next_hidden)
            quality_raw = self.quality_head(session_next).reshape(batch, len(TOOL_NAMES), 4, 2)
            cost_raw = self.cost_head(session_next).reshape(batch, len(TOOL_NAMES), 2)
            if isinstance(tool_id, torch.Tensor):
                tool_ids = tool_id.to(device=device, dtype=torch.long).reshape(-1)
                if tool_ids.numel() == 1:
                    tool_ids = tool_ids.expand(batch)
                elif tool_ids.numel() != batch:
                    raise ValueError("tool_id tensor must have one value or one value per feature row")
            else:
                tool_ids = torch.full((batch,), int(tool_id), dtype=torch.long, device=device)
            tool_embed = self.tool_embedding(tool_ids)
            task_projection = task_projection if task_projection is not None else torch.zeros(batch, 16, device=device)
            combined = torch.cat((session_next, context_next, tool_embed, task_projection), dim=-1)
            obs_raw = self.observation_head(combined)
            cursor = 0
            obs_logits = {}
            for name, size in (("status", 6), ("latency_bin", 8), ("information_gain", 5), ("semantic_agreement", 5), ("schema_valid", 2), ("image_valid", 2)):
                obs_logits[name] = obs_raw[:, cursor : cursor + size]
                cursor += size
            return {
                "encoded": encoded,
                "session_hidden": session_next,
                "context_hidden": context_next,
                "shared_hidden": shared_next,
                "session_logits": self.session_head(session_next),
                "regime_logits": self.regime_head(session_next),
                "shared_logits": shared_logits,
                "quality_raw": quality_raw,
                "cost_raw": cost_raw,
                "change_logits": self.change_head(session_next).squeeze(-1),
                "transition_logits": self.transition_head(combined),
                "observation_logits": obs_logits,
            }

        def observation_from_hidden(
            self,
            session_hidden: Tensor,
            *,
            context_hidden: Tensor | None = None,
            tool_id: int | Tensor = 0,
            task_projection: Tensor | None = None,
        ) -> dict[str, Tensor]:
            """Predict an observation without advancing the recurrent filter.

            The online runtime uses this method before executing the next
            tool.  Calling ``forward`` with a dummy feature would advance the
            GRU and make the surprise score depend on the observation it is
            supposed to predict, so prediction is kept as a pure readout.
            """

            if session_hidden.dim() == 1:
                session_hidden = session_hidden.unsqueeze(0)
            batch = session_hidden.shape[0]
            device = session_hidden.device
            if context_hidden is None:
                context_hidden = torch.zeros(batch, self.context_filter.hidden_size, device=device)
            elif context_hidden.dim() == 1:
                context_hidden = context_hidden.unsqueeze(0)
            if context_hidden.shape[0] == 1 and batch > 1:
                context_hidden = context_hidden.expand(batch, -1)
            elif context_hidden.shape[0] != batch:
                raise ValueError("context_hidden must have one row or one row per hidden row")
            if isinstance(tool_id, torch.Tensor):
                tool_ids = tool_id.to(device=device, dtype=torch.long).reshape(-1)
                if tool_ids.numel() == 1:
                    tool_ids = tool_ids.expand(batch)
                elif tool_ids.numel() != batch:
                    raise ValueError("tool_id tensor must have one value or one value per hidden row")
            else:
                tool_ids = torch.full((batch,), int(tool_id), dtype=torch.long, device=device)
            task_projection = task_projection if task_projection is not None else torch.zeros(batch, 16, device=device)
            combined = torch.cat((session_hidden, context_hidden, self.tool_embedding(tool_ids), task_projection), dim=-1)
            raw = self.observation_head(combined)
            outputs: dict[str, Tensor] = {}
            cursor = 0
            for name, size in (
                ("status", 6),
                ("latency_bin", 8),
                ("information_gain", 5),
                ("semantic_agreement", 5),
                ("schema_valid", 2),
                ("image_valid", 2),
            ):
                outputs[name] = raw[:, cursor : cursor + size]
                cursor += size
            return outputs

        def forward_sequence(
            self,
            features: Tensor,
            tool_ids: Tensor,
            task_projection: Tensor | None = None,
            mask: Tensor | None = None,
        ) -> dict[str, Any]:
            """Run the causal filter over a batch of padded trajectories."""

            if features.dim() != 3:
                raise ValueError("sequence features must have shape [batch, time, feature_dim]")
            if tool_ids.shape[:2] != features.shape[:2]:
                raise ValueError("tool_ids must have shape [batch, time]")
            if task_projection is not None and task_projection.shape[:2] != features.shape[:2]:
                raise ValueError("task_projection must have shape [batch, time, projection_dim]")
            if mask is not None and mask.shape[:2] != features.shape[:2]:
                raise ValueError("mask must have shape [batch, time]")
            if mask is None:
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
            else:
                mask = mask.to(device=features.device, dtype=torch.bool)
            outputs: dict[str, list[Tensor]] = {}
            shared_outputs: dict[str, list[Tensor]] = {family: [] for family in TOOL_FAMILIES}
            observation_outputs: dict[str, list[Tensor]] = {
                name: []
                for name in ("status", "latency_bin", "information_gain", "semantic_agreement", "schema_valid", "image_valid")
            }
            session_hidden = None
            context_hidden = None
            shared_hidden: dict[str, Tensor] = {}
            for time_index in range(features.shape[1]):
                active = mask[:, time_index]
                previous_session = session_hidden
                previous_context = context_hidden
                previous_shared = dict(shared_hidden)
                output = self.forward(
                    features[:, time_index],
                    session_hidden=session_hidden,
                    context_hidden=context_hidden,
                    shared_hidden=shared_hidden,
                    tool_id=tool_ids[:, time_index],
                    task_projection=(
                        task_projection[:, time_index]
                        if task_projection is not None
                        else None
                    ),
                )
                if previous_session is None:
                    previous_session = torch.zeros_like(output["session_hidden"])
                if previous_context is None:
                    previous_context = torch.zeros_like(output["context_hidden"])
                session_hidden = torch.where(active[:, None], output["session_hidden"], previous_session)
                context_hidden = torch.where(active[:, None], output["context_hidden"], previous_context)
                shared_hidden = {
                    family: torch.where(
                        active[:, None],
                        output["shared_hidden"][family],
                        previous_shared.get(family, torch.zeros_like(output["shared_hidden"][family])),
                    )
                    for family in TOOL_FAMILIES
                }
                for key, value in output.items():
                    if key == "observation_logits":
                        for name, logits in value.items():
                            observation_outputs[name].append(logits)
                        continue
                    if key in {"shared_logits", "shared_hidden"} or not isinstance(value, torch.Tensor):
                        continue
                    outputs.setdefault(key, []).append(value)
                for family, value in output["shared_logits"].items():
                    shared_outputs[family].append(value)
            result: dict[str, Any] = {key: torch.stack(values, dim=1) for key, values in outputs.items()}
            result["shared_logits"] = {
                family: torch.stack(values, dim=1) for family, values in shared_outputs.items()
            }
            result["observation_logits"] = {
                name: torch.stack(values, dim=1) for name, values in observation_outputs.items()
            }
            return result


    class ToolWorldSmoother(nn.Module):
        def __init__(self, network: ToolWorldFilterNetwork) -> None:
            super().__init__()
            self.encoder = network.observation_encoder
            self.bigru = nn.GRU(256, 256, num_layers=1, bidirectional=True, batch_first=True)
            self.session_head = nn.Linear(512, 4)
            self.regime_head = nn.Linear(512, 3)
            self.change_head = nn.Linear(512, 1)
            self.quality_head = nn.Linear(512, len(TOOL_NAMES) * 4 * 2)
            self.cost_head = nn.Linear(512, len(TOOL_NAMES) * 2)
            self.shared_heads = nn.ModuleDict({name: nn.Linear(512, 3) for name in TOOL_FAMILIES})

        def forward(self, features: Tensor) -> dict[str, Tensor]:
            encoded = self.encoder(features)
            hidden, _ = self.bigru(encoded)
            return {
                "session_logits": self.session_head(hidden),
                "regime_logits": self.regime_head(hidden),
                "change_logits": self.change_head(hidden).squeeze(-1),
                "quality_raw": self.quality_head(hidden).reshape(hidden.shape[0], hidden.shape[1], len(TOOL_NAMES), 4, 2),
                "cost_raw": self.cost_head(hidden).reshape(hidden.shape[0], hidden.shape[1], len(TOOL_NAMES), 2),
                "shared_logits": {name: head(hidden) for name, head in self.shared_heads.items()},
            }


    class ObservationPredictor(nn.Module):
        def __init__(self, hidden_size: int = 256, tool_dim: int = 32, task_dim: int = 16) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_size + tool_dim + task_dim, 128),
                nn.GELU(),
                nn.Linear(128, 6 + 8 + 5 + 5 + 2 + 2),
            )

        def forward(self, combined_hidden: Tensor, tool_embedding: Tensor, task_projection: Tensor) -> dict[str, Tensor]:
            raw = self.net(torch.cat((combined_hidden, tool_embedding, task_projection), dim=-1))
            outputs: dict[str, Tensor] = {}
            cursor = 0
            for name, size in (("status", 6), ("latency_bin", 8), ("information_gain", 5), ("semantic_agreement", 5), ("schema_valid", 2), ("image_valid", 2)):
                outputs[name] = raw[:, cursor : cursor + size]
                cursor += size
            return outputs

else:  # pragma: no cover - import-safe placeholders

    class ToolWorldFilterNetwork:  # type: ignore[no-redef]
        def __init__(self, config: BayesToolConfig | None = None) -> None:
            raise ImportError("PyTorch is required for ToolWorldFilterNetwork")

    class ToolWorldSmoother:  # type: ignore[no-redef]
        def __init__(self, network: Any) -> None:
            raise ImportError("PyTorch is required for ToolWorldSmoother")

    class ObservationPredictor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for ObservationPredictor")


def _beta_mean_std(alpha: float, beta: float) -> tuple[float, float]:
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    return mean, max(0.0, variance) ** 0.5


def _normalise_probs(values: Sequence[float]) -> tuple[float, ...]:
    values = [max(1e-8, float(value)) for value in values]
    total = sum(values) or 1.0
    return tuple(value / total for value in values)


def _entropy(values: Sequence[float]) -> float:
    return float(-sum(value * math.log(max(value, 1e-12)) for value in values))


class BeliefRuntime:
    """Stateful online filter with an optional trainable recurrent encoder."""

    def __init__(
        self,
        config: BayesToolConfig | None = None,
        *,
        document_digest: str = "",
        model: ToolWorldFilterNetwork | None = None,
        model_version: str = "untrained",
        seed: int = 0,
        session_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or default_config(enabled=True)
        self.document_digest = document_digest
        self.model = model
        self.model_version = str(model_version)
        self.seed = int(seed)
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.reopen_events: list[dict[str, Any]] = []
        self._last_prediction: ObservationPrediction | None = None
        self._last_prediction_tool: str | None = None
        state = dict(session_state or {})
        self._session_probs = tuple(state.get("session_probs", (0.70, 0.20, 0.08, 0.02)))
        self._regime_probs = tuple(state.get("regime_probs", (0.85, 0.10, 0.05)))
        self._family_probs = {
            name: tuple((state.get("family_probs", {}) or {}).get(name, (0.75, 0.20, 0.05)))
            for name in TOOL_FAMILIES
        }
        self._quality_params: dict[str, dict[str, list[float]]] = {
            name: {
                "availability": [9.0, 1.0],
                "semantic": [8.5, 1.5],
                "structure": [8.5, 1.5],
                "calibration": [8.0, 2.0],
            }
            for name in TOOL_NAMES
        }
        self._cost_params = {name: [0.0, 0.25] for name in TOOL_NAMES}
        self._change_probability = float(state.get("change_probability", 0.05))
        self._ood_score = float(state.get("ood_score", 0.0))
        for tool_name, dimensions in (state.get("quality_params", {}) or {}).items():
            if tool_name not in self._quality_params or not isinstance(dimensions, Mapping):
                continue
            for dimension, values in dimensions.items():
                if dimension in self._quality_params[tool_name] and isinstance(values, Sequence) and len(values) == 2:
                    self._quality_params[tool_name][dimension] = [float(values[0]), float(values[1])]
        for tool_name, values in (state.get("cost_params", {}) or {}).items():
            if tool_name in self._cost_params and isinstance(values, Sequence) and len(values) == 2:
                self._cost_params[tool_name] = [float(values[0]), float(values[1])]
        self._session_hidden: Any = None
        self._context_hidden: dict[str, Any] = {}
        self._shared_hidden: dict[str, Any] = {}
        self._last_context_key: str | None = None
        self._reopen_last_step = -10**9
        self._reopen_pending: dict[str, int] = {}
        self._reopen_cooldown_steps = 2

    @property
    def belief_model_version(self) -> str:
        return self.model_version

    @classmethod
    def from_replay_record(
        cls,
        record: Mapping[str, Any],
        config: BayesToolConfig | None = None,
        *,
        document_digest: str | None = None,
        seed: int = 0,
        model: ToolWorldFilterNetwork | None = None,
        model_version: str | None = None,
    ) -> "BeliefRuntime":
        """Restore session/shared posterior state without restoring task history."""

        state = record.get("state") if isinstance(record.get("state"), Mapping) else record
        runtime = cls(
            config,
            document_digest=str(document_digest or record.get("document_hash") or state.get("document_hash") or ""),
            model=model,
            model_version=str(model_version or record.get("belief_model_version") or "untrained"),
            seed=seed,
            session_state=state,
        )
        # Meta episodes carry only the recurrent session/shared state.  Local
        # history, page context, and navigation are deliberately not restored
        # so the next question starts with a fresh task context.
        persistent_hidden = state.get("persistent_hidden") if isinstance(state, Mapping) else None
        if isinstance(persistent_hidden, Mapping) and runtime.model is not None and torch is not None:
            try:
                device = next(runtime.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            def restore_hidden(value: Any) -> Any:
                if value is None:
                    return None
                if isinstance(value, torch.Tensor):
                    return value.detach().to(device=device)
                try:
                    return torch.as_tensor(value, dtype=torch.float32, device=device)
                except (TypeError, ValueError, RuntimeError):
                    return None

            runtime._session_hidden = restore_hidden(persistent_hidden.get("session"))
            shared_hidden = persistent_hidden.get("shared")
            if isinstance(shared_hidden, Mapping):
                runtime._shared_hidden = {
                    str(family): restored
                    for family, value in shared_hidden.items()
                    if (restored := restore_hidden(value)) is not None
                }
        return runtime

    def _posterior(self, tool_name: str) -> ToolQualityPosterior:
        values: dict[str, float] = {}
        for dimension in ("availability", "semantic", "structure", "calibration"):
            values[f"{dimension}_mean"], values[f"{dimension}_std"] = _beta_mean_std(*self._quality_params[tool_name][dimension])
        mu, sigma = self._cost_params[tool_name]
        values["cost_mean"] = math.exp(mu + 0.5 * sigma * sigma)
        values["cost_std"] = math.sqrt(max(0.0, (math.exp(sigma * sigma) - 1.0) * math.exp(2 * mu + sigma * sigma)))
        return ToolQualityPosterior(**values)

    def snapshot(self) -> BeliefSnapshot:
        qualities = {name: self._posterior(name) for name in TOOL_NAMES}
        entropy = _entropy(self._session_probs) + _entropy(self._regime_probs)
        entropy += sum(_entropy(values) for values in self._family_probs.values())
        entropy += sum(
            _entropy((_beta_mean_std(*params["availability"])[0], 1.0 - _beta_mean_std(*params["availability"])[0]))
            for params in self._quality_params.values()
        ) / max(1, len(self._quality_params))
        return BeliefSnapshot(
            version=int.from_bytes(hashlib.sha256(self.model_version.encode("utf-8")).digest()[:4], "big"),
            step=self.step,
            session_probs=_normalise_probs(self._session_probs),
            shared_family_probs={name: _normalise_probs(values) for name, values in self._family_probs.items()},
            regime_probs=_normalise_probs(self._regime_probs),
            change_probability=max(0.0, min(1.0, self._change_probability)),
            tool_quality=qualities,
            posterior_entropy=float(entropy),
            ood_score=max(0.0, min(1.0, self._ood_score)),
            document_hash=self.document_digest,
        )

    def _update_beta(self, tool_name: str, dimension: str, success_probability: float, weight: float = 1.0) -> None:
        alpha, beta = self._quality_params[tool_name][dimension]
        probability = max(0.01, min(0.99, float(success_probability)))
        self._quality_params[tool_name][dimension] = [alpha + weight * probability, beta + weight * (1.0 - probability)]

    @staticmethod
    def _task_projection(task_state: TaskStateView | Mapping[str, Any] | None) -> list[float]:
        task = task_state if isinstance(task_state, TaskStateView) else TaskStateView.from_navigation_state(task_state or {})
        question_type = task.question_type.casefold()
        question_terms = {
            token.casefold()
            for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(task.question or ""))
            if len(token) > 1
        }
        values = [
            float(question_type == "text"),
            float(question_type == "table"),
            float(question_type == "chart"),
            float(question_type in {"visual", "figure"}),
            math.log1p(task.current_page or 0),
            math.log1p(len(task.visited_pages)),
            math.log1p(task.unvisited_page_count or 0),
            math.log1p(task.remaining_tool_budget),
            float(task.evidence_sufficient),
            float(task.visual_input_required),
            float(task.phase == "search"),
            float(task.phase == "probe"),
            float(task.phase == "commit"),
            math.log1p(len(task.table_candidate_pages)),
            math.log1p(len(task.supporting_pages)),
            float(task.last_tool is not None),
        ]
        # Keep the 16-D projection contract while making the task/question
        # context explicit to Q/DVOI and the observation head.
        values[-1] = min(1.0, math.log1p(len(question_terms)) / 4.0)
        return [_clip(value) for value in values]

    def _context_state_key(
        self,
        task_state: TaskStateView | Mapping[str, Any] | None,
        event_data: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the recurrent context key for document/page/region state."""

        task = task_state if isinstance(task_state, TaskStateView) else TaskStateView.from_navigation_state(task_state or {})
        event_data = event_data or {}
        tool_name = str(event_data.get("tool") or event_data.get("tool_name") or "")
        page = event_data.get("page_number", event_data.get("page"))
        page_numbers = event_data.get("page_numbers")
        if page is None and isinstance(page_numbers, (list, tuple)) and page_numbers:
            page = page_numbers[0]
        if page is None:
            page = task.current_page
        try:
            page_key = str(int(page)) if page is not None else "document"
        except (TypeError, ValueError):
            page_key = "document"
        region = event_data.get("region", event_data.get("bbox"))
        if isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                region_key = ",".join(str(int(max(0.0, min(1.0, float(value))) * 10)) for value in region)
            except (TypeError, ValueError):
                region_key = ""
        else:
            region_key = ""
        base = self.document_digest or "<document>"
        return (
            f"{base}|tool={tool_name}|page={page_key}|region={region_key}"
            if tool_name or page_key != "document" or region_key
            else base
        )

    def _network_update(
        self,
        features: ObservationFeatures,
        tool_name: str,
        *,
        task_state: TaskStateView | Mapping[str, Any] | None = None,
        event_data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.model is None or torch is None:
            return None
        tool_id = TOOL_NAMES.index(tool_name) if tool_name in TOOL_NAMES else 0
        context_key = self._context_state_key(task_state, event_data)
        with torch.no_grad():
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = None
            tensor = torch.tensor([features.values], dtype=torch.float32, device=device)
            task_projection = torch.tensor([self._task_projection(task_state)], dtype=torch.float32, device=device)
            output = self.model(
                tensor,
                session_hidden=self._session_hidden,
                context_hidden=self._context_hidden.get(context_key),
                shared_hidden=self._shared_hidden,
                tool_id=tool_id,
                task_projection=task_projection,
            )
        self._session_hidden = output["session_hidden"]
        self._context_hidden[context_key] = output["context_hidden"]
        # Backwards-compatible document-level alias for old checkpoints and
        # diagnostics.  Tool/page-specific keys remain the canonical context
        # states used by prediction.
        if context_key != (self.document_digest or "<document>"):
            self._context_hidden.setdefault(self.document_digest or "<document>", output["context_hidden"])
        self._last_context_key = context_key
        self._shared_hidden = output["shared_hidden"]
        return output

    def _apply_network_posterior(self, output: Mapping[str, Any] | None) -> None:
        """Fuse a loaded recurrent filter prediction into the online posterior.

        The hand-coded Beta update remains the observable-data fallback and
        provides a stable prior during warm-up.  Once a Stage-A checkpoint is
        loaded, its recurrent heads are treated as a second posterior expert;
        the fixed blend keeps one bad checkpoint step from replacing the
        observable update wholesale while still making the trained filter
        affect action values, DVOI, and the prompt belief snapshot.
        """

        if not output or torch is None:
            return

        def _probabilities(value: Any, expected: int) -> tuple[float, ...] | None:
            if not isinstance(value, torch.Tensor):
                return None
            values = torch.softmax(value.detach().reshape(-1), dim=-1).cpu().tolist()
            if len(values) != expected:
                return None
            return tuple(float(item) for item in values)

        session = _probabilities(output.get("session_logits"), 4)
        if session is not None:
            self._session_probs = _normalise_probs(
                0.5 * old + 0.5 * new for old, new in zip(self._session_probs, session, strict=True)
            )
        regime = _probabilities(output.get("regime_logits"), 3)
        if regime is not None:
            self._regime_probs = _normalise_probs(
                0.5 * old + 0.5 * new for old, new in zip(self._regime_probs, regime, strict=True)
            )
        transition = _probabilities(output.get("transition_logits"), 3)
        if transition is not None:
            # The transition head is trained against the same stable /
            # abrupt / gradual contract as the regime posterior.  Fuse it
            # into both consumers instead of leaving the trained head
            # disconnected at runtime.
            self._regime_probs = _normalise_probs(
                0.75 * old + 0.25 * new for old, new in zip(self._regime_probs, transition, strict=True)
            )
            transition_change = transition[1] + transition[2]
            self._change_probability = max(
                0.0,
                min(0.99, 0.75 * self._change_probability + 0.25 * transition_change),
            )
        shared_logits = output.get("shared_logits")
        if isinstance(shared_logits, Mapping):
            for family, logits in shared_logits.items():
                if family not in self._family_probs:
                    continue
                family_probs = _probabilities(logits, 3)
                if family_probs is not None:
                    self._family_probs[family] = _normalise_probs(
                        0.5 * old + 0.5 * new
                        for old, new in zip(self._family_probs[family], family_probs, strict=True)
                    )
        change_logits = output.get("change_logits")
        if isinstance(change_logits, torch.Tensor):
            change = float(torch.sigmoid(change_logits.detach().reshape(-1)[0]).cpu().item())
            self._change_probability = max(0.0, min(0.99, 0.5 * self._change_probability + 0.5 * change))

        quality_raw = output.get("quality_raw")
        if isinstance(quality_raw, torch.Tensor):
            raw = (F.softplus(quality_raw.detach()) + 1.0).reshape(-1, len(TOOL_NAMES), 4, 2)[0]
            params = raw.cpu().tolist()
            for tool_index, tool_name in enumerate(TOOL_NAMES):
                for dimension_index, dimension in enumerate(("availability", "semantic", "structure", "calibration")):
                    alpha, beta = self._quality_params[tool_name][dimension]
                    model_alpha, model_beta = params[tool_index][dimension_index]
                    self._quality_params[tool_name][dimension] = [
                        0.75 * alpha + 0.25 * max(1.0, float(model_alpha)),
                        0.75 * beta + 0.25 * max(1.0, float(model_beta)),
                    ]

        cost_raw = output.get("cost_raw")
        if isinstance(cost_raw, torch.Tensor):
            raw_cost = cost_raw.detach().reshape(-1, len(TOOL_NAMES), 2)[0].cpu().tolist()
            for tool_index, tool_name in enumerate(TOOL_NAMES):
                model_mu = float(raw_cost[tool_index][0])
                model_sigma = float(F.softplus(torch.tensor(raw_cost[tool_index][1])).item())
                mu, sigma = self._cost_params[tool_name]
                self._cost_params[tool_name] = [0.75 * mu + 0.25 * model_mu, max(0.05, 0.75 * sigma + 0.25 * model_sigma)]

    def update(
        self,
        tool_name: str,
        observed_result: str,
        event: WorldEvent | Mapping[str, Any] | None = None,
        *,
        task_state: TaskStateView | Mapping[str, Any] | None = None,
        hidden_supervision: Mapping[str, Any] | None = None,
    ) -> BeliefSnapshot:
        features = extract_observation_features(tool_name, observed_result, event, task_state, self.history)
        event_data = _normalise_event(event, tool_name=tool_name, result=observed_result)
        if isinstance(event, WorldEvent):
            event_data["call_id"] = int(event.call_id)
        elif isinstance(event, Mapping) and event.get("call_id") is not None:
            event_data["call_id"] = event.get("call_id")
        # Predict from the pre-observation hidden state.  The resulting NLL is
        # the surprise of this observation; only after it is computed do we
        # advance the recurrent filter with the observed features.
        prediction = self.predict_observation(
            tool_name,
            task_state if isinstance(task_state, TaskStateView) else TaskStateView.from_navigation_state(task_state or {}),
        )
        status = str(event_data.get("status") or "empty").casefold()
        failed = status in {"error", "timeout", "invalid", "empty"}
        partial = status == "partial"
        availability = 0.05 if failed else (0.65 if partial else 0.95)
        semantic = _get(event_data, "semantic_agreement", default=0.15 if failed else (0.55 if partial else 0.90))
        structure = 0.15 if failed else (0.50 if partial else float(bool(event_data.get("schema_valid", True))) * 0.90)
        calibration = 0.25 if failed else (0.55 if partial else 0.85)
        self._update_beta(tool_name, "availability", availability)
        self._update_beta(tool_name, "semantic", semantic)
        self._update_beta(tool_name, "structure", structure)
        self._update_beta(tool_name, "calibration", calibration)
        latency = max(0.01, float(event_data.get("latency") or 0.01))
        mu, sigma = self._cost_params[tool_name]
        target = math.log(max(0.05, float(event_data.get("relative_cost") or latency)))
        self._cost_params[tool_name] = [0.9 * mu + 0.1 * target, max(0.05, 0.9 * sigma + 0.1 * abs(target - mu))]

        if failed:
            self._session_probs = _normalise_probs((self._session_probs[0] * 0.60, self._session_probs[1] * 1.20, self._session_probs[2] * 1.30, self._session_probs[3] * 1.10))
        else:
            self._session_probs = _normalise_probs((self._session_probs[0] * 1.04, self._session_probs[1] * 0.95, self._session_probs[2] * 0.90, self._session_probs[3] * 0.75))
        for family, members in TOOL_FAMILIES.items():
            if tool_name not in members:
                continue
            values = self._family_probs[family]
            if failed:
                self._family_probs[family] = _normalise_probs((values[0] * 0.55, values[1] * 1.30, values[2] * 1.25))
            else:
                self._family_probs[family] = _normalise_probs((values[0] * 1.04, values[1] * 0.95, values[2] * 0.75))

        before_change = self._change_probability
        surprise = self._observation_surprise(prediction, event_data)
        self._change_probability = max(0.0, min(0.99, 0.65 * before_change + 0.35 * (1.0 if surprise >= self.config.local_surprise_threshold else 0.0)))
        self._ood_score = max(0.0, min(1.0, 0.9 * self._ood_score + 0.1 * (1.0 if surprise > self.config.global_surprise_threshold else 0.0)))
        network_output = self._network_update(
            features,
            tool_name,
            task_state=task_state,
            event_data=event_data,
        )
        self._apply_network_posterior(network_output)
        self.step += 1
        record = dict(event_data)
        record["tool"] = tool_name
        record["predictive_surprise"] = surprise
        record["information_gain"] = float(event_data.get("information_gain") or 0.0)
        self.history.append(record)
        if hidden_supervision is not None:
            # Kept only for replay/training metadata.  It is never consulted by
            # extract_observation_features or to_prompt_block.
            record["hidden_supervision"] = dict(hidden_supervision)
        return self.snapshot()

    def _probability_for_event(self, event: Mapping[str, Any]) -> float:
        status = str(event.get("status") or "empty").casefold()
        tool = str(event.get("tool") or TOOL_NAMES[0])
        if tool not in self._quality_params:
            return 1e-6
        quality = self._posterior(tool)
        status_probability = quality.availability_mean if status == "ok" else max(0.02, 1.0 - quality.availability_mean)
        semantic = max(0.02, min(0.98, quality.semantic_mean))
        structure = max(0.02, min(0.98, quality.structure_mean))
        if status == "partial":
            status_probability *= 0.35 + 0.35 * semantic
        elif status in {"error", "timeout", "invalid", "empty"}:
            status_probability *= 0.55 + 0.25 * (1.0 - structure)
        return max(1e-6, min(1.0, status_probability * (0.5 + 0.5 * semantic) * (0.5 + 0.5 * structure)))

    def predictive_surprise(self, event: WorldEvent | Mapping[str, Any] | None) -> float:
        if isinstance(event, WorldEvent):
            event_data = event.visible_dict()
        else:
            event_data = dict(event or {})
        tool = str(event_data.get("tool") or event_data.get("tool_name") or "")
        if self._last_prediction is not None and tool == self._last_prediction_tool:
            return self._observation_surprise(self._last_prediction, event_data)
        return max(0.0, -math.log(self._probability_for_event(event_data)))

    @staticmethod
    def _observation_surprise(prediction: ObservationPrediction, event: Mapping[str, Any]) -> float:
        status = str(event.get("status") or "empty").casefold()
        status_index = STATUS_NAMES.index(status) if status in STATUS_NAMES else STATUS_NAMES.index("invalid")
        def _index(value: Any, scale: float, size: int) -> int:
            try:
                numeric = float(value or 0.0)
            except (TypeError, ValueError):
                numeric = 0.0
            return max(0, min(size - 1, int(numeric * scale)))

        indices = {
            "status": (status_index, prediction.status_probs),
            "latency": (_index(event.get("latency"), 2.0, len(prediction.latency_probs)), prediction.latency_probs),
            "information": (_index(event.get("information_gain"), 5.0, len(prediction.information_gain_probs)), prediction.information_gain_probs),
            "semantic": (_index(event.get("semantic_agreement"), 5.0, len(prediction.semantic_agreement_probs)), prediction.semantic_agreement_probs),
        }
        probability = 1.0
        for index, values in indices.values():
            probability *= max(1e-8, float(values[index]))
        probability *= max(1e-8, prediction.schema_valid_prob if bool(event.get("schema_valid", False)) else 1.0 - prediction.schema_valid_prob)
        probability *= max(1e-8, prediction.image_valid_prob if bool(event.get("image_valid", False)) else 1.0 - prediction.image_valid_prob)
        return max(0.0, -math.log(probability))

    def _model_observation_prediction(
        self,
        tool_name: str,
        task_state: TaskStateView | Mapping[str, Any] | None,
    ) -> ObservationPrediction | None:
        if self.model is None or torch is None or not hasattr(self.model, "observation_from_hidden"):
            return None
        try:
            device = next(self.model.parameters()).device
            if self._session_hidden is None:
                hidden_size = int(self.model.session_filter.hidden_size)
                hidden = torch.zeros(1, hidden_size, device=device)
            else:
                hidden = self._session_hidden.to(device=device)
            context_key = self._context_state_key(task_state, {"tool": tool_name})
            context_hidden = self._context_hidden.get(context_key)
            if context_hidden is not None:
                context_hidden = context_hidden.to(device=device)
            task_projection = torch.tensor([self._task_projection(task_state)], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = self.model.observation_from_hidden(
                    hidden,
                    context_hidden=context_hidden,
                    tool_id=TOOL_NAMES.index(tool_name) if tool_name in TOOL_NAMES else 0,
                    task_projection=task_projection,
                )
            return ObservationPrediction(
                status_probs=tuple(torch.softmax(logits["status"], dim=-1)[0].cpu().tolist()),
                latency_probs=tuple(torch.softmax(logits["latency_bin"], dim=-1)[0].cpu().tolist()),
                information_gain_probs=tuple(torch.softmax(logits["information_gain"], dim=-1)[0].cpu().tolist()),
                semantic_agreement_probs=tuple(torch.softmax(logits["semantic_agreement"], dim=-1)[0].cpu().tolist()),
                schema_valid_prob=float(torch.sigmoid(logits["schema_valid"][0, 1] - logits["schema_valid"][0, 0]).cpu().item()),
                image_valid_prob=float(torch.sigmoid(logits["image_valid"][0, 1] - logits["image_valid"][0, 0]).cpu().item()),
            )
        except (RuntimeError, StopIteration, ValueError, IndexError):
            return None

    def predict_observation(self, tool_name: str, task_state: TaskStateView | None = None) -> ObservationPrediction:
        model_prediction = self._model_observation_prediction(tool_name, task_state)
        if model_prediction is not None:
            self._last_prediction = model_prediction
            self._last_prediction_tool = tool_name
            return model_prediction
        posterior = self._posterior(tool_name)
        availability = posterior.availability_mean
        semantic = posterior.semantic_mean
        structure = posterior.structure_mean
        status = _normalise_probs((availability, (1.0 - availability) * 0.55, (1.0 - availability) * 0.25, (1.0 - availability) * 0.10, 0.02, 0.02))
        latency = posterior.cost_mean
        latency_probs = _normalise_probs((max(0.01, 1.2 - latency), 1.0, max(0.01, latency - 0.8), max(0.01, latency - 1.2), max(0.01, latency - 1.6), max(0.01, latency - 2.0), 0.01, 0.01))
        information = _normalise_probs((0.05, max(0.05, 1.0 - semantic), semantic * 0.8, semantic, semantic * structure))
        agreement = _normalise_probs((max(0.05, 1.0 - semantic), 0.5, semantic, semantic * structure, semantic * 0.8))
        prediction = ObservationPrediction(
            status_probs=status,
            latency_probs=latency_probs,
            information_gain_probs=information,
            semantic_agreement_probs=agreement,
            schema_valid_prob=structure,
            image_valid_prob=availability,
        )
        self._last_prediction = prediction
        self._last_prediction_tool = tool_name
        return prediction

    def hypothetical_update(
        self,
        tool_name: str,
        event: WorldEvent | Mapping[str, Any],
        observed_result: str = "hypothesis",
        *,
        task_state: TaskStateView | Mapping[str, Any] | None = None,
    ) -> "BeliefRuntime":
        clone = copy.deepcopy(self)
        clone.update(tool_name, observed_result, event, task_state=task_state)
        return clone

    def reopen(self, level: str, *, cause_tools: Sequence[str] = (), surprise: float = 0.0) -> dict[str, Any]:
        level = str(level).casefold()
        before = self.snapshot().to_dict()
        if level == "local":
            self._ood_score = min(1.0, self._ood_score + 0.05)
            self._change_probability = min(0.99, self._change_probability + 0.05)
            if self._last_context_key is not None:
                hidden = self._context_hidden.get(self._last_context_key)
                if hidden is not None and hasattr(hidden, "mul"):
                    # Blend the affected page/region context with its zero
                    # prior instead of erasing it, matching the 0.5/0.5
                    # local reopen rule while retaining useful history.
                    self._context_hidden[self._last_context_key] = hidden.mul(0.5)
                else:
                    self._context_hidden.pop(self._last_context_key, None)
            self._last_prediction = None
            self._last_prediction_tool = None
        elif level == "family":
            for family, members in TOOL_FAMILIES.items():
                if any(tool in members for tool in cause_tools):
                    values = self._family_probs[family]
                    self._family_probs[family] = _normalise_probs((0.3 * values[0] + 0.7 * 0.75, 0.3 * values[1] + 0.7 * 0.20, 0.3 * values[2] + 0.7 * 0.05))
                    hidden = self._shared_hidden.get(family)
                    if hidden is not None and hasattr(hidden, "mul"):
                        self._shared_hidden[family] = hidden.mul(0.3)
            self._last_prediction = None
            self._last_prediction_tool = None
        elif level == "global":
            self._session_probs = _normalise_probs((0.1 * self._session_probs[0] + 0.9 * 0.70, 0.1 * self._session_probs[1] + 0.9 * 0.20, 0.1 * self._session_probs[2] + 0.9 * 0.08, 0.1 * self._session_probs[3] + 0.9 * 0.02))
            self._context_hidden.clear()
            self._session_hidden = None
            self._shared_hidden.clear()
            self._last_context_key = None
            self._change_probability = 0.5
            self._ood_score = 0.0
            self._last_prediction = None
            self._last_prediction_tool = None
        else:
            raise ValueError(f"reopen level must be local, family, or global, got {level!r}")
        after = self.snapshot().to_dict()
        record = {
            "level": level,
            "surprise": float(surprise),
            "cause_tools": list(cause_tools),
            "belief_before": before,
            "belief_after": after,
        }
        self.reopen_events.append(record)
        self._reopen_last_step = int(self.step)
        self._reopen_pending.clear()
        return record

    def _detect_reopen_level_raw(self, event: WorldEvent | Mapping[str, Any]) -> str | None:
        event_data = event.to_dict() if isinstance(event, WorldEvent) else dict(event)
        tool = str(event_data.get("tool") or event_data.get("tool_name") or "")
        surprise = float(event_data.get("predictive_surprise") or self.predictive_surprise(event_data))
        if self._change_probability >= self.config.change_probability_threshold or surprise >= self.config.global_surprise_threshold:
            return "global"
        recent = self.history[-2:]
        recent_families = {
            family
            for item in recent
            for family, tools in TOOL_FAMILIES.items()
            if str(item.get("tool")) in tools
            and float(item.get("predictive_surprise", 0.0) or 0.0) >= self.config.family_surprise_threshold
        }
        current_family = next((family for family, tools in TOOL_FAMILIES.items() if tool in tools), "")
        if len(recent) >= 1 and current_family and surprise >= self.config.family_surprise_threshold:
            recent_families.add(current_family)
        if len(recent_families) >= 2 and len(recent) >= 2:
            return "global"
        if surprise >= self.config.family_surprise_threshold:
            family = next((family for family, tools in TOOL_FAMILIES.items() if tool in tools), "")
            # ``event`` is the current observation and has not been appended
            # to ``history`` yet.  Include it explicitly, otherwise the
            # second consecutive family failure can never trigger a family
            # reopen until a third call arrives.
            recent_events = list(self.history[-4:])
            last_event = recent_events[-1] if recent_events else None
            same_call = (
                isinstance(last_event, Mapping)
                and last_event.get("call_id") is not None
                and event_data.get("call_id") is not None
                and str(last_event.get("call_id")) == str(event_data.get("call_id"))
            )
            if not same_call:
                recent_events.append(event_data)
            related = [
                item
                for item in recent_events
                if str(item.get("tool")) in TOOL_FAMILIES.get(family, ())
                and str(item.get("status", "ok")).casefold() in {"error", "timeout", "invalid", "empty", "partial"}
            ]
            if len(related) >= 2:
                return "family"
        if surprise >= self.config.local_surprise_threshold:
            return "local"
        return None

    def detect_reopen_level(self, event: WorldEvent | Mapping[str, Any]) -> str | None:
        """Apply hierarchical hysteresis/cooldown to reopen evidence."""

        candidate = self._detect_reopen_level_raw(event)
        if candidate is None:
            self._reopen_pending.clear()
            return None
        if int(self.step) - int(self._reopen_last_step) < self._reopen_cooldown_steps:
            return None
        # Family/local reopen has already passed its repeated same-family
        # evidence test in the raw detector.  Global reopen remains gated by
        # two independent high-surprise signals to avoid a single spike.
        required = 2 if candidate == "global" else 1
        count = int(self._reopen_pending.get(candidate, 0)) + 1
        self._reopen_pending = {candidate: count}
        if count < required:
            return None
        self._reopen_pending.clear()
        return candidate

    def to_prompt_block(
        self,
        task_state: TaskStateView | Mapping[str, Any] | None = None,
        *,
        tokenizer: Any | None = None,
    ) -> str:
        task = task_state if isinstance(task_state, TaskStateView) else TaskStateView.from_navigation_state(task_state or {})
        task_payload = task.to_prompt_dict()
        belief_payload = self.snapshot().to_prompt_dict()
        max_tokens = max(32, int(self.config.max_belief_prompt_tokens))

        def render(task_value: Mapping[str, Any], belief_value: Mapping[str, Any]) -> str:
            # Stable compact JSON is intentionally used so the environment
            # block cannot accidentally expose world IDs or hidden labels.
            return (
                "<task_state>\n"
                + json.dumps(task_value, ensure_ascii=False, separators=(",", ":"))
                + "\n</task_state>\n"
                + "<tool_belief>\n"
                + json.dumps(belief_value, ensure_ascii=False, separators=(",", ":"))
                + "\n</tool_belief>"
            )

        def token_count(value: str) -> int:
            if tokenizer is not None and callable(tokenizer):
                try:
                    encoded = tokenizer(value, add_special_tokens=False)
                    ids = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else encoded
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                    while isinstance(ids, list) and ids and isinstance(ids[0], list):
                        ids = ids[0]
                    return max(1, len(ids) if isinstance(ids, list) else 0)
                except Exception:
                    pass
            # The fallback is deliberately conservative for environments that
            # import the package without a tokenizer.
            return max(1, math.ceil(len(value) / 4.0))

        full = render(task_payload, belief_payload)
        if token_count(full) <= max_tokens:
            return full

        # Keep the high-value state fields first and progressively reduce only
        # repeated page/tool detail.  The final fallback remains valid JSON;
        # no raw string slicing is used to create malformed prompt blocks.
        compact_task = dict(task_payload)
        for key in ("visited_pages", "table_candidate_pages", "supporting_pages"):
            values = compact_task.get(key)
            if isinstance(values, list):
                compact_task[key] = values[-8:]
        compact_belief = dict(belief_payload)
        tools = compact_belief.get("tools")
        if isinstance(tools, Mapping):
            compact_belief["tools"] = {
                str(name): value
                for name, value in tools.items()
                if isinstance(value, Mapping)
            }
        compact = render(compact_task, compact_belief)
        if token_count(compact) <= max_tokens:
            return compact

        minimal_belief = {
            key: compact_belief[key]
            for key in ("change_probability", "posterior_entropy", "ood_score")
            if key in compact_belief
        }
        minimal = render(compact_task, minimal_belief)
        if token_count(minimal) <= max_tokens:
            return minimal

        minimal_task = {
            key: compact_task[key]
            for key in (
                "question_type",
                "current_page",
                "evidence_sufficient",
                "visual_input_required",
                "remaining_tool_budget",
                "last_tool",
                "last_result_status",
                "phase",
            )
            if key in compact_task
        }
        minimal_block = render(minimal_task, minimal_belief)
        if token_count(minimal_block) <= max_tokens:
            return minimal_block

        # A caller may deliberately configure a very small budget for a smoke
        # test or an ablation.  Keep the result valid and enforce the bound
        # even when the tokenizer counts every character as a token; dropping
        # optional fields is preferable to emitting an over-budget belief.
        empty_block = json.dumps(
            {"task_state": {}, "tool_belief": {}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if token_count(empty_block) <= max_tokens:
            return empty_block
        return "{}"

    def export_replay_record(self) -> dict[str, Any]:
        def serialise_hidden(value: Any) -> Any:
            if torch is not None and isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, Mapping):
                return {str(key): serialise_hidden(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serialise_hidden(item) for item in value]
            return value

        return {
            "document_hash": self.document_digest,
            "belief_model_version": self.model_version,
            "history": [dict(item) for item in self.history],
            "reopen_events": copy.deepcopy(self.reopen_events),
            "reopen_state": {
                "last_step": int(self._reopen_last_step),
                "pending": dict(self._reopen_pending),
                "cooldown_steps": int(self._reopen_cooldown_steps),
            },
            "final_belief": self.snapshot().to_dict(),
            "state": {
                "session_probs": list(self._session_probs),
                "regime_probs": list(self._regime_probs),
                "family_probs": {name: list(values) for name, values in self._family_probs.items()},
                "quality_params": copy.deepcopy(self._quality_params),
                "cost_params": copy.deepcopy(self._cost_params),
                "change_probability": float(self._change_probability),
                "ood_score": float(self._ood_score),
                "persistent_hidden": {
                    "session": serialise_hidden(self._session_hidden),
                    "shared": serialise_hidden(self._shared_hidden),
                },
            },
        }


def _categorical_kl(target: Tensor, prediction: Tensor) -> Tensor:
    target = target.detach().clamp_min(1e-8)
    prediction = prediction.clamp_min(1e-8)
    return (target * (target.log() - prediction.log())).sum(dim=-1).mean()


def hbd_loss(filter_outputs: Mapping[str, Tensor], smoother_outputs: Mapping[str, Tensor]) -> Tensor:
    """Distil every smoother posterior into the online filter.

    The smoother is always a detached teacher.  In addition to the global
    session/regime heads, HBD covers shared tool-family categoricals, the
    four Beta quality dimensions, LogNormal cost parameters, and the
    Bernoulli change probability.
    """

    if torch is None:
        raise ImportError("PyTorch is required for hbd_loss")
    total = _categorical_kl(
        F.softmax(smoother_outputs["session_logits"].detach(), dim=-1),
        F.softmax(filter_outputs["session_logits"], dim=-1),
    )
    total = total + _categorical_kl(
        F.softmax(smoother_outputs["regime_logits"].detach(), dim=-1),
        F.softmax(filter_outputs["regime_logits"], dim=-1),
    )
    smoother_shared = smoother_outputs.get("shared_logits", {})
    filter_shared = filter_outputs.get("shared_logits", {})
    if isinstance(smoother_shared, Mapping) and isinstance(filter_shared, Mapping):
        for family in TOOL_FAMILIES:
            if family not in smoother_shared or family not in filter_shared:
                continue
            total = total + _categorical_kl(
                F.softmax(smoother_shared[family].detach(), dim=-1),
                F.softmax(filter_shared[family], dim=-1),
            )
    target_change = torch.sigmoid(smoother_outputs["change_logits"].detach()).clamp(1e-6, 1.0 - 1e-6)
    predicted_change = torch.sigmoid(filter_outputs["change_logits"]).clamp(1e-6, 1.0 - 1e-6)
    total = total + (
        target_change * (target_change.log() - predicted_change.log())
        + (1.0 - target_change) * ((1.0 - target_change).log() - (1.0 - predicted_change).log())
    ).mean()
    if "quality_raw" in filter_outputs and "quality_raw" in smoother_outputs:
        filter_params = F.softplus(filter_outputs["quality_raw"]) + 1.0
        smooth_params = F.softplus(smoother_outputs["quality_raw"].detach()) + 1.0
        # quality_raw has shape [..., tool, quality_dimension, alpha_beta].
        # Distil every availability/semantic/structure/calibration posterior;
        # selecting ``[..., 0]`` here would silently train only availability.
        q1 = torch.distributions.Beta(filter_params[..., 0], filter_params[..., 1])
        q2 = torch.distributions.Beta(smooth_params[..., 0], smooth_params[..., 1])
        total = total + torch.distributions.kl_divergence(q2, q1).mean()
    if "cost_raw" in filter_outputs and "cost_raw" in smoother_outputs:
        filter_cost = filter_outputs["cost_raw"]
        smooth_cost = smoother_outputs["cost_raw"].detach()
        filter_mu = filter_cost[..., 0]
        filter_scale = F.softplus(filter_cost[..., 1]) + 1e-4
        smooth_mu = smooth_cost[..., 0]
        smooth_scale = F.softplus(smooth_cost[..., 1]) + 1e-4
        total = total + (
            torch.log(filter_scale / smooth_scale)
            + (smooth_scale.square() + (smooth_mu - filter_mu).square()) / (2.0 * filter_scale.square())
            - 0.5
        ).mean()
    return total


def observation_prediction_loss(logits: Mapping[str, Tensor], targets: Mapping[str, Tensor]) -> Tensor:
    if torch is None:
        raise ImportError("PyTorch is required for observation_prediction_loss")
    total = torch.zeros((), device=next(iter(logits.values())).device)
    for name, prediction in logits.items():
        target = targets[name]
        mask = targets.get(f"{name}_mask")
        if prediction.shape[-1] == 2 and target.dtype in (torch.float16, torch.float32, torch.float64):
            # The binary target is the probability of class 1.  Use the
            # two-class log-odds so both logits participate; applying BCE to
            # only ``prediction[..., 1]`` would leave class 0 disconnected.
            values = F.binary_cross_entropy_with_logits(
                prediction[..., 1] - prediction[..., 0],
                target.float(),
                reduction="none",
            )
        else:
            values = F.cross_entropy(prediction, target.long(), reduction="none")
        if mask is None:
            total = total + values.mean()
            continue
        mask = mask.to(device=values.device, dtype=values.dtype)
        if bool(mask.sum() > 0):
            total = total + (values * mask).sum() / mask.sum().clamp_min(1.0)
    return total


def calibration_brier(probabilities: Tensor, targets: Tensor) -> Tensor:
    if torch is None:
        raise ImportError("PyTorch is required for calibration_brier")
    return ((probabilities.float() - targets.float()) ** 2).mean()


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "ObservationFeatures",
    "extract_observation_features",
    "ToolWorldFilterNetwork",
    "ToolWorldSmoother",
    "ObservationPredictor",
    "BeliefRuntime",
    "hbd_loss",
    "observation_prediction_loss",
    "calibration_brier",
]
