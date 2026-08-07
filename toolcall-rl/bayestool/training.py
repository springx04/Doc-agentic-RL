"""Bayes-ARPO utilities: net utility, sibling advantages, and auxiliary bundles."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from .config import BayesToolConfig, default_config
from .decision import canonical_action_key, js_divergence
from .schema import BranchRecord, SwitchPair

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    F = None  # type: ignore[assignment]


def _number(metadata: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
    return float(default)


def compute_bayestool_utility(
    metadata: Mapping[str, Any],
    quality: float,
    *,
    tool_budget: int | None = None,
    config: BayesToolConfig | None = None,
) -> dict[str, float]:
    """Compute the full-trajectory utility from the implementation plan.

    Injected world failures are excluded from ``F``; infrastructure failures
    are marked invalid by the caller and never enter this function's RL group.
    """

    config = config or default_config(enabled=True)
    quality = max(0.0, min(1.0, float(quality)))
    calls = max(0.0, _number(metadata, "tool_call_count", "call_count"))
    budget = max(1.0, float(tool_budget or _number(metadata, "tool_budget", default=max(1.0, calls))))
    latency_cost = _number(metadata, "latency_cost", "normalized_latency_cost")
    text_cost = _number(metadata, "text_token_cost", "normalized_text_token_cost")
    image_cost = _number(metadata, "image_token_cost", "normalized_image_token_cost")
    normalized_cost = 0.40 * calls / budget + 0.25 * latency_cost + 0.20 * text_cost + 0.15 * image_cost
    duplicate = _number(metadata, "duplicate_tool_calls", "duplicate_page_calls", default=0.0) + _number(metadata, "duplicate_region_calls", default=0.0)
    no_gain = _number(metadata, "no_information_gain_calls", "no_gain_calls", default=0.0)
    unnecessary = _number(metadata, "unnecessary_tool_calls", default=0.0)
    inefficiency = min(1.0, (duplicate + no_gain + unnecessary) / max(1.0, calls))
    actions = metadata.get("assistant_turns", metadata.get("actions", []))
    failure_events = metadata.get("failure_events")
    if isinstance(failure_events, Sequence) and not isinstance(failure_events, (str, bytes)):
        # Canonical failure accounting is explicit about origin.  In
        # particular a world-injected unavailable observation is a valid POMDP
        # outcome and must not become an agent penalty.
        failure = sum(
            1.0
            for event in failure_events
            if isinstance(event, Mapping)
            and bool(event.get("penalize", False))
            and str(event.get("origin") or "") not in {"world_injected", "real_infrastructure"}
        )
    else:
        failure = _number(metadata, "protocol_error_count", default=0.0) + _number(metadata, "invalid_argument_count", default=0.0)
        failure += _number(metadata, "premature_final_count", default=0.0) + _number(metadata, "unsupported_final_count", default=0.0)
        if isinstance(actions, Sequence):
            failure += sum(
                1.0
                for action in actions
                if isinstance(action, Mapping)
                and bool(action.get("accepted", True))
                and bool(action.get("used_failed_evidence", False))
            )
        if bool(metadata.get("budget_exhausted_without_terminal", False)):
            failure += 1.0
    failure = min(1.0, failure / max(1.0, calls + 1.0))
    task_score = 2.0 * quality - 1.0
    utility = task_score - config.utility.cost_weight * normalized_cost - config.utility.inefficiency_weight * inefficiency - config.utility.failure_weight * failure
    return {
        "task_score": task_score,
        "cost": normalized_cost,
        "inefficiency": inefficiency,
        "failure": failure,
        "utility": utility,
        "utility_clipped": max(-1.0, min(1.0, utility)),
    }


def attach_bayestool_utility(
    metadata: dict[str, Any],
    quality: float,
    *,
    tool_budget: int | None = None,
    config: BayesToolConfig | None = None,
) -> dict[str, Any]:
    result = compute_bayestool_utility(metadata, quality, tool_budget=tool_budget, config=config)
    metadata["bayestool_utility"] = result
    metadata["utility"] = float(result["utility"])
    return result


def weighted_sibling_advantage(
    returns: Sequence[float],
    group_ids: Sequence[str | int],
    *,
    weights: Sequence[float] | None = None,
    standardize: bool = False,
) -> list[float]:
    """Compute ``R_j - sum_l w_l R_l`` within each sibling group."""

    if len(returns) != len(group_ids):
        raise ValueError("returns and group_ids must have equal length")
    if weights is not None and len(weights) != len(returns):
        raise ValueError("weights and returns must have equal length")
    weights = list(weights) if weights is not None else [1.0] * len(returns)
    grouped: dict[str | int, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)
    advantages = [0.0] * len(returns)
    for indices in grouped.values():
        total_weight = sum(max(0.0, float(weights[index])) for index in indices) or float(len(indices))
        baseline = sum(max(0.0, float(weights[index])) * float(returns[index]) for index in indices) / total_weight
        for index in indices:
            advantages[index] = float(returns[index]) - baseline
        if standardize and len(indices) > 1:
            mean = sum(advantages[index] for index in indices) / len(indices)
            variance = sum((advantages[index] - mean) ** 2 for index in indices) / len(indices)
            scale = math.sqrt(variance) + 1e-6
            for index in indices:
                advantages[index] /= scale
    return advantages


def bayes_grpo_advantages(
    samples: Sequence[Mapping[str, Any]],
    *,
    standardize: bool = False,
) -> list[float]:
    returns = [float(sample.get("utility", sample.get("reward", 0.0))) for sample in samples]
    groups = [
        str(
            sample.get("sibling_group_id")
            or sample.get("latent_world_id")
            or sample.get("coupling_id")
            or sample.get("world_id")
            or index
        )
        for index, sample in enumerate(samples)
    ]
    weights = [float(sample.get("sibling_weight", 1.0)) for sample in samples]
    return weighted_sibling_advantage(returns, groups, weights=weights, standardize=standardize)


def _normalised_evidence(metadata: Mapping[str, Any]) -> str:
    evidence = metadata.get("evidence_candidates", metadata.get("evidence_summary", []))
    if isinstance(evidence, Mapping):
        evidence = [evidence]
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        evidence = [str(evidence)]
    cleaned: list[str] = []
    for item in evidence:
        if isinstance(item, Mapping):
            visible = {
                str(key): value
                for key, value in item.items()
                if str(key).casefold() not in {"world_id", "true_quality", "corruption_type", "clean_result", "answer_page", "answer_bbox", "label", "answers"}
            }
            cleaned.append(json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        else:
            cleaned.append(" ".join(str(item).split()))
    return "|".join(cleaned)


def content_signature(metadata: Mapping[str, Any], *, question: str | None = None) -> str:
    """Hash only task-content state; world/belief/error fields are excluded."""

    navigation = metadata.get("navigation_state", {}) if isinstance(metadata.get("navigation_state", {}), Mapping) else {}
    question = str(question if question is not None else metadata.get("question", ""))
    payload = {
        "question": question,
        "question_type": metadata.get("question_type", navigation.get("question_type", "text")),
        "visited_pages": sorted(str(value) for value in navigation.get("visited_pages", metadata.get("visited_pages", []))),
        "evidence": _normalised_evidence(metadata),
        "remaining_budget": _number(metadata, "remaining_tool_budget", "tool_budget", default=0.0),
        "phase": metadata.get("phase", navigation.get("phase", "search")),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _extract_belief_vector(value: Mapping[str, Any]) -> list[float]:
    snapshot = value.get("belief_snapshot", value)
    if not isinstance(snapshot, Mapping):
        return [0.0]
    vector: list[float] = []
    for key in ("session_probs", "regime_probs"):
        raw = snapshot.get(key, [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            vector.extend(float(item) for item in raw)
    families = snapshot.get("shared_family_probs", {})
    if isinstance(families, Mapping):
        for name in sorted(families):
            raw = families[name]
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                vector.extend(float(item) for item in raw)
    quality = snapshot.get("tool_quality", snapshot.get("tools", {}))
    if isinstance(quality, Mapping):
        for name in sorted(quality):
            raw = quality[name]
            if isinstance(raw, Mapping):
                for dim in ("availability", "semantic", "structure", "calibration", "cost"):
                    value = raw.get(dim)
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                        vector.extend(float(item) for item in value[:2])
                    elif value is not None:
                        vector.append(float(value))
    vector.append(float(snapshot.get("change_probability", 0.0)))
    return vector or [0.0]


def support_valid_pair(
    state_u: Mapping[str, Any],
    state_v: Mapping[str, Any],
    *,
    config: BayesToolConfig | None = None,
) -> bool:
    config = config or default_config(enabled=True)
    if str(state_u.get("coupling_id")) != str(state_v.get("coupling_id")):
        return False
    if str(state_u.get("content_signature")) != str(state_v.get("content_signature")):
        return False
    latent_u = str(state_u.get("latent_world_id") or "")
    latent_v = str(state_v.get("latent_world_id") or "")
    if latent_u or latent_v:
        if not latent_u or not latent_v or latent_u == latent_v:
            return False
    ood_u = float(state_u.get("ood_score", state_u.get("belief_snapshot", {}).get("ood_score", 1.0)))
    ood_v = float(state_v.get("ood_score", state_v.get("belief_snapshot", {}).get("ood_score", 1.0)))
    if ood_u >= config.pairing.max_ood_score or ood_v >= config.pairing.max_ood_score:
        return False
    js = js_divergence(_extract_belief_vector(state_u), _extract_belief_vector(state_v))
    if not (config.pairing.min_belief_js <= js <= config.pairing.max_belief_js):
        return False
    action_u = str(state_u.get("best_action") or state_u.get("action_u") or "")
    action_v = str(state_v.get("best_action") or state_v.get("action_v") or "")
    if not action_u or not action_v or action_u == action_v:
        return False
    if float(state_u.get("best_action_margin", 0.0)) < config.pairing.min_action_margin:
        return False
    if float(state_v.get("best_action_margin", 0.0)) < config.pairing.min_action_margin:
        return False
    candidate_u = {str(value) for value in state_u.get("candidate_actions", ()) or ()}
    candidate_v = {str(value) for value in state_v.get("candidate_actions", ()) or ()}
    if candidate_u or candidate_v:
        if len(candidate_u & candidate_v) < 2:
            return False
    return True


def build_switch_pair(state_u: Mapping[str, Any], state_v: Mapping[str, Any], *, config: BayesToolConfig | None = None) -> SwitchPair | None:
    if not support_valid_pair(state_u, state_v, config=config):
        return None
    return SwitchPair(
        content_signature=str(state_u["content_signature"]),
        coupling_id=str(state_u["coupling_id"]),
        state_u=dict(state_u),
        state_v=dict(state_v),
        action_u=str(state_u.get("best_action") or state_u.get("action_u")),
        action_v=str(state_v.get("best_action") or state_v.get("action_v")),
        return_u=float(state_u.get("best_return", state_u.get("utility", 0.0))),
        return_v=float(state_v.get("best_return", state_v.get("utility", 0.0))),
        belief_js=js_divergence(_extract_belief_vector(state_u), _extract_belief_vector(state_v)),
        ood_u=float(state_u.get("ood_score", 0.0)),
        ood_v=float(state_v.get("ood_score", 0.0)),
        first_distinguishing_event_step=state_u.get("first_distinguishing_event_step", state_v.get("first_distinguishing_event_step")),
        action_u_text=str(state_u.get("best_action_text") or state_u.get("best_action") or ""),
        action_v_text=str(state_v.get("best_action_text") or state_v.get("best_action") or ""),
    )


def sequence_logprob(log_probs: Sequence[float] | Tensor, *, mask: Sequence[float] | Tensor | None = None) -> Any:
    """Length-normalised log probability for teacher-forced action tokens."""

    if torch is not None and isinstance(log_probs, torch.Tensor):
        values = log_probs.float()
        if mask is not None:
            mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=values.dtype, device=values.device)
            values = values * mask_tensor
            denominator = mask_tensor.sum().clamp_min(1.0)
        else:
            denominator = torch.tensor(max(1, values.numel()), dtype=values.dtype, device=values.device)
        return values.sum() / denominator
    values = [float(value) for value in log_probs]
    if mask is not None:
        values = [value for value, enabled in zip(values, mask, strict=False) if float(enabled) > 0]
    return sum(values) / max(1, len(values))


def switch_loss(
    logprob_u_of_u: Any,
    logprob_u_of_v: Any,
    logprob_v_of_v: Any,
    logprob_v_of_u: Any,
) -> Any:
    """Exact bidirectional action-ordering loss from the method."""

    margin = logprob_u_of_u - logprob_u_of_v + logprob_v_of_v - logprob_v_of_u
    if torch is not None and isinstance(margin, torch.Tensor):
        return F.softplus(-margin)
    return math.log1p(math.exp(-max(-50.0, min(50.0, float(margin)))))


def pre_invariance_loss(log_probs_u: Sequence[Any], log_probs_v: Sequence[Any], *, temperature: float = 0.5) -> Any:
    if len(log_probs_u) != len(log_probs_v):
        raise ValueError("pre-invariance candidate lists must have equal length")
    if torch is not None and log_probs_u and isinstance(log_probs_u[0], torch.Tensor):
        p = torch.softmax(torch.stack([value / temperature for value in log_probs_u]), dim=0)
        q = torch.softmax(torch.stack([value / temperature for value in log_probs_v]), dim=0)
        midpoint = (p + q) / 2.0
        return 0.5 * F.kl_div(midpoint.log(), p, reduction="sum") + 0.5 * F.kl_div(midpoint.log(), q, reduction="sum")
    p = _softmax([float(value) for value in log_probs_u], temperature)
    q = _softmax([float(value) for value in log_probs_v], temperature)
    return js_divergence(p, q)


def build_switch_bundle(
    pair: SwitchPair,
    *,
    prompt_u: str,
    prompt_v: str,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    sequences = [
        {"prompt": prompt_u, "action": pair.action_u_text or pair.action_u, "world": "u", "preferred": True},
        {"prompt": prompt_u, "action": pair.action_v_text or pair.action_v, "world": "u", "preferred": False},
        {"prompt": prompt_v, "action": pair.action_v_text or pair.action_v, "world": "v", "preferred": True},
        {"prompt": prompt_v, "action": pair.action_u_text or pair.action_u, "world": "v", "preferred": False},
    ]
    bundle = {
        "kind": "switch",
        "switch_bundle_id": hashlib.sha256((pair.content_signature + pair.action_u + pair.action_v).encode()).hexdigest()[:24],
        "coupling_id": pair.coupling_id,
        "sequences": sequences,
        "loss_weight": 0.20,
    }
    if tokenizer is not None:
        bundle["tokenized"] = []
        for sequence in sequences:
            prompt_tokens = tokenizer(sequence["prompt"], add_special_tokens=False)["input_ids"]
            action_tokens = tokenizer(sequence["action"], add_special_tokens=False)["input_ids"]
            bundle["tokenized"].append(
                {
                    "prompt_ids": list(prompt_tokens),
                    "action_ids": list(action_tokens),
                    "sequence_ids": list(prompt_tokens) + list(action_tokens),
                    "response_start": len(prompt_tokens),
                    "response_end": len(prompt_tokens) + len(action_tokens),
                    "world": sequence["world"],
                    "preferred": bool(sequence["preferred"]),
                }
            )
    return bundle


def build_preinv_bundle(
    state_u: Mapping[str, Any],
    state_v: Mapping[str, Any],
    *,
    prompts: Sequence[str],
    actions: Sequence[str],
    config: BayesToolConfig | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any] | None:
    config = config or default_config(enabled=True)
    if not support_valid_preinv_pair(state_u, state_v, config=config):
        return None
    if len(actions) > config.max_action_candidates:
        actions = list(actions)[: config.max_action_candidates]
    bundle = {
        "kind": "pre_invariance",
        "preinv_bundle_id": hashlib.sha256((str(state_u.get("content_signature")) + "preinv").encode()).hexdigest()[:24],
        "prompts": list(prompts[:2]),
        "actions": list(actions),
        "loss_weight": config.auxiliary.preinv_loss_weight,
        "first_distinguishing_event_step": state_u.get("first_distinguishing_event_step"),
    }
    if tokenizer is not None:
        tokenized: list[dict[str, Any]] = []
        for world_index, prompt in enumerate(prompts[:2]):
            world = "u" if world_index == 0 else "v"
            for candidate_index, action in enumerate(actions):
                prompt_tokens = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                action_tokens = list(tokenizer(action, add_special_tokens=False)["input_ids"])
                tokenized.append(
                    {
                        "prompt_ids": prompt_tokens,
                        "action_ids": action_tokens,
                        "sequence_ids": prompt_tokens + action_tokens,
                        "response_start": len(prompt_tokens),
                        "response_end": len(prompt_tokens) + len(action_tokens),
                        "world": world,
                        "candidate_index": candidate_index,
                    }
                )
        bundle["tokenized"] = tokenized
    return bundle


def support_valid_preinv_pair(state_u: Mapping[str, Any], state_v: Mapping[str, Any], *, config: BayesToolConfig | None = None) -> bool:
    config = config or default_config(enabled=True)
    if str(state_u.get("coupling_id")) != str(state_v.get("coupling_id")):
        return False
    if str(state_u.get("content_signature")) != str(state_v.get("content_signature")):
        return False
    latent_u = str(state_u.get("latent_world_id") or "")
    latent_v = str(state_v.get("latent_world_id") or "")
    if latent_u or latent_v:
        if not latent_u or not latent_v or latent_u == latent_v:
            return False
    prefix_u = str(state_u.get("prefix_observation_signature") or "")
    prefix_v = str(state_v.get("prefix_observation_signature") or "")
    if prefix_u and prefix_v and prefix_u != prefix_v:
        return False
    if state_u.get("first_distinguishing_event_step") not in (None, 0) and state_v.get("first_distinguishing_event_step") not in (None, 0):
        if state_u.get("first_distinguishing_event_step") != state_v.get("first_distinguishing_event_step"):
            return False
    # The observation signatures are the exact public prefix contract.  When
    # they match, the empirical prefix divergence is zero even if an older
    # producer recorded a coarse placeholder JS value of 1.0.  Only fall back
    # to the numeric diagnostic when one side lacks a signature.
    if prefix_u and prefix_v and prefix_u == prefix_v:
        prefix_js_u = prefix_js_v = 0.0
    else:
        prefix_js_u = float(state_u.get("observed_prefix_js", state_u.get("belief_js", 1.0)))
        prefix_js_v = float(state_v.get("observed_prefix_js", state_v.get("belief_js", 1.0)))
    if prefix_js_u > config.pairing.preinv_max_js:
        return False
    if prefix_js_v > config.pairing.preinv_max_js:
        return False
    candidates_u = list(state_u.get("candidate_actions", []))
    candidates_v = list(state_v.get("candidate_actions", []))
    if candidates_u or candidates_v:
        return len(candidates_u) >= 2 and candidates_u == candidates_v
    return True


def oracle_regret(candidate_records: Sequence[BranchRecord], *, world_id: str) -> dict[str, Any]:
    world_records = [record for record in candidate_records if record.world_id == world_id]
    if not world_records:
        return {"oracle_action": None, "oracle_utility": None, "policy_utility": None, "relative_regret": None}
    oracle = max(world_records, key=lambda record: (record.utility, record.action_key))
    policy = next((record for record in world_records if record.policy_utility is not None), world_records[0])
    policy_utility = float(policy.policy_utility if policy.policy_utility is not None else policy.utility)
    return {
        "oracle_action": oracle.action_key,
        "oracle_utility": float(oracle.utility),
        "policy_utility": policy_utility,
        "relative_regret": float(oracle.utility - policy_utility),
    }


def bayes_q_gaussian_nll(prediction: Any, target: Any) -> Any:
    """Gaussian NLL for BayesQHead output ``[return_mean, log_variance]``."""

    if torch is not None and isinstance(prediction, torch.Tensor):
        target_tensor = (
            target
            if isinstance(target, torch.Tensor)
            else torch.as_tensor(target, dtype=prediction.dtype, device=prediction.device)
        )
        mean = prediction[..., 0]
        log_variance = prediction[..., 1].clamp(-10.0, 5.0)
        return 0.5 * (log_variance + (target_tensor - mean).square() * torch.exp(-log_variance)).mean()
    mean, log_variance = float(prediction[0]), max(-10.0, min(5.0, float(prediction[1])))
    return 0.5 * (log_variance + (float(target) - mean) ** 2 * math.exp(-log_variance))


def suffix_meta_returns(utilities: Sequence[float], *, discount: float = 0.95) -> list[float]:
    output = [0.0] * len(utilities)
    running = 0.0
    for index in reversed(range(len(utilities))):
        running = float(utilities[index]) + float(discount) * running
        output[index] = running
    return output


__all__ = [
    "compute_bayestool_utility",
    "attach_bayestool_utility",
    "weighted_sibling_advantage",
    "bayes_grpo_advantages",
    "content_signature",
    "support_valid_pair",
    "build_switch_pair",
    "sequence_logprob",
    "switch_loss",
    "pre_invariance_loss",
    "build_switch_bundle",
    "build_preinv_bundle",
    "support_valid_preinv_pair",
    "oracle_regret",
    "bayes_q_gaussian_nll",
    "suffix_meta_returns",
]
