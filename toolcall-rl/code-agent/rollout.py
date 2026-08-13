"""Multi-turn Code Agent rollout with exact token accounting."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

try:
    from .bayestool.belief import CodeBeliefFilter
    from .bayestool.belief_runtime import build_policy_visible_belief
    from .bayestool.schema import CodeWorldSamplingContext
    from .bayestool.world import CodeWorldRuntime
    from .bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from .bayestool.utility import compute_utility, failure_penalty, inefficiency_score, normalized_cost
    from .config import DEFAULT_CODE_CONFIG, CodeConfig, CODE_TOOL_NAMES, stable_hash
    from .data.leakage_guard import public_view
    from .env.evaluator import CleanEvaluator, EvaluatorRequest
    from .protocol import ParsedAction, parse_action
    from .schemas import CodeMessage, CodeReward, CodeToolResult, CodeTrajectory, ModelGeneration
    from .tools import DEFAULT_REGISTRY
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from bayestool.belief import CodeBeliefFilter
    from bayestool.belief_runtime import build_policy_visible_belief
    from bayestool.schema import CodeWorldSamplingContext
    from bayestool.world import CodeWorldRuntime
    from bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from bayestool.utility import compute_utility, failure_penalty, inefficiency_score, normalized_cost
    from config import DEFAULT_CODE_CONFIG, CodeConfig, CODE_TOOL_NAMES, stable_hash
    from data.leakage_guard import public_view
    from env.evaluator import CleanEvaluator, EvaluatorRequest
    from protocol import ParsedAction, parse_action
    from schemas import CodeMessage, CodeReward, CodeToolResult, CodeTrajectory, ModelGeneration
    from tools import DEFAULT_REGISTRY


class CodeModelClient(Protocol):
    async def generate(self, messages: list[dict[str, str]], *, max_tokens: int, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RolloutSettings:
    max_steps: int = 24
    max_new_tokens: int = 4096
    max_context_chars: int = 120_000
    belief_prompt_max_chars: int = 8_000


def _normalize_generation(value: Any) -> ModelGeneration:
    if isinstance(value, ModelGeneration):
        return value
    if isinstance(value, Mapping):
        text = str(value.get("text") or value.get("content") or value.get("message") or "")
        token_ids = tuple(int(item) for item in (value.get("token_ids") or value.get("tokens") or ()))
        token_mask = tuple(int(item) for item in (value.get("token_mask") or value.get("loss_mask") or ()))
        token_logprobs = tuple(float(item) for item in (value.get("token_logprobs") or value.get("logprobs") or ()))
        return ModelGeneration(text, token_ids, token_mask, token_logprobs, value)
    if isinstance(value, str):
        # Without model-provided token accounting the sample remains usable
        # for an eval smoke, but no synthetic action tokenization is invented.
        return ModelGeneration(value)
    text = getattr(value, "text", getattr(value, "content", ""))
    return ModelGeneration(
        str(text),
        tuple(int(item) for item in getattr(value, "token_ids", ()) or ()),
        tuple(int(item) for item in getattr(value, "token_mask", ()) or ()),
        tuple(float(item) for item in getattr(value, "token_logprobs", ()) or ()),
        value,
    )


def _public_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(sample.get("metadata") or {})
    if "public_instance" in metadata:
        public_instance = public_view(metadata["public_instance"])
    else:
        public_instance = public_view({key: value for key, value in metadata.items() if key != "evaluator_private"})
    return {"text": str(sample.get("text") or sample.get("problem_statement") or ""), "metadata": {"environment": "code", "public_instance": public_instance}}


def _prompt_messages(sample: Mapping[str, Any], visible_belief: str) -> list[dict[str, str]]:
    public_instance = dict(sample["metadata"]["public_instance"])
    problem = str(sample.get("text") or public_instance.get("problem_statement") or "")
    system = (
        "You are a Code Agent solving one SWE repository task.\n"
        "Use exactly one action per assistant turn. Tool calls must be exactly "
        "<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>; "
        "you may also emit <final>...</final> or <abstain>...</abstain>.\n"
        f"Available tools: {', '.join(CODE_TOOL_NAMES)}.\n"
        "Inspect and localize before editing, apply persistent edits only with apply_patch, "
        "and validate after editing. Do not use hidden evaluator information."
    )
    user = f"<swe_task>{json.dumps({'instance_id': public_instance.get('instance_id'), 'problem_statement': problem, 'task_kind': public_instance.get('task_kind', 'bugfix')}, ensure_ascii=False)}</swe_task>\n{visible_belief}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _restore_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    restored: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Code branch checkpoint messages must be mappings")
        role, content = str(item.get("role") or ""), item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Code branch checkpoint contains an invalid policy message")
        restored.append({"role": role, "content": content})
    return restored


def _restore_token_state(value: Any) -> tuple[list[int], list[int], list[float]]:
    if not isinstance(value, Mapping):
        return [], [], []
    token_ids = [int(item) for item in value.get("token_ids", ())]
    loss_mask = [int(item) for item in value.get("loss_mask", ())]
    log_probs = [float(item) for item in value.get("rollout_log_probs", ())]
    if loss_mask and len(loss_mask) != len(token_ids):
        raise ValueError("Code branch checkpoint loss_mask/token_ids mismatch")
    if log_probs and len(log_probs) != len(token_ids):
        raise ValueError("Code branch checkpoint log_probs/token_ids mismatch")
    return token_ids, loss_mask, log_probs


async def _capture_branch_checkpoint(
    *,
    client: Any,
    lease: Any,
    task_state: Any,
    runtime: Any,
    belief: CodeBeliefFilter,
    messages: list[dict[str, str]],
    token_ids: list[int],
    loss_mask: list[int],
    log_probs: list[float],
) -> Any:
    """Capture every state required to resume a decision sibling exactly."""

    try:
        from .env.checkpoint import CodeBranchCheckpoint, CodeRepoCheckpoint
    except ImportError:  # pragma: no cover - direct PYTHONPATH execution
        from env.checkpoint import CodeBranchCheckpoint, CodeRepoCheckpoint
    patch = await client.diff(lease.lease_id, cwd=lease.cwd)
    repo = CodeRepoCheckpoint.create(
        instance_id=lease.instance_id,
        image_name=lease.image_name,
        base_revision=lease.base_revision,
        cwd=lease.cwd,
        patch=patch,
    )
    return CodeBranchCheckpoint(
        repo=repo,
        messages=[dict(item) for item in messages],
        token_state={"token_ids": list(token_ids), "loss_mask": list(loss_mask), "rollout_log_probs": list(log_probs)},
        task_state=task_state.to_dict(),
        world_state=runtime.snapshot(),
        belief_state=belief.to_dict(),
        decision_state={"phase": task_state.phase, "evidence_sufficient": task_state.evidence_sufficient, "last_tool": task_state.last_tool},
        call_index=task_state.call_index,
        remaining_tool_budget=task_state.remaining_tool_budget,
        decision_prefix_hash=stable_hash(messages, prefix="code-decision-prefix-v1"),
        runtime_state_digest=runtime.runtime_state_digest,
    )


class CodeRollout:
    def __init__(self, *, settings: RolloutSettings | None = None, registry=DEFAULT_REGISTRY):
        self.settings = settings or RolloutSettings()
        self.registry = registry

    async def run(
        self,
        sample: Mapping[str, Any],
        *,
        model_client: CodeModelClient,
        code_env_client: Any,
        code_config: CodeConfig | None = None,
        world: Any = None,
        eval_script: str | None = None,
        evaluator_patch: str = "",
        seed: int = 0,
        data_source: str = "",
        interaction_lease: Any = None,
        branch_checkpoint: Any = None,
        on_decision_checkpoint: Callable[[Any], Any] | None = None,
    ) -> CodeTrajectory:
        config = code_config or DEFAULT_CODE_CONFIG
        public_sample = _public_sample(sample)
        public_instance = public_sample["metadata"]["public_instance"]
        instance_id = str(public_instance.get("instance_id") or "unknown-instance")
        image_name = str(public_instance.get("image_name") or "")
        base_revision = public_instance.get("base_revision")
        problem = str(public_sample.get("text") or public_instance.get("problem_statement") or "")
        try:
            from .bayestool.task_state import CodeTaskStateView
        except ImportError:  # pragma: no cover - direct PYTHONPATH execution
            from bayestool.task_state import CodeTaskStateView

        if branch_checkpoint is not None:
            branch_checkpoint.repo.validate()
            if branch_checkpoint.repo.instance_id != instance_id or branch_checkpoint.repo.image_name != image_name or branch_checkpoint.repo.base_revision != base_revision:
                raise ValueError("Code branch checkpoint does not match the requested task/repository")
            raw_task_state = branch_checkpoint.task_state
            if not isinstance(raw_task_state, Mapping):
                raise ValueError("Code branch checkpoint is missing serializable task state")
            task_state = CodeTaskStateView.from_dict(dict(raw_task_state))
            if task_state.instance_id != instance_id or task_state.problem_statement != problem:
                raise ValueError("Code branch checkpoint task state does not match public task")
            raw_belief_state = branch_checkpoint.belief_state
            belief = raw_belief_state.clone() if isinstance(raw_belief_state, CodeBeliefFilter) else CodeBeliefFilter.from_dict(raw_belief_state)
            messages = _restore_messages(branch_checkpoint.messages)
            token_ids, loss_mask, log_probs = _restore_token_state(branch_checkpoint.token_state)
        else:
            task_state = CodeTaskStateView(instance_id, problem, str(public_instance.get("task_kind") or "bugfix"), remaining_tool_budget=config.tool_budget)
            belief = CodeBeliefFilter()
            messages = []
            token_ids, loss_mask, log_probs = [], [], []
        settings = RolloutSettings(
            max_steps=min(self.settings.max_steps, config.max_tool_budget),
            max_new_tokens=self.settings.max_new_tokens,
            max_context_chars=min(self.settings.max_context_chars, config.context_max_chars),
            belief_prompt_max_chars=config.belief_prompt_max_chars,
        )
        lease = interaction_lease
        owns_lease = lease is None
        runtime = None
        code_messages: list[CodeMessage] = []
        events: list[dict[str, Any]] = []
        termination = "max_steps"
        failure_origin = "none"
        protocol_error = False
        invalid_action = False
        world_injected_validation_failure = False
        started = time.perf_counter()
        candidate_patch = ""
        resolved = False
        evaluator_result = None

        try:
            if lease is None:
                lease = await code_env_client.allocate(image_name, instance_id, cwd="/testbed", base_revision=base_revision)
            if branch_checkpoint is not None:
                current_patch = await code_env_client.diff(lease.lease_id, cwd=lease.cwd)
                if current_patch != branch_checkpoint.repo.patch:
                    raise ValueError("sibling lease repository state does not match its Code branch checkpoint")
            if world is None:
                world_context = public_sampling_context(tool_budget=config.tool_budget, repo_file_count=0)
                world = sample_required_worlds(instance_id=instance_id, image_name=image_name, base_revision=base_revision, context=world_context, rollout_seed=seed)[0]
            runtime = CodeWorldRuntime(world=world, client=code_env_client, lease_id=lease.lease_id, cwd=lease.cwd, task_state=task_state, registry=self.registry)
            if branch_checkpoint is not None:
                raw_world_state = branch_checkpoint.world_state
                if not isinstance(raw_world_state, Mapping):
                    raise ValueError("Code branch checkpoint is missing serializable world state")
                runtime.restore(raw_world_state)
                if runtime.call_index != task_state.call_index or branch_checkpoint.call_index != task_state.call_index:
                    raise ValueError("Code branch checkpoint call index is inconsistent")
                if branch_checkpoint.remaining_tool_budget != task_state.remaining_tool_budget:
                    raise ValueError("Code branch checkpoint budget is inconsistent")
                if branch_checkpoint.runtime_state_digest and branch_checkpoint.runtime_state_digest != runtime.runtime_state_digest:
                    raise ValueError("Code branch checkpoint runtime digest is inconsistent")
                events = [dict(item) for item in runtime.public_history]
            else:
                visible = build_policy_visible_belief(task_state, belief).to_prompt(max_chars=settings.belief_prompt_max_chars)
                messages = _prompt_messages(public_sample, visible)
            for _step in range(settings.max_steps):
                serialized_length = len(json.dumps(messages, ensure_ascii=False))
                if serialized_length > settings.max_context_chars:
                    termination = "context_overflow"
                    failure_origin = "real_infrastructure"
                    break
                if on_decision_checkpoint is not None:
                    checkpoint = await _capture_branch_checkpoint(
                        client=code_env_client,
                        lease=lease,
                        task_state=task_state,
                        runtime=runtime,
                        belief=belief,
                        messages=messages,
                        token_ids=token_ids,
                        loss_mask=loss_mask,
                        log_probs=log_probs,
                    )
                    callback_result = on_decision_checkpoint(checkpoint)
                    if inspect.isawaitable(callback_result):
                        await callback_result
                generation = _normalize_generation(await model_client.generate(messages, max_tokens=settings.max_new_tokens))
                # Never synthesize token masks or log-probabilities: training
                # must consume precisely the generation accounting emitted by
                # the model client, or reject the sample as untrainable.
                if generation.token_ids and (not generation.token_mask or not generation.token_logprobs):
                    termination = "invalid_token_accounting"
                    failure_origin = "real_infrastructure"
                    break
                parsed = parse_action(generation.text)
                assistant = CodeMessage("assistant", generation.text, generation.token_ids, generation.token_mask, generation.token_logprobs)
                code_messages.append(assistant)
                token_ids.extend(generation.token_ids)
                loss_mask.extend(generation.token_mask)
                log_probs.extend(generation.token_logprobs)
                messages.append({"role": "assistant", "content": generation.text})
                if not parsed.is_valid:
                    termination = "protocol_error"
                    failure_origin = "model_action"
                    protocol_error = True
                    invalid_action = True
                    break
                if parsed.kind == "final":
                    termination = "final"
                    break
                if parsed.kind == "abstain":
                    termination = "abstain"
                    break
                if task_state.remaining_tool_budget <= 0:
                    termination = "budget_exhausted"
                    break
                result, features, event = await runtime.execute_tool(parsed.tool_name or "", parsed.arguments or {})
                invalid_action = invalid_action or result.failure_origin == "model_action"
                if parsed.tool_name == "apply_patch" and result.status == "ok":
                    world_injected_validation_failure = False
                elif parsed.tool_name in {"run_tests", "run_checks"} and result.failure_origin == "world_injected":
                    # The agent attempted validation, but the hidden world
                    # denied/corrupted it.  It cannot make a final answer an
                    # agent-attributable premature-stop failure.
                    world_injected_validation_failure = True
                family = event.public_context.get("family")
                belief_snapshot = belief.update(parsed.tool_name or "", result, features=features, family=family)
                events.append(event.visible_dict())
                if result.failure_origin == "real_infrastructure":
                    failure_origin = "real_infrastructure"
                    termination = "real_infrastructure_failure"
                    break
                observation = json.dumps(result.visible_dict(), ensure_ascii=False, sort_keys=True)
                messages.append({"role": "user", "content": f"<interpreter>{observation}</interpreter>"})
                visible = build_policy_visible_belief(task_state, belief_snapshot).to_prompt(max_chars=settings.belief_prompt_max_chars)
                messages.append({"role": "user", "content": visible})
            else:
                termination = "max_steps"
            candidate_patch = await code_env_client.diff(lease.lease_id, cwd=lease.cwd)
        except Exception as exc:
            failure_origin = "real_infrastructure"
            termination = "real_infrastructure_failure"
            events.append({"status": "error", "failure": str(exc)})
        finally:
            if lease is not None and owns_lease:
                try:
                    if not candidate_patch:
                        candidate_patch = await code_env_client.diff(lease.lease_id, cwd=lease.cwd)
                except Exception:
                    failure_origin = "real_infrastructure"
                try:
                    await code_env_client.close(lease.lease_id)
                except Exception:
                    failure_origin = "real_infrastructure"

        if eval_script is not None and candidate_patch and failure_origin != "real_infrastructure":
            evaluator_result = await CleanEvaluator(code_env_client).evaluate(
                EvaluatorRequest(image_name=image_name, instance_id=instance_id, patch=candidate_patch, eval_script=eval_script, base_revision=base_revision, cwd="/testbed", timeout=config.evaluation_timeout, evaluator_patch=evaluator_patch)
            )
            resolved = bool(evaluator_result.resolved)
            if evaluator_result.failure_origin == "real_infrastructure":
                failure_origin = "real_infrastructure"
        latency_ms = sum(float(event.get("latency_ms", 0.0) or 0.0) for event in events)
        observation_chars = sum(len(str(event.get("output", "") or "")) for event in events)
        cost = normalized_cost(
            tool_calls=len(events),
            latency_ms=latency_ms,
            observation_chars=observation_chars,
            validation_runtime_ms=task_state.validation_runtime_ms,
            budget=config.tool_budget,
        )
        inefficiency = inefficiency_score(events)
        failure_penalty_value = failure_penalty(
            failure_origin="none",
            protocol_error=protocol_error,
            invalid_action=invalid_action,
            premature_final=(
                termination == "final"
                and not task_state.evidence_sufficient
                and not world_injected_validation_failure
                and not resolved
            ),
            budget_exhausted=termination in {"budget_exhausted", "context_overflow"},
        )
        task_quality = 1.0 if resolved else 0.0
        utility = compute_utility(
            resolved=resolved,
            cost=cost,
            inefficiency=inefficiency,
            failure_penalty_value=failure_penalty_value,
        )
        valid = failure_origin != "real_infrastructure"
        reward = CodeReward(task_quality, utility, cost, inefficiency, failure_penalty_value, resolved, valid, failure_origin)
        patch_hash = hashlib.sha256(candidate_patch.encode("utf-8")).hexdigest()
        public_metadata = {
            "environment": "code", "instance_id": instance_id, "data_source": data_source, "tool_budget": config.tool_budget,
            "tool_calls_used": len(events), "termination_reason": termination, "valid_for_rl": valid,
            "candidate_patch_sha256": patch_hash, "resolved": resolved, "belief_schema_version": config.belief_schema_version,
            "tool_schema_version": config.tool_schema_version, "world_schema_version": config.world_schema_version,
            "cost": cost, "inefficiency": inefficiency, "failure_penalty": failure_penalty_value,
            "policy_gradient_eligible": bool(token_ids) and len(token_ids) == len(loss_mask) == len(log_probs),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        trainer_metadata = {
            "environment": "code", "instance_id": instance_id, "coupling_id": world.coupling_id if world else "", "latent_world_id": world.latent_world_id if world else "",
            "world_slot_role": world.world_slot_role if world else "", "world_type": world.world_type if world else "", "world_runtime_digest": runtime.runtime_state_digest if runtime else "",
            "decision_prefix_hash": stable_hash(messages[:2], prefix="code-decision-prefix-v1"), "failure_origin": failure_origin,
            "events": getattr(runtime, "trainer_events", events),
        }
        return CodeTrajectory(tuple(token_ids), tuple(loss_mask), tuple(log_probs), reward.utility, tuple(code_messages), public_metadata, trainer_metadata)


async def generate_code_trajectory(sample: Mapping[str, Any], model_client: CodeModelClient, code_env_client: Any, code_config: CodeConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    trajectory = await CodeRollout().run(sample, model_client=model_client, code_env_client=code_env_client, code_config=code_config, **kwargs)
    return {**trajectory.to_sample(), "messages": [message.__dict__ for message in trajectory.messages], "trainer_only_metadata": trajectory.trainer_only_metadata}


__all__ = ["CodeModelClient", "CodeRollout", "RolloutSettings", "generate_code_trajectory"]
