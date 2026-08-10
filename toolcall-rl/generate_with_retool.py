# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import uuid
from dataclasses import replace
from typing import Any, Mapping

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import tool sandbox functionality
from document_reward import compute_document_reward, extract_final_answer
from tool_sandbox import TOOL_CONFIGS, tool_registry
from tool_protocol import ParsedAction, parse_assistant_action

try:
    from bayestool.belief import BeliefRuntime
    from bayestool.config import config_from_args, stage_definition, validate_stage_capabilities
    from bayestool.decision import (
        AnswerRiskCalibrator,
        BayesQHead,
        DecisionController,
        Q_FEATURE_SCHEMA_VERSION,
        canonical_action,
        canonical_action_key,
        q_feature_vectors,
    )
    from bayestool.meta_episode import build_meta_episode
    from bayestool.replay import export_canonical_replay
    from bayestool.grouping import (
        DEFAULT_POLICY_VERSION,
        QuestionRolloutPlan,
        make_question_rollout_plan,
        make_runtime_state_digest,
        slot_role_from_metadata,
    )
    from bayestool.schema import TaskStateView
    from bayestool.training import (
        attach_bayestool_utility,
        bayestool_question_id,
        content_signature,
        make_bayestool_decision_group_id,
    )
    from bayestool.world import (
        DEFAULT_TOOL_ARGUMENT_CAPABILITIES,
        CleanResultCache,
        WorldRuntime,
        document_hash,
        sample_tool_world,
        stable_seed,
    )
except Exception:  # pragma: no cover - baseline rollout remains importable without the optional package
    BeliefRuntime = None  # type: ignore[assignment]
    AnswerRiskCalibrator = None  # type: ignore[assignment]
    BayesQHead = None  # type: ignore[assignment]
    DecisionController = None  # type: ignore[assignment]
    Q_FEATURE_SCHEMA_VERSION = None  # type: ignore[assignment]
    canonical_action = None  # type: ignore[assignment]
    canonical_action_key = None  # type: ignore[assignment]
    q_feature_vectors = None  # type: ignore[assignment]
    build_meta_episode = None  # type: ignore[assignment]
    export_canonical_replay = None  # type: ignore[assignment]
    TaskStateView = None  # type: ignore[assignment]
    WorldRuntime = None  # type: ignore[assignment]
    sample_tool_world = None  # type: ignore[assignment]
    DEFAULT_TOOL_ARGUMENT_CAPABILITIES = {}  # type: ignore[assignment]
    config_from_args = None  # type: ignore[assignment]
    stage_definition = None  # type: ignore[assignment]
    validate_stage_capabilities = None  # type: ignore[assignment]
    attach_bayestool_utility = None  # type: ignore[assignment]
    bayestool_question_id = None  # type: ignore[assignment]
    content_signature = None  # type: ignore[assignment]
    make_bayestool_decision_group_id = None  # type: ignore[assignment]
    DEFAULT_POLICY_VERSION = "bayestool-policy-v1"  # type: ignore[assignment]
    QuestionRolloutPlan = None  # type: ignore[assignment]
    make_question_rollout_plan = None  # type: ignore[assignment]
    make_runtime_state_digest = None  # type: ignore[assignment]
    slot_role_from_metadata = None  # type: ignore[assignment]
    document_hash = None  # type: ignore[assignment]
    CleanResultCache = None  # type: ignore[assignment]
    stable_seed = None  # type: ignore[assignment]

try:
    from bayestool.training import compute_bayestool_utility
except Exception:  # pragma: no cover
    compute_bayestool_utility = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_STRICT_ASSISTANT_ACTION_RULE = (
    "Assistant output follows a strict action protocol. Every assistant turn must contain exactly one "
    "complete action and no other text: use <tool_call>...</tool_call> for one tool call, "
    "<final>...</final> for a grounded answer, or <abstain>...</abstain> when evidence is insufficient. "
    "Do not emit analysis, explanations, Markdown, code fences, or extra tags. "
    "<task_state>, <tool_belief>, <tool_state>, <tool_result>, and <interpreter> are read-only "
    "observation metadata; never copy or output them in an assistant turn."
)

_OBSERVATION_PROTOCOL_RULE = (
    "Observation metadata is read-only. Never copy <task_state>, <tool_belief>, <tool_state>, "
    "<tool_result>, or <interpreter> into an assistant turn. Reply with exactly one complete "
    "<tool_call>...</tool_call>, <final>...</final>, or <abstain>...</abstain> action and no other text."
)

_PRM_SEMAPHORE: asyncio.Semaphore | None = None
_PRM_TOKENIZER: Any = None
_BAYES_CLEAN_RESULT_TASKS: dict[str, asyncio.Task] = {}
_BAYES_CLEAN_RESULT_CACHE = CleanResultCache() if CleanResultCache is not None else None
# A branch checkpoint is intentionally process-local.  It contains live
# runtime objects (including the clean-result cache) and is consumed exactly
# once by its child rollout.  Only the opaque checkpoint id and public branch
# diagnostics are copied into Sample.metadata.
_BAYES_BRANCH_CHECKPOINTS: dict[str, dict[str, Any]] = {}
_BAYES_FILTER_MODEL_CACHE: dict[str, tuple[Any, str]] = {}
_BAYES_Q_HEAD_CACHE: dict[str, tuple[Any, str]] = {}
# Meta-episode questions run sequentially in one rollout worker.  Keep the
# sampled world object (including call count/regime position) alive across
# those child generations; only the task-local belief/navigation state resets.
_BAYES_META_WORLD_RUNTIMES: dict[str, Any] = {}


def _get_generation_prompt_suffix(sample_prompt: str) -> str:
    """Extract suffix after the last '<|im_start|>assistant\\n' in sample.prompt.

    sample.prompt is produced by tokenizer.apply_chat_template with the user's
    kwargs, so the suffix faithfully reflects the intended generation prompt.
    e.g. Qwen3.5 default 鈫?'<think>\\n', Qwen3 default 鈫?''.
    """
    tag = "<|im_start|>assistant\n"
    idx = sample_prompt.rfind(tag)
    if idx >= 0:
        return sample_prompt[idx + len(tag):]
    return ""

# Jinja2 template for tool-enabled conversations (Qwen3 JSON format)
TOOL_TEMPLATE_JSON = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

# Jinja2 template for tool-enabled conversations (Qwen3.5 XML format)
TOOL_TEMPLATE_XML = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}

# Tools

You have access to the following functions:

<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

# Cached tool call format: "json" (Qwen3) or "xml" (Qwen3.5)
_TOOL_CALL_FORMAT: str | None = None


def _detect_tool_call_format(tokenizer) -> str:
    """Detect whether the model uses JSON or XML tool call format.

    Qwen3.5 chat template uses '<function=' XML format;
    Qwen3 and others use JSON '{"name": ...}' format.
    """
    global _TOOL_CALL_FORMAT
    if _TOOL_CALL_FORMAT is not None:
        return _TOOL_CALL_FORMAT
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if "<function=" in chat_template or "<parameter=" in chat_template:
        _TOOL_CALL_FORMAT = "xml"
    else:
        _TOOL_CALL_FORMAT = "json"
    logger.info(f"Detected tool call format: {_TOOL_CALL_FORMAT}")
    return _TOOL_CALL_FORMAT

_PRM_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_PRM_STRICT_NUMBER_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")


def format_conversation_with_tools(
    prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None,
    tool_call_format: str = "json",
) -> str:
    """Format a Qwen tool conversation with the model's token boundaries intact.

    Qwen3-VL's native chat template is deliberately newline-sensitive: the
    role marker is followed by ``\\n`` and tool schemas are emitted as the
    standard ``{"type": "function", "function": ...}`` objects.  The old
    Jinja template used ``{%-``/``{{-`` around those boundaries, which
    silently rendered ``<|im_start|>systemYou...`` and
    ``<|im_start|>userDocument...``.  That prompt is syntactically valid text
    but is out of distribution for the model's tool instruction format.

    Build this small fixed envelope explicitly so its whitespace remains
    stable.  The full trajectory is still assembled by the rollout loop; this
    helper only formats the initial tool-enabled conversation.
    """

    # Always add a system message - use provided one or default.
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a document-understanding agent. Inspect the document path provided in the task "
            "with the available tools before answering. Choose tools deliberately: parse_document for "
            "general text, render/crop/zoom plus OCR for visual regions, detect_layout for coordinates, "
            "extract_table for tables, and chart_to_table for charts. Ground the answer in tool output, "
            "do not invent unread evidence, and wrap the supported answer in <final>...</final>. "
            "When the current page does not contain sufficient evidence and the document has unvisited "
            "pages, inspect additional pages before answering. Do not conclude that information is absent "
            "until all relevant pages have been checked. After every tool result, use the navigation state "
            "to select an unvisited page or the appropriate table/visual tool. Do not repeat a page unless "
            "a different tool is needed. When all relevant pages have been checked, do not summarize the "
            "whole document: answer concisely and end with exactly one <final>...</final> action. "
            "When the available evidence remains unreliable after the budget is used, you may instead "
            "emit exactly one <abstain>reason</abstain> action; do not abstain when reliable evidence supports an answer."
        )
    system_content = f"{str(system_content).rstrip()}\n\n{_STRICT_ASSISTANT_ACTION_RULE}"

    messages_to_render: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})
    if messages:
        messages_to_render.extend(messages)

    rendered: list[str] = ["<|im_start|>system\n"]
    rendered.append(str(messages_to_render[0].get("content", "")))

    tool_specs = list(tools or [])
    if tool_specs:
        rendered.append("\n\n")
        if tool_call_format == "xml":
            rendered.append(
                "# Tools\n\n"
                "You have access to the following functions:\n\n"
                "<tools>"
            )
        else:
            rendered.append(
                "# Tools\n\n"
                "You may call one function per assistant turn to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>"
            )
        for tool in tool_specs:
            rendered.append("\n")
            rendered.append(json.dumps(tool, ensure_ascii=False))
        rendered.append("\n</tools>\n\n")
        if tool_call_format == "xml":
            rendered.append(
                "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n"
                "<function=example_function_name>\n"
                "<parameter=example_parameter_1>\n"
                "value_1\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>"
            )
        else:
            rendered.append(
                "If you choose to call a function, return exactly one JSON object with its name and arguments "
                "within <tool_call></tool_call> XML tags and no suffix text:\n"
                "<tool_call>\n"
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                "</tool_call>"
            )
        rendered.append("<|im_end|>\n")
    else:
        rendered.append("<|im_end|>\n")

    for message in messages_to_render[1:]:
        role = str(message.get("role", ""))
        if role not in {"user", "assistant"}:
            continue
        rendered.append(f"<|im_start|>{role}\n")
        rendered.append(str(message.get("content", "")))
        rendered.append("<|im_end|>\n")
    rendered.append("<|im_start|>assistant\n")
    return "".join(rendered)


def _extract_task_prompt(prompt: str | list[dict[str, str]]) -> str:
    """Recover the user task if slime already applied a Qwen chat template."""
    if isinstance(prompt, list):
        users = [str(message.get("content", "")) for message in prompt if message.get("role") == "user"]
        return users[-1] if users else json.dumps(prompt, ensure_ascii=False)
    matches = re.findall(r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    return matches[-1].strip() if matches else prompt


def _bayestool_is_enabled(args: Any, metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    nested = metadata.get("bayestool") if isinstance(metadata.get("bayestool"), dict) else {}
    return bool(
        getattr(args, "bayestool_enable", False)
        or metadata.get("bayestool_enable", False)
        or nested.get("enabled", False)
    ) and WorldRuntime is not None and BeliefRuntime is not None


def _bayestool_meta_enabled(args: Any, metadata: dict[str, Any] | None = None) -> bool:
    """Return the configured meta-episode ablation state."""

    if config_from_args is None:
        return True
    try:
        config = config_from_args(args, enabled=True)
        return bool(getattr(config, "use_meta_episode", True))
    except (TypeError, ValueError, AttributeError):
        return True


def _bayestool_runtime_identity(sample: Sample, task_prompt: str) -> tuple[str, str]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    document_path = metadata.get("document_path")
    if not document_path:
        match = re.search(r"Document path:\s*(.*?)(?:\n\s*Question:|$)", task_prompt, re.IGNORECASE | re.DOTALL)
        document_path = match.group(1).strip() if match else task_prompt
    # Recompute the identity from the document bytes/path and the public task
    # fields.  An upstream manifest may contain stale or label-dependent
    # identity fields; trusting them here would silently break coupled-world
    # reproducibility.  Meta-episode children are the one intentional
    # exception: their wrapper assigns one coupling id to all questions so the
    # persistent cross-question session sees the same world.
    digest = str(document_hash(document_path) if document_hash else metadata.get("document_hash") or task_prompt)
    question_match = re.search(r"(?:^|\n)Question:\s*(.*?)(?:\n\s*\n|$)", task_prompt, re.IGNORECASE | re.DOTALL)
    question = question_match.group(1).strip() if question_match else task_prompt
    task_id = metadata.get("task_id", metadata.get("id", ""))
    coupling_payload = "|".join((digest, str(question), str(task_id or "")))
    computed_coupling_id = "coupling-" + hashlib.sha256(coupling_payload.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    preserve_meta_coupling = bool(metadata.get("meta_episode_id") or metadata.get("meta_episode_child"))
    coupling_id = str(metadata.get("coupling_id") or computed_coupling_id) if preserve_meta_coupling else computed_coupling_id
    return digest, coupling_id


def _checkpoint_state(payload: Any) -> tuple[dict[str, Any], str]:
    """Extract a plain model state dict and its recorded version."""

    if not isinstance(payload, dict):
        raise ValueError("BayesTool checkpoint must contain a mapping")
    state = payload.get("model_state") or payload.get("state_dict") or payload
    if not isinstance(state, dict) or not state:
        raise ValueError("BayesTool checkpoint has no model_state/state_dict")
    if any(str(key).startswith("module.") for key in state):
        state = {
            str(key)[len("module.") :] if str(key).startswith("module.") else str(key): value
            for key, value in state.items()
        }
    version = str(payload.get("belief_model_version") or payload.get("model_version") or "checkpoint")
    return state, version


def _load_bayestool_models(args: Any, config: Any) -> tuple[Any, str, Any, str]:
    """Load the small Stage-A filter and optional finite-branch Q head once per worker."""

    filter_model = None
    filter_version = str(getattr(args, "belief_model_version", "untrained") or "untrained")
    filter_path_value = getattr(args, "bayestool_belief_checkpoint", None)
    if filter_path_value:
        filter_path = str(Path(str(filter_path_value)).expanduser().resolve())
        cached = _BAYES_FILTER_MODEL_CACHE.get(filter_path)
        if cached is not None:
            filter_model, filter_version = cached
        else:
            if not Path(filter_path).is_file():
                raise FileNotFoundError(f"BayesTool belief checkpoint does not exist: {filter_path}")
            if BeliefRuntime is None:
                raise RuntimeError("BayesTool belief package is unavailable while a checkpoint was requested")
            try:
                import torch
                from bayestool.belief import ToolWorldFilterNetwork

                payload = torch.load(filter_path, map_location="cpu")
                state, filter_version = _checkpoint_state(payload)
                filter_model = ToolWorldFilterNetwork(config)
                incompatible = filter_model.load_state_dict(state, strict=False)
                if incompatible.missing_keys:
                    raise RuntimeError(
                        "BayesTool belief checkpoint is incompatible; missing keys: "
                        + ", ".join(incompatible.missing_keys[:8])
                    )
                filter_model.eval()
            except Exception as exc:
                raise RuntimeError(f"failed to load BayesTool belief checkpoint {filter_path}: {exc}") from exc
            _BAYES_FILTER_MODEL_CACHE[filter_path] = (filter_model, filter_version)

    q_head = None
    q_version = "untrained"
    q_path_value = getattr(args, "bayestool_q_checkpoint", None)
    if not q_path_value and filter_path_value:
        derived = Path(str(filter_path_value)).expanduser().with_suffix(".qhead.pt")
        if derived.is_file():
            q_path_value = str(derived)
    if q_path_value:
        q_path = str(Path(str(q_path_value)).expanduser().resolve())
        cached = _BAYES_Q_HEAD_CACHE.get(q_path)
        if cached is not None:
            q_head, q_version = cached
        else:
            if not Path(q_path).is_file():
                raise FileNotFoundError(f"BayesTool Q-head checkpoint does not exist: {q_path}")
            if BayesQHead is None:
                raise RuntimeError("BayesTool Q-head package is unavailable while a checkpoint was requested")
            try:
                import torch

                payload = torch.load(q_path, map_location="cpu")
                checkpoint_schema = str(payload.get("q_feature_schema_version") or "")
                if checkpoint_schema != Q_FEATURE_SCHEMA_VERSION:
                    raise RuntimeError(
                        "BayesTool Q-head feature schema mismatch: "
                        f"checkpoint={checkpoint_schema or 'missing'} expected={Q_FEATURE_SCHEMA_VERSION}"
                    )
                state, q_version = _checkpoint_state(payload)
                q_head = BayesQHead()
                incompatible = q_head.load_state_dict(state, strict=False)
                if incompatible.missing_keys:
                    raise RuntimeError(
                        "BayesTool Q-head checkpoint is incompatible; missing keys: "
                        + ", ".join(incompatible.missing_keys[:8])
                    )
                q_head.eval()
            except Exception as exc:
                raise RuntimeError(f"failed to load BayesTool Q-head checkpoint {q_path}: {exc}") from exc
            _BAYES_Q_HEAD_CACHE[q_path] = (q_head, q_version)
    return filter_model, filter_version, q_head, q_version


def _load_bayestool_risk_calibrator(args: Any) -> tuple[Any, str]:
    """Load a validation-fitted AnswerRiskCalibrator when configured."""

    path_value = getattr(args, "bayestool_risk_checkpoint", None)
    if not path_value or AnswerRiskCalibrator is None:
        return None, "heuristic"
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BayesTool risk calibrator checkpoint does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = str(payload.get("version") or "validation-fitted")
        if isinstance(payload.get("calibrator"), Mapping):
            payload = payload["calibrator"]
        calibrator = AnswerRiskCalibrator.from_dict(payload)
    except Exception as exc:
        raise RuntimeError(f"failed to load BayesTool risk calibrator {path}: {exc}") from exc
    return calibrator, version


def _bayestool_environment_block(
    belief_runtime: Any,
    navigation_state: dict[str, Any],
    *,
    tool_budget: int,
    tokenizer: Any | None = None,
) -> str:
    if belief_runtime is None or TaskStateView is None:
        return ""
    navigation_state["remaining_tool_budget"] = max(0, int(tool_budget))
    task_state = TaskStateView.from_navigation_state(navigation_state, tool_budget=tool_budget)
    return belief_runtime.to_prompt_block(task_state, tokenizer=tokenizer)


def _bayestool_append_observation(
    observed_result: str,
    belief_runtime: Any,
    navigation_state: dict[str, Any],
    *,
    tool_budget: int,
    tokenizer: Any | None = None,
) -> str:
    block = _bayestool_environment_block(
        belief_runtime,
        navigation_state,
        tool_budget=tool_budget,
        tokenizer=tokenizer,
    )
    if not block:
        return observed_result
    return f"{observed_result}\n{block}"


def _bayestool_token_count(value: Any, tokenizer: Any | None = None) -> int:
    """Count visible observation tokens without requiring a tokenizer."""

    text_value = str(value or "")
    if not text_value:
        return 0
    if tokenizer is not None and callable(tokenizer):
        try:
            encoded = tokenizer(text_value, add_special_tokens=False)
            ids = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else encoded
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            while isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if isinstance(ids, list):
                return len(ids)
        except Exception:
            pass
    return max(1, math.ceil(len(text_value) / 4.0))


async def _bayestool_execute_clean_cached(
    world_runtime: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Share one clean tool execution across coupled world replicas."""

    if world_runtime is None:
        return await tool_registry.execute_tool(tool_name, arguments)
    document_digest = str(getattr(world_runtime, "document_digest", ""))
    cache = getattr(world_runtime, "clean_cache", None)
    if cache is not None:
        key = cache.key(document_digest, tool_name, arguments, getattr(world_runtime, "tool_backend_version", "v1"))
    else:
        key = "|".join((document_digest, tool_name, json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)))
    task = _BAYES_CLEAN_RESULT_TASKS.get(key)
    if task is None:
        if cache is not None and hasattr(cache, "get_or_set_async"):
            task = asyncio.create_task(
                cache.get_or_set_async(
                    document_digest,
                    tool_name,
                    arguments,
                    getattr(world_runtime, "tool_backend_version", "v1"),
                    lambda: tool_registry.execute_tool(tool_name, arguments),
                )
            )
        else:
            task = asyncio.create_task(tool_registry.execute_tool(tool_name, arguments))
        _BAYES_CLEAN_RESULT_TASKS[key] = task
    try:
        return str(await asyncio.shield(task))
    except Exception:
        if _BAYES_CLEAN_RESULT_TASKS.get(key) is task:
            _BAYES_CLEAN_RESULT_TASKS.pop(key, None)
        raise


def _bayestool_action_text(action: Any) -> str:
    """Render a canonical action as strict protocol text for auxiliary replay."""

    if not isinstance(action, dict):
        return str(action or "")
    kind = str(action.get("kind") or action.get("type") or "").casefold()
    if kind in {"tool", "tool_call"}:
        return "<tool_call>" + json.dumps(
            {
                "name": str(action.get("tool") or action.get("name") or ""),
                "arguments": dict(action.get("arguments") or {}),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "</tool_call>"
    if kind == "final":
        return f"<final>{str(action.get('answer') or '')}</final>"
    if kind == "abstain":
        return f"<abstain>{str(action.get('reason') or '')}</abstain>"
    return str(action)


def _bayestool_branch_action_from_text(
    text: str,
    navigation_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, ParsedAction]:
    """Parse one policy sample into the canonical action used by BayesQ."""

    parsed = parse_assistant_action(text)
    if parsed.kind == "tool_call" and isinstance(parsed.value, dict):
        value = dict(parsed.value)
        arguments = dict(value.get("arguments", {}))
        tool_name = str(value.get("name") or "")
        arguments, _ = _tool_arguments_for_navigation(tool_name, arguments, navigation_state)
        value = {"name": tool_name, "arguments": arguments}
        if not _validate_tool_call(value)[0]:
            return None, parsed
        return {"kind": "tool", "tool": tool_name, "arguments": arguments}, parsed
    if parsed.kind == "final":
        return {"kind": "final", "answer": str(parsed.value or "")}, parsed
    if parsed.kind == "abstain":
        return {"kind": "abstain", "reason": str(parsed.value or "")}, parsed
    return None, parsed


def _bayestool_branch_prefix_hash(
    prompt_token_ids: list[int],
    context_token_ids: list[int],
    response_token_ids: list[int],
    turn: int,
) -> str:
    payload = {
        "prompt": list(prompt_token_ids),
        "context": list(context_token_ids),
        "response": list(response_token_ids),
        "turn": int(turn),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _branch_deepcopy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return dict(value)
        return value


def _capture_bayestool_branch_checkpoint(
    *,
    prompt_token_ids: list[int],
    context_token_ids: list[int],
    context_image_data: list[str],
    context_segments: list[dict[str, Any]],
    response: str,
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float],
    current_images: list[Any],
    multimodal_train_inputs_buffer: list[dict[str, Any]],
    execution_trace: list[dict[str, Any]],
    action_log: dict[str, Any],
    generation_steps: list[dict[str, Any]],
    step_action_spans: list[dict[str, int]],
    navigation_state: dict[str, Any],
    world_runtime: Any,
    belief_runtime: Any,
    bayes_document_digest: str | None,
    bayes_coupling_id: str | None,
    world_sample_index: int,
    turn: int,
    tool_call_count: int,
    max_tool_steps: int,
    config: Any,
    runtime_state_digest: str = "",
) -> dict[str, Any]:
    """Snapshot only the mutable state needed to replay a shared-prefix branch.

    The world specification and belief are copied before the current action is
    executed.  The clean-result cache is deliberately reattached after the
    copy so sibling branches reuse the exact same clean tool result.
    """

    prefix_hash = _bayestool_branch_prefix_hash(
        prompt_token_ids,
        context_token_ids,
        response_token_ids,
        turn,
    )
    copied_world = _branch_deepcopy(world_runtime)
    if copied_world is not None and world_runtime is not None:
        try:
            copied_world.clean_cache = world_runtime.clean_cache
        except Exception:
            pass
    return {
        "prefix_hash": prefix_hash,
        "prompt_token_ids": list(prompt_token_ids),
        "context_token_ids": list(context_token_ids),
        "context_image_data": list(context_image_data),
        "context_segments": _branch_deepcopy(context_segments),
        "response": str(response),
        "response_token_ids": list(response_token_ids),
        "loss_masks": list(loss_masks),
        "rollout_log_probs": list(rollout_log_probs),
        "current_images": _branch_deepcopy(current_images),
        "multimodal_train_inputs_buffer": _branch_deepcopy(multimodal_train_inputs_buffer),
        "execution_trace": _branch_deepcopy(execution_trace),
        "action_log": _branch_deepcopy(action_log),
        "generation_steps": _branch_deepcopy(generation_steps),
        "step_action_spans": _branch_deepcopy(step_action_spans),
        "navigation_state": _branch_deepcopy(navigation_state),
        "world_runtime": copied_world,
        "belief_runtime": _branch_deepcopy(belief_runtime),
        "bayes_document_digest": bayes_document_digest,
        "bayes_coupling_id": bayes_coupling_id,
        "world_sample_index": int(world_sample_index),
        "turn": int(turn),
        "tool_call_count": int(tool_call_count),
        "max_tool_steps": int(max_tool_steps),
        "config": config,
        "runtime_state_digest": str(runtime_state_digest),
    }


def _clone_bayestool_branch_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    clone = _branch_deepcopy(checkpoint)
    if clone.get("world_runtime") is not None and checkpoint.get("world_runtime") is not None:
        try:
            clone["world_runtime"].clean_cache = checkpoint["world_runtime"].clean_cache
        except Exception:
            pass
    return clone


async def _sample_bayestool_branch_candidates(
    *,
    state: Any,
    url: str,
    checkpoint: dict[str, Any],
    current_text: str,
    current_token_ids: list[int],
    current_log_probs: list[float],
    sampling_params: dict[str, Any],
    im_end_id: int | None,
    config: Any,
    target_group_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Draw actual policy candidates from one identical model prefix."""

    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    current_action, current_parsed = _bayestool_branch_action_from_text(
        current_text,
        checkpoint.get("navigation_state", {}),
    )
    if current_action is None or canonical_action_key is None:
        return candidates, errors

    def add_candidate(action: dict[str, Any], raw: str, token_ids: list[int], log_probs: list[float], *, source: str) -> None:
        key = canonical_action_key(action)
        if any(item["key"] == key for item in candidates):
            return
        candidates.append(
            {
                "key": key,
                "action": action,
                "raw": str(raw),
                "token_ids": list(token_ids),
                "log_probs": list(log_probs),
                "source": source,
            }
        )

    add_candidate(current_action, current_text, current_token_ids, current_log_probs, source="primary")
    # K=8 requires seven independent children plus the primary.  A legacy
    # max-action-candidates=4 must not silently turn that plan into K4.
    target_group_size = int(
        target_group_size
        if target_group_size is not None
        else getattr(config, "default_group_size", 4)
        or 4
    )
    max_candidates = max(
        1,
        int(getattr(config, "max_action_candidates", 4)),
        target_group_size,
    )
    attempts = max(0, max_candidates * 3 - 1)
    for offset in range(attempts):
        if len(candidates) >= max_candidates:
            break
        params = dict(sampling_params)
        base_seed = params.get("sampling_seed")
        if isinstance(base_seed, int):
            params["sampling_seed"] = base_seed + offset + 1
        else:
            # Explicit seeds make branch construction reproducible without
            # changing the caller's deterministic-inference flag.
            seed_bytes = hashlib.sha256(
                f"{checkpoint['prefix_hash']}:{offset}".encode("utf-8")
            ).digest()[:8]
            params["sampling_seed"] = int.from_bytes(seed_bytes, "big") % (2**31 - 1)
        payload: dict[str, Any] = {
            "input_ids": list(checkpoint["context_token_ids"]),
            "sampling_params": params,
            "return_logprob": True,
        }
        if checkpoint.get("context_image_data"):
            payload["image_data"] = list(checkpoint["context_image_data"])
        try:
            output = await post(url, payload)
            extracted = _extract_generation_output(output, state, im_end_id)
            raw = str(extracted["raw_generation_text"])
            action, parsed = _bayestool_branch_action_from_text(raw, checkpoint.get("navigation_state", {}))
            if action is None:
                continue
            add_candidate(
                action,
                raw,
                list(extracted["token_ids"]),
                list(extracted["log_probs"]),
                source="branch_policy_sample",
            )
        except Exception as exc:
            errors.append(str(exc))
    return candidates, errors


def _select_bayestool_policy_candidate(
    candidates: list[dict[str, Any]],
    *,
    navigation_state: dict[str, Any],
    belief_runtime: Any,
    decision_controller: Any,
    tool_budget: int,
) -> tuple[dict[str, Any] | None, Any | None, dict[str, Any]]:
    """Select one of the actual completions before any selected action runs."""

    if not candidates or decision_controller is None or TaskStateView is None:
        return candidates[0] if candidates else None, None, {
            "candidate_degenerate": len(candidates) < 2,
            "candidate_action_keys": [item.get("key") for item in candidates],
        }
    task_state = TaskStateView.from_navigation_state(
        navigation_state,
        tool_budget=max(0, int(tool_budget)),
    )
    actions = [item["action"] for item in candidates]
    diagnostic_actions = [
        action
        for action in actions
        if str(action.get("tool") or "") in {"detect_layout", "render_page", "crop_region", "zoom_region", "ocr_region"}
    ]
    report = decision_controller.select(
        task_state,
        belief_runtime,
        actions,
        diagnostic_actions=diagnostic_actions,
        risk_metadata={
            "independent_tool_count": len(set(
                str(item.get("tool"))
                for item in navigation_state.get("evidence_candidates", [])
                if isinstance(item, dict) and item.get("tool")
            )),
            "recent_surprise": float(navigation_state.get("recent_surprise", 0.0) or 0.0),
            "answer_self_consistency": float(navigation_state.get("answer_self_consistency", 0.0) or 0.0),
            "page_count": max(1, int(navigation_state.get("page_count", 0) or 0)),
        },
    )
    selected_key = str(report.selected_action or report.bayes_action or candidates[0]["key"])
    selected = next((item for item in candidates if str(item.get("key")) == selected_key), candidates[0])
    ordered_values = sorted(
        list(report.action_values),
        key=lambda value: (-float(value.value_mean), str(value.action_key)),
    )
    margin = (
        float(ordered_values[0].value_mean - ordered_values[1].value_mean)
        if len(ordered_values) >= 2
        else None
    )
    snapshot = belief_runtime.snapshot().to_dict() if belief_runtime is not None else {}
    metadata = {
        "selected_action": report.selected_action,
        "selected_mode": report.selected_mode,
        "consensus": report.consensus,
        "consensus_action": report.consensus_action,
        "bayes_action": report.bayes_action,
        "decision_regret": float(report.decision_regret),
        "dvoi": dict(report.dvoi or {}),
        "stop_decision": dict(report.stop_decision or {}),
        "candidate_degenerate": len(candidates) < 2,
        "candidate_action_keys": [str(item["key"]) for item in candidates],
        "candidate_actions": [
            {
                "key": str(item["key"]),
                "action": canonical_action(item["action"]),
                "text": str(item.get("raw", "")),
                "source": str(item.get("source", "")),
                "token_count": len(item.get("token_ids", [])),
            }
            for item in candidates
        ],
        "selected_candidate_key": str(selected.get("key")),
        "selected_candidate_source": str(selected.get("source", "")),
        "best_action_margin": margin,
        "policy_action_text": str(candidates[0].get("raw", "")),
        "selected_action_text": str(selected.get("raw", "")),
        "belief_snapshot": snapshot,
        "observed_prefix_event_count": int(snapshot.get("step", 0) or 0),
    }
    metadata["branch_eligible"] = bool(
        len(candidates) >= 2
        and bool(getattr(decision_controller.config, "use_regret_branching", True))
        and float(report.decision_regret) > float(decision_controller.config.decision_regret_threshold)
        and task_state.remaining_tool_budget >= 2
        and float(snapshot.get("ood_score", 0.0) or 0.0) < 0.15
        and canonical_action(selected["action"]).get("kind") == "tool"
    )
    return selected, report, metadata


async def _launch_bayestool_branches(
    *,
    args: Any,
    state: Any,
    url: str,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool,
    checkpoint: dict[str, Any],
    current_text: str,
    current_token_ids: list[int],
    current_log_probs: list[float],
    im_end_id: int | None,
    decision_controller: Any,
    bayes_config: Any,
    candidates_override: list[dict[str, Any]] | None = None,
    candidate_errors_override: list[str] | None = None,
    target_group_size: int | None = None,
    force_branch: bool = False,
) -> tuple[list[Sample], dict[str, Any]]:
    """Create real sibling continuations from one shared prefix."""

    if decision_controller is None or bayes_config is None or canonical_action_key is None:
        return [], {"branch_triggered": False, "reason": "bayestool decision runtime unavailable"}
    if candidates_override is not None:
        candidates = list(candidates_override)
        candidate_errors = list(candidate_errors_override or [])
    else:
        candidates, candidate_errors = await _sample_bayestool_branch_candidates(
            state=state,
            url=url,
            checkpoint=checkpoint,
            current_text=current_text,
            current_token_ids=current_token_ids,
            current_log_probs=current_log_probs,
            sampling_params=sampling_params,
            im_end_id=im_end_id,
            config=bayes_config,
            target_group_size=target_group_size,
        )
    if len(candidates) < 2:
        return [], {
            "branch_triggered": False,
            "reason": "fewer than two valid policy candidates",
            "candidate_action_keys": [item["key"] for item in candidates],
            "candidate_errors": candidate_errors,
        }

    task_state = TaskStateView.from_navigation_state(
        checkpoint.get("navigation_state", {}),
        tool_budget=max(0, int(checkpoint.get("max_tool_steps", 0) - checkpoint.get("tool_call_count", 0))),
    )
    branch_belief = checkpoint.get("belief_runtime")
    action_values = [item["action"] for item in candidates]
    diagnostic_actions = [
        action
        for action in action_values
        if str(action.get("tool") or "") in {"detect_layout", "render_page", "crop_region", "zoom_region", "ocr_region"}
    ]
    branch_controller = DecisionController(
        bayes_config,
        q_head=getattr(decision_controller, "q_head", None),
        risk_calibrator=getattr(decision_controller, "risk_calibrator", None),
        seed=int(getattr(sample, "index", 0) or 0),
    )
    if bool(getattr(decision_controller, "q_head_ready", False)):
        branch_controller.enable_q_head(True)
    report = branch_controller.select(
        task_state,
        branch_belief,
        action_values,
        diagnostic_actions=diagnostic_actions,
        risk_metadata={
            "independent_tool_count": len(checkpoint.get("navigation_state", {}).get("supporting_pages", [])),
            "recent_surprise": float(checkpoint.get("navigation_state", {}).get("recent_surprise", 0.0) or 0.0),
        },
    )
    gate_seed = int.from_bytes(
        hashlib.sha256(f"{checkpoint['prefix_hash']}:branch-gate".encode("utf-8")).digest()[:8],
        "big",
    )
    import random

    if not force_branch and not branch_controller.branch_gate(
        task_state,
        report,
        belief=branch_belief,
        rng=random.Random(gate_seed),
    ):
        return [], {
            "branch_triggered": False,
            "reason": "branch probability gate or eligibility failed",
            "candidate_action_keys": [item["key"] for item in candidates],
            "decision_regret": float(report.decision_regret),
            "candidate_errors": candidate_errors,
        }

    q_feature_records: dict[str, dict[str, Any]] = {}
    if q_feature_vectors is not None and report.particles:
        for candidate in candidates:
            particle_records: list[dict[str, Any]] = []
            for particle in report.particles:
                task_features, particle_features, action_features, budget_features = q_feature_vectors(
                    task_state,
                    particle,
                    candidate["action"],
                )
                particle_records.append(
                    {
                        "particle_id": int(getattr(particle, "particle_id", len(particle_records))),
                        "task_features": task_features,
                        "particle_features": particle_features,
                        "action_features": action_features,
                        "budget_features": budget_features,
                    }
                )
            if particle_records:
                q_feature_records[candidate["key"]] = dict(particle_records[0])
                q_feature_records[candidate["key"]]["particle_records"] = particle_records

    sibling_group_id = str(
        (sample.metadata or {}).get("sibling_group_id")
        or f"bayes-branch:{checkpoint['bayes_coupling_id']}:{checkpoint['prefix_hash'][:16]}"
    )
    branch_group_id = f"{sibling_group_id}:prefix:{checkpoint['prefix_hash'][:16]}"
    target_group_size = int(
        target_group_size
        if target_group_size is not None
        else getattr(bayes_config, "default_group_size", 4)
        or 4
    )
    # Direct legacy helper callers may provide only ``max_siblings`` and no
    # QuestionRolloutPlan.  Keep that isolated compatibility path working;
    # generated training samples always carry an explicit plan and therefore
    # take the strict K invariant below.
    explicit_plan = isinstance((sample.metadata or {}).get("question_rollout_plan"), Mapping)
    if not explicit_plan and int(getattr(bayes_config, "max_siblings", target_group_size)) < target_group_size:
        target_group_size = int(getattr(bayes_config, "max_siblings", target_group_size))
    if (explicit_plan and target_group_size not in {4, 8}) or (
        not explicit_plan and target_group_size < 2
    ):
        return [], {
            "branch_triggered": False,
            "degenerate_no_signal": True,
            "reason": f"invalid target decision-group K={target_group_size}",
        }
    max_siblings = (
        target_group_size
        if explicit_plan
        else max(target_group_size, int(getattr(bayes_config, "max_siblings", target_group_size)))
    )
    parent_action, _ = _bayestool_branch_action_from_text(
        current_text,
        checkpoint.get("navigation_state", {}),
    )
    parent_key = canonical_action_key(parent_action) if parent_action is not None else ""
    children: list[Sample] = []
    child_errors = list(candidate_errors)
    branch_candidates = [
        item for item in candidates
        if not parent_key or item["key"] != parent_key
    ][: max_siblings - 1]
    if len(branch_candidates) < target_group_size - 1:
        return [], {
            "branch_triggered": False,
            "degenerate_no_signal": True,
            "reason": "insufficient independent candidates for target K",
            "target_group_size": target_group_size,
            "available_independent_candidates": len(branch_candidates) + 1,
            "runtime_state_digest": str(checkpoint.get("runtime_state_digest", "")),
            "candidate_action_keys": [item["key"] for item in candidates],
            "candidate_errors": child_errors,
        }
    for child_number, candidate in enumerate(branch_candidates, start=1):
        resume_id = f"{checkpoint['prefix_hash']}:{child_number}:{uuid.uuid4().hex}"
        child_checkpoint = _clone_bayestool_branch_checkpoint(checkpoint)
        child_checkpoint["forced_action"] = {
            "raw": candidate["raw"],
            "token_ids": list(candidate["token_ids"]),
            "log_probs": list(candidate["log_probs"]),
            "key": candidate["key"],
        }
        child_checkpoint["branch_sibling_group_id"] = branch_group_id
        child_checkpoint["branch_horizon"] = int(getattr(bayes_config, "branch_horizon", 3))
        child_checkpoint["branch_q_features"] = q_feature_records.get(candidate["key"], {})
        _BAYES_BRANCH_CHECKPOINTS[resume_id] = child_checkpoint

        child = deepcopy(sample)
        parent_index = int(sample.index or 0)
        child.index = parent_index * 1_000_000 + int(checkpoint["turn"]) * 1_000 + child_number
        child.status = Sample.Status.PENDING
        child.tokens = list(checkpoint["prompt_token_ids"]) + list(checkpoint["response_token_ids"])
        child.response = str(checkpoint["response"])
        child.response_length = len(checkpoint["response_token_ids"])
        child.loss_mask = list(checkpoint["loss_masks"])
        child.rollout_log_probs = list(checkpoint["rollout_log_probs"])
        child.multimodal_inputs = {
            "images": _branch_deepcopy(checkpoint.get("current_images", [])),
            "videos": None,
        } if checkpoint.get("current_images") else None
        child.multimodal_train_inputs = _merge_multimodal_train_inputs(
            checkpoint.get("multimodal_train_inputs_buffer", [])
        )
        child.reward = None
        child.metadata = deepcopy(sample.metadata or {})
        child.metadata.update(
            {
                "bayestool_branch_child": True,
                "bayestool_branch_resume_id": resume_id,
                "bayestool_branch_id": f"{branch_group_id}:{child_number}",
                "bayestool_branch_parent_index": parent_index,
                "bayestool_branch_prefix_hash": checkpoint["prefix_hash"],
                "selected_decision_event": str(
                    checkpoint.get("selected_decision_event")
                    or f"branch:{checkpoint['prefix_hash'][:32]}"
                ),
                "bayestool_branch_shared_prefix": True,
                "bayestool_branch_shared_prefix_tokens": len(checkpoint["context_token_ids"]),
                "bayestool_branch_action_key": candidate["key"],
                "bayestool_branch_action_text": candidate["raw"],
                "bayestool_branch_q_features": q_feature_records.get(candidate["key"], {}),
                "bayestool_branch_horizon": int(getattr(bayes_config, "branch_horizon", 3)),
                "bayestool_branch_candidate_keys": [item["key"] for item in candidates],
                "bayestool_branch_sibling_group_id": branch_group_id,
                "bayes_content_signature": checkpoint.get("bayes_content_signature", ""),
                "bayes_aux_prompt": checkpoint.get("bayes_aux_prompt", ""),
                "sibling_group_id": branch_group_id,
                "runtime_state_digest": str(checkpoint.get("runtime_state_digest", "")),
                "bayes_runtime_state_digest": str(checkpoint.get("runtime_state_digest", "")),
                "world_slot_role": str((sample.metadata or {}).get("world_slot_role", "")),
                "variant_id": str((sample.metadata or {}).get("variant_id", "base")),
                "bayestool_realization_index": (sample.metadata or {}).get(
                    "bayestool_realization_index"
                ),
                "policy_version": str(
                    (sample.metadata or {}).get("policy_version", DEFAULT_POLICY_VERSION)
                ),
                "decision_group_size": target_group_size,
            }
        )
        child_sampling_params = dict(sampling_params)
        try:
            result = await generate(args, child, child_sampling_params, evaluation=evaluation)
            if isinstance(result, Sample):
                children.append(result)
            else:
                child_errors.append("nested branch child returned multiple samples")
        except Exception as exc:
            child_errors.append(str(exc))
            _BAYES_BRANCH_CHECKPOINTS.pop(resume_id, None)

    return children, {
        "branch_triggered": bool(children),
        "force_branch": bool(force_branch),
        "branch_sibling_group_id": branch_group_id,
        "prefix_hash": checkpoint["prefix_hash"],
        "parent_action_key": parent_key,
        "candidate_action_keys": [item["key"] for item in candidates],
        "q_features": q_feature_records,
        "selected_action": report.selected_action,
        "decision_regret": float(report.decision_regret),
        "branch_horizon": int(getattr(bayes_config, "branch_horizon", 3)),
        "child_count": len(children),
        "target_group_size": target_group_size,
        "runtime_state_digest": str(checkpoint.get("runtime_state_digest", "")),
        "candidate_errors": child_errors,
    }


def _find_last_final_span(text: str) -> tuple[int, int] | None:
    """Return a span only for a complete final-only assistant turn."""
    parsed = parse_assistant_action(text)
    if parsed.kind != "final":
        return None
    start = len(text) - len(text.lstrip())
    return start, len(text.rstrip())


def _maybe_json_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    looks_json = stripped[0] in "[{\"" or stripped in {"true", "false", "null"} or re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?", stripped
    )
    if looks_json:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def _parse_json_tool_call(json_text: str) -> dict[str, Any] | None:
    try:
        tool_call_data = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            tool_call_data = json.loads(json_text.replace("\n", "\\n"))
        except json.JSONDecodeError:
            return None
    if not isinstance(tool_call_data, dict):
        return None
    tool_name = tool_call_data.get("name")
    arguments = tool_call_data.get("arguments", {})
    if not tool_name:
        return None
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {"name": str(tool_name), "arguments": arguments}


def _parse_xml_tool_call(xml_body: str, tool_name: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    param_pattern = r"<parameter=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</parameter>"
    for match in re.finditer(param_pattern, xml_body, re.DOTALL):
        arguments[match.group(1)] = _maybe_json_value(match.group(2))
    return {"name": tool_name, "arguments": arguments}


def postprocess_predictions(prediction: str):
    """Convert a strict protocol result to the historical tuple API."""
    parsed = parse_assistant_action(prediction)
    if parsed.kind == "final":
        return "answer", parsed.value
    if parsed.kind == "tool_call":
        return "tool", parsed.value
    if parsed.kind == "protocol_error":
        return "protocol_error", {
            "reason": parsed.reason,
            "candidate_action_count": parsed.candidate_action_count,
        }
    return None, ""


def postprocess_responses(resp: str) -> str:
    """Keep the raw model turn so invalid suffixes remain observable.

    Trimming to the last tag would silently turn a multi-action response into
    a valid action.  The strict parser is the only component allowed to decide
    whether a turn is executable.
    """
    return resp


def _extract_prm_sign_from_text(text: str) -> int:
    if not text:
        return 0
    match = _PRM_BOXED_PATTERN.search(text)
    if not match:
        return 0
    boxed_content = match.group(1).strip()
    strict_number_match = _PRM_STRICT_NUMBER_PATTERN.fullmatch(boxed_content)
    if not strict_number_match:
        return 0
    try:
        value = float(strict_number_match.group(1))
    except ValueError:
        return 0
    # Strict PRM parsing: only exact +/-1 are valid; all other numeric values map to 0.
    if abs(value - 1.0) < 1e-9:
        return 1
    if abs(value + 1.0) < 1e-9:
        return -1
    return 0


def _extract_prm_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "response", "output", "content", "completion"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    return json.dumps(output, ensure_ascii=False)


def _get_prm_semaphore(args) -> asyncio.Semaphore:
    global _PRM_SEMAPHORE
    if _PRM_SEMAPHORE is None:
        prm_num_gpus = max(1, int(getattr(args, "prm_num_gpus", 1)))
        prm_num_gpus_per_engine = max(1, int(getattr(args, "prm_num_gpus_per_engine", 1)))
        max_engine_count = max(1, prm_num_gpus // prm_num_gpus_per_engine)
        _PRM_SEMAPHORE = asyncio.Semaphore(max(1, int(getattr(args, "sglang_server_concurrency", 512)) * max_engine_count))
    return _PRM_SEMAPHORE


def _get_prm_tokenizer(args):
    global _PRM_TOKENIZER
    if _PRM_TOKENIZER is None:
        from slime.utils.processing_utils import load_tokenizer

        prm_model_path = getattr(args, "prm_model_path", None)
        if prm_model_path:
            _PRM_TOKENIZER = load_tokenizer(prm_model_path, trust_remote_code=True)
        else:
            hf_ckpt = getattr(args, "hf_checkpoint", None)
            if hf_ckpt:
                _PRM_TOKENIZER = load_tokenizer(hf_ckpt, trust_remote_code=True)
    return _PRM_TOKENIZER


async def _query_prm_once(args, judge_prompt: str, vote_id: int) -> dict[str, Any]:
    prm_router_ip = getattr(args, "prm_router_ip", None)
    prm_router_port = getattr(args, "prm_router_port", None)
    if not prm_router_ip or not prm_router_port:
        return {"score": 0, "latency_ms": 0, "raw_text": "", "ok": False}

    prm_url = f"http://{prm_router_ip}:{prm_router_port}/generate"
    payload = {
        # Use text for PRM requests so PRM servers tokenize with their own tokenizer.
        # This avoids cross-model tokenizer-id mismatch when policy model != PRM model.
        "text": judge_prompt,
        "sampling_params": {
            "temperature": float(getattr(args, "prm_temperature", 1.0)),
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 2048)),
            "stop": None,
            "stop_token_ids": None,
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
            "sampling_seed": int(getattr(args, "rollout_seed", 42)) * 1000 + vote_id,
        },
        "return_logprob": False,
    }
    start = time.perf_counter()
    try:
        # Keep PRM retries low to avoid blocking rollout for long periods on PRM failures.
        output = await post(prm_url, payload, max_retries=2)
    except Exception as err:  # pragma: no cover - best effort external call
        logger.warning(f"PRM router request failed: {err}")
        return {"score": 0, "latency_ms": int((time.perf_counter() - start) * 1000), "raw_text": "", "ok": False}

    text = _extract_prm_text(output.get("text", output))
    return {
        "score": _extract_prm_sign_from_text(text),
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "raw_text": text,
        "ok": True,
    }


def _build_prm_step_messages(
    *,
    problem: str,
    history: str,
    action: str,
    observation: str,
    step_index: int,
) -> list[dict[str, str]]:
    system_content = (
        "You are a process reward model (PRM) for a document-understanding agent.\n"
        "Judge whether the current action uses the right document tool and parameters, retrieves relevant "
        "evidence, and advances a grounded answer to the user's question. Penalize fabricated evidence, "
        "irrelevant or repeated calls, invalid paths/pages/regions, and unsupported final answers.\n"
        "You may think first, but your final output MUST be a strict decision format.\n"
        "Valid decision is exactly one of: \\boxed{1} or \\boxed{-1}."
    )
    user_content = (
        f"Document task:\n{problem}\n\n"
        f"Step index: {step_index}\n\n"
        f"Trajectory so far:\n{history}\n\n"
        f"Current action:\n{action}\n\n"
        f"Next state / observation:\n{observation}\n\n"
        "Now evaluate the quality and evidential grounding of the current action, "
        "then output your final decision, \\boxed{1} or \\boxed{-1}\n"
        "Do NOT continue the trajectory. Your task is to judge the quality of the current action, not to continue the trajectory."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def _prm_vote(args, judge_prompt: str, m: int) -> dict[str, Any]:
    semaphore = _get_prm_semaphore(args)

    async def _single_vote(vote_id: int) -> dict[str, Any]:
        async with semaphore:
            _ = str(uuid.uuid4())  # ensure independent traces in async scheduling
            return await _query_prm_once(args, judge_prompt=judge_prompt, vote_id=vote_id)

    votes = await asyncio.gather(*[_single_vote(i) for i in range(max(1, m))])
    scores = [v["score"] for v in votes]
    valid_scores = [int(s) for s in scores if int(s) in (-1, 1)]
    return {
        "scores": scores,
        # Ignore unparsable/noisy PRM outputs (mapped to 0) when aggregating.
        # If no valid +/-1 votes exist, keep mean_score at 0.0.
        "valid_scores": valid_scores,
        "valid_vote_count": len(valid_scores),
        "mean_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0,
        "votes": votes,
    }


async def _judge_step_with_prm(
    args,
    sample: Sample,
    *,
    step_index: int,
    action: str,
    observation: str,
    history: str,
) -> dict[str, Any]:
    if not getattr(args, "prm_router_ip", None) or not getattr(args, "prm_router_port", None):
        return {"scores": [0], "mean_score": 0.0, "votes": [], "status": "disabled_no_router"}

    task_prompt = _extract_task_prompt(sample.prompt)
    messages = _build_prm_step_messages(
        problem=task_prompt,
        history=history,
        action=action,
        observation=observation,
        step_index=step_index,
    )

    tokenizer = _get_prm_tokenizer(args)
    if tokenizer is not None:
        judge_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        logger.warning("PRM tokenizer unavailable, falling back to plain text prompt")
        judge_prompt = "\n".join(msg["content"] for msg in messages)

    out = await _prm_vote(args, judge_prompt=judge_prompt, m=max(1, int(getattr(args, "prm_m", 3))))
    out["status"] = "ok"
    return out


def _format_tool_hint(tool_call_format: str) -> str:
    available = ", ".join(tool_registry.tools.keys())
    if tool_call_format == "xml":
        return (
            "If I want to call a tool, I should use the format: "
            "<tool_call>\n<function=render_page>\n"
            "<parameter=document_path>\n/path/to/file.pdf\n</parameter>\n"
            "<parameter=page_number>\n1\n</parameter>\n"
            "</function>\n</tool_call>. "
            f"Available tools: {available}. "
        )
    return (
        "If I want to call a tool, I should use the format: "
        '<tool_call>\n{"name": "render_page", '
        '"arguments": {"document_path": "/path/to/file.pdf", "page_number": 1}}\n</tool_call>. '
        f"Available tools: {available}. "
    )


def _set_terminal_status(action_log: dict[str, Any] | None, status: str) -> None:
    if action_log is not None:
        action_log["_terminal_status"] = status


def _record_action_candidate(
    action_log: dict[str, Any] | None,
    parsed: ParsedAction,
    *,
    turn: int | None,
    raw: str,
    **extra: Any,
) -> None:
    if action_log is None:
        return
    action_log["candidate_action_count"] = int(action_log.get("candidate_action_count", 0)) + parsed.candidate_action_count
    parsed_tool_name = parsed.parsed_tool_name
    if parsed_tool_name is None and isinstance(parsed.value, dict) and parsed.kind == "tool_call":
        parsed_tool_name = str(parsed.value.get("name") or "") or None
    action_log.setdefault("actions", []).append(
        {
            "turn": turn,
            "kind": parsed.kind,
            "candidate_action_count": parsed.candidate_action_count,
            # Keep the exact parser input separate from any rendered or
            # post-processed response.  This is the forensic source of truth
            # for empty and malformed turns.
            "raw_generation_text": raw,
            "raw": raw[:16000],
            "parsed_action_type": parsed.kind,
            "parsed_tool_name": parsed_tool_name,
            "parse_failure_reason": parsed.reason,
            "protocol_parse_error": parsed.reason if parsed.kind == "protocol_error" else None,
            "reason": parsed.reason,
            "accepted": None,
            "recovery_observation": False,
            "action_valid_for_policy_gradient": True,
            "action_reward": 0.0,
            **extra,
        }
    )


def _record_failure_event(
    action_log: dict[str, Any] | None,
    *,
    kind: str,
    origin: str,
    turn: int | None = None,
    tool: str | None = None,
    reason: str = "",
    penalize: bool | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    """Record one canonical failure outcome without conflating world faults."""

    if action_log is None:
        return
    if penalize is None:
        penalize = origin in {"model_action", "protocol", "budget"}
    action_log.setdefault("failure_events", []).append(
        {
            "kind": str(kind),
            "origin": str(origin),
            "turn": turn,
            "tool": tool,
            "reason": str(reason),
            "penalize": bool(penalize),
            "evidence": dict(evidence or {}),
        }
    )


def _looks_like_placeholder_path(value: str) -> bool:
    normalized = value.strip().casefold().replace("\\", "/")
    if normalized in {
        "/path/to/file.pdf",
        "/path/to/document.pdf",
        "path/to/file.pdf",
        "path/to/document.pdf",
        "<document-path>",
        "<absolute-document-path>",
        "...",
    }:
        return True
    return "/path/to/" in normalized or normalized.startswith("<") and normalized.endswith(">")


def _validate_tool_call(tool_call: dict[str, Any]) -> tuple[bool, str | None]:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return False, "missing tool name"
    if name not in tool_registry.tools:
        return False, f"unknown tool: {name}"
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"

    def visit(value: Any, key: str | None = None) -> str | None:
        if isinstance(value, str) and (key or "").endswith("path") and _looks_like_placeholder_path(value):
            return f"placeholder path is not a valid parameter: {value}"
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                reason = visit(child_value, str(child_key))
                if reason:
                    return reason
        elif isinstance(value, list):
            for child_value in value:
                reason = visit(child_value, key)
                if reason:
                    return reason
        return None

    reason = visit(arguments)
    return (reason is None), reason


def _parse_tool_result(result: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return not str(result).lstrip().casefold().startswith("error"), None, None
    if not isinstance(payload, dict):
        return False, None, "tool result must be an object"
    status = str(payload.get("status", "")).casefold()
    execution_succeeded = payload.get("execution_succeeded")
    observation_delivered = payload.get("observation_delivered")
    if execution_succeeded is True and observation_delivered is True and not status:
        status = "ok"
    # ``partial`` is a recoverable tool result, not a silent success: for
    # example OCR may be unavailable while the cropped image is still valid
    # vision input.  Continue the model loop and preserve the error payload.
    return status in {"ok", "partial", "partial_success", "success"}, payload, status or None


def _image_paths_from_result(payload: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not isinstance(payload, dict):
        return [], []
    candidates: list[str] = []
    for key in ("image_path", "image_paths"):
        value = payload.get(key)
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, str))
    valid = [path for path in candidates if Path(path).is_file()]
    return valid, candidates


def _update_pending_visual_input(
    navigation_state: dict[str, Any] | None,
    tool_name: str,
    image_paths: list[str],
    payload: dict[str, Any] | None,
    arguments: dict[str, Any] | None = None,
) -> None:
    """Track whether the next decision needs pixels, independently of OCR."""
    if navigation_state is None:
        return
    payload = payload if isinstance(payload, dict) else {}
    arguments = arguments if isinstance(arguments, dict) else {}
    page_number = arguments.get("page_number", payload.get("page_number"))
    try:
        page_number = int(page_number) if page_number is not None else None
    except (TypeError, ValueError):
        page_number = None
    if tool_name == "render_page" and image_paths:
        # A render is only a required visual input when the next decision
        # needs page pixels.  If a prior observation supplied a page/table
        # bbox, the controller can go directly to crop/OCR/layout without
        # making the model consume the full-page image.
        question_type = str(navigation_state.get("question_type") or "")
        region_known = _visual_region_is_known(navigation_state, page_number)
        if question_type == "visual" and not region_known:
            navigation_state["pending_visual_image_paths"] = list(image_paths)
            navigation_state["visual_input_required"] = True
            navigation_state["visual_input_reason"] = "select relevant region or read page pixels"
        else:
            navigation_state["pending_visual_image_paths"] = []
            navigation_state["visual_input_required"] = False
            navigation_state["visual_input_reason"] = (
                "bbox supplied by prior tool metadata" if region_known else None
            )
    elif tool_name in {"crop_region", "zoom_region"} and image_paths:
        navigation_state["pending_visual_image_paths"] = list(image_paths)
        navigation_state["visual_input_required"] = True
        navigation_state["visual_input_reason"] = "continue inspecting cropped pixels"
    elif tool_name in {"ocr_region", "detect_layout"}:
        # OCR/layout are valid text/geometry observations and do not require
        # the next model request to consume the full-page image.
        if tool_name == "ocr_region" and str(payload.get("text") or payload.get("raw_text") or "").strip():
            navigation_state["visual_input_consumed"] = True
        navigation_state["pending_visual_image_paths"] = []
        navigation_state["visual_input_required"] = False
        navigation_state["visual_input_reason"] = None


def _limit_tool_result(result: str, max_chars: int) -> str:
    """Limit observations without cutting a page JSON object in half."""
    if max_chars <= 0 or len(result) <= max_chars:
        return result
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return result[:max_chars] + f"\n... [truncated {len(result) - max_chars} chars]"
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        return result[:max_chars] + f"\n... [truncated {len(result) - max_chars} chars]"

    original_pages = payload["pages"]
    kept: list[Any] = []
    for page in original_pages:
        candidate_pages = kept + [page]
        candidate = dict(payload)
        candidate["pages"] = candidate_pages
        candidate["returned_pages"] = [item.get("page_number") for item in candidate_pages if isinstance(item, dict)]
        candidate["content_truncated"] = bool(payload.get("content_truncated")) or len(candidate_pages) < len(original_pages)
        candidate["truncated"] = candidate["content_truncated"]
        candidate["has_more_pages"] = len(candidate_pages) < int(candidate.get("page_count", len(original_pages)))
        candidate["document_has_unreturned_pages"] = candidate["has_more_pages"]
        encoded = json.dumps(candidate, ensure_ascii=False, indent=2, default=str)
        if len(encoded) > max_chars and kept:
            break
        if len(encoded) > max_chars:
            page_copy = dict(page) if isinstance(page, dict) else {"markdown": str(page)}
            page_copy["markdown"] = str(page_copy.get("markdown", ""))[: max(1, max_chars // 2)]
            page_copy["markdown"] += "\n... [content truncated]"
            kept = [page_copy]
            break
        kept = candidate_pages
    payload["pages"] = kept
    payload["returned_pages"] = [item.get("page_number") for item in kept if isinstance(item, dict)]
    payload.pop("markdown", None)
    payload["content_truncated"] = len(kept) < len(original_pages) or bool(payload.get("content_truncated"))
    payload["truncated"] = payload["content_truncated"]
    payload["has_more_pages"] = len(kept) < int(payload.get("page_count", len(original_pages)))
    payload["document_has_unreturned_pages"] = payload["has_more_pages"]
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


_TABLE_QUERY_TERMS = (
    "row", "column", "subject", "week", "age", "weight", "income",
    "amount", "total", "table", "group", "cholesterol", "investment",
    "time", "point", "year", "rank", "ranking", "percent", "percentage",
    "rate", "score", "continuous", "numeric", "number", "values",
)
_VISUAL_QUERY_TERMS = (
    "top", "bottom", "left", "right", "first from top", "placed at the bottom",
    "shown in the image", "image", "logo", "brand image", "picture", "visual",
)
_ABSENCE_ANSWER_RE = re.compile(
    r"\b(?:none|not found|cannot be identified|can't be identified|"
    r"unable to (?:determine|identify|find)|not mentioned|not available|"
    r"no explicit mention|does not contain|do not contain|no information|"
    r"unknown|n/?a)\b",
    re.IGNORECASE,
)

# A candidate is only usable as evidence when the extractor explicitly bound
# the observation to the question constraints.  Keep this threshold in one
# place so the online stop guard, reward calculation, and offline audit cannot
# drift apart.
_EVIDENCE_RELATION_THRESHOLD = 0.75


def _extract_question(task_prompt: str) -> str:
    matches = re.findall(r"(?:^|\n)\s*(?:question|问题)\s*:\s*(.+)", task_prompt, re.IGNORECASE)
    return matches[-1].strip() if matches else task_prompt.strip()


def _extract_document_path(task_prompt: str) -> str | None:
    match = re.search(r"(?:^|\n)\s*document\s+path\s*:\s*(\S+)", task_prompt, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _classify_question(question: str) -> str:
    lowered = question.casefold()
    if any(term in lowered for term in _VISUAL_QUERY_TERMS):
        return "visual"
    if any(term in lowered for term in _TABLE_QUERY_TERMS):
        return "table"
    return "text"


_QUESTION_STOPWORDS = {
    "what", "who", "whom", "which", "where", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "the", "a", "an",
    "of", "to", "for", "from", "in", "on", "at", "by", "with", "and",
    "or", "this", "that", "these", "those", "document", "page", "pages",
    "shown", "image", "picture", "visual", "please", "tell", "me",
}
_GENERIC_EVIDENCE_TERMS = {
    "amount", "brand", "company", "date", "document", "group", "information",
    "item", "name", "number", "page", "request", "status", "subject", "total", "value",
}
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
}


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _question_terms(question: str) -> list[str]:
    normalized = _normalized_evidence(question)
    terms = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", normalized)
    ignored = set(_QUESTION_STOPWORDS)
    ignored.update(term.casefold() for term in _TABLE_QUERY_TERMS)
    ignored.update(term.casefold() for term in _VISUAL_QUERY_TERMS)
    result: list[str] = []
    for term in terms:
        if term in ignored or (len(term) == 1 and not term.isdigit()):
            continue
        if term not in result:
            result.append(term)
    return result


def _new_navigation_state(
    task_prompt: str,
) -> dict[str, Any]:
    question = _extract_question(task_prompt)
    return {
        "document_path": _extract_document_path(task_prompt),
        "question": question,
        "question_type": _classify_question(question),
        "question_terms": _question_terms(question),
        "known_visual_region": bool(
            re.search(r"\b(?:bbox|bounding\s+box|known\s+region|predefined\s+region)\b", task_prompt, re.IGNORECASE)
        ),
        "page_count": None,
        # ``visited_pages`` is the union of pages touched by any tool.  The
        # more precise lists below drive text-page search and visual checks
        # independently.
        "visited_pages": [],
        "parsed_pages": [],
        "rendered_pages": [],
        "cropped_pages": [],
        "ocr_pages": [],
        "cropped_regions": [],
        "zoomed_regions": [],
        "ocr_regions": [],
        "unvisited_pages": [],
        "evidence_by_page": {},
        "page_metadata_by_page": {},
        "table_candidate_pages": [],
        "table_evidence_pages": [],
        "visual_evidence_pages": [],
        "visual_evidence_by_page": {},
        "supporting_regions": [],
        # Ground-truth page/bbox metadata is deliberately absent from this
        # runtime object.  It is added to the exported diagnostics only after
        # the model has finished choosing actions.
        "evidence_candidates": [],
        "evidence_reason": "no tool observation yet",
        "evidence_hits": [],
        "evidence_sufficient": False,
        "current_evidence_sufficient": False,
        "visual_evidence_sufficient": False,
        "visual_input_required": False,
        "visual_input_reason": None,
        "pending_visual_image_paths": [],
        "visual_input_consumed": False,
        "search_budget_exhausted": False,
        "supporting_pages": [],
        "final_supporting_pages": [],
        "final_supporting_regions": [],
        "prediction_found_in_document": False,
        "prediction_relation_matched": False,
        "final_supported_by_evidence": False,
        "stop_reason": None,
        "premature_final": False,
        "had_evidence_guard_recovery": False,
        "rejected_final_count": 0,
        "duplicate_page_calls": 0,
        "duplicate_region_calls": 0,
        "unnecessary_tool_calls": 0,
        "no_information_gain_calls": 0,
        "appropriate_tool_calls": 0,
        "fallback_used": False,
        "fallback_events": [],
        "image_page_by_path": {},
    }


def _requested_pages(arguments: dict[str, Any]) -> list[int]:
    values = arguments.get("page_numbers")
    if isinstance(values, list):
        pages = values
    elif arguments.get("page_number") is not None:
        pages = [arguments.get("page_number")]
    else:
        return []
    result: list[int] = []
    for value in pages:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page >= 1 and page not in result:
            result.append(page)
    return result


def _refresh_unvisited_pages(navigation_state: dict[str, Any]) -> None:
    _recompute_visited_pages(navigation_state)
    page_count = navigation_state.get("page_count")
    if isinstance(page_count, int) and page_count > 0:
        visited = {int(page) for page in navigation_state.get("visited_pages", [])}
        navigation_state["unvisited_pages"] = [
            page for page in range(1, page_count + 1) if page not in visited
        ]
    else:
        navigation_state["unvisited_pages"] = list(navigation_state.get("unvisited_pages", []))


def _add_visited_pages(navigation_state: dict[str, Any], pages: list[int]) -> None:
    # Keep the historical helper for callers, but make the source of truth the
    # per-operation page sets.  This prevents a stale hand-maintained
    # ``visited_pages`` list from disagreeing with parsed/rendered/cropped/OCR
    # diagnostics.
    navigation_state.setdefault("_pending_visited_pages", []).extend(pages)
    _recompute_visited_pages(navigation_state)
    _refresh_unvisited_pages(navigation_state)


def _recompute_visited_pages(navigation_state: dict[str, Any]) -> None:
    page_sets = []
    for field in ("parsed_pages", "rendered_pages", "cropped_pages", "ocr_pages"):
        values = navigation_state.get(field, [])
        if isinstance(values, list):
            page_sets.append(values)
    # Compatibility with callers that invoke _add_visited_pages before the
    # operation-specific field has been populated.
    pending = navigation_state.pop("_pending_visited_pages", [])
    pages = {
        int(page)
        for values in page_sets + [pending]
        for page in values
        if str(page).isdigit() and int(page) >= 1
    }
    navigation_state["visited_pages"] = sorted(pages)


def _add_page_state(navigation_state: dict[str, Any], field: str, pages: list[int]) -> None:
    values = navigation_state.setdefault(field, [])
    if not isinstance(values, list):
        values = []
        navigation_state[field] = values
    for page in pages:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number >= 1:
            _append_unique(values, page_number)
    values.sort(key=lambda value: int(value) if str(value).isdigit() else 10**9)


def _region_key(page: Any, payload: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    region: dict[str, Any] = {}
    try:
        if page is not None:
            region["page_number"] = int(page)
    except (TypeError, ValueError):
        pass
    bbox = payload.get("bbox") or payload.get("bbox_pixels") or arguments.get("bbox")
    if isinstance(bbox, (list, tuple)):
        region["bbox"] = list(bbox)
    image_path = payload.get("image_path")
    if isinstance(image_path, str):
        region["image_path"] = image_path
    return region


def _remember_evidence(
    navigation_state: dict[str, Any],
    page: Any,
    evidence: Any,
    *,
    source: str | None = None,
) -> None:
    if evidence is None:
        return
    text = str(evidence).strip()
    if not text:
        return
    page_key = str(int(page)) if page is not None and str(page).isdigit() else "tool"
    evidence_by_page = navigation_state.setdefault("evidence_by_page", {})
    previous = str(evidence_by_page.get(page_key) or "").strip()
    combined = text if not previous else f"{previous}\n{text}"
    evidence_by_page[page_key] = combined[-4800:]
    if source:
        navigation_state.setdefault("evidence_hits", []).append(
            {"page_number": int(page) if page is not None and str(page).isdigit() else None, "source": source}
        )


def _same_local_window(text: str, terms: list[str], window: int = 180) -> bool:
    if not terms:
        return False
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    matches = []
    for term in terms:
        match = re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", normalized)
        if match is None:
            return False
        matches.append(match)
    span_left = min(match.start() for match in matches)
    span_right = max(match.end() for match in matches)
    if span_right - span_left > window:
        return False
    left = max(0, span_left - window)
    right = min(len(normalized), span_right + window)
    local = normalized[left:right]
    has_field_separator = bool(re.search(r"[:=|]", local))
    # A bare copula is not enough: phrases such as ``information is listed
    # elsewhere`` contain the question term and ``is`` but do not expose an
    # answer.  Require a value-like token after the copula and explicitly
    # reject navigation/absence phrases.
    has_copula_value = bool(
        re.search(
            r"\b(?:is|are|was|were)\s+"
            r"(?!listed\b|shown\b|mentioned\b|available\b|elsewhere\b|"
            r"not\b|unknown\b|unclear\b)\S+",
            local,
        )
    )
    has_relation = bool(
        re.search(
            r"\b(?:called|named|known\s+as|located|approved|reported|"
            r"measured|amounts?\s+to|total(?:s|ed)?\s+to)\b",
            local,
        )
    ) or has_copula_value
    has_value = bool(
        re.search(r"\b(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\b", local)
    )
    # A table separator or an explicit relation is required.  A nearby
    # number alone is not evidence for generic terms such as ``company`` or
    # ``name``; that used to make unrelated page text look answer-bearing.
    return has_field_separator or has_relation or (has_value and all(term not in _GENERIC_EVIDENCE_TERMS for term in terms))


def _text_evidence_supports_question(question: str, evidence: Any) -> bool:
    text = str(evidence or "").strip()
    if not text:
        return False
    normalized = _normalized_evidence(text)
    terms = _question_terms(question)
    if not terms:
        return False
    present = [term for term in terms if re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", normalized)]
    if not present:
        return False
    if len(terms) == 1 and present:
        term = present[0]
        # Single-term generic questions need an explicit field label.  This
        # prevents an arbitrary occurrence of ``company`` or ``number`` from
        # stopping a page search early, while preserving ``SUPPLIER: BURKE``
        # and ``PROPOSAL#: 14-3006-14``.
        explicit_field = re.search(
            rf"(?<![\w])(?:page\s+)?{re.escape(term)}\s*(?:#\s*)?[:=]\s*\S+",
            normalized,
        )
        if explicit_field:
            return True
        if term in _GENERIC_EVIDENCE_TERMS:
            return False
        return term not in _GENERIC_EVIDENCE_TERMS and _same_local_window(text, [term], window=240)
    # Strong field/value evidence: ``supplier: BURKE``, a markdown table row,
    # or a clear relation between all question entities and a nearby value.
    if len(present) == len(terms) and _same_local_window(text, present):
        return True
    return False


def _question_requires_visual_region(navigation_state: dict[str, Any]) -> bool:
    question = str(navigation_state.get("question") or "").casefold()
    return any(
        term in question
        for term in (
            "top", "bottom", "left", "right", "first from top", "placed at the bottom",
            "logo", "brand image", "brand", "position", "located at",
        )
    )


def _visual_region_is_known(navigation_state: dict[str, Any], page_number: int | None = None) -> bool:
    if bool(navigation_state.get("known_visual_region")):
        return True
    visual_by_page = navigation_state.get("visual_evidence_by_page", {})
    if isinstance(visual_by_page, dict):
        items = (
            [visual_by_page.get(str(page_number))]
            if page_number is not None
            else list(visual_by_page.values())
        )
        for item in items:
            if isinstance(item, dict) and item.get("image_regions"):
                return True
    page_metadata = navigation_state.get("page_metadata_by_page", {})
    if isinstance(page_metadata, dict):
        items = (
            [page_metadata.get(str(page_number))]
            if page_number is not None
            else list(page_metadata.values())
        )
        for item in items:
            if isinstance(item, dict) and (item.get("image_regions") or item.get("table_regions")):
                return True
    # Do not let a crop/OCR on page 1 suppress the full-page visual input for
    # a later page.  Region metadata is page-scoped whenever the tool has
    # provided it; entries without a page number are treated as global legacy
    # metadata for backward compatibility.
    for key in ("cropped_regions", "zoomed_regions", "ocr_regions"):
        regions = navigation_state.get(key, [])
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                return True
            region_page = region.get("page_number", region.get("page"))
            if page_number is None or region_page is None:
                return True
            try:
                if int(region_page) == int(page_number):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _candidate_relation_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("relation_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bayestool_dvoi_score(value: Any) -> float:
    """Return a scalar DVOI score for branch ranking and telemetry.

    ``DecisionReport.dvoi`` is intentionally a mapping from candidate action
    keys to scores.  Branch selection needs one checkpoint-level score, so use
    the strongest finite candidate value while retaining the original mapping
    in the decision metadata.  Older callers may still provide a scalar.
    """
    if isinstance(value, Mapping):
        scores: list[float] = []
        for candidate in value.values():
            try:
                score = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scores.append(score)
        return max(scores, default=0.0)
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _valid_evidence_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Return only candidates that satisfy the question-bound relation rule."""
    if not isinstance(candidates, (list, tuple)):
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and bool(candidate.get("satisfies_question_constraints"))
        and _candidate_relation_score(candidate) >= _EVIDENCE_RELATION_THRESHOLD
    ]


def _evaluate_evidence_sufficiency(navigation_state: dict[str, Any]) -> None:
    """Update evidence state from structured, question-bound observations.

    A page containing a keyword, a random number, or an unrelated table is not
    sufficient evidence.  The candidates below retain the relation that made
    an observation useful so the final guard and reward code can make the same
    decision without relying on a page-level keyword hit.
    """
    question_type = str(navigation_state.get("question_type") or "text")
    candidates = [item for item in navigation_state.get("evidence_candidates", []) if isinstance(item, dict)]
    constrained = _valid_evidence_candidates(candidates)
    supporting = sorted({int(item["page"]) for item in constrained if str(item.get("page")).isdigit()})
    visual_pages = {
        int(page) for page in navigation_state.get("visual_evidence_pages", []) if str(page).isdigit()
    }
    visual_region_seen = bool(
        navigation_state.get("cropped_regions")
        or navigation_state.get("zoomed_regions")
        or navigation_state.get("ocr_regions")
    )
    # Pixel consumption is tracked separately from answer evidence.  Merely
    # rendering a page (or seeing an image) is not a structured question/value
    # relation and must not make the stop guard report sufficient evidence.
    sufficient = bool(constrained)

    navigation_state["supporting_pages"] = supporting
    navigation_state["evidence_sufficient"] = bool(sufficient)
    navigation_state["current_evidence_sufficient"] = bool(sufficient)
    navigation_state["visual_evidence_sufficient"] = bool(
        visual_pages and (not _question_requires_visual_region(navigation_state) or visual_region_seen or navigation_state.get("visual_input_consumed"))
    )
    if sufficient:
        assert constrained, "evidence_sufficient requires a valid question-bound candidate"
    if sufficient:
        navigation_state["evidence_reason"] = "question-bound supporting evidence found"
        navigation_state["stop_reason"] = "sufficient_evidence"
    elif question_type == "table" and navigation_state.get("table_evidence_pages"):
        navigation_state["evidence_reason"] = "missing requested row and column relation"
        navigation_state["stop_reason"] = "evidence_insufficient"
    elif question_type == "visual" and visual_pages:
        navigation_state["evidence_reason"] = "visual input or supporting region has not yet been consumed"
        navigation_state["stop_reason"] = "evidence_insufficient"
    elif navigation_state.get("search_budget_exhausted"):
        navigation_state["evidence_reason"] = "search budget exhausted without question-bound evidence"
        navigation_state["stop_reason"] = "search_budget_exhausted"
    else:
        navigation_state["evidence_reason"] = "no question-bound supporting evidence"
        navigation_state["stop_reason"] = "evidence_insufficient"


def _update_navigation_state(
    navigation_state: dict[str, Any] | None,
    tool_name: str,
    arguments: dict[str, Any],
    payload: dict[str, Any] | None,
    success: bool,
) -> None:
    if navigation_state is None or not success:
        return
    before_progress = (
        tuple(sorted(str(page) for page in navigation_state.get("parsed_pages", []))),
        tuple(sorted(str(page) for page in navigation_state.get("rendered_pages", []))),
        len(navigation_state.get("cropped_regions", [])),
        len(navigation_state.get("zoomed_regions", [])),
        len(navigation_state.get("ocr_regions", [])),
        tuple(sorted(str(page) for page in navigation_state.get("table_evidence_pages", []))),
        tuple(sorted(str(page) for page in navigation_state.get("visual_evidence_pages", []))),
        tuple(sorted((str(key), str(value)) for key, value in navigation_state.get("evidence_by_page", {}).items())),
    )
    evidence_was_sufficient = bool(navigation_state.get("evidence_sufficient"))
    payload = payload if isinstance(payload, dict) else {}
    # Every document tool can discover the document length.  In particular,
    # visual-first trajectories often start with render_page, so restricting
    # this update to parse_document leaves the frontier as ``unknown`` and
    # can make a model stop after inspecting only page 1.
    page_count = payload.get("page_count")
    try:
        if page_count is not None:
            navigation_state["page_count"] = int(page_count)
    except (TypeError, ValueError):
        pass
    requested = _requested_pages(arguments)
    if tool_name == "parse_document":
        returned = payload.get("returned_pages")
        pages = [int(page) for page in returned if str(page).isdigit()] if isinstance(returned, list) else []
        pages = pages or requested
        _add_page_state(navigation_state, "parsed_pages", pages)
        _add_visited_pages(navigation_state, pages)
        if navigation_state.get("question_type") == "text":
            navigation_state["appropriate_tool_calls"] = int(navigation_state.get("appropriate_tool_calls", 0)) + 1
        page_records = payload.get("pages", [])
        if isinstance(page_records, list):
            evidence_by_page = navigation_state.setdefault("evidence_by_page", {})
            page_metadata = navigation_state.setdefault("page_metadata_by_page", {})
            for record in page_records:
                if not isinstance(record, dict):
                    continue
                page_number = record.get("page_number")
                markdown = str(record.get("markdown") or "").strip()
                if page_number is None or not str(page_number).isdigit():
                    continue
                page_number = int(page_number)
                page_metadata[str(page_number)] = {
                    key: record.get(key)
                    for key in (
                        "has_tables", "table_count", "table_extraction_recommended",
                        "table_regions", "has_images", "visual_content_omitted", "image_regions",
                    )
                    if key in record
                }
                if record.get("has_tables") or record.get("table_count"):
                    _add_page_state(navigation_state, "table_candidate_pages", [page_number])
                if record.get("has_images") or record.get("visual_content_omitted"):
                    navigation_state.setdefault("visual_evidence_by_page", {}).setdefault(str(page_number), {})
                    navigation_state["visual_evidence_by_page"][str(page_number)].update(
                        {key: record.get(key) for key in ("image_regions", "has_images", "visual_content_omitted") if key in record}
                    )
                if markdown:
                    evidence_by_page[str(page_number)] = markdown[:4800]
                    _text_evidence_candidate(navigation_state, page_number, markdown)
                    table_rows = _markdown_table_rows(markdown)
                    if not table_rows and navigation_state.get("question_type") == "table":
                        table_rows = _flattened_table_rows(
                            markdown,
                            str(navigation_state.get("question") or ""),
                        )
                    if navigation_state.get("question_type") == "table" and _looks_table_like_text(
                        markdown,
                        str(navigation_state.get("question") or ""),
                    ):
                        _add_page_state(navigation_state, "table_candidate_pages", [page_number])
                    if table_rows:
                        _table_evidence_candidates(
                            navigation_state,
                            page_number,
                            table_rows,
                            supporting_region=(record.get("table_regions") or [None])[0],
                        )
        if payload.get("document_has_unreturned_pages") is False or payload.get("has_more_pages") is False:
            # Recompute from the actual returned page set.  A page-scoped call
            # can truthfully report more document pages even when it returned
            # a single complete page.
            if isinstance(navigation_state.get("page_count"), int) and len(navigation_state.get("parsed_pages", [])) >= int(navigation_state["page_count"]):
                navigation_state["unvisited_pages"] = []
        _refresh_unvisited_pages(navigation_state)
    else:
        page = arguments.get("page_number")
        if page is None:
            page = payload.get("page_number")
        if page is None:
            image_path = arguments.get("image_path")
            if isinstance(image_path, str):
                page = navigation_state.get("image_page_by_path", {}).get(image_path)
        if page is not None:
            try:
                _add_visited_pages(navigation_state, [int(page)])
            except (TypeError, ValueError):
                pass
        evidence = payload.get("markdown") or payload.get("raw_text") or payload.get("text")
        if evidence is None and payload.get("rows") is not None:
            evidence = json.dumps(payload.get("rows"), ensure_ascii=False)
        if evidence:
            _remember_evidence(navigation_state, page, evidence, source=tool_name)
        page_number: int | None
        try:
            page_number = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_number = None
        if tool_name == "render_page" and page_number is not None:
            if page_number in {int(value) for value in navigation_state.get("rendered_pages", [])}:
                navigation_state["duplicate_region_calls"] = int(navigation_state.get("duplicate_region_calls", 0)) + 1
            _add_page_state(navigation_state, "rendered_pages", [page_number])
            navigation_state["appropriate_tool_calls"] = int(navigation_state.get("appropriate_tool_calls", 0)) + 1
            if navigation_state.get("question_type") == "visual":
                _add_page_state(navigation_state, "visual_evidence_pages", [page_number])
        elif tool_name in {"crop_region", "zoom_region"} and page_number is not None:
            key = _region_key(page_number, payload, arguments)
            _append_unique(navigation_state.setdefault("cropped_regions" if tool_name == "crop_region" else "zoomed_regions", []), key)
            _add_page_state(navigation_state, "cropped_pages", [page_number])
            _add_page_state(navigation_state, "visual_evidence_pages", [page_number])
            navigation_state.setdefault("supporting_regions", []).append(key)
            navigation_state["appropriate_tool_calls"] = int(navigation_state.get("appropriate_tool_calls", 0)) + 1
        elif tool_name == "ocr_region" and page_number is not None:
            key = _region_key(page_number, payload, arguments)
            _append_unique(navigation_state.setdefault("ocr_regions", []), key)
            _add_page_state(navigation_state, "ocr_pages", [page_number])
            if str(payload.get("text") or payload.get("raw_text") or "").strip():
                _add_page_state(navigation_state, "visual_evidence_pages", [page_number])
                navigation_state.setdefault("supporting_regions", []).append(key)
            navigation_state["appropriate_tool_calls"] = int(navigation_state.get("appropriate_tool_calls", 0)) + 1
        elif tool_name == "extract_table" and page_number is not None:
            _add_page_state(navigation_state, "table_evidence_pages", [page_number])
            rows = payload.get("rows")
            if isinstance(rows, list):
                _table_evidence_candidates(
                    navigation_state,
                    page_number,
                    rows,
                    supporting_region=payload.get("bbox") or payload.get("bbox_pixels"),
                )
            navigation_state["appropriate_tool_calls"] = int(navigation_state.get("appropriate_tool_calls", 0)) + 1
        image_paths = []
        for key in ("image_path", "image_paths", "source_image_path"):
            value = payload.get(key)
            if isinstance(value, str):
                image_paths.append(value)
            elif isinstance(value, list):
                image_paths.extend(item for item in value if isinstance(item, str))
        if page is not None:
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                page_number = None
            if page_number is not None:
                mapping = navigation_state.setdefault("image_page_by_path", {})
                for image_path in image_paths:
                    mapping[str(image_path)] = page_number
        if page_number is not None and evidence:
            evidence_text = str(payload.get("text") or payload.get("raw_text") or evidence)
            _text_evidence_candidate(
                navigation_state,
                page_number,
                evidence_text,
                source_type="ocr" if tool_name == "ocr_region" else tool_name,
            )
    _refresh_unvisited_pages(navigation_state)
    _evaluate_evidence_sufficiency(navigation_state)
    if evidence_was_sufficient and tool_name in {"parse_document", "render_page", "crop_region", "zoom_region", "ocr_region", "extract_table"}:
        navigation_state["unnecessary_tool_calls"] = int(navigation_state.get("unnecessary_tool_calls", 0)) + 1
    after_progress = (
        tuple(sorted(str(page) for page in navigation_state.get("parsed_pages", []))),
        tuple(sorted(str(page) for page in navigation_state.get("rendered_pages", []))),
        len(navigation_state.get("cropped_regions", [])),
        len(navigation_state.get("zoomed_regions", [])),
        len(navigation_state.get("ocr_regions", [])),
        tuple(sorted(str(page) for page in navigation_state.get("table_evidence_pages", []))),
        tuple(sorted(str(page) for page in navigation_state.get("visual_evidence_pages", []))),
        tuple(sorted((str(key), str(value)) for key, value in navigation_state.get("evidence_by_page", {}).items())),
    )
    if after_progress == before_progress:
        navigation_state["no_information_gain_calls"] = int(navigation_state.get("no_information_gain_calls", 0)) + 1


def _navigation_status_text(navigation_state: dict[str, Any]) -> str:
    page_count = navigation_state.get("page_count")
    page_count_text = str(page_count) if page_count is not None else "unknown"
    visited = sorted({int(page) for page in navigation_state.get("visited_pages", [])})
    unvisited = sorted({int(page) for page in navigation_state.get("unvisited_pages", [])})
    lines = [
        _OBSERVATION_PROTOCOL_RULE,
        "Document navigation state:",
        f"Document page count: {page_count_text}",
        f"Visited pages: {visited}",
        f"Unvisited pages: {unvisited}",
        f"Parsed pages: {sorted({int(page) for page in navigation_state.get('parsed_pages', [])})}",
        f"Rendered pages: {sorted({int(page) for page in navigation_state.get('rendered_pages', [])})}",
        f"Cropped pages: {sorted({int(page) for page in navigation_state.get('cropped_pages', [])})}",
        f"OCR pages: {sorted({int(page) for page in navigation_state.get('ocr_pages', [])})}",
        f"Cropped regions: {len(navigation_state.get('cropped_regions', []))}",
        f"Zoomed regions: {len(navigation_state.get('zoomed_regions', []))}",
        f"OCR regions: {len(navigation_state.get('ocr_regions', []))}",
        f"Current evidence sufficient: {bool(navigation_state.get('evidence_sufficient', navigation_state.get('current_evidence_sufficient')))}",
        f"Evidence reason: {navigation_state.get('evidence_reason')}",
        f"Visual evidence sufficient: {bool(navigation_state.get('visual_evidence_sufficient'))}",
        f"Visual input required: {bool(navigation_state.get('visual_input_required'))}",
        f"Supporting pages: {sorted({int(page) for page in navigation_state.get('supporting_pages', [])})}",
    ]
    question_type = navigation_state.get("question_type")
    if question_type == "table":
        lines.append(
            "Question route: table-like. The page may contain a flattened or scanned table. "
            "After locating a relevant page, use extract_table; if row-column relations remain "
            "unclear, use render_page, detect_layout, crop_region, or ocr_region. Prefer deep "
            "inspection of the relevant page before scanning later pages."
        )
    elif question_type == "visual":
        lines.append(
            "Question route: visual/layout-like. If visual_content_omitted is true, "
            "use render_page, then crop_region/zoom_region or ocr_region. For top/bottom/left/right "
            "questions, a full-page render alone is not sufficient."
        )
    if navigation_state.get("search_budget_exhausted"):
        lines.append(
            "Search budget exhausted. Do not call another tool; produce the best "
            "grounded <final>...</final> answer now."
        )
    elif visited and not unvisited:
        lines.append(
            "All known document pages have been checked. Do not call another tool or "
            "write a document summary; answer concisely and end with <final>...</final>."
        )
    if navigation_state.get("evidence_sufficient"):
        lines.append(
            "Sufficient supporting evidence has been found. You may stop searching and emit the concise "
            "<final>...</final> answer; do not scan additional pages unless a table/visual follow-up is required."
        )
    elif question_type == "table" and navigation_state.get("table_candidate_pages"):
        lines.append(
            "A table-like page has been located. Prefer extract_table or a visual/layout fallback "
            "on that page before reading additional unvisited pages."
        )
    evidence_by_page = navigation_state.get("evidence_by_page", {})
    if isinstance(evidence_by_page, dict):
        for page in sorted(evidence_by_page, key=lambda value: int(value) if str(value).isdigit() else 10**9)[:8]:
            snippet = re.sub(r"\s+", " ", str(evidence_by_page[page])).strip()[:320]
            if snippet:
                lines.append(f"Evidence summary page {page}: {snippet}")
    if unvisited:
        lines.append(f"Next recommended page: {unvisited[0]}. Do not repeat already visited pages.")
        lines.append(
            "If the current evidence is insufficient, continue to an unvisited page "
            "instead of answering that the value is absent."
        )
    elif navigation_state.get("question_type") == "visual" and not navigation_state.get("visual_evidence_sufficient"):
        lines.append("All text pages may be checked, but visual evidence is still insufficient; inspect the rendered region before answering.")
    if navigation_state.get("fallback_used"):
        lines.append("A specialized tool failed; follow the fallback route in the latest observation.")
    return "\n".join(lines)


def _tool_arguments_for_navigation(
    tool_name: str,
    arguments: dict[str, Any],
    navigation_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if navigation_state is None or tool_name != "parse_document":
        return arguments, False
    updated = dict(arguments)
    requested = _requested_pages(updated)
    visited = {int(page) for page in navigation_state.get("parsed_pages", [])}
    unvisited = [int(page) for page in navigation_state.get("unvisited_pages", [])]
    if requested and unvisited:
        fresh_pages = [page for page in requested if page not in visited]
        if fresh_pages and len(fresh_pages) != len(requested):
            navigation_state["duplicate_page_calls"] = int(navigation_state.get("duplicate_page_calls", 0)) + (len(requested) - len(fresh_pages))
            updated["page_numbers"] = fresh_pages
            updated.pop("page_number", None)
            return updated, True
        if not fresh_pages:
            navigation_state["duplicate_page_calls"] = int(navigation_state.get("duplicate_page_calls", 0)) + 1
            updated["page_numbers"] = [unvisited[0]]
            updated.pop("page_number", None)
            return updated, True
    if not requested and unvisited:
        updated["page_numbers"] = [unvisited[0]]
        return updated, True
    return updated, False


def _normalized_evidence(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^\w\u3400-\u9fff.+/%-]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _term_variants(term: str) -> set[str]:
    """Return conservative lexical variants used for evidence binding."""
    normalized = _normalized_evidence(term)
    if not normalized:
        return set()
    variants = {normalized}
    ordinal = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", normalized)
    if ordinal:
        variants.add(ordinal.group(1))
    if normalized.startswith("#"):
        variants.add(normalized.lstrip("#"))
    if normalized.isdigit():
        variants.add(f"#{normalized}")
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
    }
    if normalized in number_words:
        variants.add(number_words[normalized])
    return variants


def _question_constraint_terms(question: str) -> list[str]:
    """Terms that must be bound by one local/table/visual evidence object."""
    normalized = _normalized_evidence(question)
    terms = re.findall(r"[a-z0-9#]+|[\u3400-\u9fff]", normalized)
    result: list[str] = []
    structural = {"table", "row", "column", "document", "page", "pages"}
    for term in terms:
        if term in _QUESTION_STOPWORDS or term in structural or (len(term) == 1 and not term.isdigit()):
            continue
        if term not in result:
            result.append(term)
    return result


def _terms_in_context(terms: list[str], context: Any) -> bool:
    normalized = _normalized_evidence(context)
    return bool(terms) and all(
        any(
            re.search(rf"(?<![\w]){re.escape(variant)}(?![\w])", normalized)
            for variant in _term_variants(term)
        )
        for term in terms
    )


def _candidate_value_matches(candidate: dict[str, Any], prediction: str) -> bool:
    value = candidate.get("value") if isinstance(candidate, dict) else None
    predicted = _normalized_evidence(prediction)
    observed = _normalized_evidence(value)
    if not predicted or not observed:
        return False
    if predicted == observed or predicted in observed or observed in predicted:
        return True
    observed_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", observed.replace(",", ""))
    predicted_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", predicted.replace(",", ""))
    if observed_numbers and predicted_numbers:
        return any(abs(float(left) - float(right)) <= 1e-9 for left in observed_numbers for right in predicted_numbers)
    return False


def _append_evidence_candidate(navigation_state: dict[str, Any], candidate: dict[str, Any]) -> None:
    candidates = navigation_state.setdefault("evidence_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
        navigation_state["evidence_candidates"] = candidates
    key = tuple(
        str(candidate.get(field, ""))
        for field in ("page", "source_type", "question_field", "row_key", "column_key", "value", "supporting_region")
    )
    for existing in candidates:
        if isinstance(existing, dict) and tuple(
            str(existing.get(field, ""))
            for field in ("page", "source_type", "question_field", "row_key", "column_key", "value", "supporting_region")
        ) == key:
            return
    candidates.append(candidate)


def _markdown_table_rows(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and not all(re.fullmatch(r":?-{1,}:?", cell) for cell in cells):
            rows.append(cells)
    return rows


def _looks_table_like_text(text: str, question: str = "") -> bool:
    """Detect table-like evidence even when a parser flattened the grid."""
    value = str(text or "")
    if len(re.findall(r"\|[^|\n]+\|", value)) >= 2:
        return True
    lowered = value.casefold()
    structural = (
        r"\b(?:week|wk\.?|month|year|age|group|row|column|rank|ranking|percent|percentage|"
        r"amount|income|weight|total|cholesterol)\b"
    )
    numeric_count = len(re.findall(r"[-+]?\d+(?:[.,]\d+)?%?", value))
    question_lowered = str(question or "").casefold()
    question_structural = bool(re.search(structural, question_lowered))
    return bool(re.search(structural, lowered) and numeric_count >= 3) or bool(
        question_structural and numeric_count >= 3
    )


def _flattened_table_rows(text: str, question: str = "") -> list[list[str]]:
    """Recover a conservative grid from common flattened PDF text output.

    This intentionally handles only rows with an explicit ordinal label (for
    example ``Week 4 103 133 111``).  If the layout is more ambiguous, the
    caller still gets the table-like route and can use render/layout/OCR.
    """
    value = str(text or "")
    if not value:
        return []
    groups = re.findall(r"(?:group\s*)?#\s*\d+", value, flags=re.IGNORECASE)
    if not groups:
        groups = re.findall(r"#\s*\d+", str(question or ""), flags=re.IGNORECASE)
    group_labels: list[str] = []
    rat_suffix = " rats" if re.search(r"\brats?\b", value, re.IGNORECASE) else ""
    for group in groups:
        match = re.search(r"#\s*\d+", group)
        if match:
            group_number = re.sub(r"\s+", "", match.group(0))
            label = f"Group {group_number}{rat_suffix}"
            if label.casefold() not in {item.casefold() for item in group_labels}:
                group_labels.append(label)
    if len(group_labels) < 2:
        return []

    metric = "Metric"
    question_terms = _question_constraint_terms(question)
    for term in question_terms:
        if not term.isdigit() and not re.fullmatch(r"\d+(?:st|nd|rd|th)", term) and re.search(
            rf"(?<![\w]){re.escape(term)}(?![\w])", value, re.IGNORECASE
        ):
            metric = term.title()
            break
    if metric == "Metric":
        for candidate in ("cholesterol", "income", "amount", "weight", "total"):
            if re.search(rf"\b{candidate}\b", value, re.IGNORECASE):
                metric = candidate.title()
                break

    # Keep the same two-header-row contract used by extracted tables: the
    # first row carries the group labels and the second binds the metric to
    # each numeric column.
    rows: list[list[str]] = [
        ["", *group_labels],
        ["Week", *([metric] * len(group_labels))],
    ]
    # Keep each explicit row on one physical line.  This avoids fabricating
    # row/column relations from a page-wide list of unrelated numbers.
    for line in re.split(r"[\r\n]+", value):
        match = re.search(
            r"\b(week|wk\.?|month|year|time|point)\s*(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b"
            r"\s+(.+)$",
            line.strip(),
            re.IGNORECASE,
        )
        if match is None:
            continue
        values = re.findall(r"[-+]?\d+(?:[.,]\d+)?%?", match.group(3))
        if len(values) < 2:
            continue
        label = f"{match.group(1).rstrip('.')} {match.group(2)}"
        rows.append([label, *values[: len(group_labels)]])
    return rows if len(rows) >= 2 else []


def _table_evidence_candidates(
    navigation_state: dict[str, Any],
    page: int,
    rows: list[list[Any]],
    *,
    source_type: str = "table",
    supporting_region: Any = None,
) -> None:
    normalized_rows = [
        [str(cell).strip() for cell in row] for row in rows
        if isinstance(row, (list, tuple)) and any(str(cell).strip() for cell in row)
    ]
    if len(normalized_rows) < 2:
        return
    question = str(navigation_state.get("question") or "")
    terms = _question_constraint_terms(question)
    header_rows = normalized_rows[: min(2, len(normalized_rows) - 1)]
    data_rows = normalized_rows[len(header_rows):]
    width = max(len(row) for row in normalized_rows)
    column_headers = []
    for column in range(width):
        labels = [row[column] for row in header_rows if column < len(row) and row[column]]
        column_headers.append(" ".join(labels).strip())
    for row in data_rows:
        row_key = row[0] if row else ""
        row_header = header_rows[-1][0] if header_rows and header_rows[-1] else ""
        for column, value in enumerate(row[1:], start=1):
            if not value:
                continue
            column_key = column_headers[column] if column < len(column_headers) else ""
            # Do not put every column header into every candidate context:
            # doing so would make a value from group #2 satisfy a question for
            # group #1 merely because both headers occurred in the table.
            context = " ".join(item for item in (row_header, row_key, column_key, value) if item)
            satisfies = _terms_in_context(terms, context)
            metric = next(
                (
                    term
                    for term in terms
                    if not term.isdigit()
                    and not re.fullmatch(r"\d+(?:st|nd|rd|th)", term)
                    and re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", column_key, re.IGNORECASE)
                ),
                row_header,
            )
            _append_evidence_candidate(
                navigation_state,
                {
                    "page": page,
                    "source_type": source_type,
                    "metric": metric,
                    "question_field": column_key or None,
                    "row_key": row_key,
                    "column_key": column_key,
                    "value": value,
                    "relation_score": 1.0 if satisfies else 0.0,
                    "satisfies_question_constraints": satisfies,
                    "supporting_region": supporting_region,
                },
            )


def _text_evidence_candidate(navigation_state: dict[str, Any], page: int, text: str, source_type: str = "text") -> None:
    question = str(navigation_state.get("question") or "")
    terms = _question_constraint_terms(question)
    if not terms or not _text_evidence_supports_question(question, text):
        return
    field = next((term for term in terms if not term.isdigit() and not re.fullmatch(r"\d+(?:st|nd|rd|th)", term)), terms[0])
    escaped_field = re.escape(field)
    match = re.search(rf"(?im)(?:^|[|;])\s*{escaped_field}\s*(?:#\s*)?[:=]\s*([^\n|;]+)", text)
    if match is None:
        match = re.search(rf"(?i)\b{escaped_field}\b\s+(?:is|was|are|were|called|named)\s+([^\n|;,.]+)", text)
    value = match.group(1).strip(" \t:;,.|\n") if match else ""
    if not value:
        return
    context = text[max(0, (match.start() if match else 0) - 180):(match.end() if match else 180) + 180]
    satisfies = _terms_in_context(terms, context)
    _append_evidence_candidate(
        navigation_state,
        {
            "page": page,
            "source_type": source_type,
            "question_field": field,
            "row_key": None,
            "column_key": None,
            "value": value,
            "relation_score": 1.0 if satisfies else 0.0,
            "satisfies_question_constraints": satisfies,
            "supporting_region": None,
        },
    )


def _supporting_pages(answer: str, navigation_state: dict[str, Any]) -> list[int]:
    normalized_answer = _normalized_evidence(answer)
    if not normalized_answer:
        return []
    answer_tokens = [token for token in normalized_answer.split() if token not in {"the", "a", "an", "is", "are", "was", "were"}]
    result: list[int] = []
    for page, evidence in navigation_state.get("evidence_by_page", {}).items():
        normalized = _normalized_evidence(evidence)
        if normalized_answer in normalized:
            result.append(int(page) if str(page).isdigit() else 0)
            continue
        evidence_tokens = set(normalized.split())
        overlap = [token for token in answer_tokens if token in evidence_tokens]
        if answer_tokens and len(overlap) >= min(2, len(answer_tokens)) and len(overlap) / len(answer_tokens) >= 0.3:
            result.append(int(page) if str(page).isdigit() else 0)
    return sorted({page for page in result if page > 0})


def _is_absence_answer(answer: str) -> bool:
    return bool(_ABSENCE_ANSWER_RE.search(answer.strip())) or _normalized_evidence(answer) in {"none", "not found", "n a"}


def _can_finish(prediction: str, navigation_state: dict[str, Any] | None) -> tuple[bool, str, list[int]]:
    if navigation_state is None:
        return True, "", []
    answer, format_ok = extract_final_answer(prediction)
    if not format_ok:
        return False, "final action could not be parsed", []
    unvisited = list(navigation_state.get("unvisited_pages", []))
    candidates = [item for item in navigation_state.get("evidence_candidates", []) if isinstance(item, dict)]
    prediction_found = any(_candidate_value_matches(item, answer) for item in candidates)
    relation_matches = [
        item for item in _valid_evidence_candidates(candidates) if _candidate_value_matches(item, answer)
    ]
    supported_pages = sorted({int(item["page"]) for item in relation_matches if str(item.get("page")).isdigit()})
    supported_regions = [
        item.get("supporting_region")
        for item in relation_matches
        if item.get("supporting_region") is not None
    ]
    navigation_state["prediction_found_in_document"] = bool(prediction_found)
    navigation_state["prediction_relation_matched"] = bool(relation_matches)
    navigation_state["final_supporting_pages"] = supported_pages
    navigation_state["final_supporting_regions"] = supported_regions
    navigation_state["final_supported_by_evidence"] = bool(
        prediction_found and relation_matches and (supported_pages or supported_regions)
    )
    # A render result is intentionally deferred until the controller knows the
    # next decision needs pixels.  A direct visual final before that image has
    # actually entered the model is not accepted.
    if navigation_state.get("pending_visual_image_paths") and navigation_state.get("question_type") == "visual":
        navigation_state["visual_input_required"] = True
        navigation_state["visual_input_reason"] = "the final decision requires the rendered page pixels"
        navigation_state["current_evidence_sufficient"] = False
        return False, "visual input is required before answering this visual question", supported_pages
    search_budget_exhausted = bool(navigation_state.get("search_budget_exhausted"))
    page_count_known = isinstance(navigation_state.get("page_count"), int) and int(navigation_state.get("page_count")) > 0
    visited_pages = list(navigation_state.get("visited_pages", []))
    has_document_observation = bool(
        visited_pages
        or candidates
        or navigation_state.get("evidence_by_page")
        or navigation_state.get("parsed_pages")
        or navigation_state.get("rendered_pages")
        or navigation_state.get("cropped_pages")
        or navigation_state.get("ocr_pages")
    )
    if not has_document_observation and not search_budget_exhausted:
        navigation_state["current_evidence_sufficient"] = False
        return False, "no document page has been inspected yet", supported_pages
    if not page_count_known and not search_budget_exhausted and _is_absence_answer(answer):
        navigation_state["current_evidence_sufficient"] = False
        return False, "document page count is unknown and an absence answer has no supporting evidence", supported_pages
    visual_pixel_support = bool(
        navigation_state.get("question_type") == "visual"
        and navigation_state.get("visual_input_consumed")
        and navigation_state.get("visual_evidence_sufficient")
        and not _is_absence_answer(answer)
    )
    if visual_pixel_support and not relation_matches:
        visual_pages = sorted(
            {int(page) for page in navigation_state.get("visual_evidence_pages", []) if str(page).isdigit()}
        )
        visual_regions = list(navigation_state.get("supporting_regions", []))
        if visual_pages or visual_regions:
            # Pixel evidence has no textual value until the model has supplied
            # its visual answer.  Materialize that observation as a structured
            # candidate so the same hard support formula applies to text,
            # tables, and vision.
            _append_evidence_candidate(
                navigation_state,
                {
                    "page": visual_pages[0] if visual_pages else None,
                    "source_type": "visual",
                    "question_field": str(navigation_state.get("question") or ""),
                    "row_key": None,
                    "column_key": None,
                    "value": answer,
                    "relation_score": 1.0,
                    "satisfies_question_constraints": True,
                    "supporting_region": visual_regions[0] if visual_regions else None,
                    "visual_verified": True,
                },
            )
            candidates = [item for item in navigation_state.get("evidence_candidates", []) if isinstance(item, dict)]
            prediction_found = any(_candidate_value_matches(item, answer) for item in candidates)
            relation_matches = [
                item for item in _valid_evidence_candidates(candidates) if _candidate_value_matches(item, answer)
            ]
            supported_pages = sorted(
                {int(item["page"]) for item in relation_matches if str(item.get("page")).isdigit()}
            )
            supported_regions = [
                item.get("supporting_region")
                for item in relation_matches
                if item.get("supporting_region") is not None
            ]
            navigation_state["prediction_found_in_document"] = bool(prediction_found)
            navigation_state["prediction_relation_matched"] = bool(relation_matches)
            navigation_state["final_supporting_pages"] = supported_pages
            navigation_state["final_supporting_regions"] = supported_regions
            navigation_state["final_supported_by_evidence"] = bool(
                prediction_found and relation_matches and (supported_pages or supported_regions)
            )
            _evaluate_evidence_sufficiency(navigation_state)
    if navigation_state.get("final_supported_by_evidence") and not _is_absence_answer(answer):
        navigation_state["stop_reason"] = "sufficient_evidence"
        return True, "", supported_pages
    if _is_absence_answer(answer) and not search_budget_exhausted and (
        not page_count_known or bool(unvisited)
    ):
        navigation_state["current_evidence_sufficient"] = False
        navigation_state["stop_reason"] = "evidence_insufficient"
        return False, "an absence answer requires all known pages to be checked", supported_pages
    if not unvisited or search_budget_exhausted:
        # A negative answer is only allowed after the complete known search
        # frontier (or the explicit budget) has been exhausted.  Positive
        # answers may also be returned here when the model found evidence that
        # our conservative relation heuristic could not bind to the exact
        # generated string.
        navigation_state["stop_reason"] = "search_budget_exhausted" if search_budget_exhausted else "all_pages_checked"
        navigation_state["current_evidence_sufficient"] = bool(navigation_state.get("evidence_sufficient"))
        return True, "", supported_pages
    if _is_absence_answer(answer):
        navigation_state["current_evidence_sufficient"] = False
        navigation_state["stop_reason"] = "evidence_insufficient"
        return False, "the answer is an absence/None answer while pages remain unvisited", supported_pages
    if not supported_pages and not navigation_state.get("final_supporting_regions"):
        navigation_state["current_evidence_sufficient"] = False
        navigation_state["stop_reason"] = "evidence_insufficient"
        return False, "the final answer has no supporting evidence in visited pages", supported_pages
    # This branch is intentionally conservative: a page-local string match
    # that did not satisfy the relation threshold is not evidence support.
    navigation_state["current_evidence_sufficient"] = bool(navigation_state.get("evidence_sufficient"))
    navigation_state["stop_reason"] = "evidence_insufficient"
    return False, "the final answer does not satisfy the question-bound evidence relation", supported_pages


def _final_guard_observation(reason: str, navigation_state: dict[str, Any]) -> str:
    return (
        "<interpreter>\n"
        "Final answer blocked by evidence guard.\n"
        f"Reason: {reason}\n"
        f"{_navigation_status_text(navigation_state)}\n"
        f"{_OBSERVATION_PROTOCOL_RULE}\n"
        "Continue with the next appropriate tool call. Do not output None or "
        "not found until all relevant pages are checked or the search budget is exhausted.\n"
        "</interpreter>"
    )


def _tool_failure_fallback(
    tool_name: str,
    arguments: dict[str, Any],
    error: str,
    navigation_state: dict[str, Any] | None,
) -> str | None:
    if navigation_state is None:
        return None
    document_path = arguments.get("document_path") or navigation_state.get("document_path") or ""
    page_number = arguments.get("page_number")
    if page_number is None:
        unvisited = navigation_state.get("unvisited_pages", [])
        page_number = unvisited[0] if unvisited else 1
    navigation_state["fallback_used"] = True
    navigation_state.setdefault("fallback_events", []).append(
        {"failed_tool": tool_name, "page_number": page_number, "error": str(error)}
    )
    if tool_name == "extract_table":
        return (
            "<interpreter>\n"
            f"Tool extract_table failed: {error}\n"
            "Fallback required: render the same page and inspect the table visually.\n"
            f"Next suggested call: render_page(document_path={document_path!r}, page_number={int(page_number)})\n"
            f"{_navigation_status_text(navigation_state)}\n"
            "</interpreter>"
        )
    if tool_name == "render_page":
        return (
            "<interpreter>\n"
            f"Tool render_page failed: {error}\n"
            "Fallback: use parse_document for this page or try ocr_region if an existing image path is available.\n"
            f"{_navigation_status_text(navigation_state)}\n"
            "</interpreter>"
        )
    if tool_name == "detect_layout":
        return (
            "<interpreter>\n"
            f"Tool detect_layout failed: {error}\n"
            "Layout backend is unavailable. Continue with the rendered page and use crop_region or zoom_region "
            "with the page image; do not terminate the rollout because layout detection failed.\n"
            f"Next suggested call: render_page(document_path={document_path!r}, page_number={int(page_number)})\n"
            f"{_navigation_status_text(navigation_state)}\n"
            "</interpreter>"
        )
    if tool_name == "ocr_region":
        return (
            "<interpreter>\n"
            f"Tool ocr_region failed: {error}\n"
            "OCR backend is unavailable, but visual evidence may still be inspected from the rendered/cropped "
            "image. Continue with the image observation or use crop_region/zoom_region; do not treat OCR failure "
            "as proof that the requested value is absent.\n"
            f"{_navigation_status_text(navigation_state)}\n"
            "</interpreter>"
        )
    if tool_name in {"crop_region", "zoom_region"}:
        return (
            "<interpreter>\n"
            f"Tool {tool_name} failed: {error}\n"
            "Region extraction failed. Re-render the page and retry a smaller visual region before answering.\n"
            f"Next suggested call: render_page(document_path={document_path!r}, page_number={int(page_number)})\n"
            f"{_navigation_status_text(navigation_state)}\n"
            "</interpreter>"
        )
    return None


def _is_infrastructure_tool_error(error: Any) -> bool:
    text = str(error or "").casefold()
    return any(
        marker in text
        for marker in (
            "libgl", "backend", "cannot import", "no module named", "connection",
            "timed out", "timeout", "processor", "ocr unavailable", "cuda",
        )
    )


async def execute_predictions(
    prediction: str,
    execution_trace: list[dict[str, Any]] | None = None,
    action_log: dict[str, Any] | None = None,
    turn: int | None = None,
    navigation_state: dict[str, Any] | None = None,
    world_runtime: Any | None = None,
    belief_runtime: Any | None = None,
    decision_controller: Any | None = None,
    tool_budget: int | None = None,
    bayes_aux_prompt: str | None = None,
    bayes_prompt_tokenizer: Any | None = None,
    cost_state: dict[str, float] | None = None,
    candidate_records: list[dict[str, Any]] | None = None,
    candidate_decision: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Parse and execute one assistant turn without silently recovering it."""
    if action_log is not None:
        for key in (
            "candidate_action_count",
            "executed_action_count",
            "valid_action_count",
            "invalid_action_count",
            "ignored_action_count",
            "protocol_error_count",
            "tool_error_count",
        ):
            action_log.setdefault(key, 0)
        action_log.setdefault("actions", [])
    parsed = parse_assistant_action(prediction)
    _record_action_candidate(action_log, parsed, turn=turn, raw=prediction)
    if action_log is not None and candidate_decision is not None:
        action_log["bayes_decision"] = dict(candidate_decision)
        action_log.setdefault("bayes_decisions", []).append(dict(candidate_decision))
        action_log["candidate_action_keys"] = [
            str(item.get("key")) for item in (candidate_records or []) if isinstance(item, dict)
        ]
        if action_log.get("actions"):
            action_log["actions"][-1]["policy_candidates"] = [
                {
                    "key": str(item.get("key")),
                    "source": str(item.get("source", "")),
                    "token_count": len(item.get("token_ids", [])),
                }
                for item in (candidate_records or [])
                if isinstance(item, dict)
            ]

    if parsed.kind == "final":
        can_finish, finish_reason, supporting_pages = _can_finish(prediction, navigation_state)
        if not can_finish:
            _record_failure_event(
                action_log,
                kind="premature_final",
                origin="protocol",
                turn=turn,
                reason=finish_reason,
                evidence={"supporting_pages": supporting_pages},
            )
            if action_log is not None:
                action_log["premature_final_count"] = int(action_log.get("premature_final_count", 0)) + 1
                action_log["actions"][-1].update(
                    {
                        "valid": True,
                        "executed": False,
                        "accepted": False,
                        "premature_final": True,
                        "recovery_observation": True,
                        "action_valid_for_policy_gradient": False,
                        "action_reward": -0.5,
                        "reason": finish_reason,
                    }
                )
            if navigation_state is not None:
                navigation_state["premature_final"] = True
                navigation_state["had_evidence_guard_recovery"] = True
                navigation_state["rejected_final_count"] = int(navigation_state.get("rejected_final_count", 0)) + 1
            if execution_trace is not None:
                execution_trace.append(
                    {
                        "kind": "final",
                        "turn": turn,
                        "raw": prediction,
                        "raw_generation_text": prediction,
                        "parsed_action_type": parsed.kind,
                        "parsed_tool_name": None,
                        "parse_failure_reason": None,
                        "valid": True,
                        "executed": False,
                        "accepted": False,
                        "premature_final": True,
                        "reason": finish_reason,
                    "supporting_pages": supporting_pages,
                    "recovery_observation": True,
                    "action_valid_for_policy_gradient": False,
                    "action_reward": -0.5,
                }
                )
            return _final_guard_observation(finish_reason, navigation_state or {}), False
        if action_log is not None:
            action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
            failed_evidence = any(
                isinstance(item, dict)
                and item.get("kind") == "tool_call"
                and item.get("executed")
                and not item.get("success")
                for item in (execution_trace or [])
            )
            if failed_evidence:
                action_log["actions"][-1]["used_failed_evidence"] = True
                _record_failure_event(
                    action_log,
                    kind="final_used_failed_evidence",
                    origin="model_action",
                    turn=turn,
                    reason="final action followed an unsuccessful tool observation",
                    penalize=not bool(supporting_pages),
                    evidence={"supporting_pages": supporting_pages},
                )
            action_log["actions"][-1].update(
                {
                    "valid": True,
                    "executed": False,
                    "accepted": True,
                    "supporting_pages": supporting_pages,
                    "action_valid_for_policy_gradient": True,
                    "action_reward": 0.0,
                }
            )
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": "final",
                    "turn": turn,
                    "raw": prediction,
                    "raw_generation_text": prediction,
                    "parsed_action_type": parsed.kind,
                    "parsed_tool_name": None,
                    "parse_failure_reason": None,
                    "valid": True,
                    "executed": False,
                    "accepted": True,
                    "supporting_pages": supporting_pages,
                    "action_valid_for_policy_gradient": True,
                    "action_reward": 0.0,
                }
            )
        _set_terminal_status(action_log, "completed")
        return "", True

    if parsed.kind == "abstain":
        reason = str(parsed.value or "insufficient reliable evidence")
        if action_log is not None:
            action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
            action_log["actions"][-1].update(
                {
                    "valid": True,
                    "executed": False,
                    "accepted": True,
                    "abstention": True,
                    "abstention_reason": reason,
                    "action_valid_for_policy_gradient": True,
                    "action_reward": 0.0,
                }
            )
            _set_terminal_status(action_log, "abstained")
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": "abstain",
                    "turn": turn,
                    "raw": prediction,
                    "raw_generation_text": prediction,
                    "parsed_action_type": parsed.kind,
                    "parsed_tool_name": None,
                    "valid": True,
                    "executed": False,
                    "accepted": True,
                    "reason": reason,
                }
            )
        if navigation_state is not None:
            navigation_state["abstention"] = True
            navigation_state["abstention_reason"] = reason
        return "", True

    if parsed.kind != "tool_call":
        invalid_count = max(1, parsed.candidate_action_count)
        _record_failure_event(
            action_log,
            kind="invalid_action",
            origin="protocol",
            turn=turn,
            reason=parsed.reason,
        )
        if action_log is not None:
            action_log["invalid_action_count"] = int(action_log.get("invalid_action_count", 0)) + invalid_count
            action_log["ignored_action_count"] = int(action_log.get("ignored_action_count", 0)) + parsed.candidate_action_count
            action_log["protocol_error_count"] = int(action_log.get("protocol_error_count", 0)) + 1
            action_log["actions"][-1].update(
                {
                    "valid": False,
                    "executed": False,
                    "accepted": False,
                    "action_valid_for_policy_gradient": False,
                    "action_reward": -0.5,
                }
            )
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": parsed.kind,
                    "turn": turn,
                    "raw": prediction,
                    "raw_generation_text": prediction,
                    "parsed_action_type": parsed.kind,
                    "parsed_tool_name": None,
                    "valid": False,
                    "executed": False,
                    "reason": parsed.reason,
                    "parse_failure_reason": parsed.reason,
                    "protocol_parse_error": parsed.reason if parsed.kind == "protocol_error" else None,
                    "candidate_action_count": parsed.candidate_action_count,
                }
            )
        _set_terminal_status(action_log, "model_protocol_error")
        return "", True

    tool_call = parsed.value if isinstance(parsed.value, dict) else {}
    valid, validation_error = _validate_tool_call(tool_call)
    if not valid:
        _record_failure_event(
            action_log,
            kind="invalid_tool_arguments",
            origin="protocol",
            turn=turn,
            tool=str(tool_call.get("name") or ""),
            reason=validation_error or "invalid tool call",
        )
        if action_log is not None:
            action_log["invalid_action_count"] = int(action_log.get("invalid_action_count", 0)) + 1
            action_log["ignored_action_count"] = int(action_log.get("ignored_action_count", 0)) + 1
            action_log["protocol_error_count"] = int(action_log.get("protocol_error_count", 0)) + 1
            action_log["actions"][-1].update(
                {
                    "valid": False,
                    "executed": False,
                    "accepted": False,
                    "action_valid_for_policy_gradient": False,
                    "action_reward": -0.5,
                    "reason": validation_error,
                }
            )
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": "tool_call",
                    "turn": turn,
                    "tool": tool_call.get("name", ""),
                    "arguments": tool_call.get("arguments", {}),
                    "raw_generation_text": prediction,
                    "parsed_action_type": parsed.kind,
                    "parsed_tool_name": tool_call.get("name", ""),
                    "valid": False,
                    "executed": False,
                    "success": False,
                    "reason": validation_error,
                    "parse_failure_reason": validation_error,
                    "protocol_parse_error": validation_error,
                }
            )
        _set_terminal_status(action_log, "model_protocol_error")
        return "", True

    if navigation_state is not None and navigation_state.get("search_budget_exhausted"):
        # Do not execute a ninth tool call after the bounded search budget has
        # been consumed.  Give the model a normal observation so it can use the
        # reserved recovery turn to emit a final answer.
        reason = "tool search budget exhausted; emit final answer"
        _record_failure_event(
            action_log,
            kind="budget_blocked_tool",
            origin="budget",
            turn=turn,
            tool=str(tool_call.get("name") or ""),
            reason=reason,
        )
        if action_log is not None:
            action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
            action_log["ignored_action_count"] = int(action_log.get("ignored_action_count", 0)) + 1
            action_log["actions"][-1].update(
                {
                    "valid": True,
                    "executed": False,
                    "accepted": False,
                    "action_valid_for_policy_gradient": False,
                    "action_reward": -0.1,
                    "recovery_observation": True,
                    "reason": reason,
                }
            )
        if navigation_state is not None:
            navigation_state["budget_blocked_tool_count"] = int(
                navigation_state.get("budget_blocked_tool_count", 0)
            ) + 1
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": "tool_call",
                    "turn": turn,
                    "tool": tool_call.get("name", ""),
                    "arguments": tool_call.get("arguments", {}),
                    "raw_generation_text": prediction,
                    "parsed_action_type": parsed.kind,
                    "parsed_tool_name": tool_call.get("name", ""),
                    "parse_failure_reason": None,
                    "protocol_parse_error": None,
                    "valid": True,
                    "executed": False,
                    "success": False,
                    "reason": reason,
                    "recovery_observation": True,
                }
            )
        return (
            "<interpreter>\n"
            "Tool search budget exhausted. Do not call another tool; output "
            "the best grounded answer inside <final>...</final>.\n"
            f"{_navigation_status_text(navigation_state)}\n"
            "</interpreter>",
            False,
        )

    tool_name = str(tool_call["name"])
    arguments = dict(tool_call.get("arguments", {}))
    arguments, auto_routed = _tool_arguments_for_navigation(tool_name, arguments, navigation_state)
    controller_selected_action: dict[str, Any] | None = None
    policy_action_key = (
        canonical_action_key({"kind": "tool", "tool": tool_name, "arguments": arguments})
        if canonical_action_key is not None
        else ""
    )
    if decision_controller is not None and TaskStateView is not None and candidate_decision is None:
        try:
            task_state = TaskStateView.from_navigation_state(
                navigation_state or {},
                tool_budget=int(tool_budget if tool_budget is not None else TOOL_CONFIGS.get("max_tool_calls", 8)),
            )
            decision_candidates = [
                {"kind": "tool", "tool": tool_name, "arguments": arguments},
            ]
            # DVOI is only allowed to compare actions that the current policy
            # actually proposed.  Do not inject a fixed diagnostic menu here:
            # the branch path already obtains a finite set of real policy
            # candidates from the shared prefix, while this single-action
            # path has at most the current tool as a diagnostic candidate.
            diagnostic_candidates = (
                [decision_candidates[0]]
                if tool_name in {"detect_layout", "render_page", "crop_region", "zoom_region", "ocr_region"}
                else []
            )
            observed_tool_names = {
                str(item.get("tool"))
                for item in (execution_trace or [])
                if isinstance(item, dict) and item.get("kind") == "tool_call" and item.get("tool")
            }
            risk_metadata: dict[str, Any] = {
                "independent_tool_count": max(
                    len(observed_tool_names),
                    len((navigation_state or {}).get("supporting_pages", [])),
                ),
                "recent_surprise": float((navigation_state or {}).get("recent_surprise", 0.0) or 0.0),
                "answer_self_consistency": float(
                    (navigation_state or {}).get("answer_self_consistency", 0.0) or 0.0
                ),
                "page_count": max(
                    1,
                    int((navigation_state or {}).get("page_count", 0) or 0),
                    len((navigation_state or {}).get("visited_pages", []))
                    + int((navigation_state or {}).get("unvisited_page_count", 0) or 0),
                ),
            }
            if belief_runtime is not None:
                risk_snapshot = belief_runtime.snapshot()
                risk_quality = risk_snapshot.tool_quality.get(tool_name)
                if risk_quality is not None:
                    risk_metadata.update(
                        {
                            "semantic_posterior": float(risk_quality.semantic_mean),
                            "structure_posterior": float(risk_quality.structure_mean),
                        }
                    )
            decision_report = decision_controller.select(
                task_state,
                belief_runtime,
                decision_candidates,
                diagnostic_actions=diagnostic_candidates,
                risk_metadata=risk_metadata,
            )
            controller_selected_action = next(
                (
                    canonical_action(candidate_action)
                    for candidate_key, candidate_action in decision_report.candidate_actions
                    if str(candidate_key) == str(decision_report.selected_action)
                ),
                None,
            )
            candidate_action_records: list[dict[str, Any]] = []
            candidate_text_by_key: dict[str, str] = {}
            for candidate_key, candidate_action in decision_report.candidate_actions:
                if canonical_action_key is None:
                    continue
                candidate_key = str(candidate_key)
                candidate_text = ""
                try:
                    if candidate_key == canonical_action_key(
                        _bayestool_branch_action_from_text(prediction, navigation_state)[0]
                    ):
                        candidate_text = prediction
                except Exception:
                    candidate_text = ""
                if not candidate_text:
                    candidate_text = _bayestool_action_text(candidate_action)
                candidate_text_by_key[candidate_key] = candidate_text
                candidate_action_records.append(
                    {"key": candidate_key, "action": canonical_action(candidate_action), "text": candidate_text}
                )
            ordered_values = list(decision_report.action_values)
            ordered_values.sort(key=lambda value: (-value.value_mean, value.action_key))
            best_action_margin = (
                float(ordered_values[0].value_mean - ordered_values[1].value_mean)
                if len(ordered_values) >= 2
                else float(max(0.0, ordered_values[0].value_mean)) if ordered_values else 0.0
            )
            belief_snapshot = belief_runtime.snapshot().to_dict() if belief_runtime is not None else {}
            prefix_event_count = len(getattr(world_runtime, "events", [])) if world_runtime is not None else 0
            prefix_observation_payload = [
                {
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments", {}),
                    "observed_result": str(item.get("observed_result") or ""),
                    "result_status": item.get("result_status"),
                }
                for item in (execution_trace or [])
                if isinstance(item, dict) and item.get("kind") == "tool_call"
            ]
            prefix_observation_signature = hashlib.sha256(
                json.dumps(
                    prefix_observation_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            current_content_signature = None
            if content_signature is not None:
                current_content_signature = content_signature(
                    {
                        "navigation_state": navigation_state or {},
                        "question_type": (navigation_state or {}).get("question_type", "text"),
                        "evidence_candidates": (navigation_state or {}).get("evidence_candidates", []),
                        "remaining_tool_budget": task_state.remaining_tool_budget,
                        "phase": (navigation_state or {}).get("phase", "search"),
                    },
                    question=str((navigation_state or {}).get("question") or ""),
                )
            branch_eligible = bool(
                bool(getattr(decision_controller.config, "use_regret_branching", True))
                and
                decision_report.decision_regret > decision_controller.config.decision_regret_threshold
                and task_state.remaining_tool_budget >= 2
                and float(belief_snapshot.get("ood_score", 0.0) or 0.0) < 0.15
            )
            decision_metadata = {
                "selected_action": decision_report.selected_action,
                "selected_mode": decision_report.selected_mode,
                "consensus": decision_report.consensus,
                "consensus_action": decision_report.consensus_action,
                "bayes_action": decision_report.bayes_action,
                "decision_regret": decision_report.decision_regret,
                "dvoi": dict(decision_report.dvoi or {}),
                "stop_decision": dict(decision_report.stop_decision or {}),
                "branch_eligible": branch_eligible,
                "branch_triggered": False,
                "branch_horizon": int(getattr(decision_controller.config, "branch_horizon", 0)),
                "candidate_action_keys": [item["key"] for item in candidate_action_records],
                "candidate_actions": candidate_action_records,
                "best_action_margin": best_action_margin,
                "best_action_text": candidate_text_by_key.get(str(decision_report.bayes_action or ""), ""),
                "policy_action_text": prediction,
                "belief_snapshot": belief_snapshot,
                "observed_prefix_event_count": prefix_event_count,
                "observed_prefix_js": 0.0 if prefix_event_count == 0 else 1.0,
                "prefix_observation_signature": prefix_observation_signature,
                "content_signature": current_content_signature or "",
                "prompt": str(bayes_aux_prompt or ""),
            }
            if controller_selected_action is not None:
                selected_key = canonical_action_key(controller_selected_action)
                decision_metadata.update(
                    {
                        "policy_action_key": policy_action_key,
                        "selected_action_kind": controller_selected_action.get("kind"),
                        "decision_action_applied": selected_key != policy_action_key,
                    }
                )
            if action_log is not None:
                action_log["bayes_decision"] = decision_metadata
                action_log.setdefault("bayes_decisions", []).append(decision_metadata)
            if navigation_state is not None:
                navigation_state["bayes_last_decision"] = decision_metadata
        except Exception as exc:  # controller diagnostics cannot invalidate a tool action
            if action_log is not None:
                action_log.setdefault("bayes_decision_errors", []).append(str(exc))

    # The normal path contains only the action sampled by the policy.  The
    # controller can therefore annotate this action, but it cannot insert a
    # synthetic final/abstain action or replace it with an unscored tool.  A
    # finite set of alternate actions is executed only by the branch path,
    # which obtains each candidate from a real policy sample.
    if controller_selected_action is not None:
        selected_key = canonical_action_key(controller_selected_action)
        if selected_key != policy_action_key:
            # This is a defensive telemetry-only guard.  The normal candidate
            # set contains only the sampled action, and branch continuations
            # are generated separately from real policy samples.  Never
            # execute a controller-only replacement here because it would be
            # absent from the policy-gradient token sequence.
            if action_log is not None:
                action_log["actions"][-1].update(
                    {
                        "controller_action_not_applied": True,
                        "controller_selected_action": selected_key,
                    }
                )
    if action_log is not None:
        action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
        action_log["executed_action_count"] = int(action_log.get("executed_action_count", 0)) + 1
        action_log["actions"][-1].update(
            {"valid": True, "executed": True, "tool": tool_name, "auto_routed": auto_routed}
        )

    tool_started_at = time.perf_counter()
    try:
        result = await _bayestool_execute_clean_cached(world_runtime, tool_name, arguments)
    except Exception as exc:  # the model action was sent, but the tool failed
        tool_elapsed_seconds = max(0.0, time.perf_counter() - tool_started_at)
        exception_budget = max(
            0,
            int(tool_budget if tool_budget is not None else TOOL_CONFIGS.get("max_tool_calls", 8)) - 1,
        )
        if navigation_state is not None:
            navigation_state["last_tool"] = tool_name
            navigation_state["last_result_status"] = "error"
            navigation_state["remaining_tool_budget"] = exception_budget
        if cost_state is not None:
            cost_state["tool_calls"] = float(cost_state.get("tool_calls", 0.0)) + 1.0
            cost_state["latency_seconds"] = float(cost_state.get("latency_seconds", 0.0)) + tool_elapsed_seconds
            cost_state["observation_text_tokens"] = float(cost_state.get("observation_text_tokens", 0.0)) + _bayestool_token_count(str(exc), bayes_prompt_tokenizer)
        if navigation_state is not None and _is_infrastructure_tool_error(exc):
            navigation_state["had_infra_error"] = True
            navigation_state["infra_error_messages"] = list(navigation_state.get("infra_error_messages", [])) + [str(exc)]
        if action_log is not None:
            action_log["tool_error_count"] = int(action_log.get("tool_error_count", 0)) + 1
            action_log["actions"][-1].update({"success": False, "tool_error": str(exc)})
        _record_failure_event(
            action_log,
            kind="tool_execution_failure",
            origin="real_infrastructure",
            turn=turn,
            tool=tool_name,
            reason=str(exc),
            penalize=False,
        )
        if execution_trace is not None:
            failure_trace = {
                    "kind": "tool_call",
                    "turn": turn,
                    "tool": tool_name,
                    "arguments": arguments,
                    "valid": True,
                    "executed": True,
                    "success": False,
                    "error": str(exc),
                    "failure_origin": "real_infrastructure",
                }
            if world_runtime is not None:
                transformed = world_runtime.record_real_infrastructure_failure(tool_name, arguments, str(exc))
                failure_trace["world_event"] = transformed.world_event.to_dict()
                failure_trace["bayes_supervision"] = transformed.hidden_supervision.to_dict()
            execution_trace.append(failure_trace)
        fallback = _tool_failure_fallback(tool_name, arguments, str(exc), navigation_state)
        if fallback is not None:
            if execution_trace is not None:
                execution_trace[-1]["recovery_observation"] = True
                execution_trace[-1]["recovered_with_fallback"] = True
            if belief_runtime is not None:
                fallback = _bayestool_append_observation(
                    fallback,
                    belief_runtime,
                    navigation_state or {},
                    tool_budget=exception_budget,
                    tokenizer=bayes_prompt_tokenizer,
                )
            return fallback, False
        _set_terminal_status(action_log, "tool_error")
        return "", True

    tool_elapsed_seconds = max(0.0, time.perf_counter() - tool_started_at)
    clean_result = str(result)
    success, parsed_result, result_status = _parse_tool_result(clean_result)
    observed_result = clean_result
    world_event = None
    hidden_supervision = None
    if world_runtime is not None:
        clean_image_paths, clean_image_candidates = _image_paths_from_result(parsed_result)
        transformed = world_runtime.transform_result(
            tool_name,
            arguments,
            clean_result,
            result_status=str(result_status or ("ok" if success else "error")),
            image_paths=clean_image_paths,
            image_valid=not clean_image_candidates or bool(clean_image_paths),
        )
        observed_result = transformed.observed_result
        world_event = transformed.world_event
        hidden_supervision = transformed.hidden_supervision
        observed_success, observed_payload, observed_status = _parse_tool_result(observed_result)
        if observed_success:
            parsed_result = observed_payload
            result_status = observed_status or result_status
        else:
            success = False
            parsed_result = {"error": observed_result}
            result_status = observed_status or "error"
    image_paths, image_candidates = _image_paths_from_result(parsed_result if success else None)
    media_error = bool(success and image_candidates and not image_paths)
    trace_item = {
        "kind": "tool_call",
        "turn": turn,
        "tool": tool_name,
        "arguments": arguments,
        "raw_generation_text": prediction,
        "parsed_action_type": parsed.kind,
        "parsed_tool_name": tool_name,
        "parse_failure_reason": None,
        "protocol_parse_error": None,
        "valid": True,
        "executed": True,
        "success": success,
        "result_status": result_status,
        "image_paths": image_paths,
        "image_candidates": image_candidates,
        "media_error": media_error,
        "result": clean_result,
        "observed_result": observed_result,
        "failure_origin": world_event.failure_origin if world_event is not None else "none",
    }
    if TaskStateView is not None:
        trace_item["task_state_before"] = TaskStateView.from_navigation_state(
            navigation_state or {},
            tool_budget=max(0, int(tool_budget if tool_budget is not None else TOOL_CONFIGS.get("max_tool_calls", 8))),
        ).to_prompt_dict()
    if world_event is not None:
        trace_item["world_event"] = world_event.to_dict()
        trace_item["bayes_supervision"] = hidden_supervision.to_dict() if hidden_supervision is not None else None
    observed_latency = (
        float(world_event.latency)
        if world_event is not None and math.isfinite(float(world_event.latency))
        else tool_elapsed_seconds
    )
    trace_item["latency_seconds"] = max(0.0, observed_latency)
    trace_item["observation_token_count"] = _bayestool_token_count(observed_result, bayes_prompt_tokenizer)
    if cost_state is not None:
        cost_state["tool_calls"] = float(cost_state.get("tool_calls", 0.0)) + 1.0
        cost_state["latency_seconds"] = float(cost_state.get("latency_seconds", 0.0)) + max(0.0, observed_latency)
        cost_state["observation_text_tokens"] = float(cost_state.get("observation_text_tokens", 0.0)) + float(trace_item["observation_token_count"])
    if isinstance(parsed_result, dict) and parsed_result.get("error"):
        trace_item["error"] = str(parsed_result["error"])
    if execution_trace is not None:
        execution_trace.append(trace_item)
    if action_log is not None:
        action_log["actions"][-1].update(
            {
                "success": success,
                "result_status": result_status,
                "image_paths": image_paths,
                "media_error": media_error,
                "auto_routed": auto_routed,
            }
        )

    # Navigation is derived from the observed result before the belief
    # filter extracts features.  This keeps task projection/budget/page state
    # causally aligned with the same observation that updates the posterior.
    budget_before_observation = int(
        tool_budget if tool_budget is not None else TOOL_CONFIGS.get("max_tool_calls", 8)
    )
    budget_after_observation = max(0, budget_before_observation - 1)
    if navigation_state is not None:
        navigation_state["last_tool"] = tool_name
        navigation_state["last_result_status"] = result_status or ("ok" if success else "error")
        navigation_state["remaining_tool_budget"] = budget_after_observation
    _update_navigation_state(navigation_state, tool_name, arguments, parsed_result, success)
    _update_pending_visual_input(
        navigation_state,
        tool_name,
        image_paths,
        parsed_result,
        arguments,
    )
    if TaskStateView is not None:
        trace_item["task_state_after"] = TaskStateView.from_navigation_state(
            navigation_state or {},
            tool_budget=budget_after_observation,
        ).to_prompt_dict()

    if not success:
        if world_event is not None and world_event.failure_origin == "world_injected":
            _record_failure_event(
                action_log,
                kind="world_observation_failure",
                origin="world_injected",
                turn=turn,
                tool=tool_name,
                reason=str(result_status or "observation unavailable"),
                penalize=False,
                evidence={"observation_delivered": bool(world_event.observation_delivered)},
            )
            # Injected failures are valid POMDP observations and must remain
            # in RL.  They are not routed through the infrastructure-error
            # exclusion path.
            if belief_runtime is not None:
                event_for_belief = world_event
                surprise = belief_runtime.predictive_surprise(event_for_belief)
                event_for_belief = replace(event_for_belief, predictive_surprise=surprise)
                if execution_trace is not None:
                    execution_trace[-1]["world_event"] = event_for_belief.to_dict()
                belief_runtime.update(
                    tool_name,
                    observed_result,
                    event_for_belief,
                    task_state=TaskStateView.from_navigation_state(navigation_state or {}) if TaskStateView else None,
                    hidden_supervision=hidden_supervision.to_dict() if hidden_supervision is not None else None,
                )
                level = belief_runtime.detect_reopen_level(event_for_belief)
                if level and bool(getattr(getattr(decision_controller, "config", None), "use_reopen", True)):
                    reopen = belief_runtime.reopen(
                        level,
                        cause_tools=[tool_name],
                        surprise=surprise,
                    )
                    if execution_trace is not None:
                        execution_trace[-1]["reopen"] = reopen
            next_obs = _bayestool_append_observation(
                observed_result,
                belief_runtime,
                navigation_state or {},
                tool_budget=budget_after_observation,
                tokenizer=bayes_prompt_tokenizer,
            )
            next_obs = f"<interpreter>\nTool: {tool_name}\n{next_obs}\n</interpreter>"
            return next_obs, False
        tool_error_text = str((parsed_result or {}).get("error") if isinstance(parsed_result, dict) else result)
        if navigation_state is not None and _is_infrastructure_tool_error(tool_error_text):
            navigation_state["had_infra_error"] = True
            navigation_state["infra_error_messages"] = list(navigation_state.get("infra_error_messages", [])) + [tool_error_text]
        if action_log is not None:
            action_log["tool_error_count"] = int(action_log.get("tool_error_count", 0)) + 1
        _record_failure_event(
            action_log,
            kind="tool_result_failure",
            origin="model_action",
            turn=turn,
            tool=tool_name,
            reason=tool_error_text,
        )
        trace_item["failure_origin"] = "real_infrastructure" if _is_infrastructure_tool_error(tool_error_text) else "model_action"
        if world_runtime is not None and world_event is None:
            transformed = world_runtime.record_real_infrastructure_failure(tool_name, arguments, tool_error_text)
            trace_item["world_event"] = transformed.world_event.to_dict()
            trace_item["bayes_supervision"] = transformed.hidden_supervision.to_dict()
        fallback = _tool_failure_fallback(
            tool_name,
            arguments,
            tool_error_text,
            navigation_state,
        )
        if fallback is not None:
            if execution_trace is not None:
                execution_trace[-1]["recovery_observation"] = True
                execution_trace[-1]["recovered_with_fallback"] = True
            return fallback, False
        _set_terminal_status(action_log, "tool_error")
        return "", True

    if world_event is not None and belief_runtime is not None:
        surprise = belief_runtime.predictive_surprise(world_event)
        world_event = replace(world_event, predictive_surprise=surprise)
        if execution_trace is not None:
            execution_trace[-1]["world_event"] = world_event.to_dict()
        belief_runtime.update(
            tool_name,
            observed_result,
            world_event,
            task_state=TaskStateView.from_navigation_state(navigation_state or {}) if TaskStateView else None,
            hidden_supervision=hidden_supervision.to_dict() if hidden_supervision is not None else None,
        )
        level = belief_runtime.detect_reopen_level(world_event)
        if level and bool(getattr(getattr(decision_controller, "config", None), "use_reopen", True)):
            reopen = belief_runtime.reopen(level, cause_tools=[tool_name], surprise=surprise)
            if execution_trace is not None:
                execution_trace[-1]["reopen"] = reopen
    limited_result = _limit_tool_result(observed_result, int(TOOL_CONFIGS.get("max_obs_chars", 8192)))
    if navigation_state is not None:
        limited_result = f"{limited_result}\n{_navigation_status_text(navigation_state)}"
    if belief_runtime is not None:
        limited_result = _bayestool_append_observation(
            limited_result,
            belief_runtime,
            navigation_state or {},
            tool_budget=budget_after_observation,
            tokenizer=bayes_prompt_tokenizer,
        )
    next_obs = f"<interpreter>\nTool: {tool_name}\n{limited_result}\n</interpreter>"
    return next_obs, False


async def _generate_bayestool_meta_episode(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    evaluation: bool = False,
) -> list[Sample]:
    """Run each meta question as a Sample while carrying session belief.

    The model still emits one strict final action per question.  The wrapper
    only shares the world identity, clean-result cache, and persistent
    session/shared posterior; navigation, evidence, and task hidden state are
    rebuilt for every question.
    """

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_questions = metadata.get("meta_questions") or []
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_questions):
        if isinstance(raw, str):
            records.append({"prompt": raw, "id": index, "metadata": {}})
        elif isinstance(raw, dict):
            record = dict(raw)
            record.setdefault("id", index)
            record.setdefault("metadata", {})
            records.append(record)
    if len(records) < 2 or build_meta_episode is None:
        child_metadata = dict(metadata)
        child_metadata.pop("meta_questions", None)
        child = deepcopy(sample)
        child.metadata = child_metadata
        return [await generate(args, child, sampling_params, evaluation=evaluation)]

    config = config_from_args(args, enabled=True) if config_from_args is not None else None
    persistent_session_belief = bool(
        getattr(config, "use_persistent_session_belief", True)
    )
    episode = build_meta_episode(records, config=config, seed=int(sample.index or 0))
    base_index = int(sample.index or 0)
    document_hash_value = str(metadata.get("document_hash") or "")
    coupling_id = str(metadata.get("coupling_id") or f"meta-coupling-{episode.meta_episode_id}")
    session_record: dict[str, Any] | None = None
    outputs: list[Sample] = []

    meta_world_key = f"{episode.meta_episode_id}:{base_index}"
    meta_world_type: str | None = None
    if str(getattr(config, "stage", "c") or "c").casefold() == "d":
        meta_world_type = (
            "healthy"
            if int(stable_seed(episode.meta_episode_id, base_index, "meta-world-type") % 2) == 0
            else "abrupt_change"
        ) if stable_seed is not None else None
    try:
        for question_index, question in enumerate(episode.questions):
            child = deepcopy(sample)
            child.index = base_index * len(episode.questions) + question_index
            child.status = Sample.Status.PENDING
            child.tokens = []
            child.response = ""
            child.response_length = 0
            child.loss_mask = None
            child.rollout_log_probs = []
            child.reward = None
            child.metadata = deepcopy(metadata)
            child.metadata.pop("meta_questions", None)
            child.metadata.update(question.metadata)
            child.metadata.update(
                {
                    "meta_episode_id": episode.meta_episode_id,
                    "episode_content_id": episode.episode_content_id,
                    "meta_trajectory_id": f"{episode.meta_episode_id}:q{question_index}",
                    "meta_question_index": question_index,
                    "meta_question_count": len(episode.questions),
                    "meta_world_sample_index": base_index,
                    "coupling_id": coupling_id,
                    "document_hash": document_hash_value or child.metadata.get("document_hash", ""),
                    "meta_episode_child": True,
                    "bayestool_meta_world_key": meta_world_key,
                }
            )
            if meta_world_type is not None:
                child.metadata["world_type"] = meta_world_type
            if persistent_session_belief and session_record is not None:
                child.metadata["bayestool_session_belief"] = deepcopy(session_record)
            child.prompt = question.prompt
            if question.label is not None:
                child.label = question.label
            child_sampling_params = sampling_params.copy()
            if isinstance(child_sampling_params.get("sampling_seed"), int):
                child_sampling_params["sampling_seed"] += question_index
            result = await generate(args, child, child_sampling_params, evaluation=evaluation)
            result_samples = result if isinstance(result, list) else [result]
            if not result_samples or not all(isinstance(item, Sample) for item in result_samples):
                raise TypeError("BayesTool meta episode child generation returned a non-Sample result")
            primary_result = result_samples[0]
            for result_sample in result_samples:
                result_sample.metadata = result_sample.metadata if isinstance(result_sample.metadata, dict) else {}
                result_sample.metadata["meta_episode_id"] = episode.meta_episode_id
                result_sample.metadata["meta_question_index"] = question_index
                result_sample.metadata["meta_question_count"] = len(episode.questions)
                result_sample.metadata["meta_world_sample_index"] = base_index
                result_sample.metadata["coupling_id"] = coupling_id
                result_sample.metadata["bayestool_meta_world_key"] = meta_world_key
                outputs.append(result_sample)
            session_record = (
                deepcopy(primary_result.metadata.get("bayestool_session_belief"))
                if persistent_session_belief
                else None
            )
    finally:
        _BAYES_META_WORLD_RUNTIMES.pop(meta_world_key, None)
    return outputs


def _image_token_count(tokenizer: Any, processor: Any, token_ids: list[int]) -> int:
    candidate_ids: set[int] = set()
    for owner in (processor, tokenizer):
        for attr in ("image_token_id", "image_token_index"):
            value = getattr(owner, attr, None)
            if isinstance(value, int) and value >= 0:
                candidate_ids.add(value)
        convert = getattr(owner, "convert_tokens_to_ids", None)
        if callable(convert):
            for token in ("<|image_pad|>", "<image>", "<|image|>"):
                try:
                    value = convert(token)
                except Exception:
                    continue
                if isinstance(value, int) and value >= 0:
                    candidate_ids.add(value)
    return sum(1 for token_id in token_ids if token_id in candidate_ids)


def _merge_multimodal_train_inputs(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not chunks:
        return None
    try:
        import torch
    except ImportError:
        return None
    values_by_key: dict[str, list[Any]] = {}
    for chunk in chunks:
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                values_by_key.setdefault(key, []).append(value)
    merged: dict[str, Any] = {}
    for key, values in values_by_key.items():
        try:
            merged[key] = torch.cat(values, dim=0)
        except (RuntimeError, TypeError):
            # A backend-specific scalar/grid field may not be concatenable;
            # dropping it is safer than passing a misaligned tensor to RL.
            continue
    return merged or None


def _resolve_training_sequence_limit(args: Any) -> int:
    """Return the hard sequence limit used by the VLM actor during training.

    The generation context is compacted independently from the training
    trajectory.  Without an explicit training limit, a successful multi-turn
    tool rollout can retain every observation and grow well beyond the
    inference context, making one dynamic microbatch require more memory than
    an 80-GiB GPU has.  Keep the real-data default conservative and allow an
    explicit override for experiments that have a larger memory budget.
    """

    configured = os.environ.get("OPENCLAW_BAYESTOOL_MAX_TRAIN_SEQUENCE_LENGTH")
    if configured:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError(
                "OPENCLAW_BAYESTOOL_MAX_TRAIN_SEQUENCE_LENGTH must be an integer"
            ) from exc
        if value <= 0:
            raise ValueError("OPENCLAW_BAYESTOOL_MAX_TRAIN_SEQUENCE_LENGTH must be positive")
        return value

    rollout_context = getattr(args, "rollout_max_context_len", None)
    if rollout_context is not None:
        try:
            return max(1024, min(8192, int(rollout_context)))
        except (TypeError, ValueError):
            pass
    return 8192


def _cap_training_trajectory(
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float] | None,
    context_segments: list[dict[str, Any]],
    multimodal_train_inputs_buffer: list[dict[str, Any]],
    step_action_spans: list[dict[str, int]],
    *,
    max_sequence_length: int,
) -> tuple[list[int], list[int], list[float] | None, list[dict[str, Any]], dict[str, Any]]:
    """Bound the sequence sent to the actor while preserving RL alignment.

    The reward and diagnostic response remain complete.  Only the actor input
    is shortened, at an assistant-action boundary whenever possible, so
    tokens, loss masks, rollout log-probabilities, and visual tensors continue
    to refer to the same suffix.  The final assistant action is always
    preferred; dropping it would turn a successful rollout into an
    untrainable prefix-only sample.
    """

    original_response_length = len(response_token_ids)
    original_total_length = len(prompt_token_ids) + original_response_length
    limit = max(1, int(max_sequence_length))
    response_budget = max(1, limit - len(prompt_token_ids))

    def _valid_span(value: Any) -> tuple[int, int] | None:
        if not isinstance(value, dict):
            return None
        try:
            start = int(value.get("action_token_start", value.get("token_start", -1)))
            end = int(value.get("response_token_end", value.get("token_end", -1)))
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start or end > original_response_length:
            return None
        return start, end

    # A complete suffix beginning at one of the retained action boundaries is
    # the least destructive compaction.  ``context_segments`` is the bounded
    # inference history, so it also prevents re-attaching images belonging to
    # observations that were already removed from the model context.
    candidate_starts: set[int] = set()
    for segment in context_segments:
        span = _valid_span(segment)
        if span is not None:
            candidate_starts.add(span[0])
    # The last action may be a terminal answer and therefore has no following
    # observation segment.  Add only that final boundary; older action spans
    # can belong to inference-compacted segments whose visual tensors are no
    # longer present in ``context_segments``.
    if step_action_spans and isinstance(step_action_spans[-1], dict):
        try:
            final_action_start = int(step_action_spans[-1].get("token_start", -1))
        except (TypeError, ValueError):
            final_action_start = -1
        if 0 <= final_action_start < original_response_length:
            candidate_starts.add(final_action_start)

    selected_start = 0
    if original_total_length > limit:
        # Walk from the oldest available boundary to the newest and keep the
        # longest suffix that fits.  This retains more useful history than
        # always keeping only the final action.
        fitting = [
            start
            for start in sorted(candidate_starts)
            if original_response_length - start <= response_budget
        ]
        if fitting:
            selected_start = fitting[0]
        else:
            selected_start = max(0, original_response_length - response_budget)

    selected_buffer: list[dict[str, Any]] = []
    retained_segment_count = 0
    retained_visual_segment_count = 0
    indexed_segments = any(
        isinstance(segment, dict) and "multimodal_train_input_index" in segment
        for segment in context_segments
    )
    selected_segment_indices: set[int] = set()
    for segment in context_segments:
        span = _valid_span(segment)
        if span is None or span[0] < selected_start:
            continue
        retained_segment_count += 1
        if segment.get("image_data"):
            retained_visual_segment_count += 1
        raw_index = segment.get("multimodal_train_input_index")
        if raw_index is None:
            continue
        try:
            buffer_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if 0 <= buffer_index < len(multimodal_train_inputs_buffer):
            selected_segment_indices.add(buffer_index)

    if original_total_length <= limit:
        # The actor sequence is unchanged, so retain every visual chunk even
        # if inference-only context compaction has already removed an older
        # segment from ``context_segments``.
        selected_buffer = list(multimodal_train_inputs_buffer)
    elif indexed_segments:
        selected_buffer = [
            multimodal_train_inputs_buffer[index]
            for index in range(len(multimodal_train_inputs_buffer))
            if index in selected_segment_indices
        ]
    else:
        # Backward-compatible path for a checkpoint created before segment
        # indices were added.  A truncated legacy trajectory cannot safely
        # remap image tensors, so keep the chunks and let the existing
        # multimodal alignment audit fail closed rather than silently train on
        # mismatched visual inputs.
        selected_buffer = list(multimodal_train_inputs_buffer)

    trimmed_response = list(response_token_ids[selected_start:])
    trimmed_masks = list(loss_masks[selected_start:])
    trimmed_log_probs = (
        list(rollout_log_probs[selected_start:]) if rollout_log_probs is not None else None
    )
    metadata = {
        "applied": bool(selected_start),
        "max_sequence_length": limit,
        "original_total_length": original_total_length,
        "original_response_length": original_response_length,
        "training_total_length": len(prompt_token_ids) + len(trimmed_response),
        "training_response_length": len(trimmed_response),
        "response_start": selected_start,
        "dropped_response_tokens": selected_start,
        "retained_segment_count": retained_segment_count,
        "retained_visual_segment_count": retained_visual_segment_count,
        "retained_multimodal_chunk_count": len(selected_buffer),
        "fallback_tail_cut": bool(
            original_total_length > limit
            and not any(start == selected_start for start in candidate_starts)
        ),
    }
    return trimmed_response, trimmed_masks, trimmed_log_probs, selected_buffer, metadata


def _compact_model_context(
    prompt_token_ids: list[int],
    segments: list[dict[str, Any]],
    *,
    max_context_length: int,
    reserve_tokens: int,
) -> tuple[list[int], list[str]]:
    """Rebuild a bounded inference context from recent action/observation pairs.

    ``sample.response`` still retains the complete trajectory for training and
    diagnostics.  The backend request only needs the task, recent tool turns,
    and the navigation/evidence summary appended to the latest observation.
    Dropping the oldest complete pair also drops its image payload, keeping
    image tokens and image data aligned instead of accumulating every page
    render until the context limit is hit.
    """

    def rebuild() -> tuple[list[int], list[str]]:
        token_ids = list(prompt_token_ids)
        image_data: list[str] = []
        for segment in segments:
            token_ids.extend(segment.get("action_token_ids", []))
            use_text_fallback = bool(segment.get("use_text_fallback"))
            if use_text_fallback and segment.get("text_observation_token_ids") is not None:
                token_ids.extend(segment.get("text_observation_token_ids", []))
            else:
                token_ids.extend(segment.get("observation_token_ids", []))
                image_data.extend(
                    value for value in segment.get("image_data", []) if isinstance(value, str)
                )
        return token_ids, image_data

    def fit_observation_to_budget(
        observation_ids: list[int],
        available: int,
        suffix_token_count: int,
    ) -> tuple[list[int], bool]:
        """Trim an observation without destroying its next-turn boundary.

        The previous implementation kept ``observation_ids[:available]``.
        That is unsafe because the observation is encoded as a complete user
        turn and its final ``<|im_end|>\n<|im_start|>assistant\n`` suffix is
        what tells the backend to start a fresh assistant action.  If the
        suffix is cut off, the next generation continues the middle of a
        JSON/path observation, which is a model-input corruption rather than
        a model protocol decision.
        """

        if len(observation_ids) <= available:
            return list(observation_ids), True
        available = max(0, int(available))
        suffix_token_count = min(
            max(0, int(suffix_token_count or 0)),
            len(observation_ids),
        )
        if suffix_token_count <= 0:
            # Synthetic/unit-test segments from older callers may not carry
            # the boundary metadata.  Preserve their historical behavior;
            # real encoded observations always provide the count below.
            return list(observation_ids[:available]), True
        if available < suffix_token_count:
            # There is no valid assistant-turn context that fits.  Returning
            # the complete observation keeps the boundary intact; the caller
            # will observe the over-budget context and terminate explicitly
            # as context_overflow instead of issuing a malformed request.
            return list(observation_ids), False
        body_count = available - suffix_token_count
        return (
            list(observation_ids[:body_count])
            + list(observation_ids[-suffix_token_count:]),
            True,
        )

    token_ids, image_data = rebuild()
    target_length = max(1, int(max_context_length) - max(32, int(reserve_tokens)))
    while len(token_ids) > target_length and len(segments) > 1:
        segments.pop(0)
        token_ids, image_data = rebuild()

    # A single rendered page can still exceed the available context because
    # vision preprocessing expands one image into thousands of placeholder
    # tokens.  Keep the textual tool observation and drop only the image from
    # the *next backend request*; the complete image remains in the training
    # trajectory and diagnostics.  This guarantees that render_page is
    # followed by another generation instead of being misclassified as a
    # context overflow.
    if len(token_ids) > target_length:
        for segment in reversed(segments):
            if segment.get("image_data") and segment.get("text_observation_token_ids") is not None:
                segment["use_text_fallback"] = True
                token_ids, image_data = rebuild()
                if len(token_ids) <= target_length:
                    break

    # Keep the prompt and the newest action intact even if a backend returns a
    # very large textual observation.  The observation is already bounded for
    # normal runs; this final guard is only a last-resort context safety net.
    if len(token_ids) > target_length and segments:
        latest = segments[-1]
        observation_key = (
            "text_observation_token_ids"
            if latest.get("use_text_fallback") and latest.get("text_observation_token_ids") is not None
            else "observation_token_ids"
        )
        action_ids = list(latest.get("action_token_ids", []))
        available = max(0, target_length - len(prompt_token_ids) - len(action_ids))
        observation_ids = list(latest.get(observation_key, []))
        if len(observation_ids) > available:
            suffix_key = (
                "text_observation_suffix_token_count"
                if observation_key == "text_observation_token_ids"
                else "observation_suffix_token_count"
            )
            fitted_observation, fits = fit_observation_to_budget(
                observation_ids,
                available,
                int(latest.get(suffix_key, 0) or 0),
            )
            latest[observation_key] = fitted_observation
            token_ids, image_data = rebuild()
            if not fits:
                # Keep the full, well-formed turn so the next loop's explicit
                # context limit check can fail closed before a backend call.
                token_ids, image_data = rebuild()
    return token_ids, image_data


def _context_image_metrics(
    tokenizer: Any,
    processor: Any,
    token_ids: list[int],
    image_data: list[str],
    segments: list[dict[str, Any]],
) -> tuple[int, int, bool, bool, int, list[str]]:
    """Return image inputs actually attached to the next backend request."""
    active_segments = [
        segment
        for segment in segments
        if not segment.get("use_text_fallback") and segment.get("image_data")
    ]
    input_count = sum(len(segment.get("image_data", [])) for segment in active_segments)
    tensor_count = sum(int(segment.get("image_tensor_count", 0) or 0) for segment in active_segments)
    token_count = sum(int(segment.get("image_token_count", 0) or 0) for segment in active_segments)
    if token_count <= 0 and image_data:
        token_count = _image_token_count(tokenizer, processor, token_ids)
    required = any(bool(segment.get("visual_input_required")) for segment in active_segments)
    consumed_paths = [
        str(path)
        for segment in active_segments
        for path in segment.get("image_paths", [])
        if isinstance(path, str)
    ]
    return (
        input_count,
        token_count,
        bool(image_data and input_count),
        bool(any(segment.get("vision_placeholder_present") for segment in active_segments)),
        input_count,
        list(dict.fromkeys(consumed_paths)),
    )


def _encode_tool_observation(
    state: GenerateState,
    observation: str,
    image_paths: list[str],
) -> tuple[list[int], str, list[str], list[Any], dict[str, Any] | None, int, int]:
    """Encode an observation as a new user turn, including real image tokens."""
    assistant_turn_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if not image_paths:
        encoded_text = f"<|im_end|>\n<|im_start|>user\n{observation}{assistant_turn_suffix}"
        token_ids = state.tokenizer(encoded_text, add_special_tokens=False)["input_ids"]
        suffix_ids = state.tokenizer(assistant_turn_suffix, add_special_tokens=False)["input_ids"]
        return token_ids, encoded_text, [], [], None, 0, len(suffix_ids)

    if state.processor is None:
        raise RuntimeError("tool returned an image but the multimodal processor is unavailable")

    try:
        from PIL import Image
        from slime.utils.processing_utils import encode_image_for_rollout_engine

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())

        image_token = str(getattr(state.processor, "image_token", "<|image_pad|>"))
        vision_start = str(getattr(state.processor, "vision_start_token", "<|vision_start|>"))
        vision_end = str(getattr(state.processor, "vision_end_token", "<|vision_end|>"))
        vision_tokens = "\n".join(f"{vision_start}{image_token}{vision_end}" for _ in images)
        encoded_text = (
            f"<|im_end|>\n<|im_start|>user\n{vision_tokens}\n{observation}"
            + assistant_turn_suffix
        )
        processor_output = state.processor(text=[encoded_text], images=images, return_tensors="pt")
        input_ids = processor_output["input_ids"][0]
        token_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        train_inputs = {
            key: value
            for key, value in processor_output.items()
            if key not in {"input_ids", "attention_mask"}
        } or None
        image_data = [encode_image_for_rollout_engine(image) for image in images]
        image_tokens_count = _image_token_count(state.tokenizer, state.processor, token_ids)
        if image_tokens_count <= 0:
            raise RuntimeError("multimodal processor produced no image token for a returned image")
        suffix_ids = state.tokenizer(assistant_turn_suffix, add_special_tokens=False)["input_ids"]
        return token_ids, encoded_text, image_data, images, train_inputs, image_tokens_count, len(suffix_ids)
    except Exception as exc:
        raise RuntimeError(f"image observation encoding failed: {exc}") from exc


async def _legacy_generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Legacy generation implementation retained only for forensic comparison.

    When ``evaluation=True`` (the dispatcher in ``slime.rollout.sglang_rollout``
    detects the kwarg via ``inspect.signature`` and forwards it for eval
    rollouts) the per-turn context budget is taken from
    ``args.eval_max_context_len`` so that eval can use a longer context window
    than training without bumping ``args.rollout_max_context_len``.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()
    tc_format = _detect_tool_call_format(state.tokenizer)
    task_prompt = _extract_task_prompt(sample.prompt)
    prompt = format_conversation_with_tools(prompt=task_prompt, tools=tool_specs, tool_call_format=tc_format)
    prompt += _get_generation_prompt_suffix(sample.prompt)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # Resolve <|im_end|> token id once for stripping later (Fix 2).
    _im_end_id: int | None = None
    try:
        _im_end_id = state.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(_im_end_id, list):
            _im_end_id = _im_end_id[0] if _im_end_id else None
    except Exception:
        pass

    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    tool_execution_trace: list[dict[str, Any]] = []
    prm_step_scores: list[float] = []
    prm_step_details: list[dict[str, Any]] = []
    prm_pending_tasks: list[tuple[int, asyncio.Task]] = []
    step_action_spans: list[dict[str, int]] = []

    if evaluation and getattr(args, "eval_max_context_len", None) is not None:
        max_context_length = args.eval_max_context_len
    elif args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = 32768

    # Reserve two turns after tool use: one gives the model an explicit
    # no-more-tools instruction, the other is a final recovery attempt.
    # Without this reserve, a rendered image can consume the last normal turn
    # and leave a rollout with observations but no final answer.
    normal_turns = TOOL_CONFIGS["max_turns"]
    for turn in range(normal_turns + 2):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        remaining_context = max_context_length - total_length
        turn_max_new_tokens = min(sampling_params["max_new_tokens"], remaining_context)
        if turn_max_new_tokens <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        turn_sampling_params = sampling_params.copy()
        turn_sampling_params["max_new_tokens"] = turn_max_new_tokens

        # Fix 1: Add </tool_call> as a stop string so the model reliably stops
        # after producing a tool call instead of continuing to hallucinate
        # results or emitting <|im_end|> which corrupts the conversation.
        existing_stop = turn_sampling_params.get("stop") or []
        if isinstance(existing_stop, str):
            existing_stop = [existing_stop]
        else:
            existing_stop = list(existing_stop)
        if "</tool_call>" not in existing_stop:
            existing_stop.append("</tool_call>")
        if "</final>" not in existing_stop:
            existing_stop.append("</final>")
        turn_sampling_params["stop"] = existing_stop

        forcing_final = turn >= normal_turns or tool_call_count >= TOOL_CONFIGS["max_tool_calls"]
        if forcing_final:
            final_reminder = (
                "\nTool use is complete. Do not call another tool. Based only on the "
                "document evidence already returned, now answer the question using exactly "
                "the supported value enclosed between <final> and </final>.\n"
            )
            reminder_ids = state.tokenizer(final_reminder, add_special_tokens=False)["input_ids"]
            if len(prompt_tokens_ids) + len(response_token_ids) + len(reminder_ids) < max_context_length:
                response += final_reminder
                response_token_ids += reminder_ids
                loss_masks += [0] * len(reminder_ids)
                if sample.rollout_log_probs is not None:
                    sample.rollout_log_probs += [0.0] * len(reminder_ids)

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": turn_sampling_params,
            "return_logprob": True,
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<interpreter>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

            # Fix 2: Strip trailing <|im_end|> token.  With no_stop_trim=True
            # the stop token stays in the output, and a mid-stream <|im_end|>
            # makes the model emit empty responses on subsequent turns.
            if (
                _im_end_id is not None
                and cur_response_token_ids
                and cur_response_token_ids[-1] == _im_end_id
            ):
                cur_response_token_ids = cur_response_token_ids[:-1]
                cur_log_probs = cur_log_probs[:-1]

            cur_response = state.tokenizer.decode(cur_response_token_ids)
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

        else:
            cur_response = output["text"]
            cur_response = postprocess_responses(cur_response)
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        action_token_start = len(response_token_ids)
        response += cur_response
        response_token_ids += cur_response_token_ids
        action_token_end = len(response_token_ids)
        loss_masks += [1] * len(cur_response_token_ids)
        step_action_spans.append(
            {
                "step_index": turn,
                "token_start": action_token_start,
                "token_end": action_token_end,
            }
        )

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        trace_count_before = len(tool_execution_trace)
        # Recovery turns must terminate with an answer.  Do not execute a new
        # tool call there, otherwise image rendering can endlessly consume the
        # final-answer budget.
        if forcing_final:
            done = _find_last_final_span(cur_response) is not None
            next_obs = "" if done else (
                "\nA final answer is still required. Do not call tools; reply only as "
                "the supported value enclosed between <final> and </final>.\n"
            )
        else:
            next_obs, done = await execute_predictions(cur_response, execution_trace=tool_execution_trace)

        if getattr(args, "prm_enable", False):
            # Run PRM for every action step, including the final "Answer" step.
            # Include next_obs in history when available for better context.
            history_for_prm = response + (next_obs if next_obs else "")
            prm_pending_tasks.append(
                (
                    turn,
                    asyncio.create_task(
                        _judge_step_with_prm(
                            args,
                            sample,
                            step_index=turn,
                            action=cur_response,
                            observation=next_obs,
                            history=history_for_prm,
                        )
                    ),
                )
            )

        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if len(tool_execution_trace) > trace_count_before:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        # Check if maximum tool call count reached
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            continue

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count
    sample.valid_tool_call_count = sum(1 for item in tool_execution_trace if item.get("success"))
    sample.tool_error_count = sum(1 for item in tool_execution_trace if not item.get("success"))
    sample.tool_execution_trace = tool_execution_trace
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["tool_execution"] = {
        "calls": tool_execution_trace,
        "call_count": sample.tool_call_count,
        "valid_call_count": sample.valid_tool_call_count,
        "error_count": sample.tool_error_count,
        "unique_tools": sorted({item["tool"] for item in tool_execution_trace if item.get("tool")}),
    }

    # Save PRM step-wise judge traces for reward composition and debugging.
    if getattr(args, "prm_enable", False):
        if prm_pending_tasks:
            done = await asyncio.gather(*[task for _, task in prm_pending_tasks], return_exceptions=True)
            for (step_idx, _), result in zip(prm_pending_tasks, done, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"PRM step task failed at step={step_idx}: {result}")
                    prm_step_details.append(
                        {"status": "exception", "step_index": step_idx, "scores": [0], "mean_score": 0.0, "votes": []}
                    )
                    prm_step_scores.append(0.0)
                    continue
                result["step_index"] = step_idx
                prm_step_details.append(result)
            prm_step_details.sort(key=lambda x: x.get("step_index", 10**9))
            prm_step_scores = [float(item.get("mean_score", 0.0)) for item in prm_step_details]

        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
        }

    # Save step-wise token spans and aligned PRM scores for downstream token-level training.
    if sample.metadata is None:
        sample.metadata = {}
    prm_score_by_step: dict[int, float] = {}
    if isinstance(sample.metadata.get("prm"), dict):
        for item in sample.metadata["prm"].get("step_details", []):
            if isinstance(item, dict) and "step_index" in item:
                prm_score_by_step[int(item["step_index"])] = float(item.get("mean_score", 0.0))
    step_wise_steps = []
    for span in step_action_spans:
        step_idx = int(span["step_index"])
        step_wise_steps.append(
            {
                "step_index": step_idx,
                "token_start": int(span["token_start"]),
                "token_end": int(span["token_end"]),
                "prm_score": float(prm_score_by_step.get(step_idx, 0.0)),
            }
        )
    sample.metadata["step_wise"] = {
        "steps": step_wise_steps,
        "step_token_spans": [[item["token_start"], item["token_end"]] for item in step_wise_steps],
        "step_scores": [item["prm_score"] for item in step_wise_steps],
    }

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


_INFRA_STATUSES = {
    "generation_empty",
    "generation_error",
    "context_overflow",
    "tool_error",
    "infra_error",
}


def _is_world_injected_observation(trace_item: Mapping[str, Any] | None) -> bool:
    """Return whether a failed tool call still delivered a valid world observation."""

    if not isinstance(trace_item, Mapping):
        return False
    if str(trace_item.get("failure_origin") or "") == "world_injected":
        return True
    world_event = trace_item.get("world_event")
    return isinstance(world_event, Mapping) and str(
        world_event.get("failure_origin") or ""
    ) == "world_injected"


def _set_rollout_status(sample: Sample, status: str, *, reason: str | None = None) -> None:
    """Persist the rollout state in both runtime fields and serialized metadata."""
    valid_for_rl = status not in _INFRA_STATUSES
    sample.rollout_status = status
    sample.valid_for_rl = valid_for_rl
    if status == "context_overflow":
        sample.status = Sample.Status.TRUNCATED
    elif status in {"completed", "abstained"}:
        sample.status = Sample.Status.COMPLETED
    elif status != "completed":
        sample.status = Sample.Status.FAILED
    if not valid_for_rl:
        sample.remove_sample = True
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["rollout_status"] = status
    sample.metadata["valid_for_rl"] = valid_for_rl
    sample.metadata["exclude_from_group_statistics"] = not valid_for_rl
    if reason:
        sample.metadata["rollout_status_reason"] = reason


def _finish_reason_type(output: dict[str, Any]) -> str | None:
    meta_info = output.get("meta_info") or {}
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        value = finish_reason.get("type")
    else:
        value = finish_reason
    return str(value) if value is not None else None


def _extract_generation_output(
    output: dict[str, Any],
    state: GenerateState,
    im_end_id: int | None,
) -> dict[str, Any]:
    """Extract model text without losing backend/stop-token diagnostics.

    SGLang can return an empty ``output_token_logprobs`` list while still
    providing text.  Treating the mere presence of that key as authoritative
    silently converted valid generations into empty rollouts.  Prefer valid
    token/logprob pairs, then fall back to the backend text and record which
    representation was used.
    """
    meta_info = output.get("meta_info") or {}
    logprob_items = meta_info.get("output_token_logprobs")
    raw_token_ids: list[int] = []
    raw_log_probs: list[float] = []
    malformed_logprob_count = 0
    if isinstance(logprob_items, list):
        for item in logprob_items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                malformed_logprob_count += 1
                continue
            try:
                raw_log_probs.append(float(item[0]))
                raw_token_ids.append(int(item[1]))
            except (TypeError, ValueError):
                malformed_logprob_count += 1

    backend_text = str(output.get("text") or output.get("output_text") or "")
    stop_token_removed_count = 0
    output_source = "text"
    backend_raw_text = backend_text

    if raw_token_ids:
        output_source = "output_token_logprobs"
        try:
            backend_raw_text = state.tokenizer.decode(raw_token_ids)
        except Exception:
            backend_raw_text = ""
        cur_token_ids = list(raw_token_ids)
        cur_log_probs = list(raw_log_probs)
        if im_end_id is not None:
            while cur_token_ids and cur_token_ids[-1] == im_end_id:
                cur_token_ids.pop()
                cur_log_probs.pop()
                stop_token_removed_count += 1
        try:
            cur_text = state.tokenizer.decode(cur_token_ids)
        except Exception:
            cur_text = ""
        # A text field is a safer recovery when the backend's logprob stream
        # only contains the stop token or has become misaligned.
        if not cur_text and backend_text.strip():
            output_source = "text_fallback_after_token_stream"
            cur_text = backend_text
            cur_token_ids = list(state.tokenizer(cur_text, add_special_tokens=False)["input_ids"])
            cur_log_probs = [0.0] * len(cur_token_ids)
    else:
        cur_text = backend_text
        cur_token_ids = list(state.tokenizer(cur_text, add_special_tokens=False)["input_ids"]) if cur_text else []
        cur_log_probs = [0.0] * len(cur_token_ids)
        if isinstance(logprob_items, list):
            output_source = "text_fallback_empty_token_stream"

    return {
        "raw_generation_text": cur_text,
        "backend_raw_generation_text": backend_raw_text,
        "backend_output_token_count": len(raw_token_ids) if isinstance(logprob_items, list) else len(cur_token_ids),
        "output_token_count": len(cur_token_ids),
        "stop_token_removed_count": stop_token_removed_count,
        "logprob_item_count": len(logprob_items) if isinstance(logprob_items, list) else 0,
        "malformed_logprob_count": malformed_logprob_count,
        "output_source": output_source,
        "token_ids": cur_token_ids,
        "log_probs": cur_log_probs,
    }


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample | list[Sample]:
    """Generate a strict, multi-turn document-tool rollout.

    Tool observations are appended as separate user turns.  When a tool
    returns an image, the observation is processed with the model processor,
    the expanded image placeholder tokens are appended to the context, and
    the accumulated base64 image data is sent on the next generation request.
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout is not supported for this function."

    if (
        isinstance(sample.metadata, dict)
        and sample.metadata.get("meta_questions")
        and not sample.metadata.get("meta_episode_child")
        and _bayestool_meta_enabled(args, sample.metadata)
    ):
        return await _generate_bayestool_meta_episode(
            args,
            sample,
            sampling_params,
            evaluation=evaluation,
        )

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_specs = tool_registry.get_tool_specs()
    tc_format = _detect_tool_call_format(state.tokenizer)
    task_prompt = _extract_task_prompt(sample.prompt)
    # Answer page/bbox are training/evaluation metadata only.  They are kept
    # outside navigation_state so they cannot leak into the prompt or alter
    # online action selection.
    diagnostic_metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    branch_resume_id = diagnostic_metadata.get("bayestool_branch_resume_id")
    branch_resume = (
        _BAYES_BRANCH_CHECKPOINTS.pop(str(branch_resume_id), None)
        if branch_resume_id
        else None
    )
    branch_child = bool(diagnostic_metadata.get("bayestool_branch_child"))
    diagnostic_answer_page = diagnostic_metadata.get("answer_page", diagnostic_metadata.get("target_page"))
    try:
        diagnostic_answer_page = int(diagnostic_answer_page) if diagnostic_answer_page is not None else None
    except (TypeError, ValueError):
        diagnostic_answer_page = None
    diagnostic_answer_bbox = diagnostic_metadata.get("answer_bbox")
    navigation_state = _new_navigation_state(task_prompt)
    bayestool_enabled = _bayestool_is_enabled(args, diagnostic_metadata)
    bayestool_config = config_from_args(args, enabled=True) if bayestool_enabled and config_from_args else None
    default_tool_budget = max(1, int(TOOL_CONFIGS.get("max_tool_calls", 8)))
    configured_bayes_budget = getattr(args, "bayestool_tool_budget", None)
    bayestool_tool_budget = (
        max(1, int(configured_bayes_budget))
        if bayestool_enabled and configured_bayes_budget is not None
        else default_tool_budget
    )
    world_runtime = None
    belief_runtime = None
    decision_controller = None
    bayes_document_digest = None
    bayes_coupling_id = None
    bayes_content_signature = None
    bayestool_filter_model = None
    bayestool_filter_version = str(getattr(args, "belief_model_version", "untrained") or "untrained")
    bayestool_q_head = None
    bayestool_q_version = "untrained"
    bayestool_risk_calibrator = None
    bayestool_risk_version = "heuristic"
    bayes_question_plan = None
    bayes_runtime_state_digest = ""
    world_sample_index = int(
        diagnostic_metadata.get("meta_world_sample_index", getattr(sample, "index", 0) or 0)
    )
    bayes_realization_index: int | None = None
    if bayestool_enabled:
        if validate_stage_capabilities is not None:
            diagnostic_metadata["bayestool_capability_manifest"] = validate_stage_capabilities(
                str(getattr(args, "bayestool_stage", "c") or "c"),
                belief_checkpoint=getattr(args, "bayestool_belief_checkpoint", None),
                q_checkpoint=getattr(args, "bayestool_q_checkpoint", None),
                risk_checkpoint=getattr(args, "bayestool_risk_checkpoint", None),
                meta_manifest=getattr(args, "bayestool_meta_manifest", None),
                allow_heuristic_belief=bool(getattr(args, "bayestool_allow_heuristic_belief", False)),
                allow_heuristic_q=bool(getattr(args, "bayestool_allow_heuristic_q", False)),
                allow_heuristic_risk=bool(getattr(args, "bayestool_allow_heuristic_risk", False)),
            )
        (
            bayestool_filter_model,
            bayestool_filter_version,
            bayestool_q_head,
            bayestool_q_version,
        ) = _load_bayestool_models(args, bayestool_config)
        bayestool_risk_calibrator, bayestool_risk_version = _load_bayestool_risk_calibrator(args)
        bayes_document_digest, bayes_coupling_id = _bayestool_runtime_identity(sample, task_prompt)
        if QuestionRolloutPlan is not None and make_question_rollout_plan is not None:
            raw_plan = diagnostic_metadata.get("question_rollout_plan")
            if isinstance(raw_plan, Mapping):
                bayes_question_plan = QuestionRolloutPlan.from_mapping(raw_plan)
            else:
                question_id = (
                    bayestool_question_id(diagnostic_metadata, default=bayes_coupling_id)
                    if bayestool_question_id is not None
                    else str(bayes_coupling_id)
                )
                configured_primary_count = int(getattr(args, "n_samples_per_prompt", 4) or 4)
                if not 4 <= configured_primary_count <= 6:
                    if evaluation:
                        configured_primary_count = 4
                    else:
                        raise ValueError(
                            "BayesTool raw dataset fallback requires n_samples_per_prompt in [4, 6]; "
                            f"got {configured_primary_count}"
                        )
                bayes_question_plan = make_question_rollout_plan(
                    question_id,
                    policy_version=str(
                        getattr(bayestool_config, "policy_version", DEFAULT_POLICY_VERSION)
                    ),
                    seed=str(diagnostic_metadata.get("rollout_id", 0)),
                    group_size=int(getattr(bayestool_config, "default_group_size", 4)),
                    realization_count=configured_primary_count,
                )
                diagnostic_metadata["question_rollout_plan"] = bayes_question_plan.to_dict()
                diagnostic_metadata["question_id"] = question_id
            if bayes_question_plan is not None:
                # The data source allocates globally increasing sample.index
                # values, but n_samples_per_prompt is now the number of
                # primary realization trajectories for each prompt.  Recover
                # the prompt-local realization index before constructing the
                # world; K-1 siblings are expanded from its checkpoint later.
                if branch_resume is None:
                    configured_primary_count = getattr(args, "n_samples_per_prompt", None)
                    if (
                        not evaluation
                        and configured_primary_count is not None
                        and int(configured_primary_count) > 0
                        and int(configured_primary_count) != bayes_question_plan.group_count
                    ):
                        raise ValueError(
                            "BayesTool n_samples_per_prompt must equal the explicit "
                            "QuestionRolloutPlan realization count; legacy world/replica "
                            f"settings cannot remap it ({configured_primary_count}!="
                            f"{bayes_question_plan.group_count})"
                        )
                    primary_count = int(
                        getattr(args, "n_samples_per_prompt", bayes_question_plan.group_count)
                        or bayes_question_plan.group_count
                    )
                    primary_count = max(1, primary_count)
                    realization_source = diagnostic_metadata.get(
                        "bayestool_realization_index", world_sample_index
                    )
                    bayes_realization_index = int(realization_source) % primary_count
                else:
                    realization_source = diagnostic_metadata.get(
                        "bayestool_realization_index",
                        branch_resume.get("world_sample_index", world_sample_index),
                    )
                    bayes_realization_index = int(realization_source) % bayes_question_plan.group_count
                if bayes_realization_index >= bayes_question_plan.group_count:
                    raise ValueError(
                        "BayesTool primary realization index exceeds the explicit question plan: "
                        f"index={bayes_realization_index} groups={bayes_question_plan.group_count}"
                    )
                diagnostic_metadata["bayestool_realization_index"] = bayes_realization_index
        bayes_output_root = diagnostic_metadata.get("bayestool_output_root") or os.getenv("OPENCLAW_TOOL_OUTPUT_DIR")
        if _BAYES_CLEAN_RESULT_CACHE is not None and bayes_output_root:
            _BAYES_CLEAN_RESULT_CACHE.set_root(
                Path(str(bayes_output_root)) / "tool_outputs" / "bayestool" / "clean_cache"
            )
        meta_world_key = diagnostic_metadata.get("bayestool_meta_world_key")
        if meta_world_key and str(meta_world_key) in _BAYES_META_WORLD_RUNTIMES:
            world_runtime = _BAYES_META_WORLD_RUNTIMES[str(meta_world_key)]
        else:
            raw_sampling_context = diagnostic_metadata.get("world_sampling_context")
            if isinstance(raw_sampling_context, dict):
                world_sampling_context = dict(raw_sampling_context)
            else:
                world_sampling_context = {}
            # ``page_count`` and the argument capabilities are public task
            # metadata.  Carry them into the sampler even for the ordinary
            # preprocessed document-qa dataset, which does not have a
            # pre-built BayesTool manifest.
            if world_sampling_context.get("page_count") is None:
                world_sampling_context["page_count"] = diagnostic_metadata.get(
                    "page_count", diagnostic_metadata.get("num_pages")
                )
            if not world_sampling_context.get("tool_argument_capabilities"):
                world_sampling_context["tool_argument_capabilities"] = {
                    str(name): sorted(str(argument) for argument in arguments)
                    for name, arguments in DEFAULT_TOOL_ARGUMENT_CAPABILITIES.items()
                }
            world_sampling_context["tool_budget"] = bayestool_tool_budget
            if bayes_question_plan is not None and not bayes_question_plan.latent_ids_finalized:
                # Raw datasets have no prebuilt world manifest.  Freeze all
                # realization worlds before constructing the first runtime so
                # every primary and continuation uses one authoritative latent
                # identity rather than a synthetic planning ID.
                raw_fixed_specs = diagnostic_metadata.get("fixed_world_specs")
                if isinstance(raw_fixed_specs, list) and len(raw_fixed_specs) == bayes_question_plan.group_count:
                    frozen_specs = list(raw_fixed_specs)
                else:
                    if sample_tool_world is None:
                        raise RuntimeError("BayesTool world sampler is unavailable for raw-plan materialization")
                    frozen_specs = [
                        sample_tool_world(
                            bayes_coupling_id,
                            world_slot=slot,
                            replica_id=0,
                            world_slot_role=realization.world_slot_role,
                            variant_id=realization.variant_id,
                            rollout_id=diagnostic_metadata.get("rollout_id", 0),
                            config=bayestool_config,
                            sampling_context=world_sampling_context,
                            tool_budget=bayestool_tool_budget,
                        ).to_dict()
                        for slot, realization in enumerate(bayes_question_plan.realizations)
                    ]
                finalized_realizations = []
                for realization, frozen_spec in zip(
                    bayes_question_plan.realizations,
                    frozen_specs,
                    strict=True,
                ):
                    latent_world_id = (
                        frozen_spec.get("latent_world_id")
                        if isinstance(frozen_spec, Mapping)
                        else getattr(frozen_spec, "latent_world_id", "")
                    )
                    if not str(latent_world_id).strip():
                        raise ValueError(
                            "raw BayesTool world materialization produced an empty latent_world_id"
                        )
                    finalized_realizations.append(
                        replace(realization, latent_world_id=str(latent_world_id))
                    )
                bayes_question_plan = replace(
                    bayes_question_plan,
                    realizations=tuple(finalized_realizations),
                    latent_ids_finalized=True,
                )
                diagnostic_metadata["fixed_world_specs"] = [
                    spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
                    for spec in frozen_specs
                ]
                diagnostic_metadata["question_rollout_plan"] = bayes_question_plan.to_dict()
            world_runtime = WorldRuntime.for_sample(
                coupling_id=bayes_coupling_id,
                sample_index=world_sample_index,
                rollout_id=diagnostic_metadata.get("rollout_id", 0),
                config=bayestool_config,
                output_root=bayes_output_root,
                world_type=diagnostic_metadata.get("world_type"),
                document_digest=bayes_document_digest,
                clean_cache=_BAYES_CLEAN_RESULT_CACHE,
                sampling_context=world_sampling_context,
                tool_budget=bayestool_tool_budget,
                question_rollout_plan=bayes_question_plan,
                realization_index=bayes_realization_index,
                fixed_world_specs=(
                    diagnostic_metadata.get("fixed_world_specs")
                    if isinstance(diagnostic_metadata.get("fixed_world_specs"), list)
                    else None
                ),
            )
            if bayes_question_plan is not None and world_runtime is not None:
                realization = bayes_question_plan.realizations[int(world_runtime.spec.world_slot)]
                diagnostic_metadata["world_slot_role"] = realization.world_slot_role
                diagnostic_metadata["variant_id"] = realization.variant_id
                diagnostic_metadata["decision_group_size"] = int(realization.k)
        if make_runtime_state_digest is not None and world_runtime is not None:
            bayes_runtime_state_digest = make_runtime_state_digest(
                {
                    "remaining_tool_budget": bayestool_tool_budget,
                    "tool_call_count": int(getattr(world_runtime, "call_count", 0)),
                    "world_schedule": world_runtime.schedule_metadata(),
                    "world_call_counters": world_runtime.public_event_metadata(),
                    "restore_state_id": "root",
                    "branch_horizon": int(getattr(bayestool_config, "branch_horizon", 0)),
                    "bootstrap": "terminal_reward",
                }
            )
            diagnostic_metadata["runtime_state_digest"] = bayes_runtime_state_digest
            diagnostic_metadata["bayes_runtime_state_digest"] = bayes_runtime_state_digest
        if meta_world_key:
            _BAYES_META_WORLD_RUNTIMES[str(meta_world_key)] = world_runtime
        session_record = diagnostic_metadata.get("bayestool_session_belief")
        if isinstance(session_record, dict) and hasattr(BeliefRuntime, "from_replay_record"):
            belief_runtime = BeliefRuntime.from_replay_record(
                session_record,
                bayestool_config,
                document_digest=bayes_document_digest,
                seed=world_sample_index,
                model=bayestool_filter_model,
                model_version=bayestool_filter_version,
            )
        else:
            belief_runtime = BeliefRuntime(
                bayestool_config,
                document_digest=bayes_document_digest,
                model=bayestool_filter_model,
                model_version=bayestool_filter_version,
                seed=world_sample_index,
            )
        decision_controller = DecisionController(
            bayestool_config,
            q_head=bayestool_q_head,
            risk_calibrator=bayestool_risk_calibrator,
            seed=int(sample.index or 0),
        )
        if bayestool_q_head is not None:
            decision_controller.enable_q_head(True)
        navigation_state["remaining_tool_budget"] = bayestool_tool_budget
        navigation_state["bayestool_enabled"] = True
        if content_signature is not None:
            bayes_content_signature = content_signature(
                {"question_type": navigation_state.get("question_type", "text")},
                question=task_prompt,
            )

    if branch_resume is not None:
        # The branch starts from the exact pre-action state.  Runtime objects
        # are copied in the checkpoint and therefore only this child's world
        # events/belief updates are mutable.  The clean-result cache remains
        # shared by _capture/_clone_bayestool_branch_checkpoint.
        navigation_state = _branch_deepcopy(branch_resume["navigation_state"])
        world_runtime = branch_resume.get("world_runtime")
        belief_runtime = branch_resume.get("belief_runtime")
        bayes_document_digest = branch_resume.get("bayes_document_digest") or bayes_document_digest
        bayes_coupling_id = branch_resume.get("bayes_coupling_id") or bayes_coupling_id
        bayes_content_signature = diagnostic_metadata.get("bayes_content_signature") or bayes_content_signature
        world_sample_index = int(branch_resume.get("world_sample_index", world_sample_index))
        frozen_resume_digest = str(
            branch_resume.get("runtime_state_digest")
            or diagnostic_metadata.get("runtime_state_digest")
            or ""
        )
        if frozen_resume_digest:
            # The child may have constructed a temporary root runtime before
            # this restore block.  The parent/children group must use the
            # digest of the selected frozen checkpoint, not that temporary
            # root state.
            bayes_runtime_state_digest = frozen_resume_digest
            diagnostic_metadata["runtime_state_digest"] = frozen_resume_digest
            diagnostic_metadata["bayes_runtime_state_digest"] = frozen_resume_digest
        if bayestool_config is not None:
            bayestool_enabled = True
        navigation_state["bayestool_enabled"] = True
    # Do not carry a model-specific <think> suffix into the strict action
    # protocol.  The assistant turn must begin with its one action tag.
    if branch_resume is None:
        initial_environment = _bayestool_environment_block(
            belief_runtime,
            navigation_state,
            tool_budget=bayestool_tool_budget,
            tokenizer=state.tokenizer,
        ) if bayestool_enabled else ""
        prompt = format_conversation_with_tools(
            prompt=f"{task_prompt}\n\n{_navigation_status_text(navigation_state)}{chr(10) + initial_environment if initial_environment else ''}",
            tools=tool_specs,
            tool_call_format=tc_format,
        )
        prompt_tokens_ids = list(state.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    else:
        # Keep the original formatted prompt tokens as the training prefix;
        # rebuilding it would change the token-level shared-prefix contract.
        initial_environment = ""
        prompt = str(diagnostic_metadata.get("bayestool_branch_prompt") or "")
        prompt_tokens_ids = list(branch_resume["prompt_token_ids"])

    im_end_id: int | None = None
    try:
        converted = state.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(converted, int) and converted >= 0:
            im_end_id = converted
    except Exception:
        pass

    response = ""
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    # Keep the complete response for training, but send a bounded recent
    # action/observation context to the generation backend.
    context_token_ids = list(prompt_tokens_ids)
    context_image_data: list[str] = []
    context_segments: list[dict[str, Any]] = []
    current_images: list[Any] = []
    multimodal_train_inputs_buffer: list[dict[str, Any]] = []
    execution_trace: list[dict[str, Any]] = []
    action_log: dict[str, Any] = {
        "candidate_action_count": 0,
        "executed_action_count": 0,
        "valid_action_count": 0,
        "invalid_action_count": 0,
        "ignored_action_count": 0,
        "protocol_error_count": 0,
        "tool_error_count": 0,
        "premature_final_count": 0,
        "had_evidence_guard_recovery": False,
        "rejected_final_count": 0,
        "failure_events": [],
        "actions": [],
    }
    cost_state: dict[str, float] = {
        "latency_seconds": 0.0,
        "observation_text_tokens": 0.0,
        "image_tokens": 0.0,
        "tool_calls": 0.0,
    }
    generation_steps: list[dict[str, Any]] = []
    prm_step_scores: list[float] = []
    prm_step_details: list[dict[str, Any]] = []
    prm_pending_tasks: list[tuple[int, asyncio.Task]] = []
    step_action_spans: list[dict[str, int]] = []
    terminal_status: str | None = None
    terminal_reason: str | None = None
    start_turn = 0
    resume_forced_action: dict[str, Any] | None = None

    if branch_resume is not None:
        response = str(branch_resume.get("response", ""))
        response_token_ids = list(branch_resume.get("response_token_ids", []))
        loss_masks = list(branch_resume.get("loss_masks", []))
        context_token_ids = list(branch_resume.get("context_token_ids", prompt_tokens_ids))
        context_image_data = list(branch_resume.get("context_image_data", []))
        context_segments = _branch_deepcopy(branch_resume.get("context_segments", []))
        current_images = _branch_deepcopy(branch_resume.get("current_images", []))
        multimodal_train_inputs_buffer = _branch_deepcopy(
            branch_resume.get("multimodal_train_inputs_buffer", [])
        )
        execution_trace = _branch_deepcopy(branch_resume.get("execution_trace", []))
        action_log = _branch_deepcopy(branch_resume.get("action_log", action_log))
        generation_steps = _branch_deepcopy(branch_resume.get("generation_steps", []))
        step_action_spans = _branch_deepcopy(branch_resume.get("step_action_spans", []))
        sample.rollout_log_probs = list(branch_resume.get("rollout_log_probs", []))
        start_turn = int(branch_resume.get("turn", 0))
        resume_forced_action = _branch_deepcopy(branch_resume.get("forced_action"))

    eval_context = getattr(args, "eval_max_context_len", None)
    train_context = getattr(args, "rollout_max_context_len", None)
    if evaluation and eval_context is not None:
        max_context_length = int(eval_context)
    elif train_context is not None:
        max_context_length = int(train_context)
    else:
        max_context_length = 32768
    max_new_tokens = int(sampling_params.get("max_new_tokens") or getattr(args, "rollout_max_response_len", 2048))
    max_tool_steps = max(
        1,
        int(
            branch_resume.get("max_tool_steps", bayestool_tool_budget)
            if branch_resume is not None
            else (bayestool_tool_budget if bayestool_enabled else default_tool_budget)
        ),
    )
    max_turns = max(max_tool_steps + 2, int(TOOL_CONFIGS.get("max_turns", max_tool_steps + 2)))
    tool_call_count = 0

    if branch_resume is not None:
        tool_call_count = int(branch_resume.get("tool_call_count", 0))
        branch_horizon = max(1, int(branch_resume.get("branch_horizon", 3)))
        max_turns = min(max_turns, start_turn + branch_horizon)

    branching_allowed = bool(
        bayestool_enabled
        and not branch_child
        and branch_resume is None
        and decision_controller is not None
        and bool(getattr(bayestool_config, "use_regret_branching", True))
    )
    # Candidate sampling is a prerequisite for action selection, independent
    # of whether the later regret gate decides to fork sibling rollouts.
    candidate_sampling_allowed = bool(
        bayestool_enabled
        and not branch_child
        and branch_resume is None
        and decision_controller is not None
        and bayestool_config is not None
    )
    branch_children: list[Sample] = []
    branch_events: list[dict[str, Any]] = []
    # Primary generation may expose several eligible checkpoints.  Exactly
    # one decision node is selected after the primary trajectory completes;
    # launching children in the middle of the loop would create multiple
    # decision groups for one realization.
    deferred_branch_candidates: list[dict[str, Any]] = []
    root_branch_candidate: dict[str, Any] | None = None

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []

    for turn in range(start_turn, max_turns):
        navigation_state["search_budget_exhausted"] = tool_call_count >= max_tool_steps
        if context_image_data:
            navigation_state["visual_input_required"] = True
            navigation_state["visual_input_reason"] = navigation_state.get("visual_input_reason") or "model must inspect attached page pixels"
        elif not navigation_state.get("pending_visual_image_paths"):
            # A render may be followed by a deterministic bbox/OCR action.  In
            # that route the next model request is text-only by design.
            navigation_state["visual_input_required"] = False
            navigation_state["visual_input_reason"] = None
        input_token_count = len(context_token_ids)
        (
            image_input_count,
            image_token_count,
            vision_input_attached,
            vision_placeholder_present,
            image_tensor_count,
            consumed_image_paths,
        ) = _context_image_metrics(
            state.tokenizer,
            state.processor,
            context_token_ids,
            context_image_data,
            context_segments,
        )
        if input_token_count >= max_context_length:
            terminal_status = "context_overflow"
            terminal_reason = "input token count reached rollout context limit"
            break

        remaining_context = max_context_length - input_token_count
        turn_max_new_tokens = min(max_new_tokens, remaining_context)
        if turn_max_new_tokens <= 0:
            terminal_status = "context_overflow"
            terminal_reason = "no generation tokens remain after context accounting"
            break

        turn_sampling_params = sampling_params.copy()
        turn_sampling_params["max_new_tokens"] = turn_max_new_tokens
        existing_stop = turn_sampling_params.get("stop") or []
        existing_stop = [existing_stop] if isinstance(existing_stop, str) else list(existing_stop)
        for stop_text in ("</tool_call>", "</final>", "</abstain>"):
            if stop_text not in existing_stop:
                existing_stop.append(stop_text)
        turn_sampling_params["stop"] = existing_stop

        payload = {
            "input_ids": context_token_ids,
            "sampling_params": turn_sampling_params,
            "return_logprob": True,
        }
        if context_image_data:
            payload["image_data"] = list(context_image_data)

        step: dict[str, Any] = {
            "step_index": turn,
            "generation_called": False,
            "input_token_count": input_token_count,
            "image_input_count": image_input_count,
            "image_token_count": image_token_count,
            "vision_input_attached": vision_input_attached,
            "vision_placeholder_present": vision_placeholder_present,
            "image_tensor_count": image_tensor_count,
            "consumed_image_paths": consumed_image_paths,
            "visual_input_required": bool(
                navigation_state.get("visual_input_required") or context_image_data
            ),
            "visual_input_reason": navigation_state.get("visual_input_reason")
            or ("consume attached visual observation" if context_image_data else None),
            "remaining_generation_tokens": turn_max_new_tokens,
            "finish_reason": None,
            "raw_generation_text": "",
            "backend_raw_generation_text": "",
            "output_token_count": 0,
            "backend_output_token_count": 0,
            "stop_token_removed_count": 0,
            "output_source": None,
            "assistant_output_token_count": 0,
            "generation_error": None,
            "protocol_parse_error": None,
        }
        pre_action_checkpoint = None
        if candidate_sampling_allowed and bayestool_config is not None:
            pre_action_checkpoint = _capture_bayestool_branch_checkpoint(
                prompt_token_ids=prompt_tokens_ids,
                context_token_ids=context_token_ids,
                context_image_data=context_image_data,
                context_segments=context_segments,
                response=response,
                response_token_ids=response_token_ids,
                loss_masks=loss_masks,
                rollout_log_probs=list(sample.rollout_log_probs or []),
                current_images=current_images,
                multimodal_train_inputs_buffer=multimodal_train_inputs_buffer,
                execution_trace=execution_trace,
                action_log=action_log,
                generation_steps=generation_steps,
                step_action_spans=step_action_spans,
                navigation_state=navigation_state,
                world_runtime=world_runtime,
                belief_runtime=belief_runtime,
                bayes_document_digest=bayes_document_digest,
                bayes_coupling_id=bayes_coupling_id,
                world_sample_index=(
                    bayes_realization_index
                    if bayes_realization_index is not None
                    else world_sample_index
                ),
                turn=turn,
                tool_call_count=tool_call_count,
                max_tool_steps=max_tool_steps,
                config=bayestool_config,
                runtime_state_digest=bayes_runtime_state_digest,
            )
            if make_runtime_state_digest is not None and world_runtime is not None:
                pre_action_checkpoint["runtime_state_digest"] = make_runtime_state_digest(
                    {
                        "remaining_tool_budget": max(0, int(max_tool_steps - tool_call_count)),
                        "tool_call_count": int(getattr(world_runtime, "call_count", tool_call_count)),
                        "world_schedule": world_runtime.schedule_metadata(),
                        "world_call_counters": world_runtime.public_event_metadata(),
                        "restore_state_id": f"prefix:{pre_action_checkpoint['prefix_hash']}",
                        "branch_horizon": int(getattr(bayestool_config, "branch_horizon", 0)),
                        "bootstrap": "terminal_reward",
                    }
                )
            pre_action_checkpoint["bayes_content_signature"] = bayes_content_signature
            pre_action_checkpoint["bayes_aux_prompt"] = prompt

        try:
            # A branch candidate was already sampled by the policy from this
            # exact prefix.  Reuse its token ids/log-probs and skip a second
            # backend request for the child action; subsequent turns are
            # generated normally.
            if resume_forced_action is not None and turn == start_turn:
                step["generation_called"] = True
                step["generation_source"] = "shared_prefix_branch_policy_sample"
                cur_response = str(resume_forced_action.get("raw", ""))
                cur_response_token_ids = list(resume_forced_action.get("token_ids", []))
                cur_log_probs = list(resume_forced_action.get("log_probs", []))
                extracted = {
                    "raw_generation_text": cur_response,
                    "backend_raw_generation_text": cur_response,
                    "backend_output_token_count": len(cur_response_token_ids),
                    "output_token_count": len(cur_response_token_ids),
                    "stop_token_removed_count": 0,
                    "output_source": "shared_prefix_branch_policy_sample",
                }
                resume_forced_action = None
            else:
                # This flag is set immediately before invoking the backend.  It
                # distinguishes a skipped generation (for example context
                # overflow) from a backend call that returned no assistant text.
                step["generation_called"] = True
                output = await post(url, payload)
                if not isinstance(output, dict):
                    raise RuntimeError("generation backend returned a non-object response")
                finish_reason = _finish_reason_type(output)
                step["finish_reason"] = finish_reason
                extracted = _extract_generation_output(output, state, im_end_id)
                cur_response = str(extracted["raw_generation_text"])
                cur_response_token_ids = list(extracted["token_ids"])
                cur_log_probs = list(extracted["log_probs"])
                if not cur_response.strip():
                    # Whitespace-only and special-token-only turns are empty from
                    # the protocol's point of view even if the backend emitted a
                    # token id for them.
                    cur_response_token_ids = []
                    cur_log_probs = []
                    extracted["output_token_count"] = 0
                for key in (
                    "raw_generation_text",
                    "backend_raw_generation_text",
                    "backend_output_token_count",
                    "output_token_count",
                    "stop_token_removed_count",
                    "output_source",
                ):
                    step[key] = extracted[key]
            for key in (
                "raw_generation_text",
                "backend_raw_generation_text",
                "backend_output_token_count",
                "output_token_count",
                "stop_token_removed_count",
                "output_source",
            ):
                if key in extracted:
                    step[key] = extracted[key]
        except Exception as exc:
            step["generation_error"] = str(exc)
            generation_steps.append(step)
            terminal_status = "generation_error"
            terminal_reason = str(exc)
            break

        step["assistant_output_token_count"] = len(cur_response_token_ids)
        generation_steps.append(step)
        if not cur_response_token_ids:
            # An empty response after a successful backend call is distinct
            # from a transport exception.  Preserve the stop reason and the
            # raw backend representation so this cannot be exported as a
            # generic invalid rollout.
            if step.get("finish_reason") is None and not step.get("generation_error"):
                step["generation_error"] = "backend returned empty output without finish_reason"
            if step.get("stop_token_removed_count", 0):
                terminal_reason = "stop_token_only_output"
            elif step.get("finish_reason"):
                terminal_reason = f"backend returned zero assistant tokens (finish_reason={step['finish_reason']})"
            else:
                terminal_reason = "backend returned zero assistant tokens"
            empty_parsed = parse_assistant_action(cur_response)
            _record_action_candidate(action_log, empty_parsed, turn=turn, raw=cur_response)
            if action_log.get("actions"):
                action_log["actions"][-1].update(
                    {
                        "valid": False,
                        "executed": False,
                        "empty_generation": True,
                    }
                )
            terminal_status = "generation_empty"
            break

        pre_action_candidates: list[dict[str, Any]] | None = None
        pre_action_candidate_errors: list[str] = []
        pre_action_decision: dict[str, Any] | None = None
        if pre_action_checkpoint is not None and candidate_sampling_allowed and bayestool_config is not None:
            try:
                current_group_size = int(
                    diagnostic_metadata.get("decision_group_size")
                    or getattr(bayestool_config, "default_group_size", 4)
                    or 4
                )
                sampled_candidates, pre_action_candidate_errors = await _sample_bayestool_branch_candidates(
                    state=state,
                    url=url,
                    checkpoint=pre_action_checkpoint,
                    current_text=cur_response,
                    current_token_ids=cur_response_token_ids,
                    current_log_probs=cur_log_probs,
                    sampling_params=turn_sampling_params,
                    im_end_id=im_end_id,
                    config=bayestool_config,
                    target_group_size=current_group_size,
                )
                pre_action_candidates = sampled_candidates
                selected_candidate, _, pre_action_decision = _select_bayestool_policy_candidate(
                    sampled_candidates,
                    navigation_state=navigation_state,
                    belief_runtime=belief_runtime,
                    decision_controller=decision_controller,
                    tool_budget=max(0, max_tool_steps - tool_call_count),
                )
                pre_action_decision["candidate_errors"] = list(pre_action_candidate_errors)
                pre_action_decision["selection_before_execution"] = True
                if selected_candidate is not None:
                    selected_key = str(selected_candidate.get("key"))
                    primary_key = str(sampled_candidates[0].get("key")) if sampled_candidates else ""
                    if selected_key != primary_key:
                        cur_response = str(selected_candidate.get("raw", ""))
                        cur_response_token_ids = list(selected_candidate.get("token_ids", []))
                        cur_log_probs = list(selected_candidate.get("log_probs", []))
                        step["generation_source"] = "bayestool_selected_policy_candidate"
                        step["selected_candidate_key"] = selected_key
                        step["selected_candidate_source"] = str(selected_candidate.get("source", ""))
                        step["policy_candidate_count"] = len(sampled_candidates)
                        step["raw_generation_text"] = cur_response
                        step["output_token_count"] = len(cur_response_token_ids)
                        step["backend_output_token_count"] = len(cur_response_token_ids)
                        step["assistant_output_token_count"] = len(cur_response_token_ids)
                    else:
                        step["policy_candidate_count"] = len(sampled_candidates)
                        step["selected_candidate_key"] = primary_key
                else:
                    step["policy_candidate_count"] = 0
            except Exception as exc:
                pre_action_candidate_errors.append(str(exc))
                pre_action_candidates = None
                pre_action_decision = {
                    "candidate_degenerate": True,
                    "candidate_errors": list(pre_action_candidate_errors),
                    "selection_before_execution": False,
                }

            if (
                root_branch_candidate is None
                and pre_action_checkpoint is not None
                and pre_action_candidates
            ):
                # Keep the first visible decision node as the bounded root
                # fallback.  It is captured after Bayes candidate selection so
                # the primary action and its log-probs are exactly the parent
                # continuation that will be compared with K-1 siblings.
                pre_action_checkpoint["selected_decision_event"] = "root"
                root_branch_candidate = {
                    "checkpoint": pre_action_checkpoint,
                    "current_text": cur_response,
                    "current_token_ids": cur_response_token_ids,
                    "current_log_probs": cur_log_probs,
                    "sampling_params": dict(turn_sampling_params),
                    "candidates": list(pre_action_candidates),
                    "candidate_errors": list(pre_action_candidate_errors),
                    "decision": dict(pre_action_decision or {}),
                    "turn": int(turn),
                }

        action_token_start = len(response_token_ids)
        response += cur_response
        response_token_ids.extend(cur_response_token_ids)
        loss_masks.extend([1] * len(cur_response_token_ids))
        sample.rollout_log_probs.extend(cur_log_probs)
        step_action_spans.append(
            {"step_index": turn, "token_start": action_token_start, "token_end": len(response_token_ids)}
        )

        trace_count_before = len(execution_trace)
        response_prefix_before_action = response
        next_obs, done = await execute_predictions(
            cur_response,
            execution_trace=execution_trace,
            action_log=action_log,
            turn=turn,
            navigation_state=navigation_state,
            world_runtime=world_runtime,
            belief_runtime=belief_runtime,
            decision_controller=decision_controller,
            tool_budget=max(0, max_tool_steps - tool_call_count),
            bayes_aux_prompt=(prompt + response_prefix_before_action) if bayestool_enabled else None,
            bayes_prompt_tokenizer=state.tokenizer,
            cost_state=cost_state,
            candidate_records=pre_action_candidates,
            candidate_decision=pre_action_decision,
        )
        if action_log.get("actions"):
            parsed_meta = action_log["actions"][-1]
            step["parsed_action_type"] = parsed_meta.get("parsed_action_type")
            step["parsed_tool_name"] = parsed_meta.get("parsed_tool_name")
            step["parse_failure_reason"] = parsed_meta.get("parse_failure_reason")
            step["protocol_parse_error"] = parsed_meta.get("protocol_parse_error")
            step["accepted"] = parsed_meta.get("accepted")
            step["action_valid_for_policy_gradient"] = bool(
                parsed_meta.get("action_valid_for_policy_gradient", True)
            )
            step["action_reward"] = float(parsed_meta.get("action_reward", 0.0) or 0.0)
            if not step["action_valid_for_policy_gradient"]:
                # A guard-rejected final is an observed recovery event, not a
                # positive policy action.  Mask only this assistant span; the
                # subsequent tool/final turns remain trainable.
                loss_masks[action_token_start:len(response_token_ids)] = [
                    0
                ] * max(0, len(response_token_ids) - action_token_start)
        branch_decision = action_log.get("bayes_decision", {})
        if (
            pre_action_checkpoint is not None
            and bayestool_config is not None
            and isinstance(branch_decision, dict)
            and bool(branch_decision.get("branch_eligible"))
            and not done
        ):
            deferred_branch_candidates.append(
                {
                    "checkpoint": pre_action_checkpoint,
                    "current_text": cur_response,
                    "current_token_ids": cur_response_token_ids,
                    "current_log_probs": cur_log_probs,
                    "sampling_params": dict(turn_sampling_params),
                    "candidates": list(pre_action_candidates),
                    "candidate_errors": list(pre_action_candidate_errors),
                    "decision": dict(branch_decision),
                    "turn": int(turn),
                }
            )
            branch_decision["branch_triggered"] = False
            branch_decision["branch_deferred"] = True
            deferred_event = {
                "branch_triggered": False,
                "branch_deferred": True,
                "reason": "eligible checkpoint retained until primary trajectory completed",
                "turn": int(turn),
                "prefix_hash": pre_action_checkpoint.get("prefix_hash", ""),
                "decision_regret": float(branch_decision.get("decision_regret", 0.0) or 0.0),
                "dvoi": _bayestool_dvoi_score(branch_decision.get("dvoi", 0.0)),
                "dvoi_values": dict(branch_decision.get("dvoi") or {})
                if isinstance(branch_decision.get("dvoi"), Mapping)
                else {},
            }
            branch_events.append(deferred_event)
            navigation_state.setdefault("bayes_branch_events", []).append(deferred_event)
        if getattr(args, "prm_enable", False) and not bayestool_enabled:
            prm_pending_tasks.append(
                (
                    turn,
                    asyncio.create_task(
                        _judge_step_with_prm(
                            args,
                            sample,
                            step_index=turn,
                            action=cur_response,
                            observation=next_obs,
                            history=response + next_obs,
                        )
                    ),
                )
            )

        if done:
            terminal_status = str(action_log.get("_terminal_status") or "model_protocol_error")
            if terminal_status in {"completed", "abstained"}:
                sample.metadata = sample.metadata or {}
                sample.metadata["final_action"] = cur_response
                sample.metadata["final_answer"] = cur_response
                sample.metadata["abstention"] = terminal_status == "abstained"
            break

        latest_tool = execution_trace[-1] if execution_trace else {}
        recovery_observation = bool(latest_tool.get("recovery_observation"))
        # World-injected failures intentionally return success=False, but the
        # returned error is a valid POMDP observation and must remain in RL data.
        if (
            not recovery_observation
            and not _is_world_injected_observation(latest_tool)
            and (
                not latest_tool.get("executed")
                or not latest_tool.get("success")
            )
        ):
            terminal_status = "infra_error"
            terminal_reason = "successful tool execution did not produce a sendable observation"
            break
        if latest_tool.get("media_error"):
            terminal_status = "infra_error"
            terminal_reason = "tool reported an image path that could not be loaded"
            break

        # Full-page renders are attached only when the pending visual state
        # says the next decision needs pixels.  Known bbox/table metadata can
        # drive a deterministic crop/OCR/layout call without consuming the
        # whole page image.  Crop/zoom results and evidence-guard recovery
        # turns always carry their image observations.  This is also the path
        # used by the deterministic visual canary.
        latest_tool_name = str(latest_tool.get("tool") or "")
        image_paths_for_next = []
        if latest_tool_name == "render_page" and navigation_state.get("pending_visual_image_paths"):
            image_paths_for_next = list(latest_tool.get("image_paths", []))
        elif latest_tool_name in {"crop_region", "zoom_region"}:
            image_paths_for_next = list(latest_tool.get("image_paths", []))
        elif recovery_observation and navigation_state.get("pending_visual_image_paths"):
            image_paths_for_next = list(navigation_state.get("pending_visual_image_paths", []))
        try:
            (
                obs_token_ids,
                encoded_obs_text,
                obs_image_data,
                obs_images,
                obs_train_inputs,
                _new_image_token_count,
                observation_suffix_token_count,
            ) = _encode_tool_observation(state, next_obs, image_paths_for_next)
        except Exception as exc:
            step["generation_error"] = str(exc)
            terminal_status = "infra_error"
            terminal_reason = str(exc)
            break

        if not obs_token_ids:
            terminal_status = "infra_error"
            terminal_reason = "tool result was not converted into model input tokens"
            break
        response += encoded_obs_text
        response_token_ids.extend(obs_token_ids)
        loss_masks.extend([0] * len(obs_token_ids))
        sample.rollout_log_probs.extend([0.0] * len(obs_token_ids))
        image_start = len(current_images)
        current_images.extend(obs_images)
        image_end = len(current_images)
        multimodal_train_input_index: int | None = None
        if obs_train_inputs:
            multimodal_train_input_index = len(multimodal_train_inputs_buffer)
            multimodal_train_inputs_buffer.append(obs_train_inputs)
        text_observation_token_ids = list(obs_token_ids)
        text_observation_suffix_token_count = int(observation_suffix_token_count)
        if obs_image_data:
            # Keep a text-only representation available for context recovery
            # when the vision-expanded observation cannot fit alongside a
            # final-answer generation request.  This does not discard the
            # image from the training trajectory.
            text_only_encoded = (
                f"<|im_end|>\n<|im_start|>user\n{next_obs}<|im_end|>\n<|im_start|>assistant\n"
            )
            text_observation_token_ids = list(
                state.tokenizer(text_only_encoded, add_special_tokens=False)["input_ids"]
            )
            text_observation_suffix_token_count = len(
                state.tokenizer(
                    "<|im_end|>\n<|im_start|>assistant\n",
                    add_special_tokens=False,
                )["input_ids"]
            )
        latest_tool["image_token_count"] = int(_new_image_token_count or 0)
        latest_tool["image_input_count"] = len(obs_image_data)
        latest_tool["vision_input_attached"] = bool(obs_image_data)
        latest_tool["vision_placeholder_present"] = bool(obs_image_data)
        latest_tool["image_tensor_count"] = len(obs_images)
        latest_tool["consumed_image_paths"] = list(image_paths_for_next)
        if obs_image_data:
            navigation_state["visual_input_consumed"] = True
            navigation_state["pending_visual_image_paths"] = []
            navigation_state["visual_input_required"] = False
            navigation_state["visual_input_reason"] = None
            _evaluate_evidence_sufficiency(navigation_state)
        elif image_paths_for_next:
            # An image path was requested for the next decision but was not
            # encoded.  Leave the requirement visible to the consistency audit;
            # the rollout will be excluded instead of silently training on a
            # text-only substitute.
            navigation_state["visual_input_required"] = True
            navigation_state["visual_input_reason"] = "requested page pixels were not attached"
        context_segments.append(
            {
                "action_token_start": int(action_token_start),
                "action_token_end": int(len(response_token_ids) - len(obs_token_ids)),
                "response_token_start": int(action_token_start),
                "response_token_end": int(len(response_token_ids)),
                "action_token_ids": list(cur_response_token_ids),
                "observation_token_ids": list(obs_token_ids),
                "text_observation_token_ids": text_observation_token_ids,
                "observation_suffix_token_count": int(observation_suffix_token_count),
                "text_observation_suffix_token_count": int(text_observation_suffix_token_count),
                "image_data": list(obs_image_data),
                "image_token_count": int(_new_image_token_count or 0),
                "image_paths": list(image_paths_for_next),
                "image_start": int(image_start),
                "image_end": int(image_end),
                "multimodal_train_input_index": multimodal_train_input_index,
                "vision_placeholder_present": bool(obs_image_data),
                "visual_input_required": bool(image_paths_for_next),
            }
        )
        context_token_ids, context_image_data = _compact_model_context(
            prompt_tokens_ids,
            context_segments,
            max_context_length=max_context_length,
            reserve_tokens=min(max_new_tokens, 256),
        )
        if len(execution_trace) > trace_count_before:
            new_tool_calls = sum(
                1
                for item in execution_trace[trace_count_before:]
                if item.get("kind") == "tool_call" and item.get("executed")
            )
            tool_call_count += new_tool_calls
            if tool_call_count >= max_tool_steps:
                navigation_state["search_budget_exhausted"] = True

    if not branch_child and bayestool_config is not None:
        explicit_plan = bayes_question_plan is not None
        selected_checkpoint: dict[str, Any] | None = None
        if deferred_branch_candidates:
            # Select one checkpoint only after the primary trajectory has
            # ended.  The score uses quantities visible at the checkpoint and
            # never consults the later terminal reward.
            selected_checkpoint = max(
                deferred_branch_candidates,
                key=lambda item: (
                    float(item["decision"].get("decision_regret", 0.0) or 0.0),
                    _bayestool_dvoi_score(item["decision"].get("dvoi", 0.0)),
                    float(item["decision"].get("best_action_margin", 0.0) or 0.0),
                    -int(item.get("turn", 0)),
                ),
            )
        elif explicit_plan:
            # Every explicit realization must produce exactly one K-group.
            # If no mid-trajectory checkpoint met the branch score, the root
            # is the required bounded fallback rather than an unpaired sample.
            selected_checkpoint = root_branch_candidate

        target_group_size = int(
            diagnostic_metadata.get("decision_group_size")
            or getattr(bayestool_config, "default_group_size", 4)
            or 4
        )
        if bayes_question_plan is not None and world_runtime is not None:
            target_group_size = int(
                bayes_question_plan.realizations[int(world_runtime.spec.world_slot)].k
            )

        def _record_branch_event(event: dict[str, Any]) -> None:
            branch_events.append(event)
            navigation_state.setdefault("bayes_branch_events", []).append(event)

        def _mark_question_skipped(reason: str) -> None:
            diagnostic_metadata["question_skipped"] = True
            diagnostic_metadata["exclude_from_group_statistics"] = True
            diagnostic_metadata["valid_for_rl"] = False
            diagnostic_metadata["degenerate_no_signal"] = True
            diagnostic_metadata["question_skip_reason"] = str(reason)
            navigation_state.setdefault("bayes_branch_events", []).append(
                {
                    "branch_triggered": False,
                    "question_skipped": True,
                    "degenerate_no_signal": True,
                    "reason": str(reason),
                    "target_group_size": target_group_size,
                }
            )

        if selected_checkpoint is not None:
            selected_prefix_hint = str(selected_checkpoint["checkpoint"].get("prefix_hash", ""))
            root_prefix_hint = str(
                root_branch_candidate["checkpoint"].get("prefix_hash", "")
                if root_branch_candidate is not None
                else ""
            )
            selected_checkpoint["checkpoint"]["selected_decision_event"] = (
                "root"
                if selected_prefix_hint == root_prefix_hint
                else f"branch:{selected_prefix_hint[:32]}"
            )
            try:
                children, branch_event = await _launch_bayestool_branches(
                    args=args,
                    state=state,
                    url=url,
                    sample=sample,
                    sampling_params=selected_checkpoint["sampling_params"],
                    evaluation=evaluation,
                    checkpoint=selected_checkpoint["checkpoint"],
                    current_text=selected_checkpoint["current_text"],
                    current_token_ids=selected_checkpoint["current_token_ids"],
                    current_log_probs=selected_checkpoint["current_log_probs"],
                    im_end_id=im_end_id,
                    decision_controller=decision_controller,
                    bayes_config=bayestool_config,
                    candidates_override=selected_checkpoint["candidates"],
                    candidate_errors_override=selected_checkpoint["candidate_errors"],
                    target_group_size=target_group_size,
                    force_branch=explicit_plan,
                )
                _record_branch_event(branch_event)

                # A middle checkpoint is allowed to fail independently.  Do
                # one bounded retry from the root under the same latent world;
                # partial children from the failed middle attempt are never
                # admitted as a smaller decision group.
                enough_children = len(children) >= max(0, target_group_size - 1)
                selected_prefix = str(selected_checkpoint["checkpoint"].get("prefix_hash", ""))
                root_prefix = str(
                    root_branch_candidate["checkpoint"].get("prefix_hash", "")
                    if root_branch_candidate is not None
                    else ""
                )
                if explicit_plan and not enough_children and root_branch_candidate is not None:
                    root_retry = dict(root_branch_candidate)
                    retry_candidates, retry_errors = await _sample_bayestool_branch_candidates(
                        state=state,
                        url=url,
                        checkpoint=root_retry["checkpoint"],
                        current_text=root_retry["current_text"],
                        current_token_ids=root_retry["current_token_ids"],
                        current_log_probs=root_retry["current_log_probs"],
                        sampling_params=root_retry["sampling_params"],
                        im_end_id=im_end_id,
                        config=bayestool_config,
                        target_group_size=target_group_size,
                    )
                    root_retry["candidates"] = retry_candidates
                    root_retry["candidate_errors"] = retry_errors
                    retry_children, retry_event = await _launch_bayestool_branches(
                        args=args,
                        state=state,
                        url=url,
                        sample=sample,
                        sampling_params=root_retry["sampling_params"],
                        evaluation=evaluation,
                        checkpoint=root_retry["checkpoint"],
                        current_text=root_retry["current_text"],
                        current_token_ids=root_retry["current_token_ids"],
                        current_log_probs=root_retry["current_log_probs"],
                        im_end_id=im_end_id,
                        decision_controller=decision_controller,
                        bayes_config=bayestool_config,
                        candidates_override=retry_candidates,
                        candidate_errors_override=retry_errors,
                        target_group_size=target_group_size,
                        force_branch=True,
                    )
                    _record_branch_event(retry_event)
                    if len(retry_children) >= max(0, target_group_size - 1):
                        children = retry_children
                        branch_event = retry_event
                        selected_checkpoint = root_retry
                        enough_children = True
                    else:
                        children = []

                if explicit_plan and not enough_children:
                    _mark_question_skipped(
                        "unable to construct the required K-sized decision group after middle/root retry"
                    )
                    children = []
                elif branch_event.get("branch_triggered") and enough_children:
                    branch_children.extend(children)
                    branch_group_id = str(branch_event.get("branch_sibling_group_id"))
                    diagnostic_metadata["sibling_group_id"] = branch_group_id
                    diagnostic_metadata["bayestool_branch_sibling_group_id"] = branch_group_id
                    diagnostic_metadata["bayestool_branch_prefix_hash"] = branch_event.get("prefix_hash")
                    diagnostic_metadata["bayestool_branch_action_key"] = branch_event.get("parent_action_key")
                    diagnostic_metadata["bayestool_branch_shared_prefix"] = True
                    diagnostic_metadata["bayestool_branch_q_features"] = (
                        branch_event.get("q_features", {}).get(branch_event.get("parent_action_key"), {})
                        if isinstance(branch_event.get("q_features"), dict)
                        else {}
                    )
                    diagnostic_metadata["bayestool_branch_horizon"] = branch_event.get("branch_horizon", 0)
                    diagnostic_metadata["runtime_state_digest"] = branch_event.get(
                        "runtime_state_digest", selected_checkpoint["checkpoint"].get("runtime_state_digest", "")
                    )
                    diagnostic_metadata["bayes_runtime_state_digest"] = diagnostic_metadata["runtime_state_digest"]
                    diagnostic_metadata["selected_decision_event"] = (
                        "root" if str(selected_checkpoint["checkpoint"].get("prefix_hash", "")) == root_prefix
                        else f"branch:{branch_event.get('prefix_hash', '')[:32]}"
                    )
                elif explicit_plan:
                    _mark_question_skipped(
                        str(branch_event.get("reason") or "branch expansion did not produce a complete decision group")
                    )
                elif not branch_event.get("branch_triggered"):
                    diagnostic_metadata["degenerate_no_signal"] = True
            except Exception as exc:
                failure_event = {
                    "branch_triggered": False,
                    "degenerate_no_signal": True,
                    "reason": f"branch expansion failed: {exc}",
                }
                _record_branch_event(failure_event)
                if explicit_plan:
                    _mark_question_skipped(failure_event["reason"])
                else:
                    diagnostic_metadata["degenerate_no_signal"] = True
        elif explicit_plan:
            _mark_question_skipped(
                "no valid root decision checkpoint was available for the explicit realization plan"
            )

    if terminal_status is None:
        # A syntactically valid final can be rejected by the evidence guard.
        # If the model keeps repeating that premature final until the turn
        # budget ends, this is a model search-policy failure, not a protocol
        # parser error.  Keep it trainable and expose the distinction in the
        # rollout status/action statistics.
        if int(action_log.get("premature_final_count", 0) or 0) > 0:
            terminal_status = "search_budget_exhausted"
            terminal_reason = "model repeated a final answer until the bounded recovery/search budget ended"
        else:
            terminal_status = "model_protocol_error"
            terminal_reason = "rollout ended before a final action"

    if navigation_state.get("had_infra_error") and terminal_status not in {"generation_error", "context_overflow"}:
        terminal_status = "infra_error"
        terminal_reason = "one or more tool calls failed because a runtime backend was unavailable"

    train_response_token_ids, train_loss_masks, train_rollout_log_probs, train_mm_chunks, train_trajectory = (
        _cap_training_trajectory(
            prompt_tokens_ids,
            response_token_ids,
            loss_masks,
            sample.rollout_log_probs,
            context_segments,
            multimodal_train_inputs_buffer,
            step_action_spans,
            max_sequence_length=_resolve_training_sequence_limit(args),
        )
    )
    sample.tokens = prompt_tokens_ids + train_response_token_ids
    sample.response_length = len(train_response_token_ids)
    sample.response = response
    sample.loss_mask = train_loss_masks
    sample.rollout_log_probs = train_rollout_log_probs
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(train_mm_chunks)
    if current_images:
        sample.multimodal_inputs = {"images": current_images, "videos": None}

    sample.tool_call_count = sum(
        1 for item in execution_trace if item.get("kind") == "tool_call" and item.get("executed")
    )
    sample.valid_tool_call_count = sum(
        1 for item in execution_trace if item.get("kind") == "tool_call" and item.get("executed") and item.get("success")
    )
    sample.tool_error_count = int(action_log.get("tool_error_count", 0))
    sample.tool_execution_trace = execution_trace
    training_response_start = int(train_trajectory.get("response_start", 0) or 0)
    training_response_end = training_response_start + len(train_response_token_ids)
    for key in (
        "candidate_action_count",
        "executed_action_count",
        "valid_action_count",
        "invalid_action_count",
        "ignored_action_count",
        "protocol_error_count",
        "premature_final_count",
        "rejected_final_count",
    ):
        action_log[key] = int(action_log.get(key, 0))
    action_log["had_evidence_guard_recovery"] = bool(
        navigation_state.get("had_evidence_guard_recovery")
        or action_log.get("had_evidence_guard_recovery")
    )
    action_log["rejected_final_count"] = max(
        int(action_log.get("rejected_final_count", 0)),
        int(navigation_state.get("rejected_final_count", 0)),
    )
    assistant_token_masks: list[dict[str, Any]] = []
    action_rewards: list[float] = []
    rejected_action_indices: list[int] = []
    for action_index, action in enumerate(action_log.get("actions", [])):
        full_span = step_action_spans[action_index] if action_index < len(step_action_spans) else {}
        try:
            full_start = int(full_span.get("token_start", 0) or 0)
            full_end = int(full_span.get("token_end", 0) or 0)
        except (TypeError, ValueError):
            full_start = 0
            full_end = 0
        # The actor receives a suffix of the response.  Drop action metadata
        # for discarded prefixes and translate the remaining token spans to
        # the new response origin so rejected-action penalties cannot land on
        # unrelated tokens.
        if full_start < training_response_start or full_end > training_response_end:
            continue
        span = {
            "token_start": full_start - training_response_start,
            "token_end": full_end - training_response_start,
        }
        valid_for_gradient = bool(action.get("action_valid_for_policy_gradient", True))
        action_reward = float(action.get("action_reward", 0.0) or 0.0)
        action["action_valid_for_policy_gradient"] = valid_for_gradient
        action["action_reward"] = action_reward
        action["full_token_start"] = full_start
        action["full_token_end"] = full_end
        action["token_start"] = int(span.get("token_start", 0))
        action["token_end"] = int(span.get("token_end", 0))
        mask_value = 1 if valid_for_gradient else 0
        assistant_token_masks.append(
            {
                "turn": action.get("turn"),
                "token_start": int(span.get("token_start", 0)),
                "token_end": int(span.get("token_end", 0)),
                "mask": mask_value,
            }
        )
        action_rewards.append(action_reward)
        if not valid_for_gradient:
            rejected_action_indices.append(len(action_rewards) - 1)

    sample.metadata = sample.metadata or {}
    sample.metadata["tool_execution"] = {
        "calls": execution_trace,
        "call_count": sample.tool_call_count,
        "valid_call_count": sample.valid_tool_call_count,
        "error_count": sample.tool_error_count,
        "unique_tools": sorted({str(item["tool"]) for item in execution_trace if item.get("tool")}),
    }
    sample.metadata["action_statistics"] = {key: action_log[key] for key in (
        "candidate_action_count",
        "executed_action_count",
        "valid_action_count",
        "invalid_action_count",
        "ignored_action_count",
        "protocol_error_count",
        "premature_final_count",
        "rejected_final_count",
    )}
    sample.metadata["failure_events"] = list(action_log.get("failure_events", []))
    sample.metadata["budget_exhausted_without_terminal"] = bool(
        navigation_state.get("search_budget_exhausted")
        and terminal_status not in {"completed", "abstained"}
    )
    sample.metadata["bayes_decisions"] = list(action_log.get("bayes_decisions", []))
    sample.metadata["training_trajectory"] = dict(train_trajectory)
    sample.metadata["full_response_length"] = len(response_token_ids)
    sample.metadata["assistant_token_masks"] = assistant_token_masks
    sample.metadata["action_rewards"] = action_rewards
    sample.metadata["rejected_action_indices"] = rejected_action_indices
    sample.metadata["policy_gradient_token_mask"] = list(loss_masks)
    sample.metadata["generation_steps"] = generation_steps
    last_generation = generation_steps[-1] if generation_steps else {
        "step_index": None,
        "generation_called": False,
        "input_token_count": len(prompt_tokens_ids),
        "image_input_count": 0,
        "image_token_count": 0,
        "vision_input_attached": False,
        "vision_placeholder_present": False,
        "image_tensor_count": 0,
        "consumed_image_paths": [],
        "visual_input_required": False,
        "visual_input_reason": None,
        "remaining_generation_tokens": 0,
        "finish_reason": None,
        "raw_generation_text": "",
        "backend_raw_generation_text": "",
        "output_token_count": 0,
        "backend_output_token_count": 0,
        "stop_token_removed_count": 0,
        "output_source": None,
        "assistant_output_token_count": 0,
        "generation_error": None,
        "protocol_parse_error": None,
    }
    for key in (
        "step_index",
        "generation_called",
        "input_token_count",
        "image_input_count",
        "image_token_count",
        "vision_input_attached",
        "vision_placeholder_present",
        "image_tensor_count",
        "consumed_image_paths",
        "visual_input_required",
        "visual_input_reason",
        "remaining_generation_tokens",
        "finish_reason",
        "raw_generation_text",
        "backend_raw_generation_text",
        "output_token_count",
        "backend_output_token_count",
        "stop_token_removed_count",
        "output_source",
        "assistant_output_token_count",
        "generation_error",
        "protocol_parse_error",
        "parsed_action_type",
        "parsed_tool_name",
        "parse_failure_reason",
    ):
        sample.metadata[key] = last_generation.get(key)
    sample.metadata["raw_generation_texts"] = [
        str(step.get("raw_generation_text") or "") for step in generation_steps
    ]
    sample.metadata["generation_called"] = any(bool(step.get("generation_called")) for step in generation_steps)
    sample.metadata["generation_call_count"] = sum(
        1 for step in generation_steps if bool(step.get("generation_called"))
    )
    sample.metadata["image_input_count"] = max(
        (int(step.get("image_input_count", 0) or 0) for step in generation_steps),
        default=0,
    )
    sample.metadata["image_token_count"] = max(
        (int(step.get("image_token_count", 0) or 0) for step in generation_steps),
        default=0,
    )
    sample.metadata["vision_input_attached"] = any(
        bool(step.get("vision_input_attached")) for step in generation_steps
    )
    sample.metadata["vision_placeholder_present"] = any(
        bool(step.get("vision_placeholder_present")) for step in generation_steps
    )
    sample.metadata["image_tensor_count"] = max(
        (int(step.get("image_tensor_count", 0) or 0) for step in generation_steps),
        default=0,
    )
    cost_state["image_tokens"] = float(
        sum(int(step.get("image_token_count", 0) or 0) for step in generation_steps)
    )
    cost_calls = max(1.0, float(cost_state.get("tool_calls", sample.tool_call_count) or sample.tool_call_count or 1.0))
    cost_metrics = {
        "latency_seconds": float(cost_state.get("latency_seconds", 0.0) or 0.0),
        "observation_text_tokens": float(cost_state.get("observation_text_tokens", 0.0) or 0.0),
        "image_tokens": float(cost_state.get("image_tokens", 0.0) or 0.0),
        "latency_cost": min(1.0, float(cost_state.get("latency_seconds", 0.0) or 0.0) / (2.0 * cost_calls)),
        "text_token_cost": min(1.0, float(cost_state.get("observation_text_tokens", 0.0) or 0.0) / (1024.0 * cost_calls)),
        "image_token_cost": min(1.0, float(cost_state.get("image_tokens", 0.0) or 0.0) / (256.0 * cost_calls)),
    }
    cost_metrics["normalized_latency_cost"] = cost_metrics["latency_cost"]
    cost_metrics["normalized_text_token_cost"] = cost_metrics["text_token_cost"]
    cost_metrics["normalized_image_token_cost"] = cost_metrics["image_token_cost"]
    sample.metadata["bayestool_cost"] = cost_metrics
    sample.metadata.update(cost_metrics)
    sample.metadata["tool_budget"] = int(
        getattr(args, "bayestool_tool_budget", None) or max_tool_steps
    )
    sample.metadata["consumed_image_paths"] = list(
        dict.fromkeys(
            path
            for step in generation_steps
            for path in (step.get("consumed_image_paths") or [])
            if isinstance(path, str)
        )
    )
    sample.metadata["assistant_turns"] = [dict(item) for item in action_log.get("actions", [])]
    sample.metadata["generation"] = dict(last_generation)
    sample.metadata["generation"]["steps"] = generation_steps
    navigation_snapshot = dict(navigation_state)
    navigation_snapshot["visited_pages"] = sorted(
        {int(page) for page in navigation_state.get("visited_pages", [])}
    )
    navigation_snapshot["unvisited_pages"] = sorted(
        {int(page) for page in navigation_state.get("unvisited_pages", [])}
    )
    navigation_snapshot["evidence_by_page"] = {
        str(page): str(value)[:2400]
        for page, value in navigation_state.get("evidence_by_page", {}).items()
    }
    navigation_snapshot["image_page_by_path"] = {
        str(path): int(page)
        for path, page in navigation_state.get("image_page_by_path", {}).items()
        if str(page).isdigit()
    }
    # Never serialize evaluation-only answer metadata into the runtime
    # navigation snapshot.  The top-level fields below are diagnostics only.
    for hidden_key in ("answer_page", "answer_bbox", "target_page"):
        navigation_snapshot.pop(hidden_key, None)
    sample.metadata["navigation_state"] = navigation_snapshot
    sample.metadata["visited_pages"] = navigation_snapshot["visited_pages"]
    for key in (
        "parsed_pages", "rendered_pages", "cropped_pages", "ocr_pages", "cropped_regions", "zoomed_regions", "ocr_regions",
        "unvisited_pages", "supporting_pages", "supporting_regions", "evidence_hits",
        "visual_evidence_sufficient", "evidence_sufficient", "stop_reason", "evidence_candidates", "evidence_reason",
        "prediction_found_in_document", "prediction_relation_matched", "final_supporting_pages",
        "final_supporting_regions", "visual_input_required", "visual_input_reason", "visual_input_consumed",
    ):
        sample.metadata[key] = navigation_snapshot.get(key)
    sample.metadata["answer_page"] = diagnostic_answer_page
    sample.metadata["answer_bbox"] = diagnostic_answer_bbox
    sample.metadata["answer_page_visited"] = bool(
        diagnostic_answer_page is not None
        and diagnostic_answer_page in {
            int(page) for page in navigation_state.get("visited_pages", []) if str(page).isdigit()
        }
    )
    sample.metadata["runtime_evidence_source"] = "tool_observation"
    sample.metadata["ground_truth_metadata_used_in_prompt"] = False
    sample.metadata["ground_truth_metadata_used_in_action_selection"] = False
    if bayestool_enabled and world_runtime is not None and belief_runtime is not None:
        snapshot = belief_runtime.snapshot()
        stage_policy = stage_definition(bayestool_config) if stage_definition is not None else None
        decision_history = list(sample.metadata.get("bayes_decisions", []))
        primary_decision = next(
            (item for item in reversed(decision_history) if isinstance(item, dict)),
            {},
        )
        world_slot = int(world_runtime.spec.world_slot)
        world_slot_role = str(
            getattr(world_runtime.spec, "world_slot_role", "")
            or (slot_role_from_metadata(diagnostic_metadata) if slot_role_from_metadata is not None else "")
        )
        variant_id = str(getattr(world_runtime.spec, "variant_id", "base") or "base")
        decision_group_size = int(getattr(bayestool_config, "default_group_size", 4) or 4)
        if bayes_question_plan is not None and world_slot < len(bayes_question_plan.realizations):
            realization = bayes_question_plan.realizations[world_slot]
            decision_group_size = int(realization.k)
            world_slot_role = world_slot_role or realization.world_slot_role
            variant_id = variant_id or realization.variant_id
        meta_episode_id = diagnostic_metadata.get("meta_episode_id")
        question_id = (
            bayestool_question_id(diagnostic_metadata, default=f"sample-{sample.index}")
            if bayestool_question_id is not None
            else str(diagnostic_metadata.get("task_id") or diagnostic_metadata.get("coupling_id") or sample.index)
        )
        # The initial input identity is deliberately independent of the
        # sampled world, replica and observed evidence.  All policy siblings
        # for one question/world therefore share the root group, while two
        # different questions can never accidentally share a baseline.
        initial_input_hash = hashlib.sha256(
            json.dumps(
                {
                    "episode_content_id": diagnostic_metadata.get("episode_content_id")
                    or diagnostic_metadata.get("document_hash")
                    or bayes_document_digest,
                    "question_id": question_id,
                    "prompt": task_prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        selected_event_hint = str(diagnostic_metadata.get("selected_decision_event") or "")
        branch_prefix_hash = str(
            diagnostic_metadata.get("bayestool_branch_prefix_hash")
            or diagnostic_metadata.get("branch_prefix_hash")
            or ""
        )
        if selected_event_hint == "root":
            # Root fallback children carry a checkpoint prefix for replay
            # diagnostics, but that prefix is not a new decision node.  Keep
            # them in the parent's root group identity.
            branch_prefix_hash = ""
        decision_event_id = (
            f"branch:{branch_prefix_hash[:32]}" if branch_prefix_hash else "root"
        )
        decision_prefix_hash = branch_prefix_hash or initial_input_hash
        frozen_runtime_digest = str(
            diagnostic_metadata.get("runtime_state_digest")
            or diagnostic_metadata.get("bayes_runtime_state_digest")
            or bayes_runtime_state_digest
            or ""
        )
        if not frozen_runtime_digest and make_runtime_state_digest is not None:
            frozen_runtime_digest = make_runtime_state_digest(
                {
                    "remaining_tool_budget": bayestool_tool_budget,
                    "tool_call_count": 0,
                    "world_schedule": world_runtime.schedule_metadata(),
                    "world_call_counters": [],
                    "restore_state_id": "root",
                    "branch_horizon": int(getattr(bayestool_config, "branch_horizon", 0)),
                    "bootstrap": "terminal_reward",
                }
            )
        policy_version = str(
            diagnostic_metadata.get("policy_version")
            or getattr(bayestool_config, "policy_version", DEFAULT_POLICY_VERSION)
        )
        selected_decision_event = str(
            diagnostic_metadata.get("selected_decision_event")
            or (f"branch:{branch_prefix_hash[:32]}" if branch_prefix_hash else "root")
        )
        # Freeze the selected node back into the per-question manifest before
        # serializing the sample.  Parent and branch children therefore carry
        # the same selected event, prefix and runtime digest; a plan that only
        # describes the initial root state cannot silently pass grouping
        # validation after a middle-node branch was selected.
        if bayes_question_plan is not None and 0 <= world_slot < bayes_question_plan.group_count:
            realization = bayes_question_plan.realizations[world_slot]
            bayes_question_plan = replace(
                bayes_question_plan,
                realizations=tuple(
                    replace(
                        item,
                        selected_decision_event=(
                            selected_decision_event
                            if index == world_slot
                            else item.selected_decision_event
                        ),
                        decision_prefix_hash=(
                            decision_prefix_hash
                            if index == world_slot
                            else item.decision_prefix_hash
                        ),
                        runtime_state_digest=(
                            frozen_runtime_digest
                            if index == world_slot
                            else item.runtime_state_digest
                        ),
                    )
                    for index, item in enumerate(bayes_question_plan.realizations)
                ),
            )
            diagnostic_metadata["question_rollout_plan"] = bayes_question_plan.to_dict()
        slot_weight = float(
            dict(bayes_question_plan.slot_weights).get(world_slot_role, 0.25)
            if bayes_question_plan is not None
            else 0.25
        )
        group_metadata = dict(diagnostic_metadata)
        group_metadata.update(
            {
                "question_id": question_id,
                "episode_content_id": diagnostic_metadata.get("episode_content_id")
                or diagnostic_metadata.get("document_hash")
                or bayes_document_digest,
                "initial_input_hash": initial_input_hash,
                "decision_event_id": decision_event_id,
                "selected_decision_event": selected_decision_event,
                "decision_prefix_hash": decision_prefix_hash,
                "runtime_state_digest": frozen_runtime_digest,
                "rng_coupling_id": bayes_coupling_id,
                "return_definition_version": "bayestool-utility-v1",
                "latent_world_id": world_runtime.spec.latent_world_id,
                "world_id": world_runtime.spec.world_id,
                "coupling_id": bayes_coupling_id,
                "world_slot_role": world_slot_role,
                "variant_id": variant_id,
                "slot_weight": slot_weight,
                "policy_version": policy_version,
                "decision_group_size": decision_group_size,
            }
        )
        decision_group_id = (
            make_bayestool_decision_group_id(group_metadata)
            if make_bayestool_decision_group_id is not None
            else str(diagnostic_metadata.get("sibling_group_id") or f"bayes-world:{bayes_coupling_id}:{world_slot}")
        )
        # Keep the legacy field as an alias for older consumers, but make the
        # strict decision-group identity the only source used by Bayes GRPO.
        sibling_group_id = decision_group_id
        aux_prompt = str(
            primary_decision.get("prompt")
            or diagnostic_metadata.get("bayes_aux_prompt")
            or prompt
            or task_prompt
        )
        aux_content_signature = str(
            primary_decision.get("content_signature")
            or diagnostic_metadata.get("bayes_content_signature")
            or bayes_content_signature
            or ""
        )
        aux_state = {
            "coupling_id": bayes_coupling_id,
            "content_signature": aux_content_signature,
            "belief_snapshot": primary_decision.get("belief_snapshot", snapshot.to_dict()),
            "ood_score": float(primary_decision.get("belief_snapshot", {}).get("ood_score", snapshot.ood_score) or 0.0)
            if isinstance(primary_decision.get("belief_snapshot", {}), dict)
            else float(snapshot.ood_score),
            "best_action": primary_decision.get("bayes_action"),
            "best_action_text": primary_decision.get("best_action_text", ""),
            "best_action_margin": float(primary_decision.get("best_action_margin", 0.0) or 0.0),
            "candidate_actions": list(primary_decision.get("candidate_action_keys", [])),
            "candidate_action_texts": {
                str(item.get("key")): str(item.get("text") or "")
                for item in primary_decision.get("candidate_actions", [])
                if isinstance(item, dict) and item.get("key")
            },
            "observed_prefix_js": float(primary_decision.get("observed_prefix_js", 1.0) or 0.0),
            "first_distinguishing_event_step": primary_decision.get("observed_prefix_event_count"),
            "prefix_observation_signature": primary_decision.get("prefix_observation_signature", ""),
            "prompt": aux_prompt,
        }
        sample.metadata["bayestool"] = {
            "enabled": True,
            "stage": str(getattr(bayestool_config, "stage", "c") or "c"),
            "stage_policy": (
                {
                    "name": stage_policy.name,
                    "objective": stage_policy.objective,
                    "runtime_mode": stage_policy.runtime_mode,
                    "branch_probability": stage_policy.branch_probability,
                    "use_meta_episode": stage_policy.use_meta_episode,
                    "use_persistent_session_belief": stage_policy.use_persistent_session_belief,
                    "expected_update_fraction": stage_policy.expected_update_fraction,
                }
                if stage_policy is not None
                else None
            ),
            "coupling_id": bayes_coupling_id,
            "document_hash": bayes_document_digest,
            "world_id": world_runtime.spec.world_id,
            "latent_world_id": world_runtime.spec.latent_world_id,
            "latent_seed": int(world_runtime.spec.latent_seed),
            "world_type": world_runtime.spec.world_type,
            "world_slot": int(world_runtime.spec.world_slot),
            "replica_id": int(world_runtime.spec.replica_id),
            "world_slot_role": world_slot_role,
            "variant_id": variant_id,
            "policy_version": policy_version,
            "decision_group_size": decision_group_size,
            "belief_model_version": belief_runtime.belief_model_version,
            "bayes_q_head_version": bayestool_q_version,
            "bayes_risk_calibrator_version": bayestool_risk_version,
            "belief_snapshot": snapshot.to_dict(),
            "posterior_entropy": snapshot.posterior_entropy,
            "ood_score": snapshot.ood_score,
            "world_events": world_runtime.public_event_metadata(),
            "bayes_supervision": world_runtime.hidden_supervision_metadata(),
            "reopen_events": list(belief_runtime.reopen_events),
            "decision_history": decision_history,
            "question_id": question_id,
            "initial_input_hash": initial_input_hash,
            "decision_event_id": decision_event_id,
            "selected_decision_event": selected_decision_event,
            "decision_prefix_hash": decision_prefix_hash,
            "runtime_state_digest": frozen_runtime_digest,
            "decision_group_id": decision_group_id,
            "sibling_group_id": sibling_group_id,
            "sibling_weight": 1.0,
            "content_signature": aux_content_signature,
            "branch_events": list(branch_events),
            "branch_prefix_hash": diagnostic_metadata.get("bayestool_branch_prefix_hash"),
            "branch_action_key": diagnostic_metadata.get("bayestool_branch_action_key"),
            "branch_q_features": diagnostic_metadata.get("bayestool_branch_q_features", {}),
            "branch_horizon": diagnostic_metadata.get("bayestool_branch_horizon", 0),
            "shared_prefix": bool(diagnostic_metadata.get("bayestool_branch_shared_prefix")),
            "context_sampling": world_runtime.context_metadata(),
            "schedule_metadata": world_runtime.schedule_metadata(),
        }
        sample.metadata["bayes_aux_state"] = aux_state
        sample.metadata["bayes_aux_prompt"] = aux_prompt
        sample.metadata["bayes_content_signature"] = aux_content_signature
        sample.metadata["bayes_branch_events"] = list(branch_events)
        if bool(getattr(bayestool_config, "use_persistent_session_belief", True)):
            sample.metadata["bayestool_session_belief"] = belief_runtime.export_replay_record()
        else:
            sample.metadata.pop("bayestool_session_belief", None)
        sample.metadata["coupling_id"] = bayes_coupling_id
        sample.metadata["world_id"] = world_runtime.spec.world_id
        sample.metadata["latent_world_id"] = world_runtime.spec.latent_world_id
        sample.metadata["latent_seed"] = int(world_runtime.spec.latent_seed)
        sample.metadata["world_slot"] = int(world_runtime.spec.world_slot)
        sample.metadata["replica_id"] = int(world_runtime.spec.replica_id)
        sample.metadata["world_slot_role"] = world_slot_role
        sample.metadata["variant_id"] = variant_id
        sample.metadata["slot_weight"] = slot_weight
        sample.metadata["policy_version"] = policy_version
        sample.metadata["decision_group_size"] = decision_group_size
        sample.metadata["sibling_group_id"] = sibling_group_id
        sample.metadata["question_id"] = question_id
        sample.metadata["initial_input_hash"] = initial_input_hash
        sample.metadata["decision_event_id"] = decision_event_id
        sample.metadata["decision_prefix_hash"] = decision_prefix_hash
        sample.metadata["selected_decision_event"] = selected_decision_event
        sample.metadata["runtime_state_digest"] = frozen_runtime_digest
        sample.metadata["decision_group_id"] = decision_group_id
        sample.metadata["belief_model_version"] = belief_runtime.belief_model_version
        sample.metadata["bayes_q_head_version"] = bayestool_q_version
        sample.metadata["bayes_supervision"] = world_runtime.hidden_supervision_metadata()
        if export_canonical_replay is not None:
            replay = export_canonical_replay(
                sample.metadata,
                trajectory_id=str(sample.metadata.get("rollout_id") or f"sample-{sample.index}"),
            )
            sample.metadata["belief_replay"] = replay
            sample.metadata["belief_replay_validation_errors"] = []
    sample.metadata["premature_final"] = bool(navigation_state.get("premature_final"))
    sample.metadata["duplicate_page_calls"] = int(navigation_state.get("duplicate_page_calls", 0))
    sample.metadata["duplicate_region_calls"] = int(navigation_state.get("duplicate_region_calls", 0))
    sample.metadata["unnecessary_tool_calls"] = int(navigation_state.get("unnecessary_tool_calls", 0))
    sample.metadata["no_information_gain_calls"] = int(navigation_state.get("no_information_gain_calls", 0))
    sample.metadata["final_supported_by_evidence"] = bool(
        navigation_state.get("final_supported_by_evidence", False)
    )
    sample.metadata["had_evidence_guard_recovery"] = bool(
        navigation_state.get("had_evidence_guard_recovery")
    )
    sample.metadata["rejected_final_count"] = int(navigation_state.get("rejected_final_count", 0) or 0)
    sample.metadata["had_infra_error"] = bool(navigation_state.get("had_infra_error"))
    sample.metadata["infra_error_messages"] = list(navigation_state.get("infra_error_messages", []))
    sample.metadata["all_pages_checked"] = bool(
        isinstance(navigation_state.get("page_count"), int)
        and int(navigation_state.get("page_count")) > 0
        and not navigation_state.get("unvisited_pages", [])
    )
    generation_turn_count = int(sample.metadata["generation_call_count"])
    unique_tool_count = len(sample.metadata["tool_execution"]["unique_tools"])
    multi_tool_rollout = sample.tool_call_count >= 2
    multi_tool_type_rollout = unique_tool_count >= 2
    sample.metadata["generation_turn_count"] = generation_turn_count
    sample.metadata["unique_tool_count"] = unique_tool_count
    sample.metadata["multi_turn_rollout"] = generation_turn_count > 1
    sample.metadata["multi_call_rollout"] = multi_tool_rollout
    sample.metadata["multi_tool_rollout"] = multi_tool_rollout
    sample.metadata["multi_tool_type_rollout"] = multi_tool_type_rollout
    sample.metadata["completed_multi_tool_rollout"] = bool(
        multi_tool_rollout and terminal_status == "completed"
    )
    sample.metadata["completed_multi_tool_type_rollout"] = bool(
        multi_tool_type_rollout and terminal_status == "completed"
    )

    _set_rollout_status(sample, terminal_status, reason=terminal_reason)
    if bool(sample.metadata.get("question_skipped")):
        # A semantically incomplete realization cannot be repaired by mixing
        # it with another question.  Keep the trajectory for diagnostics but
        # remove it from RL grouping/weight statistics.
        sample.valid_for_rl = False
        sample.remove_sample = True
        sample.metadata["valid_for_rl"] = False
        sample.metadata["exclude_from_group_statistics"] = True
        sample.metadata["question_skipped"] = True

    if getattr(args, "prm_enable", False) and not bayestool_enabled:
        if prm_pending_tasks:
            prm_results = await asyncio.gather(*[task for _, task in prm_pending_tasks], return_exceptions=True)
            for (step_idx, _), result in zip(prm_pending_tasks, prm_results, strict=False):
                if isinstance(result, Exception):
                    prm_step_details.append({"status": "exception", "step_index": step_idx, "scores": [0], "mean_score": 0.0, "votes": []})
                    continue
                result["step_index"] = step_idx
                prm_step_details.append(result)
            prm_step_details.sort(key=lambda item: item.get("step_index", 10**9))
            prm_step_scores = [float(item.get("mean_score", 0.0)) for item in prm_step_details]
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
        }

    prm_score_by_step = {
        int(item["step_index"]): float(item.get("mean_score", 0.0))
        for item in sample.metadata.get("prm", {}).get("step_details", [])
        if isinstance(item, dict) and "step_index" in item
    }
    step_wise_steps = [
        {
            "step_index": int(span["step_index"]),
            "token_start": int(span["token_start"]),
            "token_end": int(span["token_end"]),
            "prm_score": float(prm_score_by_step.get(int(span["step_index"]), 0.0)),
        }
        for span in step_action_spans
    ]
    sample.metadata["step_wise"] = {
        "steps": step_wise_steps,
        "step_token_spans": [[item["token_start"], item["token_end"]] for item in step_wise_steps],
        "step_scores": [item["prm_score"] for item in step_wise_steps],
    }
    if branch_children:
        sample.metadata["bayestool_branch_child_count"] = len(branch_children)
        return [sample, *branch_children]
    return sample


def _trajectory_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    navigation = metadata.get("navigation_state", {}) if isinstance(metadata, dict) else {}
    tool_execution = metadata.get("tool_execution", {}) if isinstance(metadata, dict) else {}
    unique_tools = tool_execution.get("unique_tools", []) if isinstance(tool_execution, dict) else []
    if not isinstance(unique_tools, list):
        unique_tools = []
    visited_pages = navigation.get("visited_pages", metadata.get("visited_pages", []))
    if not isinstance(visited_pages, list):
        visited_pages = []
    tool_call_count = int(metadata.get("tool_call_count", tool_execution.get("call_count", 0) or 0))
    generation_turn_count = int(metadata.get("generation_turn_count", metadata.get("generation_call_count", 0) or 0))
    multi_tool_rollout = bool(metadata.get("multi_tool_rollout", tool_call_count >= 2))
    multi_call_rollout = bool(metadata.get("multi_call_rollout", multi_tool_rollout))
    completed_multi_tool_rollout = bool(
        metadata.get("completed_multi_tool_rollout", multi_tool_rollout and metadata.get("rollout_status") == "completed")
    )
    multi_tool_type_rollout = bool(metadata.get("multi_tool_type_rollout", len(unique_tools) >= 2))
    completed_multi_tool_type_rollout = bool(
        metadata.get(
            "completed_multi_tool_type_rollout",
            multi_tool_type_rollout and metadata.get("rollout_status") == "completed",
        )
    )
    duplicate_page_calls = int(metadata.get("duplicate_page_calls", navigation.get("duplicate_page_calls", 0) or 0))
    duplicate_region_calls = int(metadata.get("duplicate_region_calls", navigation.get("duplicate_region_calls", 0) or 0))
    unnecessary_tool_calls = int(metadata.get("unnecessary_tool_calls", navigation.get("unnecessary_tool_calls", 0) or 0))
    no_information_gain_calls = int(metadata.get("no_information_gain_calls", navigation.get("no_information_gain_calls", 0) or 0))
    appropriate_tool_calls = int(metadata.get("appropriate_tool_calls", navigation.get("appropriate_tool_calls", 0) or 0))
    premature_final = bool(metadata.get("premature_final", navigation.get("premature_final", False)))
    answer_page = metadata.get("answer_page")
    try:
        answer_page = int(answer_page) if answer_page is not None else None
    except (TypeError, ValueError):
        answer_page = None
    answer_page_visited = bool(answer_page is not None and answer_page in {int(page) for page in visited_pages if str(page).isdigit()})
    prediction_found = bool(
        metadata.get("prediction_found_in_document", navigation.get("prediction_found_in_document", False))
    )
    prediction_relation_matched = bool(
        metadata.get("prediction_relation_matched", navigation.get("prediction_relation_matched", False))
    )
    supporting_pages = navigation.get("supporting_pages", [])
    supporting_regions = navigation.get("supporting_regions", [])
    final_supported = bool(
        prediction_found
        and prediction_relation_matched
        and bool(supporting_pages or supporting_regions)
    )
    evidence_sufficient = bool(metadata.get("evidence_sufficient", navigation.get("evidence_sufficient", False)))
    visual_evidence_sufficient = bool(metadata.get("visual_evidence_sufficient", navigation.get("visual_evidence_sufficient", False)))
    page_count = navigation.get("page_count")
    all_pages_checked = bool(
        isinstance(page_count, int)
        and page_count > 0
        and not navigation.get("unvisited_pages", [])
    )
    # Localization and alignment are deliberately separate.  A wrong answer
    # can still receive page/tool/search credit, but only a prediction that
    # satisfies the question-bound relation receives alignment/evidence credit.
    page_visit_reward = 0.5 if answer_page_visited else 0.0
    valid_candidates = _valid_evidence_candidates(metadata.get("evidence_candidates", navigation.get("evidence_candidates", [])))
    candidate_localization_reward = 0.5 if valid_candidates else 0.0
    evidence_localization_reward = max(page_visit_reward, candidate_localization_reward)
    evidence_alignment_reward = 1.0 if final_supported else 0.0
    # Keep the historical field as the strict, relation-aligned evidence term.
    evidence_reward = evidence_alignment_reward
    tool_selection_reward = 0.2 if appropriate_tool_calls > 0 else 0.0
    tool_cost = 0.1 * max(0, tool_call_count)
    process_reward = evidence_localization_reward + evidence_alignment_reward + tool_selection_reward
    process_reward -= 0.1 * unnecessary_tool_calls
    process_reward -= 0.1 * (duplicate_page_calls + duplicate_region_calls)
    process_reward -= 0.1 * no_information_gain_calls
    if premature_final:
        process_reward -= 1.0
    return {
        "visited_pages": sorted({int(page) for page in visited_pages if str(page).isdigit()}),
        "parsed_pages": sorted({int(page) for page in navigation.get("parsed_pages", []) if str(page).isdigit()}),
        "rendered_pages": sorted({int(page) for page in navigation.get("rendered_pages", []) if str(page).isdigit()}),
        "cropped_regions": navigation.get("cropped_regions", []),
        "zoomed_regions": navigation.get("zoomed_regions", []),
        "ocr_regions": navigation.get("ocr_regions", []),
        "unvisited_pages": sorted({int(page) for page in navigation.get("unvisited_pages", []) if str(page).isdigit()}),
        "answer_page_visited": answer_page_visited,
        "answer_page": answer_page,
        "page_visit_reward": page_visit_reward,
        "evidence_localization_reward": evidence_localization_reward,
        "evidence_alignment_reward": evidence_alignment_reward,
        "premature_final": premature_final,
        "duplicate_page_calls": duplicate_page_calls,
        "duplicate_region_calls": duplicate_region_calls,
        "unnecessary_tool_calls": unnecessary_tool_calls,
        "no_information_gain_calls": no_information_gain_calls,
        "final_supported_by_evidence": final_supported,
        "prediction_found_in_document": prediction_found,
        "prediction_relation_matched": prediction_relation_matched,
        "evidence_candidates": metadata.get("evidence_candidates", navigation.get("evidence_candidates", [])),
        "evidence_reason": metadata.get("evidence_reason", navigation.get("evidence_reason")),
        "evidence_sufficient": evidence_sufficient,
        "visual_evidence_sufficient": visual_evidence_sufficient,
        "supporting_pages": supporting_pages,
        "supporting_regions": supporting_regions,
        "stop_reason": navigation.get("stop_reason"),
        "all_pages_checked": all_pages_checked,
        "generation_turn_count": generation_turn_count,
        "tool_call_count": tool_call_count,
        "unique_tool_count": len(unique_tools),
        "multi_turn_rollout": generation_turn_count > 1,
        "multi_call_rollout": multi_call_rollout,
        "multi_tool_rollout": multi_tool_rollout,
        "multi_tool_type_rollout": multi_tool_type_rollout,
        "completed_multi_tool_rollout": completed_multi_tool_rollout,
        "completed_multi_tool_type_rollout": completed_multi_tool_type_rollout,
        "evidence_reward": evidence_reward,
        "tool_selection_reward": tool_selection_reward,
        "tool_cost": tool_cost,
        "latency_cost": float(metadata.get("latency_cost", 0.0) or 0.0),
        "text_token_cost": float(metadata.get("text_token_cost", 0.0) or 0.0),
        "image_token_cost": float(metadata.get("image_token_cost", 0.0) or 0.0),
        "normalized_latency_cost": float(metadata.get("normalized_latency_cost", metadata.get("latency_cost", 0.0)) or 0.0),
        "normalized_text_token_cost": float(metadata.get("normalized_text_token_cost", metadata.get("text_token_cost", 0.0)) or 0.0),
        "normalized_image_token_cost": float(metadata.get("normalized_image_token_cost", metadata.get("image_token_cost", 0.0)) or 0.0),
        "process_reward": process_reward,
        "had_evidence_guard_recovery": bool(metadata.get("had_evidence_guard_recovery", navigation.get("had_evidence_guard_recovery", False))),
        "rejected_final_count": int(metadata.get("rejected_final_count", navigation.get("rejected_final_count", 0)) or 0),
        "assistant_token_masks": metadata.get("assistant_token_masks", []),
        "action_rewards": metadata.get("action_rewards", []),
        "rejected_action_indices": metadata.get("rejected_action_indices", []),
        "runtime_evidence_source": metadata.get("runtime_evidence_source", "tool_observation"),
        "ground_truth_metadata_used_in_prompt": bool(metadata.get("ground_truth_metadata_used_in_prompt", False)),
        "ground_truth_metadata_used_in_action_selection": bool(metadata.get("ground_truth_metadata_used_in_action_selection", False)),
    }


def _reward_consistency_errors(result: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    navigation = metadata.get("navigation_state", {}) if isinstance(metadata, dict) else {}
    visited = navigation.get("visited_pages", metadata.get("visited_pages", []))
    visited_set = {int(page) for page in visited if str(page).isdigit()} if isinstance(visited, list) else set()
    answer_page = metadata.get("answer_page")
    try:
        answer_page = int(answer_page) if answer_page is not None else None
    except (TypeError, ValueError):
        answer_page = None
    expected_answer_page_visited = bool(answer_page is not None and answer_page in visited_set)
    if bool(result.get("answer_page_visited")) != expected_answer_page_visited:
        errors.append("answer_page_visited disagrees with answer_page and visited_pages")
    candidates = result.get("evidence_candidates", [])
    valid_candidates = _valid_evidence_candidates(candidates)
    if bool(result.get("evidence_sufficient")) and not valid_candidates:
        errors.append("evidence_sufficient_without_valid_candidate")
    prediction_found = bool(result.get("prediction_found_in_document"))
    prediction_relation = bool(result.get("prediction_relation_matched"))
    supporting_pages = result.get("supporting_pages") or []
    supporting_regions = result.get("supporting_regions") or []
    expected_final_supported = bool(prediction_found and prediction_relation and (supporting_pages or supporting_regions))
    if bool(result.get("final_supported_by_evidence")) and not prediction_found:
        errors.append("final_supported_without_document_match")
    if bool(result.get("final_supported_by_evidence")) and not prediction_relation:
        errors.append("final_supported_without_relation")
    if bool(result.get("final_supported_by_evidence")) and not (supporting_pages or supporting_regions):
        errors.append("final_supported_without_supporting_location")
    if bool(result.get("final_supported_by_evidence")) != expected_final_supported:
        errors.append("final_supported_by_evidence_formula_mismatch")
    if float(result.get("evidence_alignment_reward", 0.0) or 0.0) > 0 and not bool(result.get("final_supported_by_evidence")):
        errors.append("positive_evidence_alignment_reward_without_final_support")
    if float(result.get("evidence_reward", 0.0) or 0.0) > 0 and not bool(result.get("final_supported_by_evidence")):
        errors.append("positive evidence_reward without final_supported_by_evidence")
    generation = metadata.get("generation", {}) if isinstance(metadata, dict) else {}
    generation_steps = generation.get("steps", []) if isinstance(generation, dict) else []
    if not isinstance(generation_steps, list):
        generation_steps = []
    for step_index, step in enumerate(generation_steps):
        if not isinstance(step, dict) or not bool(step.get("visual_input_required")):
            continue
        if not bool(step.get("vision_placeholder_present")):
            errors.append(f"visual_step_{step_index}_missing_vision_placeholder")
        if int(step.get("image_tensor_count", 0) or 0) <= 0:
            errors.append(f"visual_step_{step_index}_missing_image_tensor")
        if not bool(step.get("vision_input_attached")):
            errors.append(f"visual_step_{step_index}_missing_attached_vision")
        if not step.get("consumed_image_paths"):
            errors.append(f"visual_step_{step_index}_missing_consumed_image_path")
    actions = metadata.get("assistant_turns", [])
    if isinstance(actions, list):
        for index, action in enumerate(actions):
            if not isinstance(action, dict) or action.get("accepted") is not False:
                continue
            if bool(action.get("action_valid_for_policy_gradient", True)) and float(action.get("action_reward", 0.0) or 0.0) >= 0:
                errors.append(f"rejected action {index} has no negative/masked training signal")
    if metadata.get("ground_truth_metadata_used_in_prompt"):
        errors.append("ground-truth metadata was used in prompt")
    if metadata.get("ground_truth_metadata_used_in_action_selection"):
        errors.append("ground-truth metadata was used in action selection")
    return errors


async def reward_func(args, sample, **kwargs):
    """Reward a grounded document answer with optional step-wise PRM scores."""
    if isinstance(sample, (list, tuple)):
        # slime's batched_async_rm calls custom reward functions with the
        # whole pending batch.  Keep the actual reward calculation below
        # single-sample so utility, consistency, and exclusion metadata remain
        # attached to the corresponding Sample, then return the same-order
        # reward list expected by the rollout manager.
        rewards = await asyncio.gather(
            *(reward_func(args, item, **kwargs) for item in sample)
        )
        return list(rewards)
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rollout_status = str(getattr(sample, "rollout_status", metadata.get("rollout_status", "")))
    valid_for_rl = bool(getattr(sample, "valid_for_rl", metadata.get("valid_for_rl", True)))
    if rollout_status in _INFRA_STATUSES or not valid_for_rl:
        # Infrastructure failures must never become the ordinary -1 answer
        # reward.  Marking the sample removed also prevents policy-gradient
        # construction from consuming its partial token trajectory.
        sample.remove_sample = True
        result = {
            "score": 0.0,
            "total_reward": 0.0,
            "answer_reward": 0.0,
            "evidence_reward": 0.0,
            "evidence_localization_reward": 0.0,
            "evidence_alignment_reward": 0.0,
            "tool_selection_reward": 0.0,
            "tool_cost": 0.0,
            "process_reward": 0.0,
            "acc": 0.0,
            "exact_acc": 0.0,
            "quality": 0.0,
            "format": 0.0,
            "answer_correctness": 0.0,
            "answer_conciseness": 0.0,
            "format_validity": 0.0,
            "pred": "",
            "metric": "invalid_rollout",
            "valid_for_rl": False,
            "exclude_from_group_statistics": True,
            "rollout_status": rollout_status or "infra_error",
        }
        for key in (
            "tool_call_count",
            "valid_tool_call_count",
            "tool_error_count",
            "candidate_action_count",
            "executed_action_count",
            "valid_action_count",
            "invalid_action_count",
            "ignored_action_count",
            "protocol_error_count",
        ):
            result[key] = int(getattr(sample, key, metadata.get(key, 0) or 0))
        result.update(_trajectory_metrics(metadata))
        consistency_errors = _reward_consistency_errors(result, metadata)
        result["reward_consistency_valid"] = not consistency_errors
        result["reward_consistency_errors"] = consistency_errors
        metadata["reward_consistency_valid"] = not consistency_errors
        metadata["reward_consistency_errors"] = consistency_errors
        if consistency_errors:
            sample.valid_for_rl = False
            sample.remove_sample = True
            sample.rollout_status = "infra_error"
            metadata["valid_for_rl"] = False
            metadata["exclude_from_group_statistics"] = True
            metadata["rollout_status"] = "infra_error"
        return result

    result = compute_document_reward(sample.response, sample.label, metadata)
    result["tool_call_count"] = int(getattr(sample, "tool_call_count", 0))
    result["valid_tool_call_count"] = int(getattr(sample, "valid_tool_call_count", 0))
    result["tool_error_count"] = int(getattr(sample, "tool_error_count", 0))
    action_statistics = metadata.get("action_statistics", {}) if isinstance(metadata, dict) else {}
    for key in (
        "candidate_action_count",
        "executed_action_count",
        "valid_action_count",
        "invalid_action_count",
        "ignored_action_count",
        "protocol_error_count",
    ):
        result[key] = int(action_statistics.get(key, 0)) if isinstance(action_statistics, dict) else 0
    result["valid_for_rl"] = True
    result["exclude_from_group_statistics"] = False
    result["rollout_status"] = rollout_status or "completed"
    trajectory_metrics = _trajectory_metrics(metadata)
    result.update(trajectory_metrics)
    bayes_mode = bool(metadata.get("bayestool", {}).get("enabled")) if isinstance(metadata.get("bayestool"), dict) else False
    if bayes_mode and compute_bayestool_utility is not None:
        configured_tool_budget = getattr(args, "bayestool_tool_budget", None)
        if configured_tool_budget is None:
            configured_tool_budget = metadata.get("tool_budget")
        if configured_tool_budget is None:
            configured_tool_budget = TOOL_CONFIGS.get("max_tool_calls", 8)
        configured_tool_budget = max(1, int(configured_tool_budget))
        utility_metadata = dict(metadata)
        utility_metadata.update(
            {
                "tool_call_count": result["tool_call_count"],
                "tool_budget": configured_tool_budget,
                "duplicate_page_calls": trajectory_metrics.get("duplicate_page_calls", 0),
                "duplicate_region_calls": trajectory_metrics.get("duplicate_region_calls", 0),
                "unnecessary_tool_calls": trajectory_metrics.get("unnecessary_tool_calls", 0),
                "no_information_gain_calls": trajectory_metrics.get("no_information_gain_calls", 0),
            }
        )
        utility_quality = float(result.get("quality", 0.0))
        if bool(result.get("abstention")) and bool(result.get("abstention_justified")):
            # A justified abstention is explicitly safer than an unsupported
            # answer, but remains below a correct final answer.
            utility_quality = 0.425
        bayes_utility = compute_bayestool_utility(
            utility_metadata,
            utility_quality,
            tool_budget=configured_tool_budget,
        )
        metadata["bayestool_utility"] = bayes_utility
        result["bayestool_utility"] = bayes_utility
        answer_reward = float(result["score"])
        total_reward = float(bayes_utility["utility"])
    else:
        answer_reward = float(result["score"])
        process_weight = float(getattr(args, "process_reward_weight", 0.1))
        cost_weight = float(getattr(args, "tool_cost_weight", 0.05))
        total_reward = answer_reward + process_weight * float(trajectory_metrics["process_reward"])
        total_reward -= cost_weight * float(trajectory_metrics["tool_cost"])
    result["answer_reward"] = answer_reward
    result["total_reward"] = total_reward
    if bayes_mode:
        # BayesTool utility is the un-clipped RL return.  The utility helper
        # records ``utility_clipped`` separately for dashboards; clipping the
        # reward here would collapse distinct high-cost trajectories before
        # sibling-relative advantages are computed.
        result["score"] = total_reward
        result["score_clipped"] = max(-1.0, min(1.0, total_reward))
    else:
        result["score"] = max(-1.0, min(1.0, total_reward))

    outcome_reward = float(result["score"])

    if getattr(args, "prm_enable", False) and not bayes_mode:
        prm_metadata = sample.metadata.get("prm", {}) if isinstance(sample.metadata, dict) else {}
        prm_step_mean = float(prm_metadata.get("step_mean_score", 0.0))

        base_score = float(result["score"])
        outcome_reward = base_score
        final_score = base_score + float(getattr(args, "prm_step_coef", 1.0)) * prm_step_mean

        result["base_score"] = base_score
        result["prm_step_score"] = prm_step_mean
        result["score"] = final_score
        # Add one concrete PRM raw output for quick sanity-check in rollout logs.
        prm_example_eval = ""
        step_details = prm_metadata.get("step_details", [])
        if isinstance(step_details, list) and step_details:
            first_step = step_details[0] if isinstance(step_details[0], dict) else {}
            votes = first_step.get("votes", []) if isinstance(first_step, dict) else []
            if isinstance(votes, list) and votes:
                first_vote = votes[0] if isinstance(votes[0], dict) else {}
                raw_text = first_vote.get("raw_text", "") if isinstance(first_vote, dict) else ""
                if isinstance(raw_text, str):
                    prm_example_eval = raw_text
        result["prm_example_eval"] = prm_example_eval

    if sample.metadata is None:
        sample.metadata = {}
    step_wise_meta = sample.metadata.get("step_wise", {})
    if not isinstance(step_wise_meta, dict):
        step_wise_meta = {}
    step_wise_meta["outcome_reward"] = outcome_reward
    raw_step_scores = step_wise_meta.get("step_scores", [])
    if isinstance(raw_step_scores, list):
        step_wise_meta["step_scores_with_outcome"] = [float(step_score) + float(outcome_reward) for step_score in raw_step_scores]
    else:
        step_wise_meta["step_scores_with_outcome"] = []
    sample.metadata["step_wise"] = step_wise_meta

    consistency_errors = _reward_consistency_errors(result, sample.metadata)
    result["reward_consistency_valid"] = not consistency_errors
    result["reward_consistency_errors"] = consistency_errors
    sample.metadata["reward_consistency_valid"] = not consistency_errors
    sample.metadata["reward_consistency_errors"] = consistency_errors
    if consistency_errors:
        sample.valid_for_rl = False
        sample.remove_sample = True
        sample.rollout_status = "infra_error"
        sample.metadata["valid_for_rl"] = False
        sample.metadata["exclude_from_group_statistics"] = True
        sample.metadata["rollout_status"] = "infra_error"
        result["valid_for_rl"] = False
        result["exclude_from_group_statistics"] = True
        result["score"] = 0.0
        result["total_reward"] = 0.0
        result["rollout_status"] = "infra_error"

    return result
