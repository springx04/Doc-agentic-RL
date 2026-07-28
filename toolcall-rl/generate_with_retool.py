# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
import json
import logging
from pathlib import Path
import re
import time
import uuid
from typing import Any

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

logger = logging.getLogger(__name__)

_PRM_SEMAPHORE: asyncio.Semaphore | None = None
_PRM_TOKENIZER: Any = None


def _get_generation_prompt_suffix(sample_prompt: str) -> str:
    """Extract suffix after the last '<|im_start|>assistant\\n' in sample.prompt.

    sample.prompt is produced by tokenizer.apply_chat_template with the user's
    kwargs, so the suffix faithfully reflects the intended generation prompt.
    e.g. Qwen3.5 default → '<think>\\n', Qwen3 default → ''.
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
    """Format conversation using Jinja2 template with tool support"""
    raw_template = TOOL_TEMPLATE_XML if tool_call_format == "xml" else TOOL_TEMPLATE_JSON
    template = Template(raw_template)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a document-understanding agent. Inspect the document path provided in the task "
            "with the available tools before answering. Choose tools deliberately: parse_document for "
            "general text, render/crop/zoom plus OCR for visual regions, detect_layout for coordinates, "
            "extract_table for tables, and chart_to_table for charts. Ground the answer in tool output, "
            "do not invent unread evidence, and wrap the supported answer in <final>...</final>."
        )

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def _extract_task_prompt(prompt: str | list[dict[str, str]]) -> str:
    """Recover the user task if slime already applied a Qwen chat template."""
    if isinstance(prompt, list):
        users = [str(message.get("content", "")) for message in prompt if message.get("role") == "user"]
        return users[-1] if users else json.dumps(prompt, ensure_ascii=False)
    matches = re.findall(r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    return matches[-1].strip() if matches else prompt


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
    action_log.setdefault("actions", []).append(
        {
            "turn": turn,
            "kind": parsed.kind,
            "candidate_action_count": parsed.candidate_action_count,
            "raw": raw[:16000],
            "reason": parsed.reason,
            **extra,
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
    return status == "ok", payload, status or None


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
        candidate["truncated"] = len(candidate_pages) < len(original_pages)
        candidate["has_more_pages"] = len(candidate_pages) < int(candidate.get("page_count", len(original_pages)))
        encoded = json.dumps(candidate, ensure_ascii=False, indent=2, default=str)
        if len(encoded) > max_chars and kept:
            break
        if len(encoded) > max_chars:
            page_copy = dict(page) if isinstance(page, dict) else {"markdown": str(page)}
            page_copy["markdown"] = str(page_copy.get("markdown", ""))[: max(1, max_chars // 2)]
            kept = [page_copy]
            break
        kept = candidate_pages
    payload["pages"] = kept
    payload["returned_pages"] = [item.get("page_number") for item in kept if isinstance(item, dict)]
    payload["truncated"] = len(kept) < len(original_pages) or bool(payload.get("truncated"))
    payload["has_more_pages"] = bool(payload.get("has_more_pages")) or len(kept) < int(payload.get("page_count", len(original_pages)))
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


async def execute_predictions(
    prediction: str,
    execution_trace: list[dict[str, Any]] | None = None,
    action_log: dict[str, Any] | None = None,
    turn: int | None = None,
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

    if parsed.kind == "final":
        if action_log is not None:
            action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
            action_log["actions"][-1].update({"valid": True, "executed": False})
        if execution_trace is not None:
            execution_trace.append({"kind": "final", "turn": turn, "raw": prediction, "valid": True, "executed": False})
        _set_terminal_status(action_log, "completed")
        return "", True

    if parsed.kind != "tool_call":
        invalid_count = max(1, parsed.candidate_action_count)
        if action_log is not None:
            action_log["invalid_action_count"] = int(action_log.get("invalid_action_count", 0)) + invalid_count
            action_log["ignored_action_count"] = int(action_log.get("ignored_action_count", 0)) + parsed.candidate_action_count
            action_log["protocol_error_count"] = int(action_log.get("protocol_error_count", 0)) + 1
            action_log["actions"][-1].update({"valid": False, "executed": False})
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": parsed.kind,
                    "turn": turn,
                    "raw": prediction,
                    "valid": False,
                    "executed": False,
                    "reason": parsed.reason,
                    "candidate_action_count": parsed.candidate_action_count,
                }
            )
        _set_terminal_status(action_log, "model_protocol_error")
        return "", True

    tool_call = parsed.value if isinstance(parsed.value, dict) else {}
    valid, validation_error = _validate_tool_call(tool_call)
    if not valid:
        if action_log is not None:
            action_log["invalid_action_count"] = int(action_log.get("invalid_action_count", 0)) + 1
            action_log["ignored_action_count"] = int(action_log.get("ignored_action_count", 0)) + 1
            action_log["protocol_error_count"] = int(action_log.get("protocol_error_count", 0)) + 1
            action_log["actions"][-1].update({"valid": False, "executed": False, "reason": validation_error})
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": "tool_call",
                    "turn": turn,
                    "tool": tool_call.get("name", ""),
                    "arguments": tool_call.get("arguments", {}),
                    "valid": False,
                    "executed": False,
                    "success": False,
                    "reason": validation_error,
                }
            )
        _set_terminal_status(action_log, "model_protocol_error")
        return "", True

    tool_name = str(tool_call["name"])
    arguments = dict(tool_call.get("arguments", {}))
    if action_log is not None:
        action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
        action_log["executed_action_count"] = int(action_log.get("executed_action_count", 0)) + 1
        action_log["actions"][-1].update({"valid": True, "executed": True, "tool": tool_name})

    try:
        result = await tool_registry.execute_tool(tool_name, arguments)
    except Exception as exc:  # the model action was sent, but the tool failed
        if action_log is not None:
            action_log["tool_error_count"] = int(action_log.get("tool_error_count", 0)) + 1
            action_log["actions"][-1].update({"success": False, "tool_error": str(exc)})
        if execution_trace is not None:
            execution_trace.append(
                {
                    "kind": "tool_call",
                    "turn": turn,
                    "tool": tool_name,
                    "arguments": arguments,
                    "valid": True,
                    "executed": True,
                    "success": False,
                    "error": str(exc),
                }
            )
        _set_terminal_status(action_log, "tool_error")
        return "", True

    success, parsed_result, result_status = _parse_tool_result(result)
    image_paths, image_candidates = _image_paths_from_result(parsed_result if success else None)
    media_error = bool(success and image_candidates and not image_paths)
    trace_item = {
        "kind": "tool_call",
        "turn": turn,
        "tool": tool_name,
        "arguments": arguments,
        "valid": True,
        "executed": True,
        "success": success,
        "result_status": result_status,
        "image_paths": image_paths,
        "image_candidates": image_candidates,
        "media_error": media_error,
        "result": result,
    }
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
            }
        )

    if not success:
        if action_log is not None:
            action_log["tool_error_count"] = int(action_log.get("tool_error_count", 0)) + 1
        _set_terminal_status(action_log, "tool_error")
        return "", True

    limited_result = _limit_tool_result(str(result), int(TOOL_CONFIGS.get("max_obs_chars", 8192)))
    next_obs = f"<interpreter>\nTool: {tool_name}\n{limited_result}\n</interpreter>"
    return next_obs, False


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


def _encode_tool_observation(
    state: GenerateState,
    observation: str,
    image_paths: list[str],
) -> tuple[list[int], str, list[str], list[Any], dict[str, Any] | None, int]:
    """Encode an observation as a new user turn, including real image tokens."""
    if not image_paths:
        encoded_text = f"<|im_end|>\n<|im_start|>user\n{observation}<|im_end|>\n<|im_start|>assistant\n"
        token_ids = state.tokenizer(encoded_text, add_special_tokens=False)["input_ids"]
        return token_ids, encoded_text, [], [], None, 0

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
            "<|im_end|>\n<|im_start|>assistant\n"
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
        return token_ids, encoded_text, image_data, images, train_inputs, image_tokens_count
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


_INFRA_STATUSES = {"generation_error", "context_overflow", "infra_error"}


def _set_rollout_status(sample: Sample, status: str, *, reason: str | None = None) -> None:
    """Persist the rollout state in both runtime fields and serialized metadata."""
    valid_for_rl = status not in _INFRA_STATUSES
    sample.rollout_status = status
    sample.valid_for_rl = valid_for_rl
    if status == "context_overflow":
        sample.status = Sample.Status.TRUNCATED
    elif status == "completed":
        sample.status = Sample.Status.COMPLETED
    elif status != "completed":
        sample.status = Sample.Status.FAILED
    if not valid_for_rl:
        sample.remove_sample = True
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["rollout_status"] = status
    sample.metadata["valid_for_rl"] = valid_for_rl
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


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Generate a strict, multi-turn document-tool rollout.

    Tool observations are appended as separate user turns.  When a tool
    returns an image, the observation is processed with the model processor,
    the expanded image placeholder tokens are appended to the context, and
    the accumulated base64 image data is sent on the next generation request.
    """
    assert not getattr(args, "partial_rollout", False), "Partial rollout is not supported for this function."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_specs = tool_registry.get_tool_specs()
    tc_format = _detect_tool_call_format(state.tokenizer)
    task_prompt = _extract_task_prompt(sample.prompt)
    # Do not carry a model-specific <think> suffix into the strict action
    # protocol.  The assistant turn must begin with its one action tag.
    prompt = format_conversation_with_tools(prompt=task_prompt, tools=tool_specs, tool_call_format=tc_format)
    prompt_tokens_ids = list(state.tokenizer(prompt, add_special_tokens=False)["input_ids"])

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
    current_image_data: list[str] = []
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
        "actions": [],
    }
    generation_steps: list[dict[str, Any]] = []
    prm_step_scores: list[float] = []
    prm_step_details: list[dict[str, Any]] = []
    prm_pending_tasks: list[tuple[int, asyncio.Task]] = []
    step_action_spans: list[dict[str, int]] = []
    terminal_status: str | None = None
    terminal_reason: str | None = None

    eval_context = getattr(args, "eval_max_context_len", None)
    train_context = getattr(args, "rollout_max_context_len", None)
    if evaluation and eval_context is not None:
        max_context_length = int(eval_context)
    elif train_context is not None:
        max_context_length = int(train_context)
    else:
        max_context_length = 32768
    max_new_tokens = int(sampling_params.get("max_new_tokens") or getattr(args, "rollout_max_response_len", 2048))
    max_turns = max(2, int(TOOL_CONFIGS.get("max_turns", 16)))

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []

    for turn in range(max_turns):
        input_token_count = len(prompt_tokens_ids) + len(response_token_ids)
        image_token_count = _image_token_count(state.tokenizer, state.processor, prompt_tokens_ids + response_token_ids)
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
        for stop_text in ("</tool_call>", "</final>"):
            if stop_text not in existing_stop:
                existing_stop.append(stop_text)
        turn_sampling_params["stop"] = existing_stop

        payload = {
            "input_ids": prompt_tokens_ids + response_token_ids,
            "sampling_params": turn_sampling_params,
            "return_logprob": True,
        }
        if current_image_data:
            payload["image_data"] = list(current_image_data)

        step: dict[str, Any] = {
            "step_index": turn,
            "input_token_count": input_token_count,
            "image_token_count": image_token_count,
            "remaining_generation_tokens": turn_max_new_tokens,
            "finish_reason": None,
            "assistant_output_token_count": 0,
            "generation_error": None,
        }

        try:
            output = await post(url, payload)
            if not isinstance(output, dict):
                raise RuntimeError("generation backend returned a non-object response")
            finish_reason = _finish_reason_type(output)
            step["finish_reason"] = finish_reason
            meta_info = output.get("meta_info") or {}
            logprob_items = meta_info.get("output_token_logprobs")
            if isinstance(logprob_items, list):
                cur_response_token_ids = [int(item[1]) for item in logprob_items if isinstance(item, (list, tuple)) and len(item) >= 2]
                cur_log_probs = [float(item[0]) for item in logprob_items if isinstance(item, (list, tuple)) and len(item) >= 2]
                if im_end_id is not None and cur_response_token_ids and cur_response_token_ids[-1] == im_end_id:
                    cur_response_token_ids.pop()
                    cur_log_probs.pop()
                cur_response = state.tokenizer.decode(cur_response_token_ids)
            else:
                cur_response = str(output.get("text") or "")
                cur_response_token_ids = list(state.tokenizer(cur_response, add_special_tokens=False)["input_ids"])
                cur_log_probs = [0.0] * len(cur_response_token_ids)
        except Exception as exc:
            step["generation_error"] = str(exc)
            generation_steps.append(step)
            terminal_status = "generation_error"
            terminal_reason = str(exc)
            break

        step["assistant_output_token_count"] = len(cur_response_token_ids)
        generation_steps.append(step)
        if not cur_response_token_ids:
            # An empty response after a successful tool result is an engine or
            # multimodal-context failure, never a valid terminal answer.
            step["generation_error"] = "empty assistant output"
            terminal_status = "generation_error"
            terminal_reason = "assistant output token count is zero"
            break

        action_token_start = len(response_token_ids)
        response += cur_response
        response_token_ids.extend(cur_response_token_ids)
        loss_masks.extend([1] * len(cur_response_token_ids))
        sample.rollout_log_probs.extend(cur_log_probs)
        step_action_spans.append(
            {"step_index": turn, "token_start": action_token_start, "token_end": len(response_token_ids)}
        )

        next_obs, done = await execute_predictions(
            cur_response,
            execution_trace=execution_trace,
            action_log=action_log,
            turn=turn,
        )
        if getattr(args, "prm_enable", False):
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
            if terminal_status == "completed":
                sample.metadata = sample.metadata or {}
                sample.metadata["final_action"] = cur_response
                sample.metadata["final_answer"] = cur_response
            break

        latest_tool = execution_trace[-1] if execution_trace else {}
        if not latest_tool.get("executed") or not latest_tool.get("success"):
            terminal_status = "infra_error"
            terminal_reason = "successful tool execution did not produce a sendable observation"
            break
        if latest_tool.get("media_error"):
            terminal_status = "infra_error"
            terminal_reason = "tool reported an image path that could not be loaded"
            break

        try:
            (
                obs_token_ids,
                encoded_obs_text,
                obs_image_data,
                obs_images,
                obs_train_inputs,
                _new_image_token_count,
            ) = _encode_tool_observation(state, next_obs, latest_tool.get("image_paths", []))
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
        current_image_data.extend(obs_image_data)
        current_images.extend(obs_images)
        if obs_train_inputs:
            multimodal_train_inputs_buffer.append(obs_train_inputs)

    if terminal_status is None:
        terminal_status = "model_protocol_error"
        terminal_reason = "rollout ended before a final action"

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    if current_images:
        sample.multimodal_inputs = {"images": current_images, "videos": None}

    sample.tool_call_count = sum(1 for item in execution_trace if item.get("kind") == "tool_call" and item.get("executed"))
    sample.valid_tool_call_count = sum(
        1 for item in execution_trace if item.get("kind") == "tool_call" and item.get("executed") and item.get("success")
    )
    sample.tool_error_count = int(action_log.get("tool_error_count", 0))
    sample.tool_execution_trace = execution_trace
    for key in (
        "candidate_action_count",
        "executed_action_count",
        "valid_action_count",
        "invalid_action_count",
        "ignored_action_count",
        "protocol_error_count",
    ):
        action_log[key] = int(action_log.get(key, 0))

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
    )}
    sample.metadata["generation_steps"] = generation_steps
    last_generation = generation_steps[-1] if generation_steps else {
        "input_token_count": len(prompt_tokens_ids),
        "image_token_count": 0,
        "remaining_generation_tokens": 0,
        "finish_reason": None,
        "assistant_output_token_count": 0,
        "generation_error": terminal_reason,
    }
    for key in (
        "input_token_count",
        "image_token_count",
        "remaining_generation_tokens",
        "finish_reason",
        "assistant_output_token_count",
        "generation_error",
    ):
        sample.metadata[key] = last_generation.get(key)
    sample.metadata["generation"] = dict(last_generation)
    sample.metadata["generation"]["steps"] = generation_steps

    _set_rollout_status(sample, terminal_status, reason=terminal_reason)

    if getattr(args, "prm_enable", False):
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
    return sample


async def reward_func(args, sample, **kwargs):
    """Reward a grounded document answer with optional step-wise PRM scores."""
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
            "acc": 0.0,
            "exact_acc": 0.0,
            "quality": 0.0,
            "format": 0.0,
            "pred": "",
            "metric": "invalid_rollout",
            "valid_for_rl": False,
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
    result["rollout_status"] = rollout_status or "completed"

    outcome_reward = float(result["score"])

    if getattr(args, "prm_enable", False):
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

    return result
