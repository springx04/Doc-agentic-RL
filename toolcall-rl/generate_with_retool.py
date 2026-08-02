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
            "do not invent unread evidence, and wrap the supported answer in <final>...</final>. "
            "When the current page does not contain sufficient evidence and the document has unvisited "
            "pages, inspect additional pages before answering. Do not conclude that information is absent "
            "until all relevant pages have been checked. After every tool result, use the navigation state "
            "to select an unvisited page or the appropriate table/visual tool. Do not repeat a page unless "
            "a different tool is needed. When all relevant pages have been checked, do not summarize the "
            "whole document: answer concisely and end with exactly one <final>...</final> action."
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
    if not visited_pages and not search_budget_exhausted:
        if _is_absence_answer(answer):
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
        can_finish, finish_reason, supporting_pages = _can_finish(prediction, navigation_state)
        if not can_finish:
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

    if parsed.kind != "tool_call":
        invalid_count = max(1, parsed.candidate_action_count)
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
    if action_log is not None:
        action_log["valid_action_count"] = int(action_log.get("valid_action_count", 0)) + 1
        action_log["executed_action_count"] = int(action_log.get("executed_action_count", 0)) + 1
        action_log["actions"][-1].update(
            {"valid": True, "executed": True, "tool": tool_name, "auto_routed": auto_routed}
        )

    try:
        result = await tool_registry.execute_tool(tool_name, arguments)
    except Exception as exc:  # the model action was sent, but the tool failed
        if navigation_state is not None and _is_infrastructure_tool_error(exc):
            navigation_state["had_infra_error"] = True
            navigation_state["infra_error_messages"] = list(navigation_state.get("infra_error_messages", [])) + [str(exc)]
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
        fallback = _tool_failure_fallback(tool_name, arguments, str(exc), navigation_state)
        if fallback is not None:
            if execution_trace is not None:
                execution_trace[-1]["recovery_observation"] = True
                execution_trace[-1]["recovered_with_fallback"] = True
            return fallback, False
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
                "auto_routed": auto_routed,
            }
        )

    if not success:
        tool_error_text = str((parsed_result or {}).get("error") if isinstance(parsed_result, dict) else result)
        if navigation_state is not None and _is_infrastructure_tool_error(tool_error_text):
            navigation_state["had_infra_error"] = True
            navigation_state["infra_error_messages"] = list(navigation_state.get("infra_error_messages", [])) + [tool_error_text]
        if action_log is not None:
            action_log["tool_error_count"] = int(action_log.get("tool_error_count", 0)) + 1
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

    _update_navigation_state(navigation_state, tool_name, arguments, parsed_result, success)
    _update_pending_visual_input(
        navigation_state,
        tool_name,
        image_paths,
        parsed_result,
        arguments,
    )
    limited_result = _limit_tool_result(str(result), int(TOOL_CONFIGS.get("max_obs_chars", 8192)))
    if navigation_state is not None:
        limited_result = f"{limited_result}\n{_navigation_status_text(navigation_state)}"
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
            latest[observation_key] = observation_ids[:available]
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


_INFRA_STATUSES = {
    "generation_empty",
    "generation_error",
    "context_overflow",
    "tool_error",
    "infra_error",
}


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
    # Answer page/bbox are training/evaluation metadata only.  They are kept
    # outside navigation_state so they cannot leak into the prompt or alter
    # online action selection.
    diagnostic_metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    diagnostic_answer_page = diagnostic_metadata.get("answer_page", diagnostic_metadata.get("target_page"))
    try:
        diagnostic_answer_page = int(diagnostic_answer_page) if diagnostic_answer_page is not None else None
    except (TypeError, ValueError):
        diagnostic_answer_page = None
    diagnostic_answer_bbox = diagnostic_metadata.get("answer_bbox")
    navigation_state = _new_navigation_state(task_prompt)
    # Do not carry a model-specific <think> suffix into the strict action
    # protocol.  The assistant turn must begin with its one action tag.
    prompt = format_conversation_with_tools(
        prompt=f"{task_prompt}\n\n{_navigation_status_text(navigation_state)}",
        tools=tool_specs,
        tool_call_format=tc_format,
    )
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
    max_tool_steps = max(1, int(TOOL_CONFIGS.get("max_tool_calls", 8)))
    max_turns = max(max_tool_steps + 2, int(TOOL_CONFIGS.get("max_turns", max_tool_steps + 2)))
    tool_call_count = 0

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []

    for turn in range(max_turns):
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
        for stop_text in ("</tool_call>", "</final>"):
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

        try:
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

        action_token_start = len(response_token_ids)
        response += cur_response
        response_token_ids.extend(cur_response_token_ids)
        loss_masks.extend([1] * len(cur_response_token_ids))
        sample.rollout_log_probs.extend(cur_log_probs)
        step_action_spans.append(
            {"step_index": turn, "token_start": action_token_start, "token_end": len(response_token_ids)}
        )

        trace_count_before = len(execution_trace)
        next_obs, done = await execute_predictions(
            cur_response,
            execution_trace=execution_trace,
            action_log=action_log,
            turn=turn,
            navigation_state=navigation_state,
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
        recovery_observation = bool(latest_tool.get("recovery_observation"))
        if not recovery_observation and (
            not latest_tool.get("executed") or not latest_tool.get("success")
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
        current_images.extend(obs_images)
        if obs_train_inputs:
            multimodal_train_inputs_buffer.append(obs_train_inputs)
        text_observation_token_ids = list(obs_token_ids)
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
                "action_token_ids": list(cur_response_token_ids),
                "observation_token_ids": list(obs_token_ids),
                "text_observation_token_ids": text_observation_token_ids,
                "image_data": list(obs_image_data),
                "image_token_count": int(_new_image_token_count or 0),
                "image_paths": list(image_paths_for_next),
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

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
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
        span = step_action_spans[action_index] if action_index < len(step_action_spans) else {}
        valid_for_gradient = bool(action.get("action_valid_for_policy_gradient", True))
        action_reward = float(action.get("action_reward", 0.0) or 0.0)
        action["action_valid_for_policy_gradient"] = valid_for_gradient
        action["action_reward"] = action_reward
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
            rejected_action_indices.append(action_index)

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
    answer_reward = float(result["score"])
    process_weight = float(getattr(args, "process_reward_weight", 0.1))
    cost_weight = float(getattr(args, "tool_cost_weight", 0.05))
    total_reward = answer_reward + process_weight * float(trajectory_metrics["process_reward"])
    total_reward -= cost_weight * float(trajectory_metrics["tool_cost"])
    result["answer_reward"] = answer_reward
    result["total_reward"] = total_reward
    result["score"] = max(-1.0, min(1.0, total_reward))

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
