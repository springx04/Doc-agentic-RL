# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import asyncio
import json
import logging
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
            "do not invent unread evidence, and return the final answer exactly as <final>answer</final>."
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
    matches = list(re.finditer(r"<final>\s*.*?\s*</final>", text, re.IGNORECASE | re.DOTALL))
    return matches[-1].span() if matches else None


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
    """Extract answer or a generic tool call from a model prediction."""
    final_answer, has_final = extract_final_answer(prediction)
    if has_final:
        return "answer", final_answer

    tool_call_json = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL)
    if tool_call_json:
        parsed = _parse_json_tool_call(tool_call_json.group(1))
        if parsed:
            return "tool", parsed

    tool_call_xml = re.search(
        r"<tool_call>\s*<function=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</function>\s*</tool_call>",
        prediction,
        re.DOTALL,
    )
    if tool_call_xml:
        return "tool", _parse_xml_tool_call(tool_call_xml.group(2), tool_call_xml.group(1))

    tool_call_xml_partial = re.search(
        r"<tool_call>\s*<function=([A-Za-z_][\w.-]*)>\s*(.*)",
        prediction,
        re.DOTALL,
    )
    if tool_call_xml_partial:
        return "tool", _parse_xml_tool_call(tool_call_xml_partial.group(2), tool_call_xml_partial.group(1))

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    if "<final>" in resp.lower():
        span = _find_last_final_span(resp)
        if span is not None:
            return resp[: span[1]]

    # Handle <tool_call> tags (JSON or XML format)
    if "<tool_call>" in resp and "</tool_call>" in resp:
        matches = list(re.finditer(r"<tool_call>.*?</tool_call>", resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

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


async def execute_predictions(prediction: str, execution_trace: list[dict[str, Any]] | None = None) -> tuple[str, bool]:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "tool":
        tool_call = content if isinstance(content, dict) else {}
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        if tool_name and isinstance(arguments, dict):
            result = await tool_registry.execute_tool(str(tool_name), arguments)
            success = False
            try:
                parsed_result = json.loads(result)
                success = isinstance(parsed_result, dict) and parsed_result.get("status") == "ok"
            except (json.JSONDecodeError, TypeError):
                success = not result.lower().startswith("error:")
            if execution_trace is not None:
                execution_trace.append(
                    {
                        "tool": str(tool_name),
                        "arguments": arguments,
                        "success": success,
                    }
                )
            max_obs_chars = TOOL_CONFIGS.get("max_obs_chars", 1024)
            if len(result) > max_obs_chars:
                result = result[:max_obs_chars] + f"\n... [truncated {len(result) - max_obs_chars} chars]"
            next_obs = f"\n\n<interpreter>\nTool: {tool_name}\n{result}\n</interpreter>\n\n"
            done = False
        else:
            if execution_trace is not None:
                execution_trace.append({"tool": "", "arguments": arguments, "success": False})
            next_obs = "\n\n<interpreter>\nError: Invalid tool call; missing tool name or arguments object\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        tc_fmt = _TOOL_CALL_FORMAT or "json"
        next_obs = (
            "\nMy previous action is invalid. "
            + _format_tool_hint(tc_fmt)
            + "If I want to give the final answer, I should use the format "
            "'<final>answer</final>'. Let me try again.\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Custom generation function supporting tool calls.

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

    for turn in range(TOOL_CONFIGS["max_turns"]):
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
            break

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


async def reward_func(args, sample, **kwargs):
    """Reward a grounded document answer with optional step-wise PRM scores."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    result = compute_document_reward(sample.response, sample.label, metadata)
    result["tool_call_count"] = int(getattr(sample, "tool_call_count", 0))
    result["valid_tool_call_count"] = int(getattr(sample, "valid_tool_call_count", 0))
    result["tool_error_count"] = int(getattr(sample, "tool_error_count", 0))

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
