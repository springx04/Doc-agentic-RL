"""Local visual test bench for the Tool-Call RL document tools.

This server is deliberately isolated from training code.  It binds only to
localhost and uses the existing ToolRegistry as the sole tool execution path.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import http.client
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import site
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse


APP_DIR = Path(__file__).resolve().parent
TOOLCALL_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
OUTPUT_DIR = APP_DIR / "outputs"
CACHE_DIR = APP_DIR / "cache"
RUNTIME_DIR = APP_DIR / "runtime"
STUDIO_VERSION = "2026.07.12.4"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Keep UI artifacts and backend caches out of training paths.
os.environ.setdefault("OPENCLAW_TOOL_OUTPUT_DIR", str(OUTPUT_DIR))
os.environ.setdefault("OPENCLAW_TOOL_CACHE_DIR", str(CACHE_DIR))
# Prevent a global Conda cmd.exe AutoRun hook from recursively activating base
# in backend worker processes started by this isolated test bench.
os.environ.setdefault("CONDA_AUTO_ACTIVATE_BASE", "false")
os.environ.setdefault("CONDA_CHANGEPS1", "false")

# These are model artifacts downloaded for the document tools, not a runtime
# import of the separately downloaded Docling source checkout.
_TEMP_BACKEND_ROOT = Path(tempfile.gettempdir())
_DEFAULT_DOCLING_ARTIFACTS = _TEMP_BACKEND_ROOT / "openclaw_docling_models"
_DEFAULT_DEPLOT_MODEL = _TEMP_BACKEND_ROOT / "openclaw_deplot_model"
_DEFAULT_MODELSCOPE_RUNTIME = _TEMP_BACKEND_ROOT / "openclaw_modelscope_runtime"
_DEFAULT_PADDLE_RUNTIME = _TEMP_BACKEND_ROOT / "openclaw_paddle_runtime"
if (_DEFAULT_DOCLING_ARTIFACTS / "docling-project--docling-layout-heron").is_dir():
    os.environ.setdefault("OPENCLAW_DOCLING_ARTIFACTS_PATH", str(_DEFAULT_DOCLING_ARTIFACTS))
if _DEFAULT_DEPLOT_MODEL.is_dir():
    os.environ.setdefault("OPENCLAW_DEPLOT_MODEL", str(_DEFAULT_DEPLOT_MODEL))
if _DEFAULT_MODELSCOPE_RUNTIME.is_dir():
    os.environ.setdefault("OPENCLAW_MODELSCOPE_RUNTIME", str(_DEFAULT_MODELSCOPE_RUNTIME))
if _DEFAULT_PADDLE_RUNTIME.is_dir():
    os.environ.setdefault("OPENCLAW_PADDLE_RUNTIME", str(_DEFAULT_PADDLE_RUNTIME))

# Optional, isolated binary dependencies used only by Tool Studio.  This is
# intentionally before the existing user-site packages so a known-good Torch
# can supersede a broken global development build without changing training.
if (RUNTIME_DIR / "torch").is_dir() and str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))
if str(TOOLCALL_DIR) not in sys.path:
    sys.path.insert(1 if sys.path and sys.path[0] == str(RUNTIME_DIR) else 0, str(TOOLCALL_DIR))

# ``python -S`` avoids automatic .pth processing that selects the globally
# installed development Torch. Add only the package locations the Studio needs,
# after the isolated runtime so its stable binary packages always win.
_store_packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Packages"
_store_sites = list(_store_packages.glob("PythonSoftwareFoundation.Python.3.11_*/LocalCache/local-packages/Python311/site-packages")) if _store_packages.is_dir() else []
for dependency_path in (Path(site.getusersitepackages()), *_store_sites, _TEMP_BACKEND_ROOT / "openclaw_pydeps"):
    if dependency_path.is_dir() and str(dependency_path) not in sys.path:
        sys.path.append(str(dependency_path))

from tool_sandbox import tool_registry  # noqa: E402
from tool_protocol import parse_assistant_action  # noqa: E402


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def normalized_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        raise ValueError("API base URL is required")
    return base if base.endswith("/v1") else f"{base}/v1"


def openai_chat_completion(config: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = str(config.get("api_key", "")).strip()
    body = {
        "model": str(config.get("model", "")).strip(),
        "messages": messages,
        "tools": tool_registry.get_tool_specs(),
        "tool_choice": "auto",
        "temperature": float(config.get("temperature", 0.1)),
        "stream": False,
    }
    if not body["model"]:
        raise ValueError("Model is required")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    endpoint = f"{normalized_base_url(str(config.get('base_url', '')))}" + "/chat/completions"
    for attempt in range(1, 4):
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(config.get("timeout", 180))) as response:
                return json.loads(response.read().decode("utf-8"))
        except http.client.IncompleteRead as exc:
            if attempt == 3:
                raise RuntimeError(f"API response ended early after {attempt} attempts: {exc}") from exc
            time.sleep(attempt)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API connection failed: {exc.reason}") from exc
    raise RuntimeError("API request did not return a response")


def text_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    """Return a text tool call only when the whole assistant turn parses."""
    parsed = parse_assistant_action(content)
    if parsed.kind != "tool_call" or not isinstance(parsed.value, dict):
        return None
    return str(parsed.value["name"]), dict(parsed.value.get("arguments", {}))


def _validate_tool_action(name: str, arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Reject placeholder/example arguments before sending them to a tool."""
    if name not in tool_registry.tools:
        return False, f"unknown tool: {name}"
    if not isinstance(arguments, dict):
        return False, "tool arguments must be an object"
    if "_invalid_arguments" in arguments:
        return False, "tool arguments are not valid JSON"

    def visit(value: Any, key: str | None = None) -> str | None:
        if isinstance(value, str) and (key or "").endswith("path"):
            normalized = value.strip().casefold().replace("\\", "/")
            if normalized in {"/path/to/file.pdf", "/path/to/document.pdf", "path/to/file.pdf", "path/to/document.pdf"}:
                return f"placeholder path is not a valid parameter: {value}"
            if "/path/to/" in normalized or (normalized.startswith("<") and normalized.endswith(">")):
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
    return reason is None, reason


def event(kind: str, **payload: Any) -> str:
    return json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n"


def tool_progress_detail(name: str, elapsed_seconds: float) -> str:
    """Describe only observable execution state; backends do not expose progress callbacks."""
    if name in {"parse_document", "detect_layout", "extract_table"}:
        phase = "Docling backend is still loading models or processing document pages"
    elif name == "ocr_region":
        phase = "OCR backend is still loading its model or recognizing the selected image"
    elif name == "chart_to_table":
        phase = "DePlot backend is still loading its model or generating the chart table"
    else:
        phase = "tool backend is still processing the request"
    return f"{phase}; elapsed {elapsed_seconds:.1f}s. No backend percentage is available."


def result_status(result: str) -> str:
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return "unknown"
    return str(payload.get("status", "unknown")) if isinstance(payload, dict) else "unknown"


def run_registered_tool(turn: int, call_id: str, name: str, arguments: dict[str, Any]) -> Iterator[str]:
    """Execute one registered tool while streaming honest timing events."""
    yield event("tool_call", turn=turn, call_id=call_id, name=name, arguments=arguments)
    yield event("tool_started", turn=turn, call_id=call_id, name=name, detail="Tool request submitted to the local registry.")
    started_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tool-studio") as executor:
        future = executor.submit(asyncio.run, tool_registry.execute_tool(name, arguments))
        while True:
            try:
                result = future.result(timeout=5)
                break
            except FutureTimeout:
                elapsed = time.monotonic() - started_at
                yield event(
                    "tool_progress",
                    turn=turn,
                    call_id=call_id,
                    name=name,
                    elapsed_seconds=round(elapsed, 1),
                    detail=tool_progress_detail(name, elapsed),
                )
            except Exception as exc:
                result = json_text({"status": "error", "tool": name, "error": str(exc)})
                break
    elapsed = round(time.monotonic() - started_at, 2)
    yield event(
        "tool_result",
        turn=turn,
        call_id=call_id,
        name=name,
        elapsed_seconds=elapsed,
        status=result_status(result),
        result=result,
    )
    return result


def _toolrl_reference_answer(page_one_text: str, page_ten_text: str) -> str:
    page_one = page_one_text.lower()
    page_ten = page_ten_text.lower()
    if "irrelevant tool detection" not in page_one or "get_date" not in page_one:
        raise RuntimeError("Page 1 OCR did not contain the Figure 1 irrelevant-tool evidence.")
    if "movie ticket" not in page_ten or "movie name" not in page_ten:
        raise RuntimeError("Page 10 OCR did not contain the Table 4 movie-ticket evidence.")
    return (
        "Figure 1 illustrates irrelevant-tool detection: the SFT model over-interprets and calls the irrelevant "
        "get_date tool for a distance question, while the RL model rejects that tool as unsuitable for calculating "
        "the distance between cities. The Figure 1 caption says SFT trajectories suffer from overthinking and limited "
        "generalization.\n\n"
        "The movie-ticket premise is not inside Figure 1; it appears in Table 4 on page 10. Its response asks for the "
        "movie name and the specific show date. The accompanying reasoning states that the date might be inferred from "
        "the current date, but the missing movie name is required to proceed with the purchase."
    )


def run_toolrl_reference_test(payload: dict[str, Any]) -> Iterator[str]:
    """Run a reproducible, no-API evidence workflow for the default ToolRL task.

    This is a reference verification flow, not a general-purpose language model.
    It exists to validate the real render/OCR tool path before RL training.
    """
    document_path = Path(str(payload.get("document_path") or "")).expanduser()
    problem = _ensure_reference_document(document_path)
    if problem:
        yield event("error", message=problem)
        return
    yield event("run_started", max_turns=2, tool_count=len(tool_registry.get_tool_specs()), mode="offline_reference")
    page_one_raw = yield from run_registered_tool(
        1,
        "reference-page-1",
        "ocr_region",
        {"document_path": str(document_path), "page_number": 1, "dpi": 200, "engine": "rapidocr", "max_lines": 180},
    )
    page_ten_raw = yield from run_registered_tool(
        2,
        "reference-page-10",
        "ocr_region",
        {"document_path": str(document_path), "page_number": 10, "dpi": 200, "engine": "rapidocr", "max_lines": 220},
    )
    try:
        page_one = json.loads(page_one_raw)
        page_ten = json.loads(page_ten_raw)
        if page_one.get("status") != "ok" or page_ten.get("status") != "ok":
            raise RuntimeError("One or more OCR calls failed; inspect the corresponding tool result above.")
        answer = _toolrl_reference_answer(str(page_one.get("text", "")), str(page_ten.get("text", "")))
    except Exception as exc:
        yield event("error", message=str(exc), trace=traceback.format_exc(limit=2))
        return
    yield event("completed", answer=answer, turns=2, mode="offline_reference")


def _ensure_reference_document(path: Path) -> str | None:
    if not str(path):
        return "Document path is required for the offline reference test."
    if not path.is_file():
        return f"Document file does not exist: {path}"
    return None


def run_agent(payload: dict[str, Any]) -> Iterator[str]:
    config = dict(payload.get("config") or {})
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        yield event("error", message="请输入任务提示。")
        return
    max_turns = max(1, min(int(payload.get("max_turns", 8)), 16))
    system_prompt = str(config.get("system_prompt") or (
        "You are a document-understanding agent. Use the provided tools when evidence from the document is needed. "
        "Use absolute local paths supplied by the user. Summarize results precisely. "
        "An output image path is not visual input to the language model. When you need text or visual evidence from a page or region, "
        "call ocr_region on that page or region and rely on the returned OCR text. Do not repeat render_page, crop_region, or zoom_region "
        "for the same region unless their arguments materially differ. Verify every Figure or Table reference against its caption and page; "
        "if the question conflates two sources, explicitly identify the mismatch rather than combining them as if they were one figure."
    ))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    action_stats = {
        "candidate_action_count": 0,
        "executed_action_count": 0,
        "valid_action_count": 0,
        "invalid_action_count": 0,
        "ignored_action_count": 0,
        "protocol_error_count": 0,
        "tool_error_count": 0,
    }
    yield event("run_started", max_turns=max_turns, tool_count=len(tool_registry.get_tool_specs()))

    for turn in range(1, max_turns + 1):
        yield event("model_request", turn=turn, message_count=len(messages))
        model_started_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-request") as executor:
            future = executor.submit(openai_chat_completion, config, messages)
            while True:
                try:
                    response = future.result(timeout=5)
                    choice = response["choices"][0]["message"]
                    break
                except FutureTimeout:
                    elapsed = round(time.monotonic() - model_started_at, 1)
                    yield event("model_progress", turn=turn, elapsed_seconds=elapsed, detail="Waiting for the configured model API response.")
                except Exception as exc:
                    yield event("error", message=str(exc), trace=traceback.format_exc(limit=2))
                    return

        content = choice.get("content") or ""
        native_calls = choice.get("tool_calls") or []
        parsed_text_action = parse_assistant_action(content) if not native_calls else None
        yield event("assistant", turn=turn, content=content, finish_reason=response["choices"][0].get("finish_reason"))

        if native_calls:
            action_stats["candidate_action_count"] += len(native_calls)
            if len(native_calls) != 1 or content.strip():
                action_stats["invalid_action_count"] += max(1, len(native_calls))
                action_stats["ignored_action_count"] += len(native_calls)
                action_stats["protocol_error_count"] += 1
                yield event(
                    "protocol_error",
                    turn=turn,
                    reason="assistant turn must contain exactly one native tool call and no suffix text",
                    **action_stats,
                )
                return
            messages.append({"role": "assistant", "content": content, "tool_calls": native_calls})
            calls = []
            for call in native_calls:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {"_invalid_arguments": str(exc), "_raw": function.get("arguments")}
                calls.append((str(call.get("id", "")), str(function.get("name", "")), arguments))
        elif parsed_text_action is not None and parsed_text_action.kind == "tool_call":
            action_stats["candidate_action_count"] += parsed_text_action.candidate_action_count
            name = str(parsed_text_action.value["name"])
            arguments = dict(parsed_text_action.value.get("arguments", {}))
            messages.append({"role": "assistant", "content": content})
            calls = [("text-call", name, arguments)]
        elif parsed_text_action is not None and parsed_text_action.kind == "final":
            action_stats["candidate_action_count"] += parsed_text_action.candidate_action_count
            action_stats["valid_action_count"] += 1
            messages.append({"role": "assistant", "content": content})
            yield event("rollout_summary", rollout_status="completed", valid_for_rl=True, **action_stats)
            yield event("completed", answer=content, turns=turn)
            return
        else:
            candidate_count = parsed_text_action.candidate_action_count if parsed_text_action is not None else 0
            action_stats["candidate_action_count"] += candidate_count
            action_stats["invalid_action_count"] += max(1, candidate_count)
            action_stats["ignored_action_count"] += candidate_count
            action_stats["protocol_error_count"] += 1
            yield event(
                "protocol_error",
                turn=turn,
                reason=(parsed_text_action.reason if parsed_text_action is not None else "no assistant action"),
                **action_stats,
            )
            return

        for call_id, name, arguments in calls:
            valid, validation_error = _validate_tool_action(name, arguments)
            if not valid:
                action_stats["invalid_action_count"] += 1
                action_stats["ignored_action_count"] += 1
                action_stats["protocol_error_count"] += 1
                yield event(
                    "protocol_error",
                    turn=turn,
                    reason=validation_error,
                    **action_stats,
                )
                return
        action_stats["valid_action_count"] += 1

        for call_id, name, arguments in calls:
            action_stats["executed_action_count"] += 1
            result = yield from run_registered_tool(turn, call_id, name, arguments)
            result_kind = result_status(result)
            if result_kind != "ok":
                action_stats["tool_error_count"] += 1
            if call_id == "text-call":
                messages.append({"role": "user", "content": f"<tool_result name=\"{name}\">\n{result}\n</tool_result>"})
            else:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    yield event("rollout_summary", rollout_status="model_protocol_error", valid_for_rl=True, **action_stats)
    yield event("completed", answer="Agent reached the configured maximum number of turns.", turns=max_turns)


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "OpenClawToolStudio/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[tool-studio] {self.address_string()} - {format % args}")

    def send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/tools":
            self.send_json(
                HTTPStatus.OK,
                {"tools": tool_registry.get_tool_specs(), "output_dir": str(OUTPUT_DIR), "studio_version": STUDIO_VERSION},
            )
            return
        if parsed.path == "/api/artifact":
            requested = parse_qs(parsed.query).get("path", [""])[0]
            path = Path(requested).resolve()
            try:
                path.relative_to(OUTPUT_DIR.resolve())
            except ValueError:
                self.send_json(HTTPStatus.FORBIDDEN, {"error": "Artifact path is outside tool_studio/outputs"})
                return
            if not path.is_file():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Artifact does not exist"})
                return
            content_type = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return
        relative = "index.html" if parsed.path in {"/", ""} else parsed.path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_types = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8"}
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path not in {"/api/run", "/api/reference-test"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON request: {exc}"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            runner = run_toolrl_reference_test(payload) if self.path == "/api/reference-test" else run_agent(payload)
            for line in runner:
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
        except BrokenPipeError:
            return


def main() -> None:
    import argparse

    if not sys.flags.no_site and os.environ.get("OPENCLAW_TOOL_STUDIO_NO_SITE") != "1":
        environment = os.environ.copy()
        environment["OPENCLAW_TOOL_STUDIO_NO_SITE"] = "1"
        os.execve(
            sys.executable,
            [sys.executable, "-S", str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )

    parser = argparse.ArgumentParser(description="OpenClaw Tool Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    print(f"OpenClaw Tool Studio: http://{args.host}:{args.port}")
    print(f"Artifacts: {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
