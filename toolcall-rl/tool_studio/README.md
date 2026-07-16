# OpenClaw Tool Studio

Local visual test bench for document tool calls. It is isolated under
`toolcall-rl/tool_studio/` and does not import training code beyond the public
`tool_sandbox.tool_registry` execution interface.

## Start

From `toolcall-rl`:

```powershell
.\tool_studio\run.ps1
```

Open `http://127.0.0.1:8765`.

The UI accepts an OpenAI-compatible Chat Completions endpoint. For a local
SGLang/vLLM server, leave the key empty and use its `/v1` base URL. API keys
are never written to disk by the UI.

The agent receives the same tool schemas as training and executes every tool
through `tool_sandbox.tool_registry`. The right-side event log shows model
turns, tool arguments, raw tool results, errors, and the final answer.

Tool artifacts and backend caches are kept in `tool_studio/outputs/` and
`tool_studio/cache/`, respectively. Existing backend environment variables,
such as `OPENCLAW_DOCLING_ARTIFACTS_PATH` and `OPENCLAW_DEPLOT_MODEL`, are
honored. Set them in the shell before starting the server when local model
artifacts are not in their default caches.

This is intended for local use only: the server binds to `127.0.0.1`, and its
artifact endpoint only serves files generated below `tool_studio/outputs/`.
