"""Line-numbered, bounded repository file reader."""

from __future__ import annotations

from ._common import ToolExecutionContext, clean_failure, invalid_result, quote, repo_path, result_from_execution
try:
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from schemas import CodeToolResult


async def execute(arguments: dict, context: ToolExecutionContext):
    try:
        path = repo_path(context.cwd, arguments["path"])
        start = int(arguments.get("start_line", 1))
        end = int(arguments.get("end_line", 240))
        if end - start + 1 > getattr(context.config, "max_read_lines", 240):
            return invalid_result("read_file", "requested line range exceeds the per-call limit")
        local_root = context.client.root_for_lease(context.lease_id) if hasattr(context.client, "root_for_lease") else None
        if local_root is not None:
            local_path = (local_root / str(arguments["path"])).resolve()
            if local_root not in local_path.parents or not local_path.is_file():
                return invalid_result("read_file", "file is outside repository root or does not exist")
            lines = local_path.read_text(encoding="utf-8", errors="replace").splitlines()
            output = "\n".join(f"{index:>6}\t{line}" for index, line in enumerate(lines[start - 1:end], start))
            return CodeToolResult("read_file", "ok" if output else "empty", output, metadata={"path": path, "start_line": start, "end_line": end, "read_line_coverage": len(lines[start - 1:end]) / max(1, end - start + 1)})
        command = f"nl -ba -- {quote(path)} | sed -n '{start},{end}p'"
        raw = await context.client.exec(context.lease_id, command, cwd=context.cwd, timeout=60)
        return result_from_execution("read_file", raw, limit=getattr(context.config, "max_output_chars", 16384), metadata={"path": path, "start_line": start, "end_line": end})
    except ValueError as exc:
        return invalid_result("read_file", str(exc))
    except Exception as exc:
        return clean_failure("read_file", exc)
