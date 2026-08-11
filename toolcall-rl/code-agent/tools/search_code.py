"""Bounded text/code search tool."""

from __future__ import annotations

from ._common import ToolExecutionContext, clean_failure, invalid_result, quote, repo_path, result_from_execution, truncate_output, validate_safe_args_text
try:
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from schemas import CodeToolResult


async def execute(arguments: dict, context: ToolExecutionContext):
    query = str(arguments.get("query", ""))
    glob = str(arguments.get("glob", "*"))
    try:
        path = repo_path(context.cwd, arguments.get("path", "."))
        max_results = int(arguments.get("max_results", 50))
        if not validate_safe_args_text(query) or not validate_safe_args_text(glob):
            return invalid_result("search_code", "query and glob contain forbidden shell characters")
        local_root = context.client.root_for_lease(context.lease_id) if hasattr(context.client, "root_for_lease") else None
        if local_root is not None:
            import fnmatch
            base = (local_root / str(arguments.get("path", "."))).resolve()
            if local_root not in base.parents and base != local_root:
                return invalid_result("search_code", "path escapes repository root")
            rows = []
            for file_path in base.rglob("*"):
                if not file_path.is_file() or ".git" in file_path.relative_to(local_root).parts or "__pycache__" in file_path.parts:
                    continue
                if glob and not fnmatch.fnmatch(file_path.name, glob):
                    continue
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                for line_no, line in enumerate(lines, 1):
                    if query.casefold() in line.casefold():
                        rows.append(f"{file_path.relative_to(local_root).as_posix()}:{line_no}:{line}")
                        if len(rows) >= max_results:
                            break
                if len(rows) >= max_results:
                    break
            output = "\n".join(rows)
            output, partial = truncate_output(output, getattr(context.config, "max_output_chars", 16384))
            return CodeToolResult("search_code", "ok" if output else "empty", output, metadata={"query": query, "path": path, "glob": glob, "match_count": len(rows), "partial": partial}, partial=partial)
        command = (
            f"rg --no-heading --line-number --color never --hidden --glob {quote(glob)} "
            f"--glob '!.git/**' --glob '!__pycache__/**' --max-count 200 -- {quote(query)} {quote(path)} "
            f"| head -n {max_results}"
        )
        raw = await context.client.exec(context.lease_id, command, cwd=context.cwd, timeout=60)
        # rg returns 1 for no matches; that is a valid empty observation.
        if raw.returncode not in (0, 1):
            return result_from_execution("search_code", raw)
        output = raw.stdout
        status = "ok" if output else "empty"
        result = result_from_execution("search_code", raw, limit=getattr(context.config, "max_output_chars", 16384), metadata={"query": query, "path": path, "glob": glob, "match_count": len(output.splitlines()) if output else 0})
        return result.__class__(**{**result.__dict__, "status": status})
    except ValueError as exc:
        return invalid_result("search_code", str(exc))
    except Exception as exc:
        return clean_failure("search_code", exc)
