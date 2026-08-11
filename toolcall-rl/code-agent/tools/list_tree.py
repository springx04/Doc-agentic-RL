"""Bounded repository tree inspection tool."""

from __future__ import annotations

import json

from ._common import ToolExecutionContext, clean_failure, invalid_result, quote, repo_path, result_from_execution, truncate_output
try:
    from ..schemas import CodeToolResult
except ImportError:  # pragma: no cover
    from schemas import CodeToolResult


async def execute(arguments: dict, context: ToolExecutionContext):
    try:
        path = repo_path(context.cwd, arguments.get("path", "."))
        max_depth = int(arguments.get("max_depth", 2))
        local_root = context.client.root_for_lease(context.lease_id) if hasattr(context.client, "root_for_lease") else None
        if local_root is not None:
            import os
            entries = []
            base = (local_root / str(arguments.get("path", "."))).resolve()
            if not base.is_dir() or (local_root not in base.parents and base != local_root):
                return invalid_result("list_tree", "path escapes repository root or is not a directory")
            for item in sorted(base.rglob("*")):
                if ".git" in item.relative_to(local_root).parts or "__pycache__" in item.parts or ".pytest_cache" in item.parts:
                    continue
                depth = len(item.relative_to(base).parts)
                if depth <= max_depth:
                    entries.append({"path": item.relative_to(local_root).as_posix(), "type": "dir" if item.is_dir() else "file", "depth": depth})
            output = "\n".join(f"{item['type']}\t{item['depth']}\t{item['path']}" for item in entries[:2000])
            output, partial = truncate_output(output, getattr(context.config, "max_output_chars", 16384))
            return CodeToolResult("list_tree", "ok" if output else "empty", output, metadata={"path": path, "max_depth": max_depth, "entry_count": len(entries), "partial": partial}, partial=partial)
        command = (
            "python -c "
            + quote(
                "import json,os,sys; root=sys.argv[1]; depth=int(sys.argv[2]); out=[]; "
                "base=os.path.abspath(root); "
                "for dirpath,dirs,files in os.walk(base): "
                " dirs[:]=sorted(d for d in dirs if d not in {'.git','__pycache__','.pytest_cache'}); "
                " rel=os.path.relpath(dirpath,base); d=0 if rel=='.' else rel.count(os.sep)+1; "
                " dirs[:]=[] if d>=depth else dirs; "
                " out.extend({'path':os.path.relpath(os.path.join(dirpath,n),base).replace(os.sep,'/'),'type':'dir','depth':d+1} for n in dirs); "
                " out.extend({'path':os.path.relpath(os.path.join(dirpath,n),base).replace(os.sep,'/'),'type':'file','depth':d+1} for n in sorted(files)); "
                "print(json.dumps(out[:2000],ensure_ascii=False))",
            )
            + f" {quote(path)} {max_depth}"
        )
        raw = await context.client.exec(context.lease_id, command, cwd=context.cwd, timeout=60)
        if not raw.ok:
            return result_from_execution("list_tree", raw)
        try:
            entries = json.loads(raw.stdout)
            output = "\n".join(f"{item['type']}\t{item['depth']}\t{item['path']}" for item in entries)
        except (TypeError, ValueError, KeyError):
            output = raw.output
        output, partial = truncate_output(output, getattr(context.config, "max_output_chars", 16384))
        return result_from_execution("list_tree", raw, limit=getattr(context.config, "max_output_chars", 16384), metadata={"path": path, "max_depth": max_depth, "entry_count": len(entries) if isinstance(entries, list) else None, "partial": partial})
    except ValueError as exc:
        return invalid_result("list_tree", str(exc))
    except Exception as exc:
        return clean_failure("list_tree", exc)
