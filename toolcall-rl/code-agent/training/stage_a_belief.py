"""Stage A scripted Code tool-world belief pretraining."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

try:
    from ..bayestool.belief import CodeBeliefFilter, save_belief_checkpoint
    from ..bayestool.schema import CodeWorldSamplingContext
    from ..bayestool.world import CodeWorldRuntime
    from ..bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from ..config import DEFAULT_CODE_CONFIG
    from ..data.schema import SWEInstance
    from ..env.client import LocalCodeEnvClient
    from ..tools import DEFAULT_REGISTRY, ToolExecutionContext
    from .common import load_instances, repository_public_context, write_json
except ImportError:  # pragma: no cover
    from bayestool.belief import CodeBeliefFilter, save_belief_checkpoint
    from bayestool.world import CodeWorldRuntime
    from bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from config import DEFAULT_CODE_CONFIG
    from env.client import LocalCodeEnvClient
    from tools import DEFAULT_REGISTRY, ToolExecutionContext
    from training.common import load_instances, repository_public_context, write_json


def _probe_file(repository_root: str | Path) -> str | None:
    """Pick one source file from public repository state for a safe read probe."""

    root = Path(repository_root)
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix in {".py", ".js", ".ts", ".go", ".rs", ".java", ".md"}
    )
    if not candidates:
        return None
    return candidates[0].relative_to(root).as_posix()


async def run_probe(instance: SWEInstance, *, repository_root: str | Path, seed: int = 0) -> dict[str, Any]:
    context_values = repository_public_context(repository_root, tool_budget=DEFAULT_CODE_CONFIG.tool_budget)
    context = public_sampling_context(**context_values)
    worlds = sample_required_worlds(instance_id=instance.public.instance_id, image_name=instance.public.image_name or "local", base_revision=instance.public.base_revision, context=context, rollout_seed=seed)
    client = LocalCodeEnvClient(repository_root)
    rows = []
    probe_file = _probe_file(repository_root)
    for world in worlds:
        lease = await client.allocate(world.image_name or "local", world.instance_id, cwd="/testbed", base_revision=world.base_revision)
        try:
            from ..bayestool.task_state import CodeTaskStateView
        except ImportError:  # pragma: no cover
            from bayestool.task_state import CodeTaskStateView
        task = CodeTaskStateView(world.instance_id, instance.public.problem_statement, remaining_tool_budget=DEFAULT_CODE_CONFIG.tool_budget)
        runtime = CodeWorldRuntime(world=world, client=client, lease_id=lease.lease_id, cwd=lease.cwd, task_state=task)
        probes = [
            ("list_tree", {"path": ".", "max_depth": 1}),
            ("search_code", {"query": instance.public.problem_statement.split()[0] if instance.public.problem_statement.split() else "", "path": ".", "glob": "*.py", "max_results": 10}),
            ("git_diff", {}),
            ("run_checks", {"check": "compile", "path": "."}),
            ("run_tests", {"target": "", "args": "-q"}),
        ]
        if probe_file:
            probes.insert(2, ("read_file", {"path": probe_file, "start_line": 1, "end_line": 80}))
        for tool_name, arguments in probes:
            if not arguments.get("query") and tool_name == "search_code":
                continue
            result, features, event = await runtime.execute_tool(tool_name, arguments)
            rows.append({"world": world.to_latent_dict(), "event": event.trainer_dict(), "features": list(features)})
        await client.close(lease.lease_id)
    return {"environment": "code", "stage": "A", "instance_id": instance.public.instance_id, "probe_rows": rows}


async def run_stage_a(instances: list[SWEInstance], *, repository_root: str | Path, output: str | Path, seed: int = 0) -> dict[str, Any]:
    result = {"environment": "code", "stage": "A", "instances": []}
    filter_ = CodeBeliefFilter()
    for index, instance in enumerate(instances):
        probe = await run_probe(instance, repository_root=repository_root, seed=seed + index)
        result["instances"].append(probe)
        for row in probe["probe_rows"]:
            event = row["event"]
            tool = str(event.get("tool") or event.get("tool_name") or "list_tree")
            filter_.update(tool, event)
    output = Path(output)
    write_json(output, result)
    save_belief_checkpoint(output.with_name("belief_code.pt"), belief=filter_, extra={"stage": "A"})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    asyncio.run(run_stage_a(load_instances(args.manifest), repository_root=args.repository_root, output=args.output, seed=args.seed))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
