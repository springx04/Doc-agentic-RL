"""Stage B single-task/single-world belief-conditioned policy runner."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

try:
    from ..bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from ..config import DEFAULT_CODE_CONFIG
    from ..env.client import LocalCodeEnvClient
    from ..rollout import generate_code_trajectory
    from .common import ScriptedModelClient, default_sample, evaluator_patch_for_instance, evaluator_script_for_instance, load_instances, repository_public_context, write_json
except ImportError:  # pragma: no cover
    from bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from config import DEFAULT_CODE_CONFIG
    from env.client import LocalCodeEnvClient
    from rollout import generate_code_trajectory
    from training.common import ScriptedModelClient, default_sample, evaluator_patch_for_instance, evaluator_script_for_instance, load_instances, repository_public_context, write_json


async def run_stage_b(instance: Any, *, repository_root: str | Path, model_client: Any, eval_script: str | None = None, seed: int = 0) -> dict[str, Any]:
    context = public_sampling_context(**repository_public_context(repository_root, tool_budget=DEFAULT_CODE_CONFIG.tool_budget))
    world = sample_required_worlds(instance_id=instance.public.instance_id, image_name=instance.public.image_name or "local", base_revision=instance.public.base_revision, context=context, rollout_seed=seed)[0]
    result = await generate_code_trajectory(default_sample(instance), model_client, LocalCodeEnvClient(repository_root), DEFAULT_CODE_CONFIG, world=world, eval_script=evaluator_script_for_instance(instance, eval_script), evaluator_patch=evaluator_patch_for_instance(instance), seed=seed, data_source=instance.public.data_source)
    return {"environment": "code", "stage": "B", "trajectory": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    instance = load_instances(args.manifest)[0]
    scripted = ScriptedModelClient(["<tool_call>{\"name\":\"list_tree\",\"arguments\":{\"path\":\".\",\"max_depth\":1}}</tool_call>", "<final>inspection complete</final>"])
    result = asyncio.run(run_stage_b(instance, repository_root=args.repository_root, model_client=scripted, seed=args.seed))
    write_json(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
