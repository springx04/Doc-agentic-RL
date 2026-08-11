"""Stage D persistent-session belief with fresh repo/task leases."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from ..bayestool.belief import CodeBeliefFilter
    from ..config import DEFAULT_CODE_CONFIG
    from ..data.manifests import load_and_validate_capabilities
    from ..env.client import LocalCodeEnvClient
    from ..rollout import generate_code_trajectory
    from .common import ScriptedModelClient, default_sample, evaluator_script_for_instance, load_instances, repository_public_context, write_json
    from ..bayestool.world_sampler import public_sampling_context, sample_required_worlds
except ImportError:  # pragma: no cover
    from bayestool.belief import CodeBeliefFilter
    from config import DEFAULT_CODE_CONFIG
    from data.manifests import load_and_validate_capabilities
    from env.client import LocalCodeEnvClient
    from rollout import generate_code_trajectory
    from training.common import ScriptedModelClient, default_sample, evaluator_script_for_instance, load_instances, repository_public_context, write_json
    from bayestool.world_sampler import public_sampling_context, sample_required_worlds


@dataclass
class PersistentCodeSession:
    belief: CodeBeliefFilter = field(default_factory=CodeBeliefFilter)
    task_count: int = 0
    carried_session_history: list[dict[str, Any]] = field(default_factory=list)

    async def run_task(self, instance: Any, *, repository_root: str | Path, model_client: Any, seed: int = 0) -> dict[str, Any]:
        # A new LocalCodeEnvClient and a new rollout lease guarantee no file,
        # patch, inspection, or path-local state crosses task boundaries.
        context = public_sampling_context(**repository_public_context(repository_root, tool_budget=DEFAULT_CODE_CONFIG.tool_budget))
        world = sample_required_worlds(instance_id=instance.public.instance_id, image_name=instance.public.image_name or "local", base_revision=instance.public.base_revision, context=context, rollout_seed=seed)[0]
        result = await generate_code_trajectory(default_sample(instance), model_client, LocalCodeEnvClient(repository_root), DEFAULT_CODE_CONFIG, world=world, eval_script=evaluator_script_for_instance(instance), seed=seed, data_source=instance.public.data_source)
        for event in result.get("trainer_only_metadata", {}).get("events", []):
            tool = str(event.get("tool") or event.get("tool_name") or "")
            if tool:
                public_event = {key: value for key, value in event.items() if key not in {"label", "failure_origin", "world_type", "world_slot_role", "latent_world_id", "corruption_type"}}
                self.belief.update(tool, public_event, family=event.get("public_context", {}).get("family"))
        self.task_count += 1
        self.carried_session_history.append({"task_index": self.task_count, "valid_for_rl": result["metadata"].get("valid_for_rl"), "tool_calls_used": result["metadata"].get("tool_calls_used")})
        self.carried_session_history = self.carried_session_history[-32:]
        return result

    def session_state(self) -> dict[str, Any]:
        return {"task_count": self.task_count, "belief": self.belief.to_dict(), "history": list(self.carried_session_history)}


async def run_stage_d(
    instances: Iterable[Any],
    *,
    repository_root: str | Path,
    model_factory,
    capability_manifest: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if capability_manifest is not None:
        capabilities = load_and_validate_capabilities(capability_manifest, stage="D")
        if capabilities.tool_budget != DEFAULT_CODE_CONFIG.tool_budget:
            raise ValueError("Stage D tool_budget must match the Code capability manifest")
    session = PersistentCodeSession()
    outputs = []
    for index, instance in enumerate(instances):
        outputs.append(await session.run_task(instance, repository_root=repository_root, model_client=model_factory(), seed=seed + index))
    return {"environment": "code", "stage": "D", "tasks": outputs, "session": session.session_state()}


def _default_model() -> ScriptedModelClient:
    return ScriptedModelClient(["<tool_call>{\"name\":\"list_tree\",\"arguments\":{\"path\":\".\",\"max_depth\":1}}</tool_call>", "<final>done</final>"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capability-manifest")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = asyncio.run(run_stage_d(load_instances(args.manifest), repository_root=args.repository_root, model_factory=_default_model, capability_manifest=args.capability_manifest, seed=args.seed))
    write_json(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
