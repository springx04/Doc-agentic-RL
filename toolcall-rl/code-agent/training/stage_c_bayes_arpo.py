"""Stage C required-world + decision-level K-sibling Bayes-ARPO smoke."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Callable

try:
    from ..bayestool.branching import close_sibling_leases, create_sibling_leases
except ImportError:
    from bayestool.branching import close_sibling_leases, create_sibling_leases

try:
    from ..bayestool.grouping import decision_group_id, decision_prefix_hash, validate_decision_group
    from ..bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from ..config import DEFAULT_CODE_CONFIG, stable_hash
    from ..data.manifests import load_and_validate_capabilities
    from ..env.client import LocalCodeEnvClient
    from ..rollout import CodeRollout, generate_code_trajectory
    from .common import ScriptedModelClient, default_sample, evaluator_patch_for_instance, evaluator_script_for_instance, load_instances, repository_public_context, write_json
except ImportError:  # pragma: no cover
    from bayestool.grouping import decision_group_id, decision_prefix_hash, validate_decision_group
    from bayestool.world_sampler import public_sampling_context, sample_required_worlds
    from config import DEFAULT_CODE_CONFIG, stable_hash
    from data.manifests import load_and_validate_capabilities
    from env.client import LocalCodeEnvClient
    from rollout import CodeRollout, generate_code_trajectory
    from training.common import ScriptedModelClient, default_sample, evaluator_patch_for_instance, evaluator_script_for_instance, load_instances, repository_public_context, write_json


async def run_stage_c_smoke(
    instance: Any,
    *,
    repository_root: str | Path,
    model_factory: Callable[[], Any],
    group_size: int = 4,
    eval_script: str | None = None,
    capability_manifest: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if group_size not in {4, 8}:
        raise ValueError("Stage C group_size must be 4 or 8")
    if capability_manifest is not None:
        capabilities = load_and_validate_capabilities(capability_manifest, stage="C")
        if capabilities.group_size != group_size:
            raise ValueError("Stage C group_size must match the Code capability manifest")
        if capabilities.tool_budget != DEFAULT_CODE_CONFIG.tool_budget:
            raise ValueError("Stage C tool_budget must match the Code capability manifest")
    context = public_sampling_context(**repository_public_context(repository_root, tool_budget=DEFAULT_CODE_CONFIG.tool_budget))
    worlds = sample_required_worlds(instance_id=instance.public.instance_id, image_name=instance.public.image_name or "local", base_revision=instance.public.base_revision, context=context, rollout_seed=seed)
    client = LocalCodeEnvClient(repository_root)
    output_worlds = []
    for world in worlds:
        parent_lease = await client.allocate(world.image_name, world.instance_id, cwd="/testbed", base_revision=world.base_revision)
        checkpoints = []
        try:
            # Capture the real root decision state through the rollout API so
            # the sibling checkpoint contains prompt, token/task/world/belief
            # state as well as the repository patch.
            await CodeRollout().run(
                default_sample(instance),
                model_client=ScriptedModelClient(["<final>checkpoint boundary</final>"]),
                code_env_client=client,
                code_config=DEFAULT_CODE_CONFIG,
                world=world,
                interaction_lease=parent_lease,
                seed=seed,
                data_source=instance.public.data_source,
                on_decision_checkpoint=checkpoints.append,
            )
            if len(checkpoints) != 1:
                raise RuntimeError("Stage C failed to capture exactly one root decision checkpoint")
            parent_checkpoint = checkpoints[0]
        finally:
            await client.close(parent_lease.lease_id)
        parent_repo = parent_checkpoint.repo
        siblings = await create_sibling_leases(client, parent_checkpoint, image_name=world.image_name, instance_id=world.instance_id, count=group_size, cwd="/testbed")
        group_id = decision_group_id(instance_id=world.instance_id, latent_world_id=world.latent_world_id, decision_event_id="root", decision_prefix_hash_value=parent_checkpoint.decision_prefix_hash, repo_state_digest=parent_repo.repo_state_digest, belief_state_digest=stable_hash(parent_checkpoint.belief_state, prefix="code-belief-state-v1"), world_runtime_digest=parent_checkpoint.runtime_state_digest, coupling_id=world.coupling_id, world_slot_role=world.world_slot_role)
        records = []
        try:
            for sibling in siblings:
                model = model_factory()
                trajectory = await generate_code_trajectory(default_sample(instance), model, client, DEFAULT_CODE_CONFIG, world=world, eval_script=evaluator_script_for_instance(instance, eval_script), evaluator_patch=evaluator_patch_for_instance(instance), seed=seed, interaction_lease=sibling.lease, branch_checkpoint=sibling.checkpoint, data_source=instance.public.data_source)
                row = {
                    "environment": "code", "instance_id": world.instance_id, "latent_world_id": world.latent_world_id, "world_slot_role": world.world_slot_role,
                    "coupling_id": world.coupling_id, "decision_group_id": group_id, "variant_id": sibling.variant_id, "repo_state_digest": parent_repo.repo_state_digest,
                    "decision_prefix_hash": parent_checkpoint.decision_prefix_hash, "reward": trajectory["reward"], "valid_for_rl": trajectory["metadata"].get("valid_for_rl", False),
                    "failure_origin": trajectory["trainer_only_metadata"].get("failure_origin", "none"), "lease_id": sibling.lease.lease_id, "trajectory": trajectory,
                }
                records.append(row)
        finally:
            await close_sibling_leases(client, siblings)
        validity = validate_decision_group(records, expected_k=group_size)
        if len({record["lease_id"] for record in records}) != group_size:
            raise AssertionError("Stage C sibling continuations must use independent leases")
        rewards = [float(record["reward"]) for record in records]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        for record in records:
            record["advantage"] = record["reward"] - mean if validity.valid_for_rl else 0.0
        output_worlds.append({"world": world.to_latent_dict(), "parent_repo_state_digest": parent_repo.repo_state_digest, "decision_group_id": group_id, "group_validity": validity.__dict__, "siblings": records})
    return {"environment": "code", "stage": "C", "instance_id": instance.public.instance_id, "group_size": group_size, "worlds": output_worlds}


def _default_model() -> ScriptedModelClient:
    return ScriptedModelClient(["<tool_call>{\"name\":\"list_tree\",\"arguments\":{\"path\":\".\",\"max_depth\":1}}</tool_call>", "<tool_call>{\"name\":\"git_diff\",\"arguments\":{}}</tool_call>", "<final>smoke</final>"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--capability-manifest")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    instance = load_instances(args.manifest)[0]
    result = asyncio.run(run_stage_c_smoke(instance, repository_root=args.repository_root, model_factory=_default_model, group_size=args.group_size, capability_manifest=args.capability_manifest, seed=args.seed))
    write_json(args.output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
