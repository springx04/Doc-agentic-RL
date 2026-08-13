"""Clean evaluator isolated from interaction-world corruption."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any

try:
    from ..schemas import CodeEvalResult
except ImportError:  # pragma: no cover
    from schemas import CodeEvalResult


@dataclass(frozen=True)
class EvaluatorRequest:
    image_name: str
    instance_id: str
    patch: str
    eval_script: str | None = None
    base_revision: str | None = None
    cwd: str = "/testbed"
    timeout: int = 300
    evaluator_patch: str = ""


_PATCH_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?([^\t\n]+)", re.M)


def patch_paths(patch: str) -> set[str]:
    """Return repository-relative files changed by a unified patch."""

    return {path.strip() for path in _PATCH_PATH_RE.findall(patch) if path.strip() and path.strip() != "/dev/null"}


class CleanEvaluator:
    """Allocate a fresh lease for every official evaluation."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    @lru_cache(maxsize=2)
    def _private_catalog(manifest_path: str) -> dict[str, tuple[str, str]]:
        """Read hidden test material only at the clean evaluator boundary."""

        try:
            from ..training.common import evaluator_patch_for_instance, evaluator_script_for_instance, load_instances
        except ImportError:  # pragma: no cover
            from training.common import evaluator_patch_for_instance, evaluator_script_for_instance, load_instances
        catalog = {
            instance.public.instance_id: (evaluator_script_for_instance(instance), evaluator_patch_for_instance(instance))
            for instance in load_instances(manifest_path)
        }
        if not catalog:
            raise ValueError("Code evaluator manifest is empty")
        return catalog

    def _resolve_private_evaluator(self, request: EvaluatorRequest) -> tuple[str, str]:
        if request.eval_script:
            return request.eval_script, request.evaluator_patch
        manifest_path = os.getenv("CODE_EVALUATOR_MANIFEST", "").strip()
        if not manifest_path:
            raise RuntimeError("CODE_EVALUATOR_MANIFEST is required when no evaluator command is supplied")
        value = self._private_catalog(str(Path(manifest_path).resolve())).get(request.instance_id)
        if value is None:
            raise KeyError(f"instance {request.instance_id!r} is absent from evaluator manifest")
        command, patch = value
        if not command:
            raise ValueError(f"instance {request.instance_id!r} has no official evaluator command")
        return command, patch

    async def evaluate(self, request: EvaluatorRequest) -> CodeEvalResult:
        lease = None
        try:
            eval_script, evaluator_patch = self._resolve_private_evaluator(request)
            lease = await self.client.allocate(
                request.image_name,
                request.instance_id,
                cwd=request.cwd,
                base_revision=request.base_revision,
            )
            # The official hidden test patch is applied only in this clean
            # evaluator lease.  Reject candidate edits to the same paths so a
            # policy cannot gain reward by changing withheld tests.
            if evaluator_patch and patch_paths(request.patch) & patch_paths(evaluator_patch):
                return CodeEvalResult(ok=True, resolved=False, output="candidate patch overlaps evaluator-private test paths", lease_id=lease.lease_id)
            # The remote evaluate endpoint resets before applying the patch;
            # both backends apply evaluator_patch only after that reset.
            result = await self.client.evaluate(
                lease.lease_id,
                request.patch,
                eval_script,
                cwd=lease.cwd,
                timeout=request.timeout,
                evaluator_patch=evaluator_patch,
            )
            return result
        except Exception as exc:
            return CodeEvalResult(
                ok=False,
                resolved=False,
                output="",
                error=str(exc),
                lease_id=getattr(lease, "lease_id", None),
                failure_origin="real_infrastructure",
            )
        finally:
            if lease is not None:
                try:
                    await self.client.close(lease.lease_id)
                except Exception:
                    # The evaluation result is already marked by the actual
                    # evaluator; cleanup failure must not be mistaken for a
                    # passing task.
                    pass


async def evaluate_clean_patch(
    client: Any,
    *,
    image_name: str,
    instance_id: str,
    patch: str,
    eval_script: str,
    base_revision: str | None = None,
    cwd: str = "/testbed",
    timeout: int = 300,
) -> CodeEvalResult:
    return await CleanEvaluator(client).evaluate(
        EvaluatorRequest(image_name, instance_id, patch, eval_script, base_revision, cwd, timeout)
    )


__all__ = ["CleanEvaluator", "EvaluatorRequest", "evaluate_clean_patch", "patch_paths"]
