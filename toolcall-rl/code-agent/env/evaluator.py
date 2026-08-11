"""Clean evaluator isolated from interaction-world corruption."""

from __future__ import annotations

from dataclasses import dataclass
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
    eval_script: str
    base_revision: str | None = None
    cwd: str = "/testbed"
    timeout: int = 300


class CleanEvaluator:
    """Allocate a fresh lease for every official evaluation."""

    def __init__(self, client: Any):
        self.client = client

    async def evaluate(self, request: EvaluatorRequest) -> CodeEvalResult:
        lease = None
        try:
            lease = await self.client.allocate(
                request.image_name,
                request.instance_id,
                cwd=request.cwd,
                base_revision=request.base_revision,
            )
            # The remote evaluate endpoint resets before applying the patch;
            # local clients implement the same contract explicitly.
            result = await self.client.evaluate(
                lease.lease_id,
                request.patch,
                request.eval_script,
                cwd=lease.cwd,
                timeout=request.timeout,
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


__all__ = ["CleanEvaluator", "EvaluatorRequest", "evaluate_clean_patch"]
