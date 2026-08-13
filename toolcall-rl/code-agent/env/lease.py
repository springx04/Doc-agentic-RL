"""Lease records and lifecycle helpers for Code interaction environments."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Awaitable, Callable

try:
    from ..schemas import CodeLease
except ImportError:  # pragma: no cover
    from schemas import CodeLease


class LeaseLost(RuntimeError):
    """Raised when a lease can no longer be used for a rollout."""


class LeaseHeartbeat:
    """Optional background heartbeat; it never translates world failures into lease loss."""

    def __init__(self, lease: CodeLease, heartbeat: Callable[[str], Awaitable[object]], interval: float = 30.0):
        self.lease = lease
        self._heartbeat = heartbeat
        self.interval = max(0.1, float(interval))
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self.error: BaseException | None = None

    async def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    await self._heartbeat(self.lease.lease_id)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # transport-level failure is surfaced to caller
            self.error = exc

    async def __aenter__(self) -> "LeaseHeartbeat":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._stopped.set()
        if self._task is not None:
            await self._task


def touch_lease(lease: CodeLease) -> CodeLease:
    return replace(lease, last_heartbeat=time.time())


__all__ = ["LeaseHeartbeat", "LeaseLost", "touch_lease"]
