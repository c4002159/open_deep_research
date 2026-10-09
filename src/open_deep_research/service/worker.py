"""Worker pool: claim -> lease -> run -> release, plus the lease reaper.

A worker holds a short-lived lease key and renews it while it works. If the process dies
the key expires on its own; the reaper (one elected instance at a time) then moves the task
id back to the queue and the next worker resumes it from its checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from open_deep_research.service import store as S
from open_deep_research.service.runner import TaskRunner
from open_deep_research.service.settings import Settings
from open_deep_research.service.store import TaskStore

logger = logging.getLogger(__name__)


class WorkerPool:
    def __init__(self, store: TaskStore, runner: TaskRunner, settings: Settings):
        self.store, self.runner, self.settings = store, runner, settings
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    def start(self, workers: int | None = None) -> None:
        self._stopping = False
        for i in range(workers if workers is not None else self.settings.workers):
            self._tasks.append(asyncio.create_task(self._worker_loop(i), name=f"odr-worker-{i}"))
        self._tasks.append(asyncio.create_task(self._reaper_loop(), name="odr-reaper"))

    async def stop(self) -> None:
        self._stopping = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()

    async def kill_workers(self) -> None:
        """Test hook: stop workers abruptly WITHOUT releasing leases or processing entries,
        which is exactly what a crashed process leaves behind."""
        await self.stop()

    async def _worker_loop(self, n: int) -> None:
        while not self._stopping:
            try:
                tid = await self.store.claim(timeout_s=1)
                if tid is None:
                    continue
                await self._handle(tid)
            except asyncio.CancelledError:
                raise
            except Exception:                                    # noqa: BLE001
                logger.exception("worker %d loop error", n)
                await asyncio.sleep(1)

    async def _handle(self, tid: str) -> None:
        await self.store.lease(tid, self.settings.lease_ttl_s)
        if not await self.store.transition(tid, S.RUNNING):      # cancelled while queued, or already terminal
            await self.store.release(tid)
            return
        await self.store.bump_attempts(tid)
        renew = asyncio.create_task(self._renew(tid))
        try:
            await self.runner.run(tid)
        except asyncio.CancelledError:
            raise                                                # simulated/real crash: leave lease to expire
        finally:
            renew.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renew
        await self.store.release(tid)

    async def _renew(self, tid: str) -> None:
        while True:
            await asyncio.sleep(self.settings.lease_refresh_s)
            await self.store.lease(tid, self.settings.lease_ttl_s)

    async def _reaper_loop(self) -> None:
        lock = f"{self.store.p}reaper_lock"
        while not self._stopping:
            await asyncio.sleep(self.settings.reaper_interval_s)
            try:
                # only one instance reaps per interval (works across API replicas)
                if await self.store.r.set(lock, "1", nx=True, px=int(self.settings.reaper_interval_s * 900)):
                    await self.store.reap_expired(self.settings.max_attempts)
            except asyncio.CancelledError:
                raise
            except Exception:                                    # noqa: BLE001
                logger.exception("reaper error")
