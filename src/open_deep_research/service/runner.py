"""Executes one task on the LangGraph agent: streams progress, pauses for approval, resumes.

Crash safety comes from the checkpointer, not from this file: a task id re-delivered by the
reaper runs against the *same thread id*, so ``aget_state`` finds the saved progress and the
graph continues from the last finished node instead of starting over.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Optional

from langgraph.types import Command

from open_deep_research.service import store as S
from open_deep_research.telemetry import UsageTracker
from open_deep_research.service.settings import Settings
from open_deep_research.service.store import TaskStore

logger = logging.getLogger(__name__)


class GraphProvider:
    """Owns the compiled agent graph and its SQLite checkpointer for the process lifetime."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph: Any = None
        self._cm = None

    async def start(self) -> None:
        import os
        from pathlib import Path

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from open_deep_research.deep_researcher import deep_researcher_builder

        Path(self.settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        self._cm = AsyncSqliteSaver.from_conn_string(self.settings.checkpoint_path)
        saver = await self._cm.__aenter__()
        self.graph = deep_researcher_builder.compile(checkpointer=saver)

    async def stop(self) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)


class _UserCancelled(Exception):
    pass


def thread_id_for(owner: str, task_id: str) -> str:
    return f"{owner}:{task_id}"           # user-prefixed: one user can never address another's thread


def final_text(values: dict) -> str:
    if values.get("final_report"):
        return values["final_report"]
    msgs = values.get("messages") or []
    return str(getattr(msgs[-1], "content", msgs[-1])) if msgs else ""


class TaskRunner:
    def __init__(self, store: TaskStore, provider: Any, settings: Settings):
        self.store, self.provider, self.settings = store, provider, settings

    def _config(self, owner: str, task_id: str) -> dict:
        return {"configurable": {
            "thread_id": thread_id_for(owner, task_id),
            "allow_clarification": False,
            "approval_mode": "interrupt",       # fixed server-side: callers cannot waive the approval gate
            "mem0_user_id": owner,              # long-term memory is per authenticated user
        }}

    async def run(self, task_id: str) -> None:
        """Run (or continue) a task that is already in state RUNNING. Never raises."""
        task = await self.store.get(task_id)
        if not task:
            return
        owner, graph = task["owner"], self.provider.graph
        cfg = self._config(owner, task_id)
        tracker = UsageTracker(trace_path=f"{self.settings.trace_dir}/{task_id}.jsonl", task_id=task_id)
        cfg["callbacks"] = [tracker]
        try:
            resume = task.get("resume")
            if resume:
                await self.store.r.hdel(self.store._task(task_id), "resume")
                graph_input: Any = Command(resume=resume)
                await self.store.add_event(task_id, "resumed", {"approved": bool(resume.get("approved"))})
            else:
                snap = await graph.aget_state(cfg)
                if snap.next:                                     # progress exists: continue it
                    graph_input = None
                    await self.store.add_event(task_id, "continued", {"from_nodes": list(snap.next)})
                elif snap.values:                                 # finished before the crash
                    return await self._finish(task_id, snap.values)
                else:
                    graph_input = {"messages": [{"role": "user", "content": task["question"]}]}

            outcome = await self._run_with_cancel(task_id, graph, graph_input, cfg)
            if outcome is not None:
                await self._finish(task_id, outcome)
        except asyncio.TimeoutError:
            await self._fail(task_id, f"timed out after {self.settings.task_timeout_s:.0f}s")
        except _UserCancelled:
            if await self.store.transition(task_id, S.CANCELLED):
                await self.store.add_event(task_id, "cancelled", {})
        except Exception as e:                                    # noqa: BLE001 - report, never crash the worker
            logger.exception("task %s failed", task_id)
            await self._fail(task_id, f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            # usage is recorded for every outcome (success, failure, pause, cancel, worker kill):
            # tokens were spent either way, and the daily budget must see them
            await asyncio.shield(self._record_usage(task_id, owner, tracker))
        # asyncio.CancelledError (the worker itself being killed) is deliberately NOT caught:
        # the task must stay RUNNING so the lease expires and another worker takes it over.

    async def _record_usage(self, task_id: str, owner: str, tracker: UsageTracker) -> None:
        try:
            await self.store.record_usage(task_id, owner, tracker.summary(), time.strftime("%Y%m%d", time.gmtime()))
        except Exception:                                         # noqa: BLE001 - never fail a task over bookkeeping
            logger.exception("could not record usage for %s", task_id)

    async def _observe_latency(self, task_id: str) -> None:
        task = await self.store.get(task_id)
        if task:
            await self.store.observe_latency(max(0.0, time.time() - float(task["created_at"])))

    async def _run_with_cancel(self, task_id: str, graph: Any, graph_input: Any, cfg: dict) -> Optional[dict]:
        """Run the graph as a child task so a user's cancel request stops only that child."""
        flag = {"user_cancel": False}
        child = asyncio.create_task(self._consume(task_id, graph, graph_input, cfg))
        watcher = asyncio.create_task(self._watch_cancel(task_id, child, flag))
        try:
            return await asyncio.wait_for(child, self.settings.task_timeout_s)
        except asyncio.CancelledError:
            if flag["user_cancel"] and child.cancelled():
                raise _UserCancelled() from None
            child.cancel()                                        # worker is being killed: take the child down too
            raise
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    async def _consume(self, task_id: str, graph: Any, graph_input: Any, cfg: dict) -> Optional[dict]:
        async for chunk in graph.astream(graph_input, cfg, stream_mode="updates"):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                if await self.store.transition(task_id, S.AWAITING, awaiting=payload):
                    await self.store.r.sadd(f"{self.store.p}awaiting", task_id)
                    await self.store.add_event(task_id, "approval_required", payload)
                return None
            for node in chunk:
                await self.store.add_event(task_id, "progress", {"node": node})
        return (await graph.aget_state(cfg)).values

    async def _watch_cancel(self, task_id: str, child: asyncio.Task, flag: dict) -> None:
        while not child.done():
            await asyncio.sleep(0.3)
            if await self.store.cancel_requested(task_id):
                flag["user_cancel"] = True
                child.cancel()
                return

    async def _finish(self, task_id: str, values: dict) -> None:
        if await self.store.transition(task_id, S.SUCCEEDED, result={"report": final_text(values)}):
            await self.store.add_event(task_id, "succeeded", {})
            await self._observe_latency(task_id)

    async def _fail(self, task_id: str, error: str) -> None:
        if await self.store.transition(task_id, S.FAILED, error=error):
            await self.store.add_event(task_id, "failed", {"error": error})
            await self._observe_latency(task_id)
