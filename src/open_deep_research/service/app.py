"""FastAPI service: async research tasks with idempotency, rate limiting and approvals.

    POST /v1/tasks                  submit (202) - honours Idempotency-Key
    GET  /v1/tasks/{id}             status / result
    GET  /v1/tasks                  my recent tasks
    GET  /v1/tasks/{id}/events      Server-Sent Events (progress; resumable via Last-Event-ID)
    POST /v1/tasks/{id}/cancel
    POST /v1/tasks/{id}/approval    approve / reject a paused task (approvers only)
    GET  /healthz, /metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from open_deep_research.service import store as S
from open_deep_research.service.auth import AuthError, Authenticator
from open_deep_research.service.ratelimit import SlidingWindowLimiter
from open_deep_research.service.runner import GraphProvider, TaskRunner
from open_deep_research.service.settings import Settings
from open_deep_research.service.store import TaskStore
from open_deep_research.service.worker import WorkerPool

PUBLIC_FIELDS = ("id", "state", "question", "created_at", "updated_at", "result", "error", "awaiting", "attempts")


class TaskIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ApprovalIn(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=500)


def create_app(settings: Optional[Settings] = None, provider: Any = None, start_workers: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    auth = Authenticator(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        r = aioredis.from_url(settings.redis_url, max_connections=settings.redis_max_connections)
        rb = aioredis.from_url(settings.redis_url, max_connections=settings.redis_blocking_connections)
        app.state.redis = r
        app.state.store = TaskStore(r, settings.prefix, settings.task_ttl_s, blocking_redis=rb)
        app.state.limiter = SlidingWindowLimiter(r, settings.prefix, settings.rate_limit_per_window, settings.rate_window_s)
        app.state.provider = provider or GraphProvider(settings)
        await app.state.provider.start()
        app.state.runner = TaskRunner(app.state.store, app.state.provider, settings)
        app.state.pool = WorkerPool(app.state.store, app.state.runner, settings)
        if start_workers:
            app.state.pool.start()
        try:
            yield
        finally:
            await app.state.pool.stop()
            await app.state.provider.stop()
            await r.aclose()
            await rb.aclose()

    app = FastAPI(title="Open Deep Research service", lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(RedisError)
    async def redis_down(request: Request, exc: RedisError):
        # a Redis problem is an availability problem (retryable), not a bug in the caller's request
        return JSONResponse({"detail": "backing store unavailable, retry shortly"}, status_code=503, headers={"Retry-After": "2"})

    # ------------------------------------------------------------ helpers
    async def current_user(authorization: Optional[str] = Header(default=None)) -> str:
        try:
            return auth.authenticate(authorization)
        except AuthError as e:
            raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})

    async def metric(request: Request, name: str, n: int = 1) -> None:
        await request.app.state.redis.hincrby(f"{settings.prefix}metrics", name, n)

    def is_approver(user: str, task: dict) -> bool:
        return user in settings.approvers or (settings.owner_can_approve and user == task["owner"])

    async def load_visible_task(request: Request, task_id: str, user: str, *, as_approver: bool = False) -> dict:
        task = await request.app.state.store.get(task_id)
        # 404 (not 403) for other people's tasks: do not reveal that the id exists
        if not task or (task["owner"] != user and not (as_approver and user in settings.approvers)):
            raise HTTPException(404, "task not found")
        return task

    def public(task: dict) -> dict:
        return {k: task[k] for k in PUBLIC_FIELDS if k in task}

    # ------------------------------------------------------------ endpoints
    @app.get("/healthz")
    async def healthz(request: Request):
        try:
            await request.app.state.redis.ping()
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"status": "degraded", "redis": str(e)[:100]}, status_code=503)
        return {"status": "ok", "workers": len(request.app.state.pool._tasks)}

    @app.post("/v1/tasks", status_code=202)
    async def submit_task(body: TaskIn, request: Request, response: Response, user: str = Depends(current_user),
                          idempotency_key: Optional[str] = Header(default=None, max_length=128)):
        st: TaskStore = request.app.state.store
        body_hash = hashlib.sha256(body.question.encode()).hexdigest()

        async def replay(info: dict):
            if info["body_hash"] != body_hash:
                raise HTTPException(409, "Idempotency-Key was already used with a different request body")
            await metric(request, "idempotent_replays")
            task = await st.get(info["task_id"])
            response.status_code = 200
            response.headers["Idempotent-Replay"] = "true"
            return public(task) if task else {"id": info["task_id"], "state": "expired"}

        # replays are free: check before spending rate-limit quota
        if idempotency_key:
            existing = await st.find_idempotent(user, idempotency_key)
            if existing:
                return await replay(existing)

        decision = await request.app.state.limiter.hit(user)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_window)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        if not decision.allowed:
            await metric(request, "rate_limited")
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(max(1, int(decision.retry_after_s + 0.999)))})

        res = await st.submit(user, body.question, body_hash, idempotency_key, settings.max_active_per_user,
                              settings.idempotency_ttl_s, settings.queue_max_depth)
        if res.status == "EXISTING":                       # lost a race against a concurrent identical request
            return await replay({"task_id": res.task_id, "body_hash": res.body_hash})
        if res.status == "LIMIT":
            await metric(request, "active_limit")
            raise HTTPException(429, f"too many active tasks (max {settings.max_active_per_user})", headers={"Retry-After": "5"})
        if res.status == "BUSY":
            await metric(request, "queue_full")
            raise HTTPException(503, "service busy, try again shortly", headers={"Retry-After": "10"})
        await st.record_submitted(res.task_id, f"{settings.prefix}metrics")
        return {"id": res.task_id, "state": S.QUEUED}

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str, request: Request, user: str = Depends(current_user)):
        return public(await load_visible_task(request, task_id, user, as_approver=True))

    @app.get("/v1/tasks")
    async def list_tasks(request: Request, user: str = Depends(current_user)):
        st: TaskStore = request.app.state.store
        tasks = [await st.get(i) for i in await st.list_for_user(user)]
        return {"tasks": [public(t) for t in tasks if t]}

    @app.post("/v1/tasks/{task_id}/cancel", status_code=202)
    async def cancel_task(task_id: str, request: Request, user: str = Depends(current_user)):
        task = await load_visible_task(request, task_id, user)
        if task["state"] in S.TERMINAL:
            raise HTTPException(409, f"task already {task['state']}")
        return {"id": task_id, "cancel": await request.app.state.store.request_cancel(task_id)}

    @app.post("/v1/tasks/{task_id}/approval", status_code=202)
    async def approve_task(task_id: str, body: ApprovalIn, request: Request, user: str = Depends(current_user)):
        st: TaskStore = request.app.state.store
        task = await st.get(task_id)
        if not task or not (user == task["owner"] or user in settings.approvers):
            raise HTTPException(404, "task not found")
        if not is_approver(user, task):
            raise HTTPException(403, "you are not allowed to approve requests")
        if task["state"] != S.AWAITING:
            raise HTTPException(409, f"task is {task['state']}, not awaiting approval")
        # reviewer identity comes from the token, never from the body
        resume = {"approved": body.approved, "reviewer": user, "note": body.note}
        await st.set_field(task_id, resume=resume)
        if not await st.transition(task_id, S.QUEUED):
            raise HTTPException(409, "task state changed, retry")
        await st.r.srem(f"{settings.prefix}awaiting", task_id)
        await st.r.lpush(st.queue_key, task_id)
        await st.add_event(task_id, "approval_decided", {"approved": body.approved, "reviewer": user})
        return {"id": task_id, "state": S.QUEUED}

    @app.get("/v1/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request, user: str = Depends(current_user),
                          last_event_id: Optional[str] = Header(default=None)):
        await load_visible_task(request, task_id, user, as_approver=True)
        st: TaskStore = request.app.state.store

        async def stream():
            cursor, idle = last_event_id or "0", 0
            while True:
                if await request.is_disconnected():
                    return
                events = await st.read_events(task_id, cursor, block_ms=1000)
                for eid, type_, data in events:
                    cursor, idle = eid, 0
                    yield f"id: {eid}\nevent: {type_}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                task = await st.get(task_id)
                if not events:
                    idle += 1
                    if idle % 15 == 0:
                        yield ": keep-alive\n\n"
                    if task is None or (task["state"] in S.TERMINAL):
                        yield "event: end\ndata: {}\n\n"
                        return

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request):
        raw = await request.app.state.redis.hgetall(f"{settings.prefix}metrics")
        lines = [f"odr_{k.decode()} {int(v)}" for k, v in sorted(raw.items())]
        lines.append(f"odr_queue_depth {await request.app.state.redis.llen(request.app.state.store.queue_key)}")
        return "\n".join(lines) + "\n"

    return app
