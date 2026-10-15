"""FastAPI service: async research tasks with idempotency, rate limiting and approvals.

    POST /v1/tasks                  submit (202) - honours Idempotency-Key
    GET  /v1/tasks/{id}             status / result
    GET  /v1/tasks                  my recent tasks
    GET  /v1/tasks/{id}/events      Server-Sent Events (progress; resumable via Last-Event-ID)
    POST /v1/tasks/{id}/cancel
    POST /v1/tasks/{id}/approval    approve / reject a paused task (approvers only)
    GET  /v1/me                     who am I: role, budget, today's usage
    GET  /v1/approvals              tasks paused for confidential-data approval (approvers)
    GET  /v1/refunds?view=mine|inbox|all, GET /v1/refunds/{id}
    POST /v1/refunds/{id}/decision  sign / reject a refund proposal (role from the token; never your own)
    POST /v1/refunds/{id}/reverse   reverse an applied refund within 24h (finance / director)
    GET  /v1/audit                  refund events + data-access decisions (approvers, admins)
    GET  /healthz, /metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from open_deep_research import refunds as RF
from open_deep_research.service import store as S
from open_deep_research.service.dashboard import DASHBOARD_HTML
from open_deep_research.service.auth import AuthError, Authenticator
from open_deep_research.service.ratelimit import SlidingWindowLimiter
from open_deep_research.service.runner import GraphProvider, TaskRunner
from open_deep_research.service.settings import Settings
from open_deep_research.service.store import TaskStore
from open_deep_research.service.worker import WorkerPool
from open_deep_research.telemetry import estimate_cost

logger = logging.getLogger(__name__)
PUBLIC_FIELDS = ("id", "state", "question", "created_at", "updated_at", "result", "error", "awaiting", "attempts")


class TaskIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ApprovalIn(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=500)


class RefundDecisionIn(BaseModel):
    approve: bool
    note: str = Field(default="", max_length=500)


class ReverseIn(BaseModel):
    note: str = Field(default="", max_length=500)


_STATUS_FOR = {"not_found": 404, "forbidden": 403, "conflict": 409, "invalid": 422}


def _utc_iso(ts: str) -> Optional[str]:
    """approvals.jsonl stamps local time with an offset; the refund events are UTC. Compare them in UTC."""
    from datetime import datetime, timezone

    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def read_data_audit(path: str, limit: int) -> list[dict]:
    """Data-access approvals and triage refusals from the JSONL audit log (newest last in the file)."""
    from collections import deque
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    items = []
    with p.open(encoding="utf-8") as f:
        for line in deque(f, maxlen=max(1, limit * 3)):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            at = _utc_iso(r.get("ts", ""))
            if not at:
                continue
            subject = str(r.get("thread_id", "")).split(":")[0]
            if r.get("event") == "request_refused_by_triage":
                items.append({"kind": "refused", "at": at, "actor": "triage", "role": "", "action": "refused",
                              "note": r.get("reason", ""), "detail": str(r.get("request", ""))[:200], "user": subject})
            elif "approved" in r:
                items.append({"kind": "data_access", "at": at, "actor": r.get("reviewer", ""), "role": "",
                              "action": "approved" if r["approved"] else "denied", "note": r.get("note", ""),
                              "detail": str(r.get("query", ""))[:200], "user": subject})
    return items


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
        if settings.business_db_url:
            try:
                await asyncio.to_thread(RF.ensure_schema, RF.sqlite_path(settings.business_db_url))
            except Exception as e:  # noqa: BLE001 - the refund endpoints will report the problem per request
                logger.error("could not prepare the refund tables: %s", e)
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

    async def metric_quiet(request: Request, name: str) -> None:
        """Counters must never fail a business operation that has already been committed."""
        try:
            await metric(request, name)
        except Exception:  # noqa: BLE001
            logger.warning("could not record metric %s", name)

    def is_approver(user: str, task: dict) -> bool:
        return user in settings.approvers or (settings.owner_can_approve and user == task["owner"])

    async def load_visible_task(request: Request, task_id: str, user: str, *, as_approver: bool = False) -> dict:
        task = await request.app.state.store.get(task_id)
        # 404 (not 403) for other people's tasks: do not reveal that the id exists
        if not task or (task["owner"] != user and not (as_approver and user in settings.approvers)):
            raise HTTPException(404, "task not found")
        return task

    def cost_of(u: dict):
        return estimate_cost(u.get("tokens_in", 0), u.get("tokens_out", 0), settings.price_in_per_m, settings.price_out_per_m)

    def utc_day() -> str:
        return time.strftime("%Y%m%d", time.gmtime())

    def public(task: dict) -> dict:
        out = {k: task[k] for k in PUBLIC_FIELDS if k in task}
        if "usage" in task:
            out["usage"] = {**task["usage"], "est_cost_usd": cost_of(task["usage"])}
        return out

    async def current_admin(user: str = Depends(current_user)) -> str:
        if user not in settings.admins:
            raise HTTPException(403, "admin only")
        return user

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

        if settings.daily_token_budget:
            used = await st.usage_day(user, utc_day())
            spent = used.get("tokens_in", 0) + used.get("tokens_out", 0)
            if spent >= settings.daily_token_budget:
                await metric(request, "budget_exceeded")
                now = time.gmtime()
                until_midnight = 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
                raise HTTPException(429, f"daily token budget exhausted ({spent}/{settings.daily_token_budget})",
                                    headers={"Retry-After": str(until_midnight)})

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

    @app.get("/v1/usage")
    async def my_usage(request: Request, days: int = 7, user: str = Depends(current_user)):
        st: TaskStore = request.app.state.store
        today = utc_day()
        history = []
        for i in range(max(1, min(days, 30))):
            day = time.strftime("%Y%m%d", time.gmtime(time.time() - i * 86400))
            u = await st.usage_day(user, day)
            if u:
                history.append({"day": day, **u, "est_cost_usd": cost_of(u)})
        t = await st.usage_day(user, today)
        spent = t.get("tokens_in", 0) + t.get("tokens_out", 0)
        budget = settings.daily_token_budget or None
        return {"today": {"day": today, **t, "est_cost_usd": cost_of(t)},
                "budget": {"daily_tokens": budget, "remaining": None if budget is None else max(0, budget - spent)},
                "history": history}

    @app.get("/v1/admin/summary")
    async def admin_summary(request: Request, admin: str = Depends(current_admin)):
        st: TaskStore = request.app.state.store
        day = utc_day()
        users = []
        for u in await st.users_with_usage(day):
            d = await st.usage_day(u, day)
            users.append({"user": u, **d, "est_cost_usd": cost_of(d)})
        users.sort(key=lambda x: x.get("tokens_in", 0) + x.get("tokens_out", 0), reverse=True)
        totals = await st.global_usage(day)
        tools = [{"tool": t, **v, "error_rate": (v["errors"] / v["calls"]) if v["calls"] else 0.0}
                 for t, v in sorted((await st.tool_stats()).items(), key=lambda kv: -kv[1]["calls"])]
        raw = await request.app.state.redis.hgetall(f"{settings.prefix}metrics")
        return {"day": day, "totals": {**totals, "est_cost_usd": cost_of(totals)}, "users": users, "tools": tools,
                "latency": await st.latency_hist(), "counters": {k.decode(): int(v) for k, v in sorted(raw.items())},
                "queue_depth": await request.app.state.redis.llen(st.queue_key)}

    # ------------------------------------------------------------ identity, approvals, refunds
    def role_of(user: str) -> Optional[str]:
        return settings.roles.get(user)

    def biz_path() -> str:
        if not settings.business_db_url:
            raise HTTPException(404, "the refund workflow is not enabled")
        try:
            return RF.sqlite_path(settings.business_db_url)
        except ValueError as e:
            raise HTTPException(503, str(e))

    def raise_for(out: RF.Outcome) -> None:
        if not out.ok:
            raise HTTPException(_STATUS_FOR.get(out.code, 400), out.message)

    @app.get("/v1/me")
    async def me(request: Request, user: str = Depends(current_user)):
        st: TaskStore = request.app.state.store
        t = await st.usage_day(user, utc_day())
        spent = t.get("tokens_in", 0) + t.get("tokens_out", 0)
        budget = settings.daily_token_budget or None
        return {"user": user, "role": role_of(user), "is_admin": user in settings.admins,
                "can_approve_data": user in settings.approvers, "actions_enabled": bool(settings.business_db_url),
                "budget": {"daily_tokens": budget, "used": spent, "remaining": None if budget is None else max(0, budget - spent)},
                "today": {**t, "est_cost_usd": cost_of(t)}}

    @app.get("/v1/approvals")
    async def pending_data_approvals(request: Request, user: str = Depends(current_user)):
        st: TaskStore = request.app.state.store
        if user not in settings.approvers:
            return {"tasks": []}
        out = []
        for raw in await st.r.smembers(f"{settings.prefix}awaiting"):
            task = await st.get(raw.decode() if isinstance(raw, bytes) else raw)
            if task and task["state"] == S.AWAITING and is_approver(user, task):
                out.append({**public(task), "owner": task["owner"]})
        out.sort(key=lambda t: t["updated_at"])
        return {"tasks": out}

    @app.get("/v1/refunds")
    async def list_refunds(view: str = "mine", status: Optional[str] = None, user: str = Depends(current_user)):
        path, role = biz_path(), role_of(user)
        if view == "mine":
            rows = await asyncio.to_thread(lambda: RF.list_refunds(path, status=status, requested_by=user))
        elif view == "inbox":
            rows = [] if not role else [r for r in await asyncio.to_thread(lambda: RF.list_refunds(path, awaiting_role=role))
                                        if r["requested_by"] != user]
        elif view == "all":
            if not (role or user in settings.admins):
                raise HTTPException(403, "approvers and admins only")
            rows = await asyncio.to_thread(lambda: RF.list_refunds(path, status=status))
        else:
            raise HTTPException(422, "view must be mine, inbox or all")
        return {"refunds": rows}

    @app.get("/v1/refunds/{refund_id}")
    async def get_refund(refund_id: int, user: str = Depends(current_user)):
        r = await asyncio.to_thread(RF.get, biz_path(), refund_id)
        if not r or not (r["requested_by"] == user or role_of(user) or user in settings.admins):
            raise HTTPException(404, "refund not found")        # 404, not 403: do not reveal that the id exists
        return r

    @app.post("/v1/refunds/{refund_id}/decision")
    async def decide_refund(refund_id: int, body: RefundDecisionIn, request: Request, user: str = Depends(current_user)):
        path, role = biz_path(), role_of(user)
        if role is None:
            raise HTTPException(403, "you do not hold an approver role")
        # actor and role come from the token; nothing in the body (or in anything the model wrote) can change them
        out = await asyncio.to_thread(lambda: RF.decide(path, refund_id=refund_id, user=user, role=role,
                                                         approve=body.approve, note=body.note))
        raise_for(out)
        await metric_quiet(request, "refund_signed" if body.approve else "refund_rejected")
        if out.refund["status"] == "applied":
            await metric_quiet(request, "refund_applied")
        return out.refund

    @app.post("/v1/refunds/{refund_id}/reverse")
    async def reverse_refund(refund_id: int, body: ReverseIn, request: Request, user: str = Depends(current_user)):
        path, role = biz_path(), role_of(user)
        if role is None:
            raise HTTPException(403, "you do not hold an approver role")
        out = await asyncio.to_thread(lambda: RF.reverse(path, refund_id=refund_id, user=user, role=role, note=body.note))
        raise_for(out)
        await metric_quiet(request, "refund_reversed")
        return out.refund

    @app.get("/v1/audit")
    async def audit_feed(limit: int = 100, user: str = Depends(current_user)):
        if not (role_of(user) or user in settings.admins):
            raise HTTPException(403, "approvers and admins only")
        limit = max(1, min(limit, 300))
        items: list[dict] = []
        if settings.business_db_url:
            for e in await asyncio.to_thread(RF.audit_trail, biz_path(), limit):
                items.append({"kind": "refund", "at": e["at"], "actor": e["actor"], "role": e["role"], "action": e["action"],
                              "note": e["note"], "refund_id": e["refund_id"], "order_id": e["order_id"], "amount": e["amount"]})
        items += await asyncio.to_thread(read_data_audit, settings.approval_log_path, limit)
        items.sort(key=lambda x: x["at"], reverse=True)
        return {"items": items[:limit]}

    if settings.demo_mode:
        @app.get("/demo/users")
        async def demo_users():
            """Demo only: lists the sample accounts and their tokens so the UI can switch persona. Never enable in production."""
            return {"users": [{"user": u, "role": role_of(u), "key": k} for k, u in settings.api_keys.items()]}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request):
        raw = await request.app.state.redis.hgetall(f"{settings.prefix}metrics")
        lines = [f"odr_{k.decode()} {int(v)}" for k, v in sorted(raw.items())]
        lines.append(f"odr_queue_depth {await request.app.state.redis.llen(request.app.state.store.queue_key)}")
        st: TaskStore = request.app.state.store
        g = await st.global_usage(utc_day())
        lines += [f"odr_tokens_in_today {g.get('tokens_in', 0)}", f"odr_tokens_out_today {g.get('tokens_out', 0)}"]
        for tool, v in (await st.tool_stats()).items():
            lines += [f'odr_tool_calls_total{{tool="{tool}"}} {v["calls"]}', f'odr_tool_errors_total{{tool="{tool}"}} {v["errors"]}']
        for bucket, n in (await st.latency_hist()).items():
            lines.append(f'odr_task_latency_bucket{{le="{bucket[3:]}"}} {n}')
        return "\n".join(lines) + "\n"

    return app
