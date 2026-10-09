"""End-to-end service tests: real Redis, real FastAPI app, real LangGraph (a small stand-in graph
with the same interrupt / checkpoint semantics as the agent, so no LLM is needed)."""

import asyncio
import contextlib
import operator
import time
import uuid
from typing import Annotated

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from open_deep_research.service.app import create_app
from open_deep_research.service.settings import Settings
from tests.redis_fixtures import REDIS_URL, needs_redis

pytestmark = needs_redis

KEYS = {"ka": "alice", "kb": "bob", "kc": "carol"}
A, B, C = ({"Authorization": f"Bearer {k}"} for k in ("ka", "kb", "kc"))


class St(TypedDict):
    messages: list
    final_report: str
    log: Annotated[list, operator.add]


class FakeProvider:
    """Stand-in for GraphProvider: same interface, a tiny graph, a shared in-memory checkpointer."""

    def __init__(self):
        self.counts = {"a": 0, "b": 0, "c": 0}
        self.saver = InMemorySaver()           # survives "worker restarts" like a durable checkpointer
        self.graph = None

    async def start(self):
        counts = self.counts

        async def a(state):
            counts["a"] += 1
            return {"log": ["a"]}

        async def b(state):
            counts["b"] += 1
            q = state["messages"][-1]["content"]
            if "SLOW" in q:
                await asyncio.sleep(30 if "FOREVER" in q else 2.0)
            if "SENSITIVE" in q:
                decision = interrupt({"type": "approval_required", "requests": [{"tool": "sql_executor", "reason": "row-level"}]})
                return {"log": [f"approved={decision['approved']}:{decision['reviewer']}"]}
            return {"log": ["b"]}

        async def c(state):
            counts["c"] += 1
            return {"final_report": f"done:{state['messages'][-1]['content']}:{','.join(state['log'])}"}

        g = StateGraph(St)
        for n, f in (("a", a), ("b", b), ("c", c)):
            g.add_node(n, f)
        g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", "c"); g.add_edge("c", END)
        self.graph = g.compile(checkpointer=self.saver)

    async def stop(self):
        pass


def make_settings(**kw) -> Settings:
    base = dict(redis_url=REDIS_URL, prefix=f"odrtest:{uuid.uuid4().hex[:8]}:", api_keys=KEYS,
                approvers=frozenset({"carol"}), rate_limit_per_window=100, max_active_per_user=50, workers=2,
                lease_ttl_s=1, lease_refresh_s=0.3, reaper_interval_s=0.2, task_timeout_s=30)
    base.update(kw)
    return Settings(**base)


@contextlib.asynccontextmanager
async def running(settings=None, provider=None, start_workers=True):
    settings = settings or make_settings()
    provider = provider or FakeProvider()
    app = create_app(settings, provider, start_workers=start_workers)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            try:
                yield client, app, provider
            finally:
                keys = [k async for k in app.state.redis.scan_iter(match=f"{settings.prefix}*")]
                if keys:
                    await app.state.redis.delete(*keys)


async def wait_state(client, tid, states, headers=A, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t = (await client.get(f"/v1/tasks/{tid}", headers=headers)).json()
        if t.get("state") in states:
            return t
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {tid} never reached {states}; last={t}")


def run(coro_fn):
    def wrapper():
        return asyncio.run(coro_fn())
    wrapper.__name__ = coro_fn.__name__
    return wrapper


# ---------------------------------------------------------------- basics / auth
@run
async def test_submit_poll_and_result():
    async with running() as (c, app, prov):
        r = await c.post("/v1/tasks", json={"question": "hello"}, headers=A)
        assert r.status_code == 202 and r.json()["state"] == "queued"
        t = await wait_state(c, r.json()["id"], {"succeeded"})
        assert t["result"]["report"] == "done:hello:a,b"
        assert (await c.get("/healthz")).json()["status"] == "ok"


@run
async def test_authentication_required():
    async with running() as (c, *_):
        assert (await c.post("/v1/tasks", json={"question": "x"})).status_code == 401
        assert (await c.post("/v1/tasks", json={"question": "x"}, headers={"Authorization": "Bearer nope"})).status_code == 401
        assert (await c.post("/v1/tasks", json={"question": "x"}, headers={"Authorization": "Basic ka"})).status_code == 401


@run
async def test_input_validation():
    async with running() as (c, *_):
        assert (await c.post("/v1/tasks", json={"question": ""}, headers=A)).status_code == 422
        assert (await c.post("/v1/tasks", json={"question": "x" * 4001}, headers=A)).status_code == 422


# ---------------------------------------------------------------- idempotency
@run
async def test_idempotency_key_runs_once_and_replays_result():
    async with running() as (c, app, prov):
        h = {**A, "Idempotency-Key": "order-42"}
        first = await c.post("/v1/tasks", json={"question": "same"}, headers=h)
        second = await c.post("/v1/tasks", json={"question": "same"}, headers=h)
        assert first.status_code == 202 and second.status_code == 200
        assert second.headers["Idempotent-Replay"] == "true" and first.json()["id"] == second.json()["id"]
        await wait_state(c, first.json()["id"], {"succeeded"})
        assert prov.counts["a"] == 1, "the duplicate request must not run the agent again"


@run
async def test_idempotency_key_with_different_body_is_a_conflict():
    async with running() as (c, *_):
        h = {**A, "Idempotency-Key": "k"}
        await c.post("/v1/tasks", json={"question": "one"}, headers=h)
        assert (await c.post("/v1/tasks", json={"question": "two"}, headers=h)).status_code == 409


@run
async def test_concurrent_duplicate_submissions_create_one_task():
    async with running() as (c, app, prov):
        h = {**A, "Idempotency-Key": "burst"}
        rs = await asyncio.gather(*(c.post("/v1/tasks", json={"question": "burst"}, headers=h) for _ in range(20)))
        assert len({r.json()["id"] for r in rs}) == 1
        assert sum(r.status_code == 202 for r in rs) == 1
        await wait_state(c, rs[0].json()["id"], {"succeeded"})
        assert prov.counts["a"] == 1


@run
async def test_replays_do_not_consume_rate_limit():
    async with running(make_settings(rate_limit_per_window=2)) as (c, *_):
        h = {**A, "Idempotency-Key": "r"}
        for _ in range(5):
            assert (await c.post("/v1/tasks", json={"question": "q"}, headers=h)).status_code in (200, 202)
        assert (await c.post("/v1/tasks", json={"question": "other"}, headers=A)).status_code == 202   # quota intact


# ---------------------------------------------------------------- rate limit / concurrency
@run
async def test_rate_limit_returns_429_with_retry_after():
    async with running(make_settings(rate_limit_per_window=3)) as (c, *_):
        codes = [(await c.post("/v1/tasks", json={"question": f"q{i}"}, headers=A)) for i in range(5)]
        assert [r.status_code for r in codes] == [202, 202, 202, 429, 429]
        assert int(codes[3].headers["Retry-After"]) >= 1 and codes[2].headers["X-RateLimit-Remaining"] == "0"
        assert (await c.post("/v1/tasks", json={"question": "bob"}, headers=B)).status_code == 202  # other user unaffected


@run
async def test_active_task_cap_frees_up_on_cancel():
    async with running(make_settings(max_active_per_user=1)) as (c, *_):
        first = await c.post("/v1/tasks", json={"question": "SLOW FOREVER"}, headers=A)
        assert (await c.post("/v1/tasks", json={"question": "again"}, headers=A)).status_code == 429
        await c.post(f"/v1/tasks/{first.json()['id']}/cancel", headers=A)
        await wait_state(c, first.json()["id"], {"cancelled"})
        assert (await c.post("/v1/tasks", json={"question": "again"}, headers=A)).status_code == 202


# ---------------------------------------------------------------- isolation
@run
async def test_users_cannot_see_or_touch_each_others_tasks():
    async with running() as (c, *_):
        tid = (await c.post("/v1/tasks", json={"question": "secret"}, headers=A)).json()["id"]
        await wait_state(c, tid, {"succeeded"})
        assert (await c.get(f"/v1/tasks/{tid}", headers=B)).status_code == 404
        assert (await c.get(f"/v1/tasks/{tid}/events", headers=B)).status_code == 404
        assert (await c.post(f"/v1/tasks/{tid}/cancel", headers=B)).status_code == 404
        assert (await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=B)).status_code == 404
        assert [t["id"] for t in (await c.get("/v1/tasks", headers=B)).json()["tasks"]] == []
        assert [t["id"] for t in (await c.get("/v1/tasks", headers=A)).json()["tasks"]] == [tid]


@run
async def test_thread_ids_and_memory_user_are_namespaced_by_user():
    from open_deep_research.service.runner import TaskRunner, thread_id_for

    async with running(start_workers=False) as (c, app, prov):
        cfg = app.state.runner._config("alice", "T1")["configurable"]
        assert cfg["thread_id"] == thread_id_for("alice", "T1") == "alice:T1"
        assert cfg["mem0_user_id"] == "alice" and cfg["approval_mode"] == "interrupt"


# ---------------------------------------------------------------- cancel
@run
async def test_cancel_running_task_keeps_the_worker_alive():
    async with running(make_settings(workers=1)) as (c, app, prov):
        slow = (await c.post("/v1/tasks", json={"question": "SLOW FOREVER"}, headers=A)).json()["id"]
        await wait_state(c, slow, {"running"})
        assert (await c.post(f"/v1/tasks/{slow}/cancel", headers=A)).status_code == 202
        await wait_state(c, slow, {"cancelled"})
        nxt = (await c.post("/v1/tasks", json={"question": "after"}, headers=A)).json()["id"]
        assert (await wait_state(c, nxt, {"succeeded"}))["result"]["report"].startswith("done:after")  # same single worker


@run
async def test_cancel_finished_task_is_rejected():
    async with running() as (c, *_):
        tid = (await c.post("/v1/tasks", json={"question": "x"}, headers=A)).json()["id"]
        await wait_state(c, tid, {"succeeded"})
        assert (await c.post(f"/v1/tasks/{tid}/cancel", headers=A)).status_code == 409


# ---------------------------------------------------------------- approval
@run
async def test_approval_flow_pauses_then_resumes_without_redoing_work():
    async with running() as (c, app, prov):
        tid = (await c.post("/v1/tasks", json={"question": "SENSITIVE rows"}, headers=A)).json()["id"]
        t = await wait_state(c, tid, {"awaiting_approval"})
        assert t["awaiting"]["type"] == "approval_required"
        assert prov.counts["c"] == 0
        # the requester is not an approver -> forbidden; a stranger sees nothing
        assert (await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=A)).status_code == 403
        assert (await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=B)).status_code == 404
        r = await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True, "note": "audit"}, headers=C)
        assert r.status_code == 202
        done = await wait_state(c, tid, {"succeeded"})
        assert "approved=True:carol" in done["result"]["report"], "reviewer identity comes from the token"
        assert prov.counts["a"] == 1, "node a must not rerun after resume"


@run
async def test_rejection_is_recorded_and_task_still_completes():
    async with running() as (c, *_):
        tid = (await c.post("/v1/tasks", json={"question": "SENSITIVE rows"}, headers=A)).json()["id"]
        await wait_state(c, tid, {"awaiting_approval"})
        await c.post(f"/v1/tasks/{tid}/approval", json={"approved": False}, headers=C)
        done = await wait_state(c, tid, {"succeeded"})
        assert "approved=False:carol" in done["result"]["report"]


@run
async def test_approving_a_task_that_is_not_waiting_is_a_conflict():
    async with running() as (c, *_):
        tid = (await c.post("/v1/tasks", json={"question": "plain"}, headers=A)).json()["id"]
        await wait_state(c, tid, {"succeeded"})
        assert (await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=C)).status_code == 409


@run
async def test_owner_self_approval_only_when_explicitly_enabled():
    async with running(make_settings(owner_can_approve=True)) as (c, *_):
        tid = (await c.post("/v1/tasks", json={"question": "SENSITIVE"}, headers=A)).json()["id"]
        await wait_state(c, tid, {"awaiting_approval"})
        assert (await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=A)).status_code == 202


# ---------------------------------------------------------------- crash recovery
@run
async def test_task_survives_worker_death_and_resumes_from_checkpoint():
    async with running(make_settings(workers=1)) as (c, app, prov):
        tid = (await c.post("/v1/tasks", json={"question": "SLOW job"}, headers=A)).json()["id"]
        for _ in range(100):                                   # wait until node b is executing
            if prov.counts["b"] >= 1:
                break
            await asyncio.sleep(0.05)
        assert prov.counts["a"] == 1 and prov.counts["c"] == 0
        await app.state.pool.kill_workers()                    # process "dies": lease and queue entry left behind
        assert (await c.get(f"/v1/tasks/{tid}", headers=A)).json()["state"] == "running"
        app.state.pool.start(1)                                # a replacement worker + reaper come up
        done = await wait_state(c, tid, {"succeeded"}, timeout=10)
        assert done["attempts"] == 2
        assert prov.counts["a"] == 1, "finished node a was NOT redone after the crash"
        assert prov.counts["b"] == 2, "interrupted node b was retried"
        types = [e["event"] for e in await _read_sse(c, tid)]
        assert "requeued" in types and "continued" in types


# ---------------------------------------------------------------- SSE
async def _read_sse(c, tid, headers=A, last_id=None):
    h = dict(headers)
    if last_id:
        h["Last-Event-ID"] = last_id
    events, cur = [], {}
    async with c.stream("GET", f"/v1/tasks/{tid}/events", headers=h) as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if line.startswith("id: "): cur["id"] = line[4:]
            elif line.startswith("event: "): cur["event"] = line[7:]
            elif line.startswith("data: "): cur["data"] = line[6:]
            elif line == "" and cur:
                events.append(cur)
                if cur.get("event") == "end":
                    break
                cur = {}
    return events


@run
async def test_sse_streams_progress_in_order_and_supports_resume():
    async with running() as (c, *_):
        tid = (await c.post("/v1/tasks", json={"question": "stream me"}, headers=A)).json()["id"]
        events = await _read_sse(c, tid)
        types = [e["event"] for e in events]
        assert types[0] == "queued" and types[-1] == "end" and "succeeded" in types
        nodes = [e["data"] for e in events if e["event"] == "progress"]
        assert [n for n in ("a", "b", "c") if any(f'"{n}"' in d for d in nodes)] == ["a", "b", "c"]
        first_id = events[0]["id"]
        later = await _read_sse(c, tid, last_id=first_id)
        assert "queued" not in [e["event"] for e in later] and later[-1]["event"] == "end"


# ---------------------------------------------------------------- metrics
@run
async def test_metrics_endpoint_counts_submissions_and_throttling():
    async with running(make_settings(rate_limit_per_window=1)) as (c, *_):
        await c.post("/v1/tasks", json={"question": "a"}, headers=A)
        await c.post("/v1/tasks", json={"question": "b"}, headers=A)
        text = (await c.get("/metrics")).text
        assert "odr_submitted 1" in text and "odr_rate_limited 1" in text and "odr_queue_depth" in text
