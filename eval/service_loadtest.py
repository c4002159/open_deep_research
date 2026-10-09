"""Load / correctness test of the service layer over real HTTP (uvicorn) and real Redis.

    python -m eval.service_loadtest

The agent is replaced by a tiny LangGraph graph (3 nodes, ~20 ms each) so this measures the
service itself - admission control, queueing, Redis round-trips, leases - not LLM latency.
Scenarios:
  1 throughput      600 tasks from 40 users, 64 concurrent clients; all must finish exactly once
  2 rate limit      one user fires 100 concurrent requests against a 10/min limit
  3 idempotency     300 concurrent requests with the same Idempotency-Key
  4 worker crash    kill every worker mid-load; replacements must finish all tasks, none lost
"""

from __future__ import annotations

import asyncio
import json
import operator
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated

import httpx
import uvicorn
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from open_deep_research.service.app import create_app
from open_deep_research.service.settings import Settings

REDIS = "redis://localhost:6390/0"
OUT = Path(__file__).resolve().parent / "service_loadtest_report.json"


class St(TypedDict):
    messages: list
    final_report: str
    log: Annotated[list, operator.add]


class FakeProvider:
    def __init__(self, node_delay=0.02):
        self.counts = {"a": 0, "b": 0, "c": 0}
        self.node_delay = node_delay
        self.graph = None

    async def start(self):
        d, counts = self.node_delay, self.counts

        def mk(name):
            async def f(state):
                counts[name] += 1
                await asyncio.sleep(d)
                if name == "c":
                    return {"final_report": f"ok:{state['messages'][-1]['content']}"}
                return {"log": [name]}
            return f

        g = StateGraph(St)
        for n in "abc":
            g.add_node(n, mk(n))
        g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", "c"); g.add_edge("c", END)
        self.graph = g.compile(checkpointer=InMemorySaver())

    async def stop(self):
        pass


class Server:
    """uvicorn in a background thread (own event loop) so the load generator can use its own."""

    def __init__(self, settings: Settings, provider: FakeProvider, port: int):
        self.app = create_app(settings, provider)
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="warning", loop="asyncio"))
        self.thread = threading.Thread(target=lambda: asyncio.run(self.server.serve()), daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/healthz", timeout=1).status_code == 200:
                    return self
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def _serve(port: int, settings: Settings, node_delay: float) -> None:        # runs in a child process
    import asyncio as _a
    import sys as _s

    if _s.platform == "win32":
        _a.set_event_loop_policy(_a.WindowsSelectorEventLoopPolicy())
    uvicorn.run(create_app(settings, FakeProvider(node_delay)), host="127.0.0.1", port=port, log_level="warning")


class ProcServer:
    """Server in its own OS process: no shared GIL with the load generator, so latencies are honest."""

    def __init__(self, settings: Settings, port: int, node_delay: float = 0.02):
        import multiprocessing as mp

        self.port = port
        self.proc = mp.get_context("spawn").Process(target=_serve, args=(port, settings, node_delay), daemon=True)

    def __enter__(self):
        self.proc.start()
        for _ in range(300):
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/healthz", timeout=1).status_code == 200:
                    return self
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("server process did not start")

    def __exit__(self, *a):
        self.proc.terminate()
        self.proc.join(timeout=10)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def base_settings(**kw) -> Settings:
    d = dict(redis_url=REDIS, prefix=f"odrload:{uuid.uuid4().hex[:6]}:",
             api_keys={f"key{i}": f"user{i}" for i in range(40)} | {"keyX": "solo"},
             rate_limit_per_window=10**6, max_active_per_user=10**6, workers=16,
             lease_ttl_s=2, lease_refresh_s=0.5, reaper_interval_s=0.3, task_timeout_s=60)
    d.update(kw)
    return Settings(**d)


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p / 100))] * 1000, 1)


async def wait_all(c, headers_by_task, timeout=120):
    pending, t0, states = dict(headers_by_task), time.monotonic(), {}
    while pending and time.monotonic() - t0 < timeout:
        async def one(tid, h):
            return tid, (await c.get(f"/v1/tasks/{tid}", headers=h)).json()["state"]
        done = await asyncio.gather(*(one(t, h) for t, h in list(pending.items())[:200]))
        for tid, st in done:
            if st in ("succeeded", "failed", "cancelled"):
                states[tid] = st
                pending.pop(tid)
        await asyncio.sleep(0.05)
    return states, len(pending)


async def cleanup(settings):
    import redis.asyncio as aioredis

    r = aioredis.from_url(REDIS)
    keys = [k async for k in r.scan_iter(match=f"{settings.prefix}*")]
    if keys:
        await r.delete(*keys)
    await r.aclose()


# ------------------------------------------------------------------ scenarios
async def s_throughput(port=8771):
    s = base_settings()
    with ProcServer(s, port) as srv:                              # separate process: honest latency numbers
        async with httpx.AsyncClient(base_url=srv.url, timeout=60, limits=httpx.Limits(max_connections=100)) as c:
            sem, lat, owners = asyncio.Semaphore(64), [], {}

            async def submit(i):
                h = {"Authorization": f"Bearer key{i % 40}"}
                async with sem:
                    t = time.perf_counter()
                    r = await c.post("/v1/tasks", json={"question": f"job {i}"}, headers=h)
                    lat.append(time.perf_counter() - t)
                    assert r.status_code == 202, r.text
                    owners[r.json()["id"]] = h

            t0 = time.perf_counter()
            await asyncio.gather(*(submit(i) for i in range(600)))
            t_submit = time.perf_counter() - t0
            states, left = await wait_all(c, owners)
            t_total = time.perf_counter() - t0
            metrics = (await c.get("/metrics")).text
    await cleanup(s)
    return {"tasks": 600, "submit_rps": round(600 / t_submit), "end_to_end_tasks_per_s": round(600 / t_total, 1),
            "submit_p50_ms": pct(lat, 50), "submit_p95_ms": pct(lat, 95), "submit_p99_ms": pct(lat, 99),
            "succeeded": sum(v == "succeeded" for v in states.values()), "unfinished": left,
            "server_metrics": " ".join(metrics.split()), "total_seconds": round(t_total, 2)}


async def s_rate_limit(port=8772):
    s = base_settings(rate_limit_per_window=10)
    with Server(s, FakeProvider(), port) as srv:
        async with httpx.AsyncClient(base_url=srv.url, timeout=30) as c:
            rs = await asyncio.gather(*(c.post("/v1/tasks", json={"question": f"q{i}"}, headers={"Authorization": "Bearer keyX"})
                                        for i in range(100)))
            other = await c.post("/v1/tasks", json={"question": "x"}, headers={"Authorization": "Bearer key1"})
    await cleanup(s)
    codes = [r.status_code for r in rs]
    return {"requests": 100, "limit_per_minute": 10, "accepted_202": codes.count(202), "throttled_429": codes.count(429),
            "all_429_have_retry_after": all("retry-after" in r.headers for r in rs if r.status_code == 429),
            "other_user_unaffected": other.status_code == 202}


async def s_idempotency(port=8773):
    s, prov = base_settings(), FakeProvider()
    with Server(s, prov, port) as srv:
        async with httpx.AsyncClient(base_url=srv.url, timeout=30, limits=httpx.Limits(max_connections=100)) as c:
            h = {"Authorization": "Bearer keyX", "Idempotency-Key": "storm"}
            rs = await asyncio.gather(*(c.post("/v1/tasks", json={"question": "same"}, headers=h) for _ in range(300)))
            ids = {r.json()["id"] for r in rs}
            await wait_all(c, {i: h for i in ids})
            await asyncio.sleep(0.3)
    await cleanup(s)
    codes = [r.status_code for r in rs]
    return {"concurrent_requests": 300, "created_202": codes.count(202), "replayed_200": codes.count(200),
            "distinct_task_ids": len(ids), "agent_executions": prov.counts["c"]}


async def s_worker_crash(port=8774):
    s, prov = base_settings(), FakeProvider(node_delay=0.15)
    with Server(s, prov, port) as srv:
        async with httpx.AsyncClient(base_url=srv.url, timeout=60, limits=httpx.Limits(max_connections=100)) as c:
            owners = {}
            for i in range(100):
                h = {"Authorization": f"Bearer key{i % 40}"}
                r = await c.post("/v1/tasks", json={"question": f"job {i}"}, headers=h)
                owners[r.json()["id"]] = h
            await asyncio.sleep(1.0)                                   # let the workers get busy
            pool = srv.app.state.pool                                  # lives on the server thread's event loop
            loop = pool._tasks[0].get_loop()
            asyncio.run_coroutine_threadsafe(pool.kill_workers(), loop).result(10)   # crash: leases/queue entries stay behind
            asyncio.run_coroutine_threadsafe(_restart(pool), loop).result(10)         # replacement workers + reaper
            states, left = await wait_all(c, owners, timeout=60)
    await cleanup(s)
    return {"tasks": 100, "succeeded": sum(v == "succeeded" for v in states.values()), "lost_or_unfinished": left,
            "node_a_runs": prov.counts["a"], "node_c_runs": prov.counts["c"],
            "redundant_node_a_runs": prov.counts["a"] - 100, "redundant_node_c_runs": prov.counts["c"] - 100,
            "note": "runs beyond 100 are at-least-once redeliveries of tasks that were in flight at the crash"}


async def _restart(pool):
    pool.start(16)


async def main():
    report = {}
    for name, fn in (("1_throughput", s_throughput), ("2_rate_limit", s_rate_limit),
                     ("3_idempotency_storm", s_idempotency), ("4_worker_crash", s_worker_crash)):
        print(f"running {name} ...", flush=True)
        report[name] = await fn()
        print(json.dumps(report[name], indent=2), flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
