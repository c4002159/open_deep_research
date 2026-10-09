"""Usage accounting, trace files, budgets and the admin dashboard - real Redis, real LangGraph,
a scripted chat model that reports token usage the way real providers do."""

import asyncio
import json
import operator
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated

import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from open_deep_research.service.app import create_app
from open_deep_research.service.settings import Settings
from open_deep_research.telemetry import UsageTracker, estimate_cost
from tests.redis_fixtures import REDIS_URL, needs_redis

pytestmark = needs_redis
A = {"Authorization": "Bearer ka"}
B = {"Authorization": "Bearer kb"}
ADMIN = {"Authorization": "Bearer kadmin"}


class St(TypedDict):
    messages: list
    final_report: str
    log: Annotated[list, operator.add]


def usage_msg(i=100, o=20):
    return AIMessage(content="ok", usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o})


@tool
async def good_tool(x: str) -> str:
    """works"""
    return "rows"


@tool
async def bad_tool(x: str) -> str:
    """fails the way our SQL tool reports failures"""
    return "SQL execution error: no such column: foo"


class Provider:
    def __init__(self, fail=False):
        self.fail, self.graph = fail, None

    async def start(self):
        fail = self.fail

        async def work(state, config):
            llm = GenericFakeChatModel(messages=iter([usage_msg(100, 20)]))
            await llm.ainvoke("hello", config)
            await good_tool.ainvoke({"x": "1"}, config)
            await bad_tool.ainvoke({"x": "1"}, config)
            if fail:
                raise RuntimeError("boom after spending tokens")
            return {"final_report": "done", "log": ["work"]}

        g = StateGraph(St)
        g.add_node("work", work)
        g.add_edge(START, "work"); g.add_edge("work", END)
        self.graph = g.compile(checkpointer=InMemorySaver())

    async def stop(self):
        pass


def settings(**kw) -> Settings:
    base = dict(redis_url=REDIS_URL, prefix=f"odrtest:{uuid.uuid4().hex[:8]}:",
                api_keys={"ka": "alice", "kb": "bob", "kadmin": "root"}, admins=frozenset({"root"}),
                rate_limit_per_window=100, max_active_per_user=50, workers=2, lease_ttl_s=2, lease_refresh_s=0.5,
                reaper_interval_s=0.3, trace_dir=tempfile.mkdtemp(prefix="odr_trace_"))
    base.update(kw)
    return Settings(**base)


class running:
    def __init__(self, s=None, provider=None):
        self.s, self.provider = s or settings(), provider or Provider()

    async def __aenter__(self):
        self.app = create_app(self.s, self.provider)
        self.cm = self.app.router.lifespan_context(self.app)
        await self.cm.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t")
        return self.client, self.app

    async def __aexit__(self, *a):
        await self.client.aclose()
        r = self.app.state.redis
        keys = [k async for k in r.scan_iter(match=f"{self.s.prefix}*")]
        if keys:
            await r.delete(*keys)
        await self.cm.__aexit__(*a)


async def done(c, tid, headers=A, states=("succeeded", "failed")):
    for _ in range(200):
        t = (await c.get(f"/v1/tasks/{tid}", headers=headers)).json()
        if t["state"] in states:
            # usage is written just after the terminal transition; give it a moment
            for _ in range(40):
                if "usage" in t:
                    return t
                await asyncio.sleep(0.05)
                t = (await c.get(f"/v1/tasks/{tid}", headers=headers)).json()
            return t
        await asyncio.sleep(0.05)
    raise AssertionError("task did not finish")


def run(fn):
    def wrapper():
        return asyncio.run(fn())
    wrapper.__name__ = fn.__name__
    return wrapper


# ------------------------------------------------------------------ tracker unit tests
def test_tracker_counts_tokens_tools_and_nodes():
    from uuid import uuid4

    tr = UsageTracker()
    rid = uuid4()
    tr.on_tool_start({"name": "sql_executor"}, "q", run_id=rid)
    tr.on_tool_end("SQL execution error: boom", run_id=rid)
    rid = uuid4()
    tr.on_tool_start({"name": "sql_executor"}, "q", run_id=rid)
    tr.on_tool_end("a | b\n1 | 2", run_id=rid)
    s = tr.summary()
    assert s["tool_calls"] == 2 and s["tool_errors"] == 1
    assert s["by_tool"]["sql_executor"]["calls"] == 2 and s["by_tool"]["sql_executor"]["errors"] == 1


def test_cost_is_never_guessed():
    assert estimate_cost(1_000_000, 1_000_000, None, None) is None
    assert estimate_cost(1_000_000, 500_000, 2.0, 4.0) == 4.0


# ------------------------------------------------------------------ per-task usage
@run
async def test_task_reports_tokens_tool_stats_and_cost():
    async with running(settings(price_in_per_m=1000.0, price_out_per_m=2000.0)) as (c, app):
        tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=A)).json()["id"]
        t = await done(c, tid)
        u = t["usage"]
        assert (u["tokens_in"], u["tokens_out"], u["llm_calls"]) == (100, 20, 1)
        assert (u["tool_calls"], u["tool_errors"]) == (2, 1)
        assert u["est_cost_usd"] == round((100 * 1000 + 20 * 2000) / 1e6, 6)


@run
async def test_cost_is_null_when_prices_not_configured():
    async with running() as (c, app):
        tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=A)).json()["id"]
        assert (await done(c, tid))["usage"]["est_cost_usd"] is None


@run
async def test_usage_is_recorded_even_when_the_task_fails():
    async with running(provider=Provider(fail=True)) as (c, app):
        tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=A)).json()["id"]
        t = await done(c, tid)
        assert t["state"] == "failed" and t["usage"]["tokens_in"] == 100, "tokens were spent, so they must be counted"


@run
async def test_trace_file_has_llm_tool_and_node_spans():
    s = settings()
    async with running(s) as (c, app):
        tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=A)).json()["id"]
        await done(c, tid)
    lines = [json.loads(l) for l in (Path(s.trace_dir) / f"{tid}.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {l["type"] for l in lines}
    assert {"llm", "tool", "node"} <= kinds
    assert [l["ok"] for l in lines if l["type"] == "tool"] == [True, False]
    assert all(l["task"] == tid for l in lines)


# ------------------------------------------------------------------ user totals + budget
@run
async def test_my_usage_endpoint_and_budget_remaining():
    async with running(settings(daily_token_budget=1000)) as (c, app):
        tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=A)).json()["id"]
        await done(c, tid)
        u = (await c.get("/v1/usage", headers=A)).json()
        assert u["today"]["tokens_in"] == 100 and u["today"]["tokens_out"] == 20 and u["today"]["runs"] == 1
        assert u["budget"] == {"daily_tokens": 1000, "remaining": 880}
        assert (await c.get("/v1/usage", headers=B)).json()["today"].get("tokens_in", 0) == 0     # per-user totals


@run
async def test_daily_budget_blocks_new_tasks_but_not_replays_or_other_users():
    async with running(settings(daily_token_budget=100)) as (c, app):          # one task costs 120 tokens
        first = await c.post("/v1/tasks", json={"question": "q"}, headers={**A, "Idempotency-Key": "k"})
        await done(c, first.json()["id"])
        blocked = await c.post("/v1/tasks", json={"question": "again"}, headers=A)
        assert blocked.status_code == 429 and "budget" in blocked.json()["detail"] and int(blocked.headers["Retry-After"]) > 0
        replay = await c.post("/v1/tasks", json={"question": "q"}, headers={**A, "Idempotency-Key": "k"})
        assert replay.status_code == 200                                         # replays cost nothing
        assert (await c.post("/v1/tasks", json={"question": "bob"}, headers=B)).status_code == 202


@run
async def test_budget_disabled_by_default():
    async with running() as (c, app):
        for i in range(3):
            tid = (await c.post("/v1/tasks", json={"question": f"q{i}"}, headers=A)).json()["id"]
            await done(c, tid)
        assert (await c.post("/v1/tasks", json={"question": "more"}, headers=A)).status_code == 202


# ------------------------------------------------------------------ admin
@run
async def test_admin_summary_requires_admin_and_aggregates_everything():
    async with running(settings(price_in_per_m=1000.0, price_out_per_m=1000.0)) as (c, app):
        for h in (A, A, B):
            tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=h)).json()["id"]
            await done(c, tid, headers=h)
        assert (await c.get("/v1/admin/summary", headers=A)).status_code == 403
        assert (await c.get("/v1/admin/summary")).status_code == 401
        d = (await c.get("/v1/admin/summary", headers=ADMIN)).json()
        assert d["totals"]["tokens_in"] == 300 and d["totals"]["tool_errors"] == 3
        assert [u["user"] for u in d["users"]] == ["alice", "bob"] and d["users"][0]["tokens_in"] == 200
        bad = next(t for t in d["tools"] if t["tool"] == "bad_tool")
        assert bad["calls"] == 3 and bad["errors"] == 3 and bad["error_rate"] == 1.0
        assert next(t for t in d["tools"] if t["tool"] == "good_tool")["error_rate"] == 0.0
        assert sum(d["latency"].values()) == 3
        assert d["counters"]["submitted"] == 3 and d["totals"]["est_cost_usd"] is not None


@run
async def test_dashboard_page_and_prometheus_metrics():
    async with running() as (c, app):
        tid = (await c.post("/v1/tasks", json={"question": "q"}, headers=A)).json()["id"]
        await done(c, tid)
        page = await c.get("/dashboard")
        assert page.status_code == 200 and "Usage dashboard" in page.text and "/v1/admin/summary" in page.text
        m = (await c.get("/metrics")).text
        assert 'odr_tool_errors_total{tool="bad_tool"} 1' in m and "odr_tokens_in_today 100" in m
        assert 'odr_task_latency_bucket{le="5"} 1' in m
