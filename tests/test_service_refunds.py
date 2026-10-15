"""HTTP side of the refund workflow: roles come from the token, rules are enforced server-side.

Most tests need no Redis (the refund endpoints talk to SQLite only); the ones that do are marked.
"""

import asyncio
import contextlib
import json
import sqlite3
import time
import uuid
from datetime import date

import httpx
import pytest

from open_deep_research import refunds as RF
from open_deep_research.service.app import create_app
from open_deep_research.service.settings import Settings
from tests.redis_fixtures import needs_redis

TODAY = date(2026, 6, 10)
KEYS = {"ka": "alice", "kb": "bob", "kc": "carol", "kd": "dave", "ke": "eve", "kr": "root"}
ROLES = {"bob": "manager", "carol": "finance", "dave": "director", "root": "director"}


def H(key):
    return {"Authorization": f"Bearer {key}"}


A, B, C, D, E, ROOT = (H(k) for k in ("ka", "kb", "kc", "kd", "ke", "kr"))


class NoGraph:
    graph = None

    async def start(self):
        pass

    async def stop(self):
        pass


@pytest.fixture()
def biz(tmp_path):
    path = tmp_path / "biz.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sales (id INTEGER PRIMARY KEY, date TEXT, product_line TEXT, region TEXT, revenue REAL, units INTEGER);
        INSERT INTO sales VALUES (1, '2026-06-01', 'AI Tools',       'North',  3000.0, 10);
        INSERT INTO sales VALUES (2, '2026-06-05', 'Cloud Services', 'East',  20000.0, 50);
        INSERT INTO sales VALUES (3, '2026-06-08', 'Data Platform',  'West',  80000.0, 90);
    """)
    con.commit(); con.close()
    RF.ensure_schema(str(path))
    return str(path), "sqlite:///" + path.as_posix(), tmp_path


def propose(db, order, amount, user="alice"):
    out = RF.propose(db, order_id=order, amount=amount, reason="unhappy", outage_claim=False, requested_by=user, today=TODAY)
    assert out.ok, out.message
    return out.refund["id"]


def make_settings(biz, **kw) -> Settings:
    base = dict(api_keys=KEYS, roles=ROLES, admins=frozenset({"root"}), business_db_url=biz[1],
                approval_log_path=str(biz[2] / "approvals.jsonl"), rate_limit_per_window=100,
                prefix=f"odrtest:{uuid.uuid4().hex[:8]}:")
    base.update(kw)
    return Settings(**base)


@contextlib.asynccontextmanager
async def running(settings, provider=None, start_workers=False, redis_cleanup=False):
    app = create_app(settings, provider or NoGraph(), start_workers=start_workers)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            try:
                yield client, app
            finally:
                if redis_cleanup:                    # only tests that really used Redis (no 10 s connect timeout otherwise)
                    keys = [k async for k in app.state.redis.scan_iter(match=f"{settings.prefix}*")]
                    if keys:
                        await app.state.redis.delete(*keys)


def run(fn):
    def wrapper(biz):
        return asyncio.run(fn(biz))
    wrapper.__name__ = fn.__name__
    return wrapper


# ------------------------------------------------------------------ approvals
@run
async def test_a_manager_alone_approves_a_small_refund(biz):
    rid = propose(biz[0], 1, 1000)
    async with running(make_settings(biz)) as (c, _):
        r = await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True, "note": "ok"}, headers=B)
        assert r.status_code == 200 and r.json()["status"] == "applied" and r.json()["signed_roles"] == ["manager"]


@run
async def test_a_mid_size_refund_needs_two_people_and_each_sign_is_visible(biz):
    rid = propose(biz[0], 2, 20000)
    async with running(make_settings(biz)) as (c, _):
        first = (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=C)).json()
        assert first["status"] == "pending" and first["missing_roles"] == ["manager"]
        second = (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=B)).json()
        assert second["status"] == "applied"
        detail = (await c.get(f"/v1/refunds/{rid}", headers=A)).json()
        assert [(e["action"], e["actor"]) for e in detail["events"]] == [
            ("propose", "alice"), ("approve", "carol"), ("approve", "bob"), ("applied", "system")]


@run
async def test_authority_comes_from_the_token_not_the_request_body(biz):
    rid = propose(biz[0], 3, 80000)                                  # needs the director
    async with running(make_settings(biz)) as (c, _):
        forged = {"approve": True, "user": "dave", "role": "director", "reviewer": "dave", "requested_by": "x"}
        assert (await c.post(f"/v1/refunds/{rid}/decision", json=forged, headers=B)).status_code == 403
        assert (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=D)).json()["status"] == "applied"


@run
async def test_nobody_approves_their_own_request_and_unauthorised_users_are_refused(biz):
    rid = propose(biz[0], 1, 1000, user="bob")
    async with running(make_settings(biz)) as (c, _):
        own = await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=B)
        assert own.status_code == 403 and "separation of duties" in own.json()["detail"]
        assert (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=E)).status_code == 403   # no role
        assert (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=A)).status_code == 403
        assert (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True})).status_code == 401
        assert (await c.get(f"/v1/refunds/{rid}", headers=B)).json()["status"] == "pending"


@run
async def test_reject_needs_a_reason_and_closes_the_request(biz):
    rid = propose(biz[0], 1, 1000)
    async with running(make_settings(biz)) as (c, _):
        assert (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": False}, headers=B)).status_code == 422
        done = await c.post(f"/v1/refunds/{rid}/decision", json={"approve": False, "note": "not covered"}, headers=B)
        assert done.status_code == 200 and done.json()["status"] == "rejected"
        assert (await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=B)).status_code == 409


@run
async def test_unknown_refund_is_404(biz):
    async with running(make_settings(biz)) as (c, _):
        assert (await c.post("/v1/refunds/999/decision", json={"approve": True}, headers=B)).status_code == 404
        assert (await c.get("/v1/refunds/999", headers=B)).status_code == 404


@run
async def test_a_failing_metrics_store_never_fails_a_committed_decision(biz):
    class Broken:
        async def hincrby(self, *a, **k):
            raise ConnectionError("redis is down")

    rid = propose(biz[0], 1, 1000)
    async with running(make_settings(biz)) as (c, app):
        app.state.redis = Broken()
        r = await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=B)
        assert r.status_code == 200 and r.json()["status"] == "applied"


# ------------------------------------------------------------------ reverse
@run
async def test_reverse_is_for_finance_or_director_with_a_reason(biz):
    rid = propose(biz[0], 1, 1000)
    async with running(make_settings(biz)) as (c, _):
        assert (await c.post(f"/v1/refunds/{rid}/reverse", json={"note": "x"}, headers=C)).status_code == 409   # not applied yet
        await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True}, headers=B)
        assert (await c.post(f"/v1/refunds/{rid}/reverse", json={"note": "x"}, headers=B)).status_code == 403    # manager
        assert (await c.post(f"/v1/refunds/{rid}/reverse", json={"note": ""}, headers=C)).status_code == 422
        ok = await c.post(f"/v1/refunds/{rid}/reverse", json={"note": "customer kept the product"}, headers=C)
        assert ok.status_code == 200 and ok.json()["status"] == "reversed"
        assert (await c.post(f"/v1/refunds/{rid}/reverse", json={"note": "again"}, headers=C)).status_code == 409


# ------------------------------------------------------------------ visibility
@run
async def test_inbox_mine_all_and_detail_visibility(biz):
    small, mid = propose(biz[0], 1, 1000), propose(biz[0], 2, 20000)
    bobs = propose(biz[0], 3, 80000, user="bob")
    async with running(make_settings(biz)) as (c, _):
        ids = lambda r: sorted(x["id"] for x in r.json()["refunds"])
        assert ids(await c.get("/v1/refunds", headers=A)) == [small, mid]                          # mine
        assert ids(await c.get("/v1/refunds?view=inbox", headers=B)) == [small, mid]               # manager signs both
        assert ids(await c.get("/v1/refunds?view=inbox", headers=C)) == [mid]                      # finance: only the mid one
        assert ids(await c.get("/v1/refunds?view=inbox", headers=D)) == [bobs]                     # director: the large one
        assert ids(await c.get("/v1/refunds?view=inbox", headers=A)) == []                         # no role, empty inbox
        assert (await c.get("/v1/refunds?view=all", headers=A)).status_code == 403
        assert len((await c.get("/v1/refunds?view=all", headers=B)).json()["refunds"]) == 3
        assert (await c.get("/v1/refunds?view=bogus", headers=B)).status_code == 422
        await c.post(f"/v1/refunds/{mid}/decision", json={"approve": True}, headers=B)
        assert ids(await c.get("/v1/refunds?view=inbox", headers=B)) == [small]                    # already signed: gone
        # strangers cannot tell that someone else's refund exists
        assert (await c.get(f"/v1/refunds/{small}", headers=E)).status_code == 404
        assert (await c.get(f"/v1/refunds/{small}", headers=A)).status_code == 200


@run
async def test_the_inbox_never_offers_your_own_request(biz):
    mine = propose(biz[0], 1, 1000, user="bob")
    async with running(make_settings(biz)) as (c, _):
        assert (await c.get("/v1/refunds?view=inbox", headers=B)).json()["refunds"] == []
        assert [r["id"] for r in (await c.get("/v1/refunds", headers=B)).json()["refunds"]] == [mine]


@run
async def test_the_refund_endpoints_are_off_without_a_business_database(biz):
    async with running(make_settings(biz, business_db_url="")) as (c, _):
        assert (await c.get("/v1/refunds", headers=B)).status_code == 404
        assert (await c.post("/v1/refunds/1/decision", json={"approve": True}, headers=B)).status_code == 404


@run
async def test_a_non_sqlite_business_database_is_a_503_not_a_crash(biz):
    async with running(make_settings(biz, business_db_url="postgresql://u:p@h/db")) as (c, _):
        assert (await c.get("/v1/refunds", headers=B)).status_code == 503


# ------------------------------------------------------------------ audit
@run
async def test_audit_feed_merges_refund_events_and_data_access_decisions_in_time_order(biz):
    rid = propose(biz[0], 1, 1000)
    log = biz[2] / "approvals.jsonl"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() + 3600))   # later than the refund events
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"ts": stamp, "thread_id": "alice:t1", "tool": "sql_executor", "reason": "row-level", "query": "SELECT * FROM sales",
         "approved": False, "reviewer": "carol", "note": "no need"},
        {"ts": stamp, "event": "request_refused_by_triage", "reason": "data_mutation", "request": "drop table sales",
         "thread_id": "alice:t2"},
        {"ts": "garbage", "approved": True},
    ]) + "\n", encoding="utf-8")
    async with running(make_settings(biz)) as (c, _):
        await c.post(f"/v1/refunds/{rid}/decision", json={"approve": True, "note": "fine"}, headers=B)
        items = (await c.get("/v1/audit", headers=B)).json()["items"]
        kinds = [(i["kind"], i["action"]) for i in items]
        assert ("data_access", "denied") in kinds and ("refused", "refused") in kinds and ("refund", "applied") in kinds
        assert [i["at"] for i in items] == sorted((i["at"] for i in items), reverse=True)
        assert kinds[0][0] in ("data_access", "refused"), "log entries are an hour newer than the refund events"
        denied = next(i for i in items if i["action"] == "denied")
        assert denied["actor"] == "carol" and denied["user"] == "alice" and "SELECT" in denied["detail"]
        assert len((await c.get("/v1/audit?limit=2", headers=B)).json()["items"]) == 2
        assert (await c.get("/v1/audit", headers=A)).status_code == 403
        assert (await c.get("/v1/audit", headers=ROOT)).status_code == 200


# ------------------------------------------------------------------ demo users
@run
async def test_demo_users_are_exposed_only_in_demo_mode(biz):
    async with running(make_settings(biz)) as (c, _):
        assert (await c.get("/demo/users")).status_code == 404
    async with running(make_settings(biz, demo_mode=True)) as (c, _):
        users = {u["user"]: u for u in (await c.get("/demo/users")).json()["users"]}
        assert users["bob"] == {"user": "bob", "role": "manager", "key": "kb"} and users["alice"]["role"] is None


# ------------------------------------------------------------------ needs Redis
@needs_redis
def test_me_reports_role_budget_and_usage():
    async def go():
        from tests.test_service_api import make_settings as redis_settings

        s = redis_settings(api_keys=KEYS, roles=ROLES, daily_token_budget=1000, business_db_url="", admins=frozenset({"root"}))
        async with running(s, redis_cleanup=True) as (c, _):
            me = (await c.get("/v1/me", headers=B)).json()
            assert me["user"] == "bob" and me["role"] == "manager" and me["actions_enabled"] is False
            assert me["budget"] == {"daily_tokens": 1000, "used": 0, "remaining": 1000}
            assert (await c.get("/v1/me", headers=ROOT)).json()["is_admin"] is True
            assert (await c.get("/v1/me")).status_code == 401

    asyncio.run(go())


@needs_redis
def test_pending_data_approvals_are_listed_for_approvers_only():
    async def go():
        from tests.test_service_api import FakeProvider, make_settings as redis_settings, wait_state

        s = redis_settings(api_keys=KEYS, roles=ROLES, approvers=frozenset({"carol"}), business_db_url="")
        async with running(s, FakeProvider(), start_workers=True, redis_cleanup=True) as (c, _):
            tid = (await c.post("/v1/tasks", json={"question": "SENSITIVE rows"}, headers=A)).json()["id"]
            await wait_state(c, tid, {"awaiting_approval"})
            listed = (await c.get("/v1/approvals", headers=C)).json()["tasks"]
            assert [t["id"] for t in listed] == [tid] and listed[0]["owner"] == "alice"
            assert listed[0]["awaiting"]["type"] == "approval_required"
            assert (await c.get("/v1/approvals", headers=A)).json()["tasks"] == []        # not an approver
            await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=C)
            await wait_state(c, tid, {"succeeded"})
            assert (await c.get("/v1/approvals", headers=C)).json()["tasks"] == []        # decided: gone

    asyncio.run(go())


# ------------------------------------------------------------------ live events (what the UI shows while a task runs)
def test_tracker_collects_live_events_and_only_previews_action_tools():
    from uuid import uuid4

    from open_deep_research.telemetry import UsageTracker

    tr = UsageTracker()
    for name, out in (("sql_executor", "id | revenue\n80 | 39700.37"), ("propose_refund", "Refund proposal #1 created")):
        rid = uuid4()
        tr.on_tool_start({"name": name}, "SELECT * FROM sales WHERE id = 80 LIMIT 1", run_id=rid)
        tr.on_tool_end(out, run_id=rid)
    evs = list(tr.live_events)
    assert [(e["type"], e["name"]) for e in evs] == [("tool_start", "sql_executor"), ("tool_end", "sql_executor"),
                                                     ("tool_start", "propose_refund"), ("tool_end", "propose_refund")]
    assert "preview" not in evs[1], "query results are business data: never streamed"
    assert evs[3]["preview"].startswith("Refund proposal #1")


def test_runner_publishes_live_events_in_order_and_survives_a_failing_store():
    from open_deep_research.service.runner import TaskRunner
    from open_deep_research.telemetry import UsageTracker

    class Store:
        def __init__(self, fail=False):
            self.events, self.fail = [], fail

        async def add_event(self, task_id, type_, data):
            if self.fail:
                raise ConnectionError("redis down")
            self.events.append((task_id, type_, data))

    tr = UsageTracker()
    tr.live_events.extend([{"type": "tool_start", "name": "a", "args": "x"}, {"type": "usage", "tokens_in": 5, "tokens_out": 1, "llm_calls": 1}])
    ok = Store()
    asyncio.run(TaskRunner(ok, None, Settings())._drain("t1", tr))
    assert [e[1] for e in ok.events] == ["tool_start", "usage"] and ok.events[1][2]["tokens_in"] == 5
    tr.live_events.append({"type": "usage", "tokens_in": 1, "tokens_out": 1, "llm_calls": 1})
    asyncio.run(TaskRunner(Store(fail=True), None, Settings())._drain("t1", tr))      # must not raise
