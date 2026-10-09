import asyncio

from open_deep_research.service import store as S
from open_deep_research.service.ratelimit import SlidingWindowLimiter
from open_deep_research.service.store import TaskStore
from tests.redis_fixtures import with_redis


def mk(r, prefix):
    return TaskStore(r, prefix)


async def submit(st, owner="alice", q="q", body="h", idem=None, max_active=5):
    return await st.submit(owner, q, body, idem, max_active=max_active, idem_ttl_s=60, queue_max=100)


# ------------------------------------------------------------------ rate limiter
@with_redis
async def test_rate_limit_blocks_after_limit_and_reports_retry_after(r, p):
    rl = SlidingWindowLimiter(r, p, limit=3, window_s=60)
    results = [await rl.hit("alice") for _ in range(4)]
    assert [x.allowed for x in results] == [True, True, True, False]
    assert results[2].remaining == 0 and 0 < results[3].retry_after_s <= 60


@with_redis
async def test_rate_limit_is_per_user(r, p):
    rl = SlidingWindowLimiter(r, p, limit=1, window_s=60)
    assert (await rl.hit("alice")).allowed and not (await rl.hit("alice")).allowed
    assert (await rl.hit("bob")).allowed


@with_redis
async def test_rate_limit_window_slides(r, p):
    rl = SlidingWindowLimiter(r, p, limit=2, window_s=1)
    assert (await rl.hit("a")).allowed and (await rl.hit("a")).allowed and not (await rl.hit("a")).allowed
    await asyncio.sleep(1.15)
    assert (await rl.hit("a")).allowed


@with_redis
async def test_rate_limit_is_atomic_under_concurrency(r, p):
    rl = SlidingWindowLimiter(r, p, limit=5, window_s=60)
    res = await asyncio.gather(*(rl.hit("alice") for _ in range(50)))
    assert sum(x.allowed for x in res) == 5, "check-then-add must not let concurrent requests over the limit"


# ------------------------------------------------------------------ submit / idempotency
@with_redis
async def test_idempotent_submit_returns_same_task_and_enqueues_once(r, p):
    st = mk(r, p)
    a = await submit(st, idem="k1")
    b = await submit(st, idem="k1")
    assert a.status == "CREATED" and b.status == "EXISTING" and a.task_id == b.task_id
    assert await r.llen(st.queue_key) == 1


@with_redis
async def test_idempotency_keys_are_scoped_per_user(r, p):
    st = mk(r, p)
    a = await submit(st, owner="alice", idem="same")
    b = await submit(st, owner="bob", idem="same")
    assert a.status == b.status == "CREATED" and a.task_id != b.task_id


@with_redis
async def test_concurrent_same_key_creates_exactly_one_task(r, p):
    st = mk(r, p)
    res = await asyncio.gather(*(submit(st, idem="race", max_active=100) for _ in range(30)))
    assert sum(x.status == "CREATED" for x in res) == 1
    assert len({x.task_id for x in res}) == 1
    assert await r.llen(st.queue_key) == 1


@with_redis
async def test_existing_reports_original_body_hash_for_conflict_detection(r, p):
    st = mk(r, p)
    await submit(st, idem="k", body="hash-A")
    again = await submit(st, idem="k", body="hash-B")
    assert again.status == "EXISTING" and again.body_hash == "hash-A"


# ------------------------------------------------------------------ concurrency cap + state machine
@with_redis
async def test_active_cap_and_slot_release(r, p):
    st = mk(r, p)
    a = await submit(st, max_active=2)
    await submit(st, max_active=2)
    assert (await submit(st, max_active=2)).status == "LIMIT"
    assert await st.request_cancel(a.task_id) == "cancelled"        # terminal -> frees a slot
    assert (await submit(st, max_active=2)).status == "CREATED"


@with_redis
async def test_queue_depth_backpressure(r, p):
    st = mk(r, p)
    await st.submit("a", "q", "h", None, max_active=10, idem_ttl_s=60, queue_max=1)
    res = await st.submit("b", "q", "h", None, max_active=10, idem_ttl_s=60, queue_max=1)
    assert res.status == "BUSY"


@with_redis
async def test_cas_prevents_late_success_overwriting_cancel(r, p):
    st = mk(r, p)
    t = (await submit(st)).task_id
    assert await st.transition(t, S.RUNNING)
    assert await st.request_cancel(t) == "requested"          # running: flagged, the worker finishes the job
    assert await st.cancel_requested(t) and (await st.get(t))["state"] == "running"
    assert await st.transition(t, S.CANCELLED)                # what the worker does once it has stopped the graph
    assert not await st.transition(t, S.SUCCEEDED, result={"x": 1}), "terminal states are final"
    assert (await st.get(t))["state"] == "cancelled"


@with_redis
async def test_invalid_transition_rejected(r, p):
    st = mk(r, p)
    t = (await submit(st)).task_id
    assert not await st.transition(t, S.SUCCEEDED)    # queued -> succeeded is not a legal jump


@with_redis
async def test_result_roundtrip(r, p):
    st = mk(r, p)
    t = (await submit(st, q="你好")).task_id
    await st.transition(t, S.RUNNING)
    await st.transition(t, S.SUCCEEDED, result={"report": "完成", "n": 3})
    got = await st.get(t)
    assert got["state"] == "succeeded" and got["result"] == {"report": "完成", "n": 3} and got["question"] == "你好"


# ------------------------------------------------------------------ queue, leases, reaper
@with_redis
async def test_claim_moves_to_processing_and_release_clears(r, p):
    st = mk(r, p)
    t = (await submit(st)).task_id
    assert await st.claim(1) == t
    assert await r.llen(st.queue_key) == 0 and await r.llen(st.processing_key) == 1
    await st.release(t)
    assert await r.llen(st.processing_key) == 0


@with_redis
async def test_reaper_requeues_task_with_expired_lease_but_not_live_one(r, p):
    st = mk(r, p)
    live = (await submit(st)).task_id
    dead = (await submit(st)).task_id
    for _ in range(2):
        await st.claim(1)
    for t in (live, dead):
        await st.transition(t, S.RUNNING)
    await st.lease(live, 30)                      # the dead worker never renewed its lease
    out = await st.reap_expired(max_attempts=3)
    assert out["requeued"] == [dead] and out["failed"] == []
    assert (await st.get(dead))["state"] == "queued" and (await st.get(live))["state"] == "running"
    assert await r.llen(st.queue_key) == 1


@with_redis
async def test_reaper_fails_task_after_max_attempts(r, p):
    st = mk(r, p)
    t = (await submit(st)).task_id
    await st.claim(1)
    await st.transition(t, S.RUNNING)
    for _ in range(3):
        await st.bump_attempts(t)
    out = await st.reap_expired(max_attempts=3)
    assert out["failed"] == [t] and (await st.get(t))["state"] == "failed"


# ------------------------------------------------------------------ events
@with_redis
async def test_events_are_ordered_and_resumable(r, p):
    st = mk(r, p)
    t = (await submit(st)).task_id
    ids = [await st.add_event(t, "node", {"n": i}) for i in range(3)]
    first = await st.read_events(t, "0", block_ms=100)
    assert [e[2]["n"] for e in first] == [0, 1, 2]
    rest = await st.read_events(t, ids[0], block_ms=100)     # resume after the first event
    assert [e[2]["n"] for e in rest] == [1, 2]
