"""Redis-backed task store: state machine, idempotent submit, reliable queue, leases, events.

Everything that must be atomic is a Lua script:
  submit      idempotency check + active-task cap + create + index + enqueue, in one step
  transition  compare-and-set on the task state (a late "succeeded" cannot overwrite "cancelled")
Reliable queue: workers BLMOVE a task id from ``queue`` to ``processing`` and hold a short
lease key while working. If a worker dies, the lease expires and the reaper moves the id
back to ``queue`` - the next worker resumes from the LangGraph checkpoint, not from scratch.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

import redis.asyncio as aioredis

QUEUED, RUNNING, AWAITING, SUCCEEDED, FAILED, CANCELLED = (
    "queued", "running", "awaiting_approval", "succeeded", "failed", "cancelled")
TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})
ACTIVE = frozenset({QUEUED, RUNNING, AWAITING})

# who may move a task into which state
ALLOWED_FROM = {
    RUNNING: {QUEUED, AWAITING},
    AWAITING: {RUNNING},
    SUCCEEDED: {RUNNING},
    FAILED: {QUEUED, RUNNING, AWAITING},
    CANCELLED: {QUEUED, RUNNING, AWAITING},
    QUEUED: {RUNNING, AWAITING},          # requeue after a lost lease
}

_SUBMIT = """
-- KEYS: 1 idem  2 task  3 active-set  4 queue  5 user-index
-- ARGV: 1 idem_value  2 task_id  3 owner  4 question  5 now  6 max_active  7 idem_ttl  8 task_ttl
--       9 queue_max  10 body_hash
if ARGV[1] ~= '' then
  local existing = redis.call('GET', KEYS[1])
  if existing then return {'EXISTING', existing} end
end
if redis.call('LLEN', KEYS[4]) >= tonumber(ARGV[9]) then return {'BUSY', ''} end
if redis.call('SCARD', KEYS[3]) >= tonumber(ARGV[6]) then return {'LIMIT', ''} end
redis.call('HSET', KEYS[2], 'id', ARGV[2], 'owner', ARGV[3], 'state', 'queued', 'question', ARGV[4],
           'created_at', ARGV[5], 'updated_at', ARGV[5], 'attempts', 0, 'body_hash', ARGV[10])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[8]))
redis.call('SADD', KEYS[3], ARGV[2])
redis.call('ZADD', KEYS[5], ARGV[5], ARGV[2])
redis.call('LPUSH', KEYS[4], ARGV[2])
if ARGV[1] ~= '' then redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[7])) end
return {'CREATED', ARGV[2]}
"""

_TRANSITION = """
-- KEYS: 1 task  2 active-set
-- ARGV: 1 to_state  2 now  3 task_id  4 terminal(0/1)  5 n_allowed  6..5+n allowed-from states  then field,value pairs
-- The state change and its payload fields (result / error / awaiting ...) are ONE atomic step, so a
-- reader can never observe "succeeded" without its result or "awaiting_approval" without its request.
local cur = redis.call('HGET', KEYS[1], 'state')
if not cur then return 0 end
local n = tonumber(ARGV[5])
local ok = false
for i = 6, 5 + n do if ARGV[i] == cur then ok = true end end
if not ok then return 0 end
redis.call('HSET', KEYS[1], 'state', ARGV[1], 'updated_at', ARGV[2])
for i = 6 + n, #ARGV, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1]) end
if ARGV[4] == '1' then redis.call('SREM', KEYS[2], ARGV[3]) else redis.call('SADD', KEYS[2], ARGV[3]) end
return 1
"""


class SubmitResult:
    def __init__(self, status: str, task_id: str = "", body_hash: str = ""):
        self.status, self.task_id, self.body_hash = status, task_id, body_hash  # CREATED|EXISTING|LIMIT|BUSY


class TaskStore:
    def __init__(self, redis: aioredis.Redis, prefix: str = "odr:", task_ttl_s: int = 7 * 86400,
                 blocking_redis: Optional[aioredis.Redis] = None):
        # blocking commands park a connection for up to a second; keeping them on their own pool
        # means a burst of SSE clients or idle workers can never starve normal requests
        self.r, self.p, self.task_ttl_s = redis, prefix, task_ttl_s
        self.rb = blocking_redis or redis
        self._submit = redis.register_script(_SUBMIT)
        self._transition = redis.register_script(_TRANSITION)

    # ---- keys
    def _task(self, tid: str) -> str: return f"{self.p}task:{tid}"
    def _active(self, owner: str) -> str: return f"{self.p}active:{owner}"
    def _events(self, tid: str) -> str: return f"{self.p}events:{tid}"
    def _lease(self, tid: str) -> str: return f"{self.p}lease:{tid}"
    @property
    def queue_key(self) -> str: return f"{self.p}queue"
    @property
    def processing_key(self) -> str: return f"{self.p}processing"

    # ---- submit
    async def submit(self, owner: str, question: str, body_hash: str, idem_key: Optional[str],
                     max_active: int, idem_ttl_s: int, queue_max: int) -> SubmitResult:
        tid = uuid.uuid4().hex
        idem_val = json.dumps({"task_id": tid, "body_hash": body_hash}) if idem_key else ""
        status, payload = await self._submit(
            keys=[f"{self.p}idem:{owner}:{idem_key or '-'}", self._task(tid), self._active(owner),
                  self.queue_key, f"{self.p}user_tasks:{owner}"],
            args=[idem_val, tid, owner, question, time.time(), max_active, idem_ttl_s, self.task_ttl_s, queue_max, body_hash],
        )
        status = status.decode() if isinstance(status, bytes) else status
        payload = payload.decode() if isinstance(payload, bytes) else payload
        if status == "EXISTING":
            info = json.loads(payload)
            return SubmitResult("EXISTING", info["task_id"], info["body_hash"])
        return SubmitResult(status, payload)

    async def find_idempotent(self, owner: str, idem_key: str) -> Optional[dict]:
        raw = await self.r.get(f"{self.p}idem:{owner}:{idem_key}")
        return json.loads(raw) if raw else None

    # ---- reads
    async def get(self, tid: str) -> Optional[dict[str, Any]]:
        h = await self.r.hgetall(self._task(tid))
        if not h:
            return None
        d = {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in h.items()}
        for k in ("result", "awaiting", "resume"):
            if d.get(k):
                d[k] = json.loads(d[k])
        d["attempts"] = int(d.get("attempts", 0))
        usage = {k[2:]: float(d.pop(k)) if k == "u_seconds" else int(d.pop(k)) for k in [k for k in d if k.startswith("u_")]}
        if usage:
            d["usage"] = usage
        return d

    async def list_for_user(self, owner: str, limit: int = 20) -> list[str]:
        ids = await self.r.zrevrange(f"{self.p}user_tasks:{owner}", 0, limit - 1)
        return [i.decode() if isinstance(i, bytes) else i for i in ids]

    async def active_count(self, owner: str) -> int:
        return await self.r.scard(self._active(owner))

    # ---- state machine
    async def transition(self, tid: str, to: str, only_from: Optional[set] = None, **fields: Any) -> bool:
        """Atomic compare-and-set. Returns False if the task is not in an allowed source state."""
        task = self._task(tid)
        owner = (await self.r.hget(task, "owner")) or b""
        owner = owner.decode() if isinstance(owner, bytes) else owner
        allowed = sorted(only_from or ALLOWED_FROM[to])
        payload: list[str] = []
        for k, v in fields.items():
            if v is not None:
                payload += [k, json.dumps(v) if isinstance(v, (dict, list)) else str(v)]
        ok = await self._transition(
            keys=[task, self._active(owner)],
            args=[to, time.time(), tid, "1" if to in TERMINAL else "0", len(allowed), *allowed, *payload],
        )
        return bool(ok)

    async def set_field(self, tid: str, **fields: Any) -> None:
        enc = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in fields.items()}
        await self.r.hset(self._task(tid), mapping=enc)

    async def request_cancel(self, tid: str) -> str:
        """Queued/paused tasks are cancelled at once. A RUNNING task is only flagged: its worker must
        stop the graph (and free itself) before the state becomes CANCELLED, otherwise the API would
        report "cancelled" while a worker keeps burning tokens on it."""
        if await self.transition(tid, CANCELLED, only_from={QUEUED, AWAITING}):
            await self.add_event(tid, "cancelled", {})
            return "cancelled"
        await self.r.hset(self._task(tid), "cancel_requested", "1")
        return "requested"

    async def cancel_requested(self, tid: str) -> bool:
        return (await self.r.hget(self._task(tid), "cancel_requested")) in (b"1", "1")

    # ---- events (SSE)
    async def add_event(self, tid: str, type_: str, data: dict) -> str:
        async with self.r.pipeline(transaction=False) as pipe:          # one round trip instead of two
            pipe.xadd(self._events(tid), {"type": type_, "data": json.dumps(data, ensure_ascii=False)},
                      maxlen=1000, approximate=True)
            pipe.expire(self._events(tid), self.task_ttl_s)
            eid, _ = await pipe.execute()
        return eid.decode() if isinstance(eid, bytes) else eid

    async def record_submitted(self, tid: str, metrics_key: str) -> None:
        """Post-submit bookkeeping (metric + first event) in a single round trip."""
        async with self.r.pipeline(transaction=False) as pipe:
            pipe.hincrby(metrics_key, "submitted", 1)
            pipe.xadd(self._events(tid), {"type": "queued", "data": "{}"}, maxlen=1000, approximate=True)
            pipe.expire(self._events(tid), self.task_ttl_s)
            await pipe.execute()

    async def read_events(self, tid: str, last_id: str = "0", block_ms: int = 1000) -> list[tuple[str, str, dict]]:
        res = await self.rb.xread({self._events(tid): last_id}, block=block_ms, count=100)
        out = []
        for _, entries in res or []:
            for eid, fields in entries:
                eid = eid.decode() if isinstance(eid, bytes) else eid
                f = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                     for k, v in fields.items()}
                out.append((eid, f["type"], json.loads(f["data"])))
        return out

    # ---- reliable queue + leases
    async def claim(self, timeout_s: int = 1) -> Optional[str]:
        tid = await self.rb.blmove(self.queue_key, self.processing_key, timeout_s, "RIGHT", "LEFT")
        return tid.decode() if isinstance(tid, bytes) else tid

    async def lease(self, tid: str, ttl_s: int) -> None:
        await self.r.set(self._lease(tid), "1", ex=ttl_s)

    async def release(self, tid: str) -> None:
        await self.r.delete(self._lease(tid))
        await self.r.lrem(self.processing_key, 0, tid)

    async def reap_expired(self, max_attempts: int) -> dict[str, list[str]]:
        """Requeue (or fail) tasks whose worker stopped renewing its lease."""
        requeued, failed = [], []
        for raw in await self.r.lrange(self.processing_key, 0, -1):
            tid = raw.decode() if isinstance(raw, bytes) else raw
            if await self.r.exists(self._lease(tid)):
                continue
            task = await self.get(tid)
            await self.r.lrem(self.processing_key, 0, tid)
            if not task or task["state"] in TERMINAL:
                continue
            if task["attempts"] >= max_attempts:
                if await self.transition(tid, FAILED, error=f"worker lost {task['attempts']} times"):
                    await self.add_event(tid, "failed", {"error": "worker lost too many times"})
                failed.append(tid)
                continue
            if task["state"] != QUEUED:
                await self.transition(tid, QUEUED)
            await self.r.lpush(self.queue_key, tid)
            await self.add_event(tid, "requeued", {"reason": "worker lease expired"})
            requeued.append(tid)
        return {"requeued": requeued, "failed": failed}

    # ---- usage, tool stats, latency (observability / cost control)
    def _usage_key(self, owner: str, day: str) -> str:
        return f"{self.p}usage:{owner}:{day}"

    async def record_usage(self, tid: str, owner: str, usage: dict, day: str) -> None:
        """Add one run's usage to the task, the user's daily totals and the global tool statistics."""
        fields = {"tokens_in": usage["tokens_in"], "tokens_out": usage["tokens_out"], "llm_calls": usage["llm_calls"],
                  "tool_calls": usage["tool_calls"], "tool_errors": usage["tool_errors"]}
        async with self.r.pipeline(transaction=False) as pipe:
            for k, v in fields.items():
                pipe.hincrby(self._task(tid), f"u_{k}", v)
                pipe.hincrby(self._usage_key(owner, day), k, v)
                pipe.hincrby(f"{self.p}usage:all:{day}", k, v)
            pipe.hincrbyfloat(self._task(tid), "u_seconds", usage["seconds"])
            pipe.hincrby(self._usage_key(owner, day), "runs", 1)
            pipe.expire(self._usage_key(owner, day), 45 * 86400)
            pipe.expire(f"{self.p}usage:all:{day}", 45 * 86400)
            pipe.sadd(f"{self.p}usage_users:{day}", owner)
            pipe.expire(f"{self.p}usage_users:{day}", 45 * 86400)
            for tool, st in usage["by_tool"].items():
                pipe.hincrby(f"{self.p}toolstats", f"{tool}|calls", st["calls"])
                pipe.hincrby(f"{self.p}toolstats", f"{tool}|errors", st["errors"])
            await pipe.execute()

    async def usage_day(self, owner: str, day: str) -> dict:
        raw = await self.r.hgetall(self._usage_key(owner, day))
        return {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in raw.items()}

    async def users_with_usage(self, day: str) -> list[str]:
        return sorted((m.decode() if isinstance(m, bytes) else m) for m in await self.r.smembers(f"{self.p}usage_users:{day}"))

    async def global_usage(self, day: str) -> dict:
        raw = await self.r.hgetall(f"{self.p}usage:all:{day}")
        return {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in raw.items()}

    async def tool_stats(self) -> dict:
        raw = await self.r.hgetall(f"{self.p}toolstats")
        out: dict = {}
        for k, v in raw.items():
            k = k.decode() if isinstance(k, bytes) else k
            tool, _, kind = k.partition("|")
            out.setdefault(tool, {"calls": 0, "errors": 0})[kind] = int(v)
        return out

    LAT_BUCKETS = (5, 15, 30, 60, 120, 300, 900)

    async def observe_latency(self, seconds: float) -> None:
        bucket = next((f"le_{b}" for b in self.LAT_BUCKETS if seconds <= b), "le_inf")
        await self.r.hincrby(f"{self.p}latency", bucket, 1)

    async def latency_hist(self) -> dict:
        raw = await self.r.hgetall(f"{self.p}latency")
        got = {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in raw.items()}
        return {f"le_{b}": got.get(f"le_{b}", 0) for b in self.LAT_BUCKETS} | {"le_inf": got.get("le_inf", 0)}

    async def bump_attempts(self, tid: str) -> int:
        return int(await self.r.hincrby(self._task(tid), "attempts", 1))
