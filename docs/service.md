# HTTP service

The agent behind an asynchronous task API: submit a question, get a task id, poll or stream
progress, approve confidential-data requests, cancel.

```bash
docker run -d --name odr-redis -p 6390:6379 redis:7          # or point ODR_REDIS_URL at your Redis
export ODR_API_KEYS="key-alice:alice,key-bob:bob,key-carol:carol"
export ODR_APPROVERS="carol"                                  # who may approve row-level order data
export INTERNAL_DB_URL="sqlite:///.../test.db"
python -m open_deep_research.service                          # http://127.0.0.1:8080
```

```bash
curl -X POST localhost:8080/v1/tasks -H "Authorization: Bearer key-alice" \
     -H "Idempotency-Key: report-2026-06" -H "Content-Type: application/json" \
     -d '{"question": "sales 表总共有多少条记录？"}'              # 202 {"id": "...", "state": "queued"}
curl localhost:8080/v1/tasks/<id>          -H "Authorization: Bearer key-alice"
curl -N localhost:8080/v1/tasks/<id>/events -H "Authorization: Bearer key-alice"   # SSE
```

## API

| Method & path | Notes |
|---|---|
| `POST /v1/tasks` | `202` new task · `200` + `Idempotent-Replay: true` for a repeated `Idempotency-Key` · `409` same key, different body · `429` rate limit or too many active tasks (`Retry-After`) · `503` queue full |
| `GET /v1/tasks/{id}` | `queued → running → (awaiting_approval →) succeeded / failed / cancelled`. Someone else's task is `404`, not `403` |
| `GET /v1/tasks` | my recent tasks |
| `GET /v1/tasks/{id}/events` | Server-Sent Events: `queued, progress, approval_required, approval_decided, resumed, requeued, continued, succeeded/failed/cancelled, end`. Resume with `Last-Event-ID` |
| `POST /v1/tasks/{id}/cancel` | queued/paused tasks cancel at once; a running task is stopped by its worker |
| `POST /v1/tasks/{id}/approval` | `{"approved": true, "note": "..."}` — approvers only (`ODR_APPROVERS`). The reviewer recorded is the token's user, never a body field |
| `GET /v1/tasks/{id}/events` (live) | besides the events above, `tool_start`, `tool_end` and `usage` stream while a task runs (tool arguments, success, seconds; running token totals). Query *results* are never streamed; only `propose_refund`/`withdraw_refund` results are, because they contain no business data |
| `GET /v1/me` | who am I: role, admin flag, today's tokens, budget remaining |
| `GET /v1/approvals` | tasks paused for confidential-data approval, with the SQL to be approved (approvers only) |
| `GET /v1/refunds?view=mine\|inbox\|all`, `GET /v1/refunds/{id}` | refund proposals: my own / waiting for my role to sign / everything (role holders and admins). Someone else's refund is `404` |
| `POST /v1/refunds/{id}/decision` | `{"approve": bool, "note": "..."}`. Actor **and role** come from the token. `403` own request (separation of duties) or wrong role · `409` already decided / already signed · `422` rejecting without a reason |
| `POST /v1/refunds/{id}/reverse` | finance or director, within 24 h of the refund taking effect, reason required |
| `GET /v1/audit?limit=` | refund events + data-access decisions + triage refusals, newest first, in UTC (role holders and admins) |
| `GET /` | the web client (`static/`); `GET /demo/users` exists **only** with `ODR_DEMO_MODE=1` |
| `GET /healthz`, `GET /metrics` | Redis ping + worker count; plain-text counters |

The refund endpoints exist when `ODR_BUSINESS_DB_URL` (or, with `ENABLE_ACTIONS=1`, `INTERNAL_DB_URL`) is a SQLite URL.
Roles are `ODR_ROLES="bob=manager,carol=finance,dave=director"`. The design and its trade-offs are in
[nl_operations_design.md](nl_operations_design.md); the 3-minute walk-through is [demo_script.md](demo_script.md).

## How the pieces work

* **Idempotency** — `submit` is one Lua script: check key → check caps → create task → enqueue. 300 concurrent
  requests with one key create exactly one task (measured). Replays do not consume rate-limit quota.
* **Rate limit** — per-user sliding-window *log* in a Redis ZSET (no 2× burst at window edges), checked and
  updated atomically in Lua using Redis server time. Separate cap on simultaneously active tasks per user.
* **Queue + leases** — workers `BLMOVE` task ids into a `processing` list and renew a short lease key while
  working. A dead worker's lease expires; the reaper (one elected instance) requeues the task; the next worker
  continues **from the LangGraph checkpoint** (`aget_state`), so finished nodes are not redone. Delivery is
  at-least-once: a node that was mid-flight at the crash runs again — side-effecting tools need idempotency keys.
* **Approval** — the graph pauses with `interrupt()`; the task becomes `awaiting_approval` and the worker is free.
  Approving writes the decision, re-queues the task, and the graph resumes with `Command(resume=...)`.
  `approval_mode` is fixed to `interrupt` server-side, so callers cannot waive it.
* **Isolation** — identity comes only from the bearer token (API key or HS256 JWT). It namespaces the LangGraph
  thread id (`<user>:<task>`), the long-term memory user (`mem0_user_id`, which now beats a process-wide
  `MEM0_USER_ID`), idempotency keys and task ownership checks.
* **Schema cache** — introspection queries (`sqlite_master`, `information_schema`, `pragma_table_info`) are cached
  for `schema_cache_ttl_s` (default 300 s; in-process, or shared via `ODR_CACHE_REDIS_URL`). Business-data queries
  are never cached.
* **Redis pools** — request traffic and blocking calls (`BLMOVE`, SSE `XREAD`) use separate, explicitly sized pools.
  A burst test with one shared default pool returned HTTP 500 (`MaxConnectionsError`); Redis errors now map to `503`.

## Measured (local, one uvicorn process, Redis in Docker, stand-in graph of 3×20 ms nodes, `python -m eval.service_loadtest`)

| Scenario | Result |
|---|---|
| 600 tasks, 40 users, 64 concurrent clients | 228 submits/s · submit p50 179 ms / p95 792 ms · all 600 succeeded in 6.2 s |
| 100 concurrent requests, 10/min limit, one user | exactly 10 accepted, 90 × `429` with `Retry-After`; other user unaffected |
| 300 concurrent requests, same `Idempotency-Key` | 1 × `202`, 299 × `200`, 1 task, 1 agent execution |
| every worker killed mid-load, replacements started | 100/100 finished, 0 lost; 1 node-a and 4 node-c re-executions (at-least-once) |

These measure the service layer only; real latency is dominated by the LLM (tens of seconds per task).

Usage accounting, daily token budgets, the admin dashboard and hybrid retrieval are described in
[observability_and_rag.md](observability_and_rag.md).

## Limits & production notes

* **Code sandbox needs Docker.** `python_executor` runs in a container (no network, 256 MB, non-root) and
  *refuses to execute* if no Docker daemon is reachable - the in-process restrictions are not a security
  boundary. Build the image with `docker build -t agent-sandbox-python:latest -f docker/sandbox.Dockerfile .`.
  For local development only, `CODE_SANDBOX_BACKEND=auto` falls back to a weaker subprocess sandbox.
* Point `INTERNAL_DB_URL` at a **read-only database account** as well; the SQL guard and read-only
  connection are defence in depth, a DB-level grant is the real boundary.
* Checkpoints use a SQLite file (`ODR_CHECKPOINT_PATH`): fine for one host. Several hosts need the Postgres
  checkpointer, otherwise a task re-delivered to another machine has no checkpoint and restarts.
* Task state lives in Redis with a TTL (7 days). It is a work queue, not a system of record.
* Tests that need Redis use `redis://localhost:6390/0` (override with `ODR_TEST_REDIS_URL`) and skip if absent.
* Redis keys are built in Python and passed to Lua scripts; this assumes a single Redis node (not Cluster).
