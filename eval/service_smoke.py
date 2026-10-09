"""End-to-end smoke test: the REAL agent behind the HTTP API (costs LLM tokens).

    python -m eval.service_smoke

1. alice asks an aggregate question          -> runs to completion, no approval needed
2. alice asks for row-level order detail     -> task pauses in `awaiting_approval`
3. carol (approver) approves via the API     -> task resumes from its checkpoint and finishes
Also shows that bob cannot see alice's tasks and that a duplicate Idempotency-Key does not rerun anything.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

from open_deep_research.service.app import create_app
from open_deep_research.service.settings import Settings

A, B, C = ({"Authorization": f"Bearer {k}"} for k in ("ka", "kb", "kc"))
Q1 = "sales 表总共有多少条记录？不用问我连接信息，直接用工具查。"
Q2 = "请列出 East 大区的所有订单明细，包含 id、date、product_line、revenue。不用问我连接信息，直接用工具查。"


async def wait(c, tid, states, headers=A, timeout=420):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        t = (await c.get(f"/v1/tasks/{tid}", headers=headers)).json()
        if t["state"] in states:
            return t, time.monotonic() - t0
        await asyncio.sleep(1)
    raise TimeoutError(f"{tid} stuck in {t['state']}")


async def main():
    load_dotenv()
    assert os.getenv("INTERNAL_DB_URL"), "set INTERNAL_DB_URL"
    work = Path(tempfile.mkdtemp(prefix="svc_smoke_"))
    settings = Settings(
        redis_url=os.environ.get("ODR_REDIS_URL", "redis://localhost:6390/0"), prefix=f"odrsmoke:{uuid.uuid4().hex[:6]}:",
        api_keys={"ka": "alice", "kb": "bob", "kc": "carol"}, approvers=frozenset({"carol"}),
        workers=2, checkpoint_path=str(work / "ckpt.sqlite"), task_timeout_s=900, lease_ttl_s=60, lease_refresh_s=15,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc", timeout=60) as c:
            t0 = time.monotonic()
            r = await c.post("/v1/tasks", json={"question": Q1}, headers={**A, "Idempotency-Key": "smoke-1"})
            t1 = r.json()["id"]
            dup = await c.post("/v1/tasks", json={"question": Q1}, headers={**A, "Idempotency-Key": "smoke-1"})
            print(f"submit -> {r.status_code}; duplicate with same key -> {dup.status_code} same id={dup.json()['id'] == t1}")
            task, took = await wait(c, t1, {"succeeded", "failed"})
            print(f"[1] aggregate question: {task['state']} in {took:.0f}s  attempts={task['attempts']}")
            print("    answer:", str(task.get("result", {}).get("report", task.get("error")))[:200].replace("\n", " "))

            tid = (await c.post("/v1/tasks", json={"question": Q2}, headers=A)).json()["id"]
            task, took = await wait(c, tid, {"awaiting_approval", "succeeded", "failed"})
            print(f"[2] row-level question: {task['state']} after {took:.0f}s")
            if task["state"] == "awaiting_approval":
                print("    pending request:", task["awaiting"]["requests"][0]["reason"][:110])
                print("    bob sees alice's task? ->", (await c.get(f"/v1/tasks/{tid}", headers=B)).status_code)
                print("    alice approves herself ->", (await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True}, headers=A)).status_code)
                ok = await c.post(f"/v1/tasks/{tid}/approval", json={"approved": True, "note": "quarterly audit"}, headers=C)
                print("    carol approves ->", ok.status_code)
                task, took = await wait(c, tid, {"succeeded", "failed"})
                print(f"[3] after approval: {task['state']} in {took:.0f}s  attempts={task['attempts']}")
                print("    answer:", str(task.get("result", {}).get("report", task.get("error")))[:200].replace("\n", " "))

            ev = await c.get(f"/v1/tasks/{tid}/events", headers=A)
            print("    event types:", [l[7:] for l in ev.text.splitlines() if l.startswith("event: ")])
            print("\n/metrics:\n" + (await c.get("/metrics")).text)
            print(f"total {time.monotonic() - t0:.0f}s")

        r = app.state.redis
        keys = [k async for k in r.scan_iter(match=f"{settings.prefix}*")]
        if keys:
            await r.delete(*keys)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
