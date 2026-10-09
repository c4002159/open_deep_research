"""End-to-end human-approval demo against the REAL agent (costs LLM tokens).

    python -m eval.hitl_demo --decision approve
    python -m eval.hitl_demo --decision reject

Run 1 asks for row-level order data -> the graph pauses and prints what it wants to run.
The checkpoint is stored in a SQLite file; run 2 opens a *new* connection and resumes the
same thread with the reviewer's decision, which is what a web UI / ticket system would do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

QUESTION = "请列出 East 大区的所有订单明细，包含 id、date、product_line、revenue。不用问我连接信息，直接用工具查。"


async def main(decision: str) -> None:
    load_dotenv()
    from open_deep_research.deep_researcher import deep_researcher_builder

    db = str(Path(tempfile.mkdtemp(prefix="hitl_")) / "checkpoints.sqlite")
    log = Path(tempfile.mkdtemp(prefix="hitl_log_")) / "approvals.jsonl"
    cfg = {"configurable": {"thread_id": "hitl-demo", "allow_clarification": False,
                            "approval_mode": "interrupt", "approval_log_path": str(log)}}

    t0 = time.perf_counter()
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = deep_researcher_builder.compile(checkpointer=saver)
        out = await graph.ainvoke({"messages": [{"role": "user", "content": QUESTION}]}, cfg)
    paused = out.get("__interrupt__")
    print(f"[{time.perf_counter() - t0:5.1f}s] run 1 finished; paused for approval: {bool(paused)}")
    if not paused:
        print("The agent never asked for approval. Final:", str(out.get("final_report"))[:300])
        return
    print(json.dumps(paused[0].value, ensure_ascii=False, indent=2)[:900])

    reviewer = {"approved": decision == "approve", "reviewer": "demo-reviewer", "note": f"demo {decision}"}
    async with AsyncSqliteSaver.from_conn_string(db) as saver:  # fresh connection == "after a restart"
        graph = deep_researcher_builder.compile(checkpointer=saver)
        out = await graph.ainvoke(Command(resume=reviewer), cfg)
    print(f"\n[{time.perf_counter() - t0:5.1f}s] run 2 finished (resumed with decision={decision})")
    print("still paused?", bool(out.get("__interrupt__")))
    print("FINAL REPORT (first 700 chars):\n", str(out.get("final_report"))[:700])
    print("\nAUDIT LOG:\n", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", choices=["approve", "reject"], default="approve")
    a = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    assert os.getenv("INTERNAL_DB_URL") or load_dotenv() or True
    asyncio.run(main(a.decision))
