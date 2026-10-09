"""Run the agent over the evaluation suite and record quality / cost / latency metrics.

    python -m eval.run_agent_eval --name baseline                 # all cases (costs LLM tokens!)
    python -m eval.run_agent_eval --name smoke --tags simple safety --limit 6
    python -m eval.run_agent_eval --name pipeline-check --dry-run # no LLM, validates the harness
    python -m eval.compare eval/baselines/baseline.json eval/results/smoke.json

Results go to eval/results/<name>.json (git-ignored); promote a run to a baseline by
copying it into eval/baselines/. Finished cases are cached in the result file, so an
interrupted run resumes (use --force to redo).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.cases import ALL_CASES
from eval.harness import TraceHandler, aggregate, build_result, db_fingerprint, run_metadata

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


async def _real_invoke(case: dict, config: dict) -> str:
    from open_deep_research.deep_researcher import deep_researcher

    result = await deep_researcher.ainvoke({"messages": [{"role": "user", "content": case["question"]}]}, config=config)
    if result.get("final_report"):
        return result["final_report"]
    msgs = result.get("messages") or []  # e.g. a triage refusal or a clarification question
    return getattr(msgs[-1], "content", None) or str(result) if msgs else str(result)


async def _fake_invoke(case: dict, config: dict) -> str:
    """Offline stand-in: answers correctly for ~2/3 of the cases so every code path is exercised."""
    handler = config["callbacks"][0]
    handler.llm_calls += 3
    handler.tokens_in += 1200
    handler.tokens_out += 300
    handler.tool_calls.append({"tool": "sql_executor", "ok": True, "seconds": 0.01})
    if sum(map(ord, case["id"])) % 3 == 0:
        return "无法回答"
    ch = case["checks"]
    parts = [str(n) for n in ch.get("numbers", [])] + ch.get("contains_all", []) + ch.get("contains_any", [])[:1]
    return " ".join(parts) or "ok"


MEMORY_USER = "eval-default"            # replaced per run in main_async (see --shared-memory)

PROVIDER_FATAL = re.compile(r"error code: 40[12]|insufficient[ _]balance|insufficient_quota|invalid_api_key|incorrect api key|authentication", re.I)
MAX_CONSECUTIVE_PROVIDER_FAILURES = 3


class Abort:
    """Shared across concurrent cases: set once the model provider is evidently unusable."""

    def __init__(self) -> None:
        self.consecutive = 0
        self.reason = ""

    @property
    def tripped(self) -> bool:
        return self.consecutive >= MAX_CONSECUTIVE_PROVIDER_FAILURES


async def run_one(case: dict, invoke, sem: asyncio.Semaphore, db_url: str | None, abort: Abort | None = None) -> dict | None:
    async with sem:
        if abort is not None and abort.tripped:
            return None                       # not run, not recorded -> a resume will run it
        trace = TraceHandler()
        config = {"callbacks": [trace], "configurable": {
            "thread_id": f"eval-{case['id']}", "allow_clarification": False,
            # batch runs cannot answer an approval prompt; the audit log still records every decision
            "approval_mode": os.environ.get("EVAL_APPROVAL_MODE", "auto_approve"),
            # Long-term memory is shared state: facts saved by earlier runs (some of them wrong) steer later
            # runs, so evaluations would not be independent. Each eval run gets its own empty memory.
            "mem0_user_id": MEMORY_USER}}
        before = db_fingerprint(db_url)
        t0, answer, error = time.perf_counter(), "", None
        try:
            answer = await invoke(case, config)
        except Exception as e:  # infrastructure failure: record it, keep going
            error = f"{type(e).__name__}: {str(e)[:300]}"
        if abort is not None:
            if error and PROVIDER_FATAL.search(error):
                abort.consecutive += 1
                abort.reason = error[:200]
            else:
                abort.consecutive = 0
        res = build_result(case, answer, trace, time.perf_counter() - t0, before, db_fingerprint(db_url), error)
        print(f"  {'PASS' if res.passed else 'FAIL'}  {case['id']:<28} {res.latency_s:>6.1f}s  tools={res.tool_calls}(err {res.tool_errors})")
        return res.to_dict()


async def main_async(args) -> int:
    global MEMORY_USER
    load_dotenv()
    MEMORY_USER = "default_user" if getattr(args, "shared_memory", False) else f"eval-{args.name}-{int(time.time())}"
    print(f"long-term memory namespace: {MEMORY_USER}")
    db_url = os.getenv("INTERNAL_DB_URL")
    if not args.dry_run and not db_url:
        print("INTERNAL_DB_URL is not set - the suite needs the sales database. (Use --dry-run to test the harness.)")
        return 2

    cases = [c for c in ALL_CASES
             if (not args.tags or c["tag"] in args.tags) and (not args.ids or c["id"] in args.ids)]
    if args.limit:
        cases = cases[: args.limit]
    out_path = RESULTS / f"{args.name}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists() and not args.force:
        # cases that died of an infrastructure error (provider outage, no credit...) are NOT finished work
        done = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))["results"] if not r.get("error")}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} selected, {len(done)} cached, {len(todo)} to run (concurrency {args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)
    invoke = _fake_invoke if args.dry_run else _real_invoke
    selected = {c["id"] for c in cases}

    def save() -> dict:
        results = [r for r in done.values() if r["id"] in selected]
        report = {"meta": {**run_metadata(ALL_CASES), "name": args.name, "dry_run": args.dry_run},
                  "summary": aggregate(results), "results": results}
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    abort = Abort()

    async def run_and_save(c: dict) -> None:
        res = await run_one(c, invoke, sem, db_url, abort)
        if res is None:
            return
        if res.get("error") and PROVIDER_FATAL.search(res["error"]):
            return                            # provider outage / bad key: not a result, a resume will run it
        done[c["id"]] = res
        save()  # persist after every case so an interrupted run loses nothing

    await asyncio.gather(*(run_and_save(c) for c in todo))
    report = save()
    if abort.tripped:
        print(f"\n!! ABORTED: the model provider failed {MAX_CONSECUTIVE_PROVIDER_FAILURES} times in a row "
              f"({abort.reason}).\n!! Fix it (top up credit / check the key), then re-run the same command: "
              f"finished cases are cached and only the missing ones run.")

    o = report["summary"]["overall"]
    print(f"\n=== {args.name}: {o['cases']} cases ===")
    for k, v in o.items():
        print(f"  {k:<26} {v}")
    print("\n  by tag:")
    for tag, b in report["summary"]["by_tag"].items():
        print(f"    {tag:<15} success {b['task_success_rate']:<6} n={b['cases']}")
    print(f"\nsaved -> {out_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shared-memory", action="store_true",
                    help="use the shared 'default_user' memory (old behaviour; runs then influence each other)")
    args = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
