"""Run the business-action cases against the real agent and score the resulting *state*.

    python -m eval.run_action_eval --name actions                 # all cases (costs LLM tokens)
    python -m eval.run_action_eval --name actions --ids refund_expired_is_refused

Each case gets its own fresh copy of the demo database, so cases cannot influence each other. Results go to
eval/results/<name>.json; promote one to eval/baselines/ when you want it as a reference.
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

from eval.action_cases import ACTION_CASES, EVAL_USER
from eval.harness import TraceHandler
from eval.run_agent_eval import PROVIDER_FATAL

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def score(case: dict, rows: list[dict], answer: str) -> dict:
    checks = dict(case["state"](rows))
    low = answer.lower()
    for i, group in enumerate(case["say"]):
        checks[f"says_{i}"] = any(w.lower() in low for w in group)
    return checks


async def run_case(case: dict, run_id: str) -> dict:
    from open_deep_research import demo_data, refunds as RF
    from open_deep_research.deep_researcher import deep_researcher

    work = Path(tempfile.mkdtemp(prefix="odr_action_eval_"))
    db = demo_data.build_demo_db(str(work / "demo.db"), source=str(ROOT / "test.db"), seed=False)
    if case["seed"] == "pending80":
        out = RF.propose(db, order_id=80, amount=39700.37, reason="seed", outage_claim=False, requested_by=EVAL_USER,
                         today=__import__("datetime").date.fromisoformat(demo_data.BUSINESS_DATE))
        assert out.ok, out.message
    os.environ.update({"INTERNAL_DB_URL": "sqlite:///" + Path(db).as_posix(), "ENABLE_ACTIONS": "1",
                       "BUSINESS_DATE": demo_data.BUSINESS_DATE})
    os.environ.pop("BUSINESS_DB_URL", None)

    trace = TraceHandler()
    config = {"callbacks": [trace], "configurable": {
        "thread_id": f"{EVAL_USER}:{case['id']}-{run_id}", "allow_clarification": False,
        "approval_mode": "auto_approve",          # data-access approvals cannot be answered in a batch run
        "mem0_user_id": f"eval-actions-{run_id}", "requested_by": EVAL_USER}}
    t0, answer, error = time.perf_counter(), "", None
    try:
        result = await deep_researcher.ainvoke({"messages": [{"role": "user", "content": case["question"]}]}, config=config)
        if result.get("final_report"):
            answer = result["final_report"]
        else:
            msgs = result.get("messages") or []
            answer = (getattr(msgs[-1], "content", None) or str(result)) if msgs else str(result)
    except Exception as e:                          # noqa: BLE001 - infrastructure failure: record it, keep going
        error = f"{type(e).__name__}: {str(e)[:300]}"
    rows = RF.list_refunds(db)
    checks = {} if error else score(case, rows, answer)
    return {"id": case["id"], "tag": case["tag"], "passed": error is None and all(checks.values()), "checks": checks,
            "latency_s": round(time.perf_counter() - t0, 1), "tool_calls": len(trace.tool_calls),
            "tool_errors": sum(1 for t in trace.tool_calls if not t["ok"]), "llm_calls": trace.llm_calls,
            "tokens_in": trace.tokens_in, "tokens_out": trace.tokens_out, "error": error,
            "refunds_after": [{k: r[k] for k in ("id", "order_id", "amount", "status", "required_roles", "outage_claim")} for r in rows],
            "answer": answer[:3000]}


async def main_async(args) -> int:
    load_dotenv()
    run_id = str(int(time.time()))
    cases = [c for c in ACTION_CASES if not args.ids or c["id"] in args.ids]
    out_path = RESULTS / f"{args.name}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists() and not args.force:
        done = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))["results"] if not r.get("error")}
    consecutive = 0
    for c in cases:
        if c["id"] in done:
            continue
        res = await run_case(c, run_id)
        print(f"  {'PASS' if res['passed'] else 'FAIL'}  {c['id']:<34} {res['latency_s']:>6.1f}s  tools={res['tool_calls']}"
              f"  tokens={res['tokens_in'] + res['tokens_out']}")
        if not res["passed"]:
            for k, v in res["checks"].items():
                if not v:
                    print(f"        failed check: {k}")
            if res["error"]:
                print(f"        error: {res['error']}")
        if res["error"] and PROVIDER_FATAL.search(res["error"]):
            consecutive += 1
            if consecutive >= 3:
                print("!! the model provider failed 3 times in a row; fix it and re-run (finished cases are cached)")
                break
            continue
        consecutive = 0
        done[c["id"]] = res
        out_path.write_text(json.dumps(_report(args.name, done), indent=2, ensure_ascii=False), encoding="utf-8")
    report = _report(args.name, done)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print(f"\n=== {args.name}: {s['passed']}/{s['cases']} passed ({s['pass_rate']:.0%}); "
          f"avg tools {s['avg_tool_calls']}, max tools {s['max_tool_calls']}, avg tokens {s['avg_tokens']}, "
          f"applied-by-agent {s['applied_by_agent']} ===\nsaved -> {out_path}")
    return 0


def _report(name: str, done: dict) -> dict:
    rs = list(done.values())
    n = len(rs) or 1
    return {"name": name, "summary": {
        "cases": len(rs), "passed": sum(r["passed"] for r in rs), "pass_rate": sum(r["passed"] for r in rs) / n,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rs) / n, 1), "max_tool_calls": max((r["tool_calls"] for r in rs), default=0),
        "avg_tokens": round(sum(r["tokens_in"] + r["tokens_out"] for r in rs) / n),
        "avg_latency_s": round(sum(r["latency_s"] for r in rs) / n, 1),
        "applied_by_agent": sum(1 for r in rs for x in r["refunds_after"] if x["status"] == "applied"),
    }, "results": rs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
