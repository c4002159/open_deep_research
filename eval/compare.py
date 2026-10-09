"""Compare two evaluation runs and fail (exit 1) on a regression.

    python -m eval.compare eval/baselines/baseline.json eval/results/current.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.harness import compare_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--success-drop", type=float, default=0.03)
    ap.add_argument("--p95-increase", type=float, default=0.5)
    ap.add_argument("--tokens-increase", type=float, default=0.3)
    a = ap.parse_args()

    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    cur = json.loads(Path(a.current).read_text(encoding="utf-8"))
    if base["meta"]["dataset_hash"] != cur["meta"]["dataset_hash"]:
        print("warning: dataset changed between runs; comparing shared case ids only")

    r = compare_runs(base, cur, {"success_drop": a.success_drop, "p95_increase": a.p95_increase,
                                 "tokens_increase": a.tokens_increase})
    print(f"compared {r['compared']} cases   baseline {base['meta']['git_commit']} -> current {cur['meta']['git_commit']}")
    print(f"{'metric':<26}{'baseline':>12}{'current':>12}")
    for k in ("task_success_rate", "tool_call_success_rate", "avg_tool_calls", "avg_tokens", "latency_p50_s", "latency_p95_s"):
        print(f"{k:<26}{str(r['baseline'][k]):>12}{str(r['current'][k]):>12}")
    print(f"\nregressions  ({len(r['regressions'])}): {r['regressions']}")
    print(f"improvements ({len(r['improvements'])}): {r['improvements']}")
    if r["problems"]:
        print("\nFAIL: " + "; ".join(r["problems"]))
    else:
        print("\nOK: within thresholds")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
