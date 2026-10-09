"""Scoring, tracing and aggregation for the agent evaluation (no LLM judge needed).

Why not the old substring metric: ``"80" in "1180"`` passes. Numbers are now parsed out
of the answer as numbers (commas, 万 handled) and compared with a small tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

# ------------------------------------------------------------------ scoring
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*万)?")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text):
        tok = m.group(0)
        mult = 10_000 if tok.rstrip().endswith("万") else 1
        try:
            out.append(float(tok.replace(",", "").replace("万", "").strip()) * mult)
        except ValueError:
            continue
    return out


def number_present(expected: float, found: list[float]) -> bool:
    # exact for small integers; otherwise allow rounding differences (0.2% / 0.01 abs)
    tol = 0 if float(expected).is_integer() and abs(expected) < 1000 else max(0.011, abs(expected) * 0.002)
    return any(abs(expected - f) <= tol for f in found)


def _norm(s: str) -> str:
    return re.sub(r"[,\s，]", "", str(s)).lower()


def score_text(checks: dict, output: str) -> dict:
    """Evaluate the text-based checks; returns {check_name: bool}."""
    res: dict[str, bool] = {}
    n = _norm(output)
    if "numbers" in checks:
        found = extract_numbers(output)
        res["numbers"] = all(number_present(e, found) for e in checks["numbers"])
    if "contains_all" in checks:
        res["contains_all"] = all(_norm(s) in n for s in checks["contains_all"])
    if "contains_any" in checks:
        res["contains_any"] = any(_norm(s) in n for s in checks["contains_any"])
    if checks.get("cited_docs_exist"):
        # every knowledge-base document the answer cites must really exist (no fabricated sources)
        from open_deep_research.citation_check import cited_documents

        known = {p.name for p in (Path(__file__).resolve().parent.parent / "docs" / "internal_kb").glob("*.md")}
        res["cited_docs_exist"] = all(d in known for d in cited_documents(output))
    return res


def db_fingerprint(db_url: Optional[str]) -> Optional[str]:
    """SHA-256 of a SQLite file, or None for other backends / missing file."""
    if not db_url or not db_url.startswith("sqlite"):
        return None
    path = Path(db_url.split("///", 1)[-1])
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ------------------------------------------------------------------ tracing
from open_deep_research.telemetry import TOOL_ERROR_PREFIXES, UsageTracker as TraceHandler  # noqa: E402,F401


# ------------------------------------------------------------------ per-case result
@dataclass
class CaseResult:
    id: str
    tag: str
    passed: bool
    checks: dict = field(default_factory=dict)
    latency_s: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None  # infrastructure failure (exception), counts as a failed case
    answer_preview: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_result(case: dict, answer: str, trace: TraceHandler, latency: float,
                 db_before: Optional[str], db_after: Optional[str], error: Optional[str] = None) -> CaseResult:
    checks = score_text(case["checks"], answer) if error is None else {}
    if case["checks"].get("db_unchanged"):
        # None == None when the backend cannot be fingerprinted; treat as "not verifiable" -> pass
        checks["db_unchanged"] = db_before == db_after
    passed = error is None and bool(checks) and all(checks.values())
    return CaseResult(
        id=case["id"], tag=case["tag"], passed=passed, checks=checks, latency_s=round(latency, 2),
        tool_calls=len(trace.tool_calls), tool_errors=sum(1 for t in trace.tool_calls if not t["ok"]),
        llm_calls=trace.llm_calls, tokens_in=trace.tokens_in, tokens_out=trace.tokens_out,
        error=error, answer_preview=answer[:300],
    )


# ------------------------------------------------------------------ aggregation
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)], 2)


def aggregate(results: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        n = len(rs)
        calls = sum(r["tool_calls"] for r in rs)
        errs = sum(r["tool_errors"] for r in rs)
        lat = [r["latency_s"] for r in rs]
        tin, tout = sum(r["tokens_in"] for r in rs), sum(r["tokens_out"] for r in rs)
        price_in = float(os.environ.get("EVAL_PRICE_IN_PER_M", "nan"))
        price_out = float(os.environ.get("EVAL_PRICE_OUT_PER_M", "nan"))
        cost = None if math.isnan(price_in) or math.isnan(price_out) else (tin * price_in + tout * price_out) / 1e6
        return {
            "cases": n,
            "task_success_rate": round(sum(r["passed"] for r in rs) / n, 3) if n else 0,
            "tool_call_success_rate": round((calls - errs) / calls, 3) if calls else None,
            "avg_tool_calls": round(calls / n, 2) if n else 0,
            "avg_llm_calls": round(sum(r["llm_calls"] for r in rs) / n, 2) if n else 0,
            "avg_tokens": round((tin + tout) / n) if n else 0,
            "est_cost_per_case_usd": None if cost is None or not n else round(cost / n, 5),
            "latency_p50_s": _pct(lat, 50),
            "latency_p95_s": _pct(lat, 95),
            "infra_errors": sum(1 for r in rs if r["error"]),
            # a provider outage must not masquerade as the agent getting answers wrong
            "success_rate_excl_infra": (round(sum(r["passed"] for r in rs if not r["error"]) /
                                              max(1, sum(1 for r in rs if not r["error"])), 3)),
        }

    tags = sorted({r["tag"] for r in results})
    return {"overall": block(results), "by_tag": {t: block([r for r in results if r["tag"] == t]) for t in tags}}


def run_metadata(cases: list[dict]) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "src", "eval")),
        "models": {k: os.environ.get(k, "") for k in
                   ("RESEARCH_MODEL", "SUMMARIZATION_MODEL", "COMPRESSION_MODEL", "FINAL_REPORT_MODEL")},
        "dataset_hash": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12],
    }


# ------------------------------------------------------------------ baseline comparison
DEFAULT_THRESHOLDS = {"success_drop": 0.03, "p95_increase": 0.5, "tokens_increase": 0.3}


def compare_runs(base: dict, cur: dict, thresholds: Optional[dict] = None) -> dict:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    b = {r["id"]: r for r in base["results"]}
    c = {r["id"]: r for r in cur["results"]}
    shared = sorted(set(b) & set(c))
    regressions = [i for i in shared if b[i]["passed"] and not c[i]["passed"]]
    improvements = [i for i in shared if not b[i]["passed"] and c[i]["passed"]]

    bo = aggregate([b[i] for i in shared])["overall"]
    co = aggregate([c[i] for i in shared])["overall"]
    problems = []
    if bo["task_success_rate"] - co["task_success_rate"] > th["success_drop"]:
        problems.append(f"task success {bo['task_success_rate']} -> {co['task_success_rate']}")
    if bo["latency_p95_s"] and co["latency_p95_s"] > bo["latency_p95_s"] * (1 + th["p95_increase"]):
        problems.append(f"p95 latency {bo['latency_p95_s']}s -> {co['latency_p95_s']}s")
    if bo["avg_tokens"] and co["avg_tokens"] > bo["avg_tokens"] * (1 + th["tokens_increase"]):
        problems.append(f"avg tokens {bo['avg_tokens']} -> {co['avg_tokens']}")
    return {"compared": len(shared), "baseline": bo, "current": co, "regressions": regressions,
            "improvements": improvements, "problems": problems, "ok": not problems}
