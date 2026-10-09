"""Fault-injection benchmark for the reliability layer (offline, deterministic, virtual time).

    python -m eval.run_fault_injection

Each scenario runs the same fault pattern against two configurations:
  naive      - one attempt, no breaker, no model fallback (the pre-stage-3 behaviour)
  resilient  - retries with backoff + jitter, circuit breaker, model fallback
Time is simulated (a failing attempt "costs" its timeout), so the run takes milliseconds
and gives identical numbers every time.

Metrics are per unit of *time*, not per call count: failing fast is the whole point of a
breaker, so a fixed call count would reward the slow configuration for burning more time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path

from open_deep_research.model_fallback import ainvoke_with_fallback
from open_deep_research.resilience import CircuitBreaker, ResiliencePolicy, call_with_resilience

OUT = Path(__file__).resolve().parent / "fault_injection_report.json"
TIMEOUT_S = 5.0
HORIZON_S = 600.0
N_MODEL_CALLS = 200


class VClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, d):
        self.t += d


def configs():
    naive = ResiliencePolicy(timeout_s=TIMEOUT_S, max_retries=0, breaker_threshold=10**9)
    resilient = ResiliencePolicy(timeout_s=TIMEOUT_S, max_retries=2, backoff_base_s=0.5, backoff_max_s=4.0,
                                 breaker_threshold=3, breaker_cooldown_s=15.0)
    return {"naive": naive, "resilient": resilient}


async def run_tool_scenario(policy: ResiliencePolicy, down, blip_p=0.0, seed=7, spacing_s=1.0):
    """An agent keeps calling a tool for HORIZON_S virtual seconds.

    ``down(t)``  - True while the dependency is in a hard outage (every attempt fails)
    ``blip_p``   - probability that any single attempt fails randomly
      success_when_up      successes among calls started while the dependency was healthy
      avg_failed_call_s    how long the agent waits on a call that ends up failing
      dependency_attempts  load that actually reached the dependency
    """
    clk, rng = VClock(), random.Random(seed)
    br = CircuitBreaker(policy.breaker_threshold, policy.breaker_cooldown_s, clock=clk)
    stats: dict = {}
    attempts = {"n": 0}
    up_calls = up_ok = calls = 0
    failed_durations: list[float] = []

    async def dependency():
        attempts["n"] += 1
        if down(clk.t) or rng.random() < blip_p:
            clk.t += TIMEOUT_S  # a failing attempt burns the full timeout
            return "SQL execution error: connection refused (timeout)"
        clk.t += 0.2
        return "rows"

    while clk.t < HORIZON_S:
        started, was_up = clk.t, not down(clk.t)
        out = await call_with_resilience("sql_executor", dependency, policy, breaker=br, sleep=clk.sleep,
                                         rng=rng.random, stats=stats)
        calls += 1
        if was_up:
            up_calls += 1
            up_ok += out == "rows"
        if out != "rows":
            failed_durations.append(clk.t - started)
        clk.t += spacing_s
    return {
        "calls": calls,
        "success_when_up": round(up_ok / up_calls, 3) if up_calls else None,
        "avg_failed_call_s": round(sum(failed_durations) / len(failed_durations), 2) if failed_durations else 0.0,
        "dependency_attempts": attempts["n"],
        "retries": stats.get("retries", 0),
        "fast_fails": stats.get("short_circuited", 0),
    }


async def run_model_scenario(use_fallback: bool, p_fail: float, seed=11):
    rng = random.Random(seed)

    class RateLimitError(Exception):
        pass

    class Primary:
        async def ainvoke(self, _):
            if rng.random() < p_fail:
                raise RateLimitError("429")
            return "ok"

    class Backup:
        async def ainvoke(self, _):
            return "ok-backup"

    ok, stats = 0, {}
    for _ in range(N_MODEL_CALLS):
        try:
            await ainvoke_with_fallback(Primary(), Backup() if use_fallback else None, [], stats=stats)
            ok += 1
        except RateLimitError:
            pass
    return {"recovery_rate": round(ok / N_MODEL_CALLS, 3), "fallbacks_used": stats.get("fallbacks", 0)}


async def run_bad_sql_scenario(policy):
    """The model writes wrong SQL 40% of the time; a healthy database must NOT be tripped offline."""
    clk, rng = VClock(), random.Random(3)
    br = CircuitBreaker(policy.breaker_threshold, policy.breaker_cooldown_s, clock=clk)
    stats: dict = {}

    async def dependency():
        return "SQL execution error: no such column: foo" if rng.random() < 0.4 else "rows"

    for _ in range(200):
        await call_with_resilience("sql_executor", dependency, policy, breaker=br, sleep=clk.sleep, stats=stats)
    return {"breaker_state": br.state, "short_circuited": stats.get("short_circuited", 0), "retries": stats.get("retries", 0)}


async def main_async():
    cfg = configs()
    report: dict = {}
    scenarios = {
        "A_random_blips_30pct": (lambda t: False, 0.30),
        "B_outage_t100s_to_t300s": (lambda t: 100 <= t < 300, 0.0),
        "C_flapping_60s_up_60s_down": (lambda t: (t // 60) % 2 == 1, 0.0),
    }
    for name, (down, blip) in scenarios.items():
        report[name] = {label: await run_tool_scenario(p, down, blip) for label, p in cfg.items()}

    report["D_model_429_30pct"] = {
        "naive": await run_model_scenario(False, 0.30),
        "resilient": await run_model_scenario(True, 0.30),
    }
    report["E_bad_sql_does_not_trip_breaker"] = {label: await run_bad_sql_scenario(p) for label, p in cfg.items()}
    return report


def main():
    logging.getLogger("open_deep_research").setLevel(logging.ERROR)  # keep the table readable
    report = asyncio.run(main_async())
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"tool scenarios: {HORIZON_S:.0f}s virtual time, a failing attempt costs {TIMEOUT_S}s; "
          f"model scenario: {N_MODEL_CALLS} calls\n")
    print(f"{'scenario':<30}{'config':<11}{'ok when up':>11}{'fail wait':>11}{'dep. hits':>10}{'retries':>9}{'fast-fail':>10}")
    for name, res in report.items():
        for label, r in res.items():
            if "dependency_attempts" in r:
                print(f"{name:<30}{label:<11}{r['success_when_up']:>11}{r['avg_failed_call_s']:>10}s"
                      f"{r['dependency_attempts']:>10}{r['retries']:>9}{r['fast_fails']:>10}")
            elif "fallbacks_used" in r:
                print(f"{name:<30}{label:<11}{r['recovery_rate']:>11}   (fallbacks used: {r['fallbacks_used']})")
            else:
                print(f"{name:<30}{label:<11}   breaker={r['breaker_state']} fast-fails={r['short_circuited']} retries={r['retries']}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
