"""Deterministic tests for timeout / retry / circuit breaker (fake clock, no real sleeping)."""

import asyncio

import pytest

from open_deep_research.policy import fatal_denial
from open_deep_research.resilience import (
    UNAVAILABLE_PREFIX,
    CircuitBreaker,
    ResiliencePolicy,
    backoff_delay,
    call_with_resilience,
    is_transient_text,
    reset_breakers,
)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


async def no_sleep(_):
    return None


@pytest.fixture(autouse=True)
def _clean():
    reset_breakers()
    yield
    reset_breakers()


def flaky(fail_times, fail_text="SQL execution error: connection refused", ok="rows"):
    state = {"calls": 0}

    async def fn():
        state["calls"] += 1
        return fail_text if state["calls"] <= fail_times else ok

    return fn, state


# ------------------------------------------------------------ classification
@pytest.mark.parametrize("text,expected", [
    ("SQL execution error: (psycopg.OperationalError) connection refused", True),
    ("SQL execution error: query exceeded 10s and was cancelled (timeout)", True),
    ("Error executing tool: TimeoutError", True),
    ("SQL execution error: no such column: foo", False),         # model's mistake
    ("SQL execution error: syntax error near FROM", False),
    ("Query rejected by read-only guard: comments", False),
    (fatal_denial("DROP not allowed"), False),
    ("a | b\n1 | 2", False),
])
def test_transient_classification(text, expected):
    assert is_transient_text(text) is expected


# ------------------------------------------------------------ retry
def test_retries_transient_failure_then_succeeds():
    fn, st = flaky(2)
    stats = {}
    out = asyncio.run(call_with_resilience("sql_executor", fn, ResiliencePolicy(max_retries=2), sleep=no_sleep, stats=stats))
    assert out == "rows" and st["calls"] == 3 and stats["retries"] == 2


def test_gives_up_after_max_retries():
    fn, st = flaky(99)
    out = asyncio.run(call_with_resilience("sql_executor", fn, ResiliencePolicy(max_retries=2), sleep=no_sleep))
    assert "connection refused" in out and st["calls"] == 3


def test_model_errors_are_not_retried_and_do_not_trip_breaker():
    fn, st = flaky(99, fail_text="SQL execution error: no such column: foo")
    policy = ResiliencePolicy(breaker_threshold=2)
    for _ in range(5):
        asyncio.run(call_with_resilience("sql_executor", fn, policy, sleep=no_sleep))
    assert st["calls"] == 5  # one attempt each, breaker never opened


def test_non_idempotent_tool_is_never_retried():
    fn, st = flaky(99, fail_text="Memory save error: timeout")
    asyncio.run(call_with_resilience("add_memory", fn, ResiliencePolicy(max_retries=3), sleep=no_sleep))
    assert st["calls"] == 1


def test_timeout_is_enforced_and_retried():
    state = {"calls": 0}

    async def slow():
        state["calls"] += 1
        await asyncio.sleep(5)
        return "late"

    out = asyncio.run(call_with_resilience("sql_executor", slow, ResiliencePolicy(timeout_s=0.05, max_retries=1), sleep=no_sleep))
    assert state["calls"] == 2 and "Timeout" in out


def test_unexpected_exception_is_reported_not_raised():
    async def boom():
        raise ValueError("bad args")

    out = asyncio.run(call_with_resilience("sql_executor", boom, ResiliencePolicy(), sleep=no_sleep))
    assert out.startswith("Error executing tool: ValueError")


# ------------------------------------------------------------ backoff
def test_backoff_is_exponential_capped_and_jittered():
    p = ResiliencePolicy(backoff_base_s=1, backoff_max_s=8)
    assert [backoff_delay(a, p, rng=lambda: 1.0) for a in range(5)] == [1, 2, 4, 8, 8]
    assert backoff_delay(3, p, rng=lambda: 0.0) == 0.0
    assert 0 <= backoff_delay(2, p) <= 4


# ------------------------------------------------------------ circuit breaker
def test_breaker_state_machine():
    clk = Clock()
    b = CircuitBreaker(threshold=3, cooldown_s=10, clock=clk)
    assert b.state == "closed" and b.allow()
    for _ in range(3):
        b.record_failure()
    assert b.state == "open" and not b.allow()
    clk.t = 10
    assert b.state == "half-open"
    assert b.allow() and not b.allow()          # exactly one probe while half-open
    b.record_failure()                           # probe fails -> open again, cooldown restarts
    assert b.state == "open"
    clk.t = 20
    assert b.allow()
    b.record_success()
    assert b.state == "closed" and b.failures == 0


def test_open_breaker_fails_fast_without_calling_tool():
    clk = Clock()
    policy = ResiliencePolicy(max_retries=0, breaker_threshold=2, breaker_cooldown_s=30)
    br = CircuitBreaker(2, 30, clock=clk)
    fn, st = flaky(99)
    for _ in range(2):
        asyncio.run(call_with_resilience("sql_executor", fn, policy, breaker=br, sleep=no_sleep))
    calls_when_opened = st["calls"]
    stats = {}
    out = asyncio.run(call_with_resilience("sql_executor", fn, policy, breaker=br, sleep=no_sleep, stats=stats))
    assert out.startswith(UNAVAILABLE_PREFIX) and st["calls"] == calls_when_opened and stats["short_circuited"] == 1
    clk.t = 31                                   # cooldown over: one probe goes through and heals
    fn2, _ = flaky(0)
    assert asyncio.run(call_with_resilience("sql_executor", fn2, policy, breaker=br, sleep=no_sleep)) == "rows"
    assert br.state == "closed"
