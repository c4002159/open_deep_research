"""Timeout, retry-with-backoff and circuit breaking for tool calls.

Design rules (each one exists because the naive version causes an incident):

* Only *transient* failures are retried (timeouts, dropped connections, 429/503).
  A wrong column name or a policy refusal is the model's problem; retrying it just
  burns time, and counting it against the breaker would take a healthy database
  offline because the model wrote bad SQL.
* Only tools known to be idempotent are retried. ``add_memory`` writes; running it
  twice stores the fact twice. Unknown (e.g. MCP) tools get a timeout and breaker but
  no retries.
* Backoff is exponential with full jitter so a fleet of agents does not retry in
  lock-step against a struggling dependency.
* The breaker is per tool name: when a dependency is down, calls fail fast instead of
  every agent waiting out the full timeout on every step.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from open_deep_research.policy import is_fatal_denial

logger = logging.getLogger(__name__)

UNAVAILABLE_PREFIX = "[TOOL_UNAVAILABLE]"

# Read-only / pure tools: safe to run again after a transient failure.
IDEMPOTENT_TOOLS = frozenset({
    "sql_executor", "python_executor", "search_memory", "search_knowledge_base",
    "tavily_search", "web_search", "think_tool",
})

_TRANSIENT_MARKERS = (
    "timeout", "timed out", "time out", "connection refused", "connection reset",
    "connection aborted", "could not connect", "server closed the connection",
    "connection was closed", "database is locked", "temporarily unavailable",
    "rate limit", "too many requests", "429", "502", "503", "504", "service unavailable",
    "operationalerror", "interfaceerror", "network is unreachable",
)
_TRANSIENT_EXCEPTIONS = (asyncio.TimeoutError, TimeoutError, ConnectionError)


def is_transient_text(text: str) -> bool:
    """True if a tool's *returned* error text describes an infrastructure problem."""
    if not isinstance(text, str) or is_fatal_denial(text):
        return False
    head = text[:400].lower()
    looks_like_error = head.startswith(("sql execution error", "error", "python execution error",
                                        "knowledge base search error", "memory search error"))
    return looks_like_error and any(m in head for m in _TRANSIENT_MARKERS)


@dataclass
class ResiliencePolicy:
    timeout_s: float = 90.0
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    breaker_threshold: int = 3
    breaker_cooldown_s: float = 15.0


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (cooldown) -> half-open -> closed/open."""

    def __init__(self, threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self._clock = threshold, cooldown_s, clock
        self.failures = 0
        self.opened_at: Optional[float] = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if self._clock() - self.opened_at >= self.cooldown_s else "open"

    def retry_in(self) -> float:
        return max(0.0, self.cooldown_s - (self._clock() - (self.opened_at or 0.0)))

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half-open" and not self._probe_in_flight:
            self._probe_in_flight = True  # exactly one trial call while half-open
            return True
        return False

    def record_success(self) -> None:
        self.failures, self.opened_at, self._probe_in_flight = 0, None, False

    def record_failure(self) -> None:
        self._probe_in_flight = False
        self.failures += 1
        if self.opened_at is not None or self.failures >= self.threshold:
            self.opened_at = self._clock()  # (re)open and restart the cooldown


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, policy: ResiliencePolicy) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(policy.breaker_threshold, policy.breaker_cooldown_s)
    return _breakers[name]


def reset_breakers() -> None:
    _breakers.clear()


def backoff_delay(attempt: int, policy: ResiliencePolicy, rng: Callable[[], float] = random.random) -> float:
    """Exponential backoff with full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return rng() * min(policy.backoff_max_s, policy.backoff_base_s * (2 ** attempt))


async def call_with_resilience(
    name: str,
    fn: Callable[[], Awaitable[str]],
    policy: ResiliencePolicy,
    *,
    breaker: Optional[CircuitBreaker] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
    stats: Optional[dict] = None,
) -> str:
    """Run ``fn`` with timeout, retries (idempotent tools only) and a circuit breaker."""
    stats = stats if stats is not None else {}
    breaker = breaker or get_breaker(name, policy)

    # The breaker is consulted once per *call*, and judged on the call's final outcome:
    # a blip that a retry absorbs must not count as a failure, otherwise random 30% packet
    # loss would open the circuit even though 97% of calls succeed.
    was_half_open = breaker.state == "half-open"
    if not breaker.allow():
        stats["short_circuited"] = stats.get("short_circuited", 0) + 1
        return (f"{UNAVAILABLE_PREFIX} Tool '{name}' is temporarily unavailable (circuit open after "
                f"{breaker.failures} consecutive failed calls; retry in ~{breaker.retry_in():.0f}s). "
                "Do not call it again; continue with what you already have or report the limitation.")
    # the single half-open probe gets no retries: it should answer "is it back?" quickly
    retries_allowed = 0 if was_half_open else (policy.max_retries if name in IDEMPOTENT_TOOLS else 0)

    last = ""
    for attempt in range(retries_allowed + 1):
        stats["attempts"] = stats.get("attempts", 0) + 1
        try:
            result = await asyncio.wait_for(fn(), timeout=policy.timeout_s)
            transient = is_transient_text(result)
            last = result
        except _TRANSIENT_EXCEPTIONS as e:
            transient, last = True, f"Error executing tool: {type(e).__name__}: {e}"
        except Exception as e:  # non-transient bug / bad arguments: report, don't retry, don't trip the breaker
            breaker.record_success()  # the dependency itself answered; release a half-open probe
            return f"Error executing tool: {type(e).__name__}: {e}"

        if not transient:
            breaker.record_success()
            return last
        if attempt < retries_allowed:
            delay = backoff_delay(attempt, policy, rng)
            stats["retries"] = stats.get("retries", 0) + 1
            logger.warning("tool %s transient failure (%s), retry %d in %.2fs", name, last[:80], attempt + 1, delay)
            await sleep(delay)
    breaker.record_failure()
    return last
