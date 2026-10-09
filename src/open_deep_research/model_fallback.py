"""Fail over to a backup model when the primary keeps failing for infrastructure reasons.

Only *availability* errors trigger the switch (timeouts, rate limits, 5xx, dropped
connections). Auth errors, bad requests and context-length errors would fail the same
way on the backup (or need a different remedy), so they are re-raised untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_RETRYABLE_NAMES = {
    "ratelimiterror", "apitimeouterror", "apiconnectionerror", "internalservererror",
    "serviceunavailableerror", "timeouterror", "connecterror", "connecttimeout", "readtimeout",
    "timeoutexception", "connectionerror", "overloadederror", "resourceexhausted",
    "serviceunavailable", "deadlineexceeded", "toomanyrequests",
}
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


def is_retryable_model_error(exc: BaseException) -> bool:
    for cls in type(exc).__mro__:
        if cls.__name__.lower() in _RETRYABLE_NAMES:
            return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status in _RETRYABLE_STATUS


async def ainvoke_with_fallback(
    primary: Any,
    fallback: Optional[Any],
    messages: list,
    *,
    stats: Optional[dict] = None,
    on_fallback: Optional[Callable[[BaseException], None]] = None,
) -> Any:
    """``await primary.ainvoke(messages)``; on an availability error use ``fallback`` once."""
    try:
        return await primary.ainvoke(messages)
    except Exception as e:
        if fallback is None or not is_retryable_model_error(e):
            raise
        logger.warning("primary model failed (%s: %s); switching to fallback", type(e).__name__, str(e)[:120])
        if stats is not None:
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
        if on_fallback:
            on_fallback(e)
        return await fallback.ainvoke(messages)
