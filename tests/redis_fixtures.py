"""Shared helpers for tests that need a real Redis (skipped when none is reachable)."""

import asyncio
import functools
import inspect
import os
import uuid

import pytest
import redis
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("ODR_TEST_REDIS_URL", "redis://localhost:6390/0")


def _reachable() -> bool:
    try:
        return bool(redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1).ping())
    except Exception:
        return False


needs_redis = pytest.mark.skipif(not _reachable(), reason=f"no Redis at {REDIS_URL}")


def with_redis(fn):
    """Run an async test body as ``fn(r, prefix)`` against an isolated key prefix, then clean up."""

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        async def go():
            r = aioredis.from_url(REDIS_URL)
            prefix = f"odrtest:{uuid.uuid4().hex[:8]}:"
            try:
                return await fn(r, prefix, *a, **kw)
            finally:
                keys = [k async for k in r.scan_iter(match=f"{prefix}*")]
                if keys:
                    await r.delete(*keys)
                await r.aclose()

        return asyncio.run(go())

    wrapper.__signature__ = inspect.Signature()  # hide (r, prefix) from pytest's fixture lookup
    return needs_redis(wrapper)
