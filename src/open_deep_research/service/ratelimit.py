"""Per-user sliding-window rate limiter, atomic in Redis (Lua).

Sliding *log* rather than a fixed window: a fixed window lets a client burst 2x the limit
across a window boundary (N at 00:59, N more at 01:00). The log keeps one ZSET member per
accepted request and counts those inside the trailing window, so the limit holds for any
60-second interval. Time comes from the Redis server (``TIME``), so API replicas with
skewed clocks agree, and the check-then-add is one script so concurrent requests cannot
both squeeze through the last slot.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import redis.asyncio as aioredis

_SLIDING_WINDOW = """
local key     = KEYS[1]
local window  = tonumber(ARGV[1])   -- ms
local limit   = tonumber(ARGV[2])
local member  = ARGV[3]
local t       = redis.call('TIME')
local now     = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, member)
  redis.call('PEXPIRE', key, window)
  return {1, limit - count - 1, 0}
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local retry  = tonumber(oldest[2]) + window - now
if retry < 1 then retry = 1 end
return {0, 0, retry}
"""


@dataclass
class RateDecision:
    allowed: bool
    remaining: int
    retry_after_s: float


class SlidingWindowLimiter:
    def __init__(self, redis: aioredis.Redis, prefix: str, limit: int, window_s: int):
        self.redis, self.prefix, self.limit, self.window_ms = redis, prefix, limit, window_s * 1000
        self._script = redis.register_script(_SLIDING_WINDOW)

    async def hit(self, user: str) -> RateDecision:
        allowed, remaining, retry_ms = await self._script(
            keys=[f"{self.prefix}rl:{user}"], args=[self.window_ms, self.limit, uuid.uuid4().hex]
        )
        return RateDecision(bool(allowed), int(remaining), round(int(retry_ms) / 1000, 1))
