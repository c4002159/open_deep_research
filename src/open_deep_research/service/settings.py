from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_keys(raw: str) -> dict[str, str]:
    """ODR_API_KEYS="key1:alice,key2:bob" -> {"key1": "alice", "key2": "bob"}"""
    out = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        key, _, user = pair.partition(":")
        if key and user:
            out[key] = user
    return out


@dataclass
class Settings:
    redis_url: str = "redis://localhost:6390/0"
    prefix: str = "odr:"
    redis_max_connections: int = 256          # request/response traffic
    redis_blocking_connections: int = 128     # BLMOVE (workers) and XREAD (SSE) hold a connection while waiting

    # admission control (per user)
    rate_limit_per_window: int = 10
    rate_window_s: int = 60
    max_active_per_user: int = 3
    queue_max_depth: int = 1000

    # workers / leases
    workers: int = 2
    lease_ttl_s: int = 30
    lease_refresh_s: float = 10.0
    reaper_interval_s: float = 5.0
    max_attempts: int = 3
    task_timeout_s: float = 1800.0

    # retention
    idempotency_ttl_s: int = 86_400
    task_ttl_s: int = 7 * 86_400

    # auth
    api_keys: dict[str, str] = field(default_factory=dict)
    jwt_secret: str = ""
    approvers: frozenset = frozenset()  # users who may approve/reject confidential-data requests
    owner_can_approve: bool = False     # approving your own request defeats the control; dev/demo only

    checkpoint_path: str = "data/checkpoints.sqlite"

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        return cls(
            redis_url=e("ODR_REDIS_URL", cls.redis_url),
            prefix=e("ODR_PREFIX", cls.prefix),
            rate_limit_per_window=int(e("ODR_RATE_LIMIT", cls.rate_limit_per_window)),
            rate_window_s=int(e("ODR_RATE_WINDOW_S", cls.rate_window_s)),
            max_active_per_user=int(e("ODR_MAX_ACTIVE", cls.max_active_per_user)),
            workers=int(e("ODR_WORKERS", cls.workers)),
            lease_ttl_s=int(e("ODR_LEASE_TTL_S", cls.lease_ttl_s)),
            api_keys=_parse_keys(e("ODR_API_KEYS", "")),
            jwt_secret=e("ODR_JWT_SECRET", ""),
            approvers=frozenset(filter(None, e("ODR_APPROVERS", "").split(","))),
            owner_can_approve=e("ODR_OWNER_CAN_APPROVE", "0") == "1",
            checkpoint_path=e("ODR_CHECKPOINT_PATH", cls.checkpoint_path),
        )
