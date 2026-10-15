from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _parse_roles(raw: str) -> dict[str, str]:
    """ODR_ROLES="bob=manager,carol=finance" -> {"bob": "manager", "carol": "finance"}"""
    out = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        user, _, role = pair.partition("=")
        if user and role:
            out[user.strip()] = role.strip()
    return out


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
    workers: int = 4        # tasks mostly wait on the LLM, so a few more than CPU cores is fine; ODR_WORKERS
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

    # business workflow (refunds)
    roles: dict[str, str] = field(default_factory=dict)   # user -> manager | finance | director
    business_db_url: str = ""                             # sqlite:///... ; empty disables the refund endpoints
    approval_log_path: str = "logs/approvals.jsonl"
    demo_mode: bool = False                               # exposes /demo/users (tokens!) - never in production

    checkpoint_path: str = "data/checkpoints.sqlite"

    # cost control & observability
    daily_token_budget: int = 0                 # per user per UTC day; 0 disables
    price_in_per_m: Optional[float] = None      # USD per million tokens - configure yours, we never guess
    price_out_per_m: Optional[float] = None
    admins: frozenset = frozenset()             # may open /dashboard and /v1/admin/summary
    trace_dir: str = "logs/traces"

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
            roles=_parse_roles(e("ODR_ROLES", "")),
            business_db_url=e("ODR_BUSINESS_DB_URL") or e("BUSINESS_DB_URL") or (
                e("INTERNAL_DB_URL", "") if e("ENABLE_ACTIONS", "0") in ("1", "true", "True") else ""),
            approval_log_path=e("APPROVAL_LOG_PATH", cls.approval_log_path),
            demo_mode=e("ODR_DEMO_MODE", "0") == "1",
            checkpoint_path=e("ODR_CHECKPOINT_PATH", cls.checkpoint_path),
            daily_token_budget=int(e("ODR_DAILY_TOKEN_BUDGET", 0)),
            price_in_per_m=float(e("ODR_PRICE_IN_PER_M")) if e("ODR_PRICE_IN_PER_M") else None,
            price_out_per_m=float(e("ODR_PRICE_OUT_PER_M")) if e("ODR_PRICE_OUT_PER_M") else None,
            admins=frozenset(filter(None, e("ODR_ADMINS", "").split(","))),
            trace_dir=e("ODR_TRACE_DIR", cls.trace_dir),
        )
