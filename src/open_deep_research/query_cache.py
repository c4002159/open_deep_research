"""TTL cache for schema-introspection queries (sqlite_master, information_schema, pragma_table_info).

Why only introspection: every data-analyst run starts by asking the database what tables
and columns exist, and that answer changes rarely. Business-data queries are never cached -
a stale revenue number is worse than a slow one. Backend is an in-process dict by default;
set ODR_CACHE_REDIS_URL to share the cache between workers/replicas.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Callable, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_INTROSPECTION_TABLES = {"sqlite_master", "sqlite_schema", "sqlite_temp_master"}
_INTROSPECTION_SCHEMAS = {"information_schema", "pg_catalog"}
_INTROSPECTION_FUNCS = {"pragma_table_info", "pragma_table_list", "pragma_table_xinfo"}


def is_introspection(sql: str, dialect: Optional[str] = None) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError:
        return False
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return False
    touched = False
    for node in tree.walk():
        if isinstance(node, exp.Table):
            if node.name.lower() in _INTROSPECTION_TABLES or (node.db or "").lower() in _INTROSPECTION_SCHEMAS:
                touched = True
            elif node.name and not isinstance(node.this, exp.Anonymous):
                return False                      # also reads a real business table -> not pure introspection
        elif isinstance(node, exp.Anonymous) and node.name.lower() in _INTROSPECTION_FUNCS:
            touched = True
    return touched


class QueryCache:
    def __init__(self, ttl_s: float = 300.0, redis_url: Optional[str] = None, clock: Callable[[], float] = time.monotonic):
        self.ttl_s, self._clock = ttl_s, clock
        self._mem: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self._redis = None
        if redis_url:
            import redis

            self._redis = redis.Redis.from_url(redis_url, socket_timeout=1, socket_connect_timeout=1)

    @staticmethod
    def key(db_url: str, sql: str) -> str:
        norm = " ".join(sql.split()).lower()
        return "odr:qcache:" + hashlib.sha256(f"{db_url}\x00{norm}".encode()).hexdigest()

    def get(self, key: str):
        if self.ttl_s <= 0:
            return None
        raw = None
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                raw = raw.decode() if raw else None
            except Exception:                     # cache trouble must never break a query
                raw = None
        else:
            with self._lock:
                entry = self._mem.get(key)
                if entry and self._clock() - entry[0] < self.ttl_s:
                    raw = entry[1]
                elif entry:
                    del self._mem[key]
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def put(self, key: str, value) -> None:
        if self.ttl_s <= 0:
            return
        raw = json.dumps(value, default=str)
        if self._redis is not None:
            try:
                self._redis.set(key, raw, ex=int(self.ttl_s))
            except Exception:
                pass
            return
        with self._lock:
            if len(self._mem) > 512:              # crude bound: drop everything rather than grow forever
                self._mem.clear()
            self._mem[key] = (self._clock(), raw)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


_default: Optional[QueryCache] = None


def get_cache(ttl_s: float) -> QueryCache:
    global _default
    if _default is None or _default.ttl_s != ttl_s:
        _default = QueryCache(ttl_s, os.environ.get("ODR_CACHE_REDIS_URL"))
    return _default
