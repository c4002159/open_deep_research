import asyncio
import sqlite3

import pytest

from open_deep_research.configuration import Configuration
from open_deep_research.query_cache import QueryCache, is_introspection
from tests.redis_fixtures import REDIS_URL, needs_redis


@pytest.mark.parametrize("sql,expected", [
    ("SELECT name FROM sqlite_master WHERE type = 'table'", True),
    ("SELECT sql FROM sqlite_master WHERE name = 'sales'", True),
    ("SELECT name, type FROM pragma_table_info('sales')", True),
    ("SELECT column_name FROM information_schema.columns WHERE table_name = 'sales'", True),
    ("SELECT COUNT(*) FROM sales", False),
    ("SELECT * FROM sales LIMIT 3", False),
    # touches schema AND data -> must not be cached
    ("SELECT m.name, (SELECT COUNT(*) FROM sales) FROM sqlite_master m", False),
    ("DROP TABLE sales", False),
])
def test_introspection_detection(sql, expected):
    assert is_introspection(sql) is expected


def test_hit_miss_and_ttl_expiry():
    now = [0.0]
    c = QueryCache(ttl_s=10, clock=lambda: now[0])
    k = QueryCache.key("sqlite:///x.db", "SELECT name FROM sqlite_master")
    assert c.get(k) is None
    c.put(k, [["name"], [["sales"]], False])
    assert c.get(k) == [["name"], [["sales"]], False]
    now[0] = 11
    assert c.get(k) is None
    assert (c.hits, c.misses) == (1, 2) and c.hit_rate == pytest.approx(1 / 3)


def test_key_normalises_whitespace_and_separates_databases():
    a = QueryCache.key("db1", "SELECT  name\nFROM sqlite_master")
    assert a == QueryCache.key("db1", "select name from sqlite_master")
    assert a != QueryCache.key("db2", "select name from sqlite_master")


def test_ttl_zero_disables():
    c = QueryCache(ttl_s=0)
    c.put("k", [1])
    assert c.get("k") is None


@needs_redis
def test_redis_backend_shares_entries_between_instances():
    k = QueryCache.key("db", "select name from sqlite_master -- redis-test")
    a, b = QueryCache(60, REDIS_URL), QueryCache(60, REDIS_URL)
    a.put(k, [["n"], [["x"]], False])
    assert b.get(k) == [["n"], [["x"]], False]
    a._redis.delete(k)


def test_sql_executor_serves_introspection_from_cache_but_never_data(tmp_path, monkeypatch):
    import open_deep_research.query_cache as qc
    from open_deep_research import utils

    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE sales (id INTEGER, revenue REAL); INSERT INTO sales VALUES (1, 10.0);")
    con.commit(); con.close()
    qc._default = None
    calls = []
    real = utils.run_readonly_query
    monkeypatch.setattr(utils, "run_readonly_query", lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
    cfg = {"configurable": {"internal_db_url": "sqlite:///" + db.as_posix(), "schema_cache_ttl_s": 300}}

    async def go():
        schema = "SELECT name FROM sqlite_master WHERE type = 'table'"
        for _ in range(3):
            assert "sales" in await utils.sql_executor.ainvoke({"query": schema}, config=cfg)
        for _ in range(3):
            await utils.sql_executor.ainvoke({"query": "SELECT SUM(revenue) FROM sales"}, config=cfg)

    asyncio.run(go())
    assert calls.count("SELECT name FROM sqlite_master WHERE type = 'table'") == 1, "schema query hit the DB once"
    assert calls.count("SELECT SUM(revenue) FROM sales") == 3, "business data is never served from cache"


def test_request_identity_beats_process_environment(monkeypatch):
    """A MEM0_USER_ID in the environment must not override the authenticated user."""
    monkeypatch.setenv("MEM0_USER_ID", "shared-account")
    cfg = Configuration.from_runnable_config({"configurable": {"mem0_user_id": "alice"}})
    assert cfg.mem0_user_id == "alice"
    assert Configuration.from_runnable_config({"configurable": {}}).mem0_user_id == "shared-account"
