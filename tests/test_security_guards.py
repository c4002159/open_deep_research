"""Regression tests for the sql_executor / python_executor hardening.

Run:  .venv/Scripts/python -m pytest tests/test_security_guards.py -q
(Subprocess backend is forced so the tests do not need Docker.)
"""

import shutil
import sqlite3

import pytest

from eval.security_cases import CODE_CASES, SQL_CASES
from open_deep_research.code_sandbox import format_for_agent, run_python_sandboxed_sync
from open_deep_research.sql_guard import (
    ReadOnlyViolation,
    get_readonly_engine,
    run_readonly_query,
    validate_readonly_sql,
)

SQL_ATTACKS = [c for c in SQL_CASES if c["expect"] == "block"]
SQL_CONTROLS = [c for c in SQL_CASES if c["expect"] == "allow"]
CODE_ATTACKS = [c for c in CODE_CASES if c["cat"] != "control"]
CODE_CONTROLS = [c for c in CODE_CASES if c["cat"] == "control"]


@pytest.fixture()
def db_url(tmp_path):
    src = sqlite3.connect(":memory:")
    src.executescript(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product_line TEXT, region TEXT, revenue REAL, units INTEGER, date TEXT);"
        "CREATE TABLE products (product_line TEXT PRIMARY KEY, category TEXT, launch_year INTEGER);"
        "INSERT INTO products VALUES ('AI Tools','Basic',2020);"
        "INSERT INTO sales VALUES (1,'AI Tools','East',100.0,2,'2025-01-01');"
    )
    path = tmp_path / "t.db"
    dst = sqlite3.connect(path)
    src.backup(dst)
    dst.close()
    get_readonly_engine.cache_clear()
    yield "sqlite:///" + path.as_posix()
    get_readonly_engine.cache_clear()


@pytest.mark.parametrize("case", SQL_ATTACKS, ids=lambda c: c["id"])
def test_parser_blocks_sql_attack(case):
    with pytest.raises(ReadOnlyViolation):
        validate_readonly_sql(case["sql"], case.get("dialect", "sqlite"))


@pytest.mark.parametrize("case", SQL_CONTROLS, ids=lambda c: c["id"])
def test_parser_allows_legit_sql(case):
    validate_readonly_sql(case["sql"], "sqlite")


@pytest.mark.parametrize("case", [c for c in SQL_ATTACKS if c.get("runnable", True)], ids=lambda c: c["id"])
def test_pipeline_never_mutates_db(case, db_url):
    path = db_url.replace("sqlite:///", "")
    before = "\n".join(sqlite3.connect(path).iterdump())
    with pytest.raises(Exception):
        run_readonly_query(db_url, case["sql"])
    assert "\n".join(sqlite3.connect(path).iterdump()) == before


def test_readonly_connection_is_independent_boundary(db_url):
    """Even if the parser were bypassed, the connection itself refuses writes."""
    from sqlalchemy import text

    with pytest.raises(Exception):
        with get_readonly_engine(db_url).connect() as conn:
            conn.execute(text("DROP TABLE sales"))


def test_query_timeout(db_url):
    q = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r"
    with pytest.raises(TimeoutError):
        run_readonly_query(db_url, q, timeout_s=1)


def test_row_cap(db_url):
    q = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 1000) SELECT n FROM r"
    _, rows, truncated = run_readonly_query(db_url, q, max_rows=200)
    assert len(rows) == 200 and truncated


def _won(case, text):
    kind = case.get("kind")
    if kind == "hang":
        return "exceeded" not in text  # should have been killed
    if kind == "flood":
        return len(text) > 100_000
    return "PWNED" in text


@pytest.mark.parametrize("case", CODE_ATTACKS, ids=lambda c: c["id"])
def test_code_attack_blocked(case):
    res = run_python_sandboxed_sync(case["code"], timeout=5, backend="subprocess")
    assert not _won(case, format_for_agent(res)), format_for_agent(res)[:200]


@pytest.mark.parametrize("case", CODE_CONTROLS, ids=lambda c: c["id"])
def test_legit_analysis_still_works(case):
    res = run_python_sandboxed_sync(case["code"], timeout=30, backend="subprocess")
    assert case["expect"] in format_for_agent(res)


def test_secrets_not_inherited(monkeypatch):
    """The child process starts with an empty environment (no API keys)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    res = run_python_sandboxed_sync("print('x')", timeout=20, backend="subprocess")
    assert "sk-should-not-leak" not in format_for_agent(res)


# ------------------------------------------------------------ Docker backend (skipped when no daemon / image)
def _docker_ready() -> bool:
    from open_deep_research.code_sandbox import docker_available

    if not docker_available():
        return False
    import docker

    try:
        docker.from_env().images.get("agent-sandbox-python:latest")
        return True
    except Exception:
        return False


docker_only = pytest.mark.skipif(not _docker_ready(), reason="Docker daemon or agent-sandbox-python image not available")


@docker_only
def test_docker_runs_legit_code():
    res = run_python_sandboxed_sync("df = pd.DataFrame({'a':[1,2,3]})\nresult = int(df.a.sum())", timeout=60, backend="docker")
    assert res.backend == "docker" and "result = 6" in format_for_agent(res)


@docker_only
def test_docker_kills_infinite_loop():
    res = run_python_sandboxed_sync("while True:\n    pass", timeout=8, backend="docker")
    assert res.blocked and not res.ok


@docker_only
@pytest.mark.parametrize("name,payload,must_not_contain", [
    ("no_network", "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); print('NET_OPEN')\nexcept OSError: print('blocked')", "NET_OPEN"),
    ("non_root_user", "import os\nprint('ROOT' if os.getuid() == 0 else 'nonroot')", "ROOT"),
    ("rootfs_not_writable", "try:\n    open('/etc/pwned', 'w'); print('WRITTEN')\nexcept OSError: print('denied')", "WRITTEN"),
    ("memory_cap", "x = bytearray(900 * 1024 * 1024); print('ALLOCATED')", "ALLOCATED"),
])
def test_container_boundary_without_python_level_restrictions(name, payload, must_not_contain):
    """Raw payloads straight into the container: the boundary must hold even if the
    static policy and restricted builtins were bypassed."""
    from llm_sandbox import SandboxSession

    from open_deep_research.code_sandbox import DOCKER_RUNTIME_CONFIGS

    with SandboxSession(lang="python", image="agent-sandbox-python:latest", skip_environment_setup=True,
                        execution_timeout=30, runtime_configs=DOCKER_RUNTIME_CONFIGS) as s:
        r = s.run(payload)
    assert must_not_contain not in (r.stdout or "")
