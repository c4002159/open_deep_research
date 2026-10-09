"""Read-only protection for the agent's ``sql_executor`` tool.

Three layers, any one of which is enough to stop a write on its own:

1. ``validate_readonly_sql``  - parse with sqlglot and allow exactly one SELECT
   (CTEs / UNIONs included). Anything else is rejected before it reaches the DB.
2. ``get_readonly_engine``    - the connection itself is read-only
   (SQLite ``mode=ro`` + ``query_only``, PostgreSQL ``default_transaction_read_only``),
   so a parser bypass still cannot modify data.
3. ``run_readonly_query``     - statement timeout and a row cap, so a legal but
   expensive query cannot hang the agent.

The parser is the fast, explainable layer (it gives the model an error it can
correct). The connection settings are the real boundary; for production use a
dedicated read-only database account as well.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import sqlalchemy
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine, make_url


logging.getLogger("sqlglot").setLevel(logging.ERROR)  # parse fallbacks are expected for rejected syntax


class ReadOnlyViolation(Exception):
    """Raised when a statement is not a single, side-effect-free SELECT."""


_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}

# Node types that must never appear anywhere in the tree (covers writes hidden in
# CTEs, ``SELECT ... INTO``, ``FOR UPDATE`` locks, etc.).
_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
        "TruncateTable", "Command", "Pragma", "Set", "Into", "Copy", "Lock",
        "Transaction", "Commit", "Rollback", "Use", "Attach", "Detach",
        "Grant", "Revoke", "LoadData", "Kill",
    )
    if hasattr(exp, name)
)

# sqlglot models common functions as dedicated classes (safe). Anything it does not
# recognise comes back as ``Anonymous``; those are denied unless listed here. This
# is what blocks pg_read_file, pg_sleep, load_extension, lo_import, dblink, ...
_SAFE_ANONYMOUS_FUNCTIONS = frozenset({
    "date", "datetime", "time", "julianday", "strftime", "unixepoch",
    "round", "abs", "ifnull", "nullif", "iif", "typeof", "length", "lower",
    "upper", "substr", "replace", "trim", "ltrim", "rtrim", "instr", "printf",
    "group_concat", "total", "random", "sign", "ceil", "floor", "power", "sqrt",
    "pragma_table_info", "pragma_table_list", "pragma_table_xinfo",
})


def _dialect_for(engine_or_name) -> Optional[str]:
    name = getattr(getattr(engine_or_name, "dialect", None), "name", engine_or_name)
    return _DIALECTS.get(name)


def validate_readonly_sql(sql: str, dialect: Optional[str] = None) -> str:
    """Return ``sql`` unchanged if it is one read-only SELECT, else raise.

    Fails closed: anything the parser cannot understand is rejected.
    """
    if not sql or not sql.strip():
        raise ReadOnlyViolation("empty statement")

    try:
        dialect_obj = sqlglot.Dialect.get_or_raise(dialect)
        tokens = dialect_obj.tokenize(sql)
        statements = sqlglot.parse(sql, read=dialect)
    except SqlglotError as e:
        raise ReadOnlyViolation(f"could not parse statement: {str(e)[:200]}")

    # Comments are a classic way to smuggle a second statement past naive filters.
    if any(tok.comments for tok in tokens):
        raise ReadOnlyViolation("SQL comments are not allowed")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ReadOnlyViolation("exactly one statement is allowed (no ';' chaining)")

    root = statements[0]
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        raise ReadOnlyViolation(
            f"only SELECT statements are allowed, got {type(root).__name__.upper()}"
        )

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise ReadOnlyViolation(f"{type(node).__name__.upper()} is not allowed in a read-only query")
        if isinstance(node, exp.Anonymous) and node.name.lower() not in _SAFE_ANONYMOUS_FUNCTIONS:
            raise ReadOnlyViolation(f"function '{node.name}' is not allowed")

    return sql


@lru_cache(maxsize=8)
def get_readonly_engine(db_url: str) -> Engine:
    """Create (and cache) an engine whose connections cannot write."""
    url = make_url(db_url)
    backend = url.get_backend_name()

    if backend == "sqlite":
        db = url.database
        if db and db != ":memory:":
            path = quote(Path(db).resolve().as_posix(), safe="/:")
            engine = sqlalchemy.create_engine(f"sqlite:///file:{path}?mode=ro&uri=true")
        else:
            engine = sqlalchemy.create_engine(url)

        @sqlalchemy.event.listens_for(engine, "connect")
        def _sqlite_readonly(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA query_only = ON")

        return engine

    if backend == "postgresql":
        return sqlalchemy.create_engine(
            url,
            pool_pre_ping=True,
            connect_args={
                "options": "-c default_transaction_read_only=on -c statement_timeout=10000"
            },
        )

    if backend == "mysql":
        engine = sqlalchemy.create_engine(url, pool_pre_ping=True)

        @sqlalchemy.event.listens_for(engine, "connect")
        def _mysql_readonly(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute("SET SESSION max_execution_time = 10000")
            cur.close()

        return engine

    raise ReadOnlyViolation(f"unsupported database backend '{backend}'")


def run_readonly_query(db_url: str, query: str, max_rows: int = 200, timeout_s: float = 10.0):
    """Validate and execute ``query``; returns ``(columns, rows, truncated)``.

    Raises ``ReadOnlyViolation`` for rejected statements; database errors propagate
    so the caller can show them to the model for self-correction.
    """
    engine = get_readonly_engine(db_url)
    validate_readonly_sql(query, _dialect_for(engine))

    with engine.connect() as conn:
        timer: Optional[threading.Timer] = None
        if engine.dialect.name == "sqlite":
            # SQLite has no statement_timeout; interrupt the connection instead.
            raw = conn.connection.driver_connection
            timer = threading.Timer(timeout_s, raw.interrupt)
            timer.daemon = True
            timer.start()
        try:
            result = conn.execute(sql_text(query))
            columns = list(result.keys())
            rows = result.fetchmany(max_rows + 1)
        except sqlalchemy.exc.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise TimeoutError(f"query exceeded {timeout_s:.0f}s and was cancelled")
            raise
        finally:
            if timer:
                timer.cancel()

    truncated = len(rows) > max_rows
    return columns, rows[:max_rows], truncated
