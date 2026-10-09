"""Measure the attack-block rate of the legacy executors vs the hardened ones.

    python -m eval.run_security_eval            # prints a table, writes eval/security_report.json

Legacy behaviour is re-implemented here (never imported from src) purely as the
baseline: raw SQLAlchemy execution and an ``exec()`` with full builtins.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import sqlalchemy
from sqlalchemy import text

from eval.security_cases import CODE_CASES, FLOOD_LIMIT, SQL_CASES
from open_deep_research.code_sandbox import (
    MAX_OUTPUT_CHARS,
    _run_subprocess,
    format_for_agent,
    run_python_sandboxed_sync,
)
from open_deep_research.sql_guard import (
    ReadOnlyViolation,
    get_readonly_engine,
    run_readonly_query,
    validate_readonly_sql,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = ROOT / "test.db"
CODE_TIMEOUT = float(os.environ.get("EVAL_CODE_TIMEOUT", "5"))
BACKEND = os.environ.get("EVAL_BACKEND", "subprocess")  # subprocess | docker


# ------------------------------------------------------------------- SQL
def _fresh_db() -> tuple[Path, str]:
    d = Path(tempfile.mkdtemp(prefix="sqleval_"))
    shutil.copy(SOURCE_DB, d / "t.db")
    return d, "sqlite:///" + (d / "t.db").as_posix()


def _fingerprint(db_file: Path) -> str:
    import sqlite3

    con = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
    try:
        return hashlib.sha256("\n".join(con.iterdump()).encode()).hexdigest()
    finally:
        con.close()


def eval_sql() -> list[dict]:
    import os

    os.chdir(tempfile.mkdtemp(prefix="sqlcwd_"))  # ATTACH payloads create files relative to cwd
    rows = []
    for c in SQL_CASES:
        row = {"id": c["id"], "cat": c["cat"], "expect": c["expect"]}
        dialect = c.get("dialect", "sqlite")

        # parser layer on its own (works for every dialect)
        try:
            validate_readonly_sql(c["sql"], dialect)
            row["parser_blocked"] = False
        except ReadOnlyViolation:
            row["parser_blocked"] = True

        if c.get("runnable", True):
            # legacy: raw execution, no guard, writable connection
            d, url = _fresh_db()
            before = _fingerprint(d / "t.db")
            try:
                engine = sqlalchemy.create_engine(url)
                with engine.connect() as conn:
                    conn.execute(text(c["sql"]))
                row["legacy_executed"] = True
            except Exception:
                row["legacy_executed"] = False
            row["legacy_mutated"] = _fingerprint(d / "t.db") != before
            shutil.rmtree(d, ignore_errors=True)

            # hardened pipeline (parser + read-only connection + timeout)
            d, url = _fresh_db()
            before = _fingerprint(d / "t.db")
            try:
                run_readonly_query(url, c["sql"])
                row["guarded_executed"] = True
            except Exception:
                row["guarded_executed"] = False
            row["guarded_mutated"] = _fingerprint(d / "t.db") != before

            # ablation: skip the parser, rely on the read-only connection only
            try:
                with get_readonly_engine(url).connect() as conn:
                    conn.execute(text(c["sql"]))
                row["db_layer_executed"] = True
            except Exception:
                row["db_layer_executed"] = False
            row["db_layer_mutated"] = _fingerprint(d / "t.db") != before
            get_readonly_engine.cache_clear()
            shutil.rmtree(d, ignore_errors=True)
        rows.append(row)
    return rows


# ------------------------------------------------------------------- code
_LEGACY_RUNNER = r"""
import sys, io, contextlib, traceback
import pandas as pd, numpy as np, scipy
g = {"pd": pd, "np": np, "scipy": scipy, "__builtins__": __builtins__}
loc = {}
buf = io.StringIO()
code = sys.stdin.read()
try:
    with contextlib.redirect_stdout(buf):
        exec(code, g, loc)
    out = buf.getvalue()
    if loc.get("result") is not None:
        out += "\nresult = " + repr(loc["result"])
except Exception:
    out = "Python execution error:\n" + traceback.format_exc()
sys.__stdout__.write(out)
"""


def _legacy_exec(code: str) -> str:
    with tempfile.TemporaryDirectory(prefix="legacy_") as wd:
        try:
            p = subprocess.run(
                [sys.executable, "-c", _LEGACY_RUNNER],
                input=code.encode(), capture_output=True, timeout=CODE_TIMEOUT, cwd=wd,
            )
            return p.stdout.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            return "__HUNG__"


def _attacker_won(case: dict, output: str) -> bool:
    kind = case.get("kind")
    if kind == "hang":
        return output == "__HUNG__"
    if kind == "flood":
        return len(output) > FLOOD_LIMIT
    return "PWNED" in output


def eval_code() -> list[dict]:
    rows = []
    for c in CODE_CASES:
        row = {"id": c["id"], "cat": c["cat"]}
        if c["cat"] == "control":
            t = time.perf_counter()
            res = run_python_sandboxed_sync(c["code"], 30.0, backend=BACKEND)
            row["latency_s"] = round(time.perf_counter() - t, 2)
            row["control_ok"] = c["expect"] in format_for_agent(res)
            rows.append(row)
            continue

        # legacy: exec() with full builtins (run in a child so a hang/crash is survivable)
        row["legacy_won"] = _attacker_won(c, _legacy_exec(c["code"]))

        # hardened: static policy + isolated process + timeout + output cap
        res = run_python_sandboxed_sync(c["code"], CODE_TIMEOUT, backend=BACKEND)
        row["guarded_won"] = _attacker_won(c, format_for_agent(res) if not res.blocked else res.error or "")
        row["guarded_by"] = res.backend

        # ablation: skip the static policy, rely on process/runtime restrictions only
        res2 = _run_subprocess(c["code"], CODE_TIMEOUT)
        row["runtime_only_won"] = _attacker_won(c, format_for_agent(res2) if not res2.blocked else res2.error or "")
        rows.append(row)
    return rows


# ------------------------------------------------------------------- report
def _rate(rows, key, invert=False):
    n = len(rows)
    hits = sum(1 for r in rows if (not r[key] if invert else r[key]))
    return hits, n


def main():
    sql_rows = eval_sql()
    code_rows = eval_code()

    sql_attacks = [r for r in sql_rows if r["expect"] == "block"]
    sql_runnable = [r for r in sql_attacks if "legacy_executed" in r]
    sql_controls = [r for r in sql_rows if r["expect"] == "allow"]
    code_attacks = [r for r in code_rows if r["cat"] != "control"]
    code_controls = [r for r in code_rows if r["cat"] == "control"]

    summary = {
        "sql": {
            "attacks": len(sql_attacks),
            "attacks_runnable_on_sqlite": len(sql_runnable),
            "parser_block_rate": _rate(sql_attacks, "parser_blocked"),
            "legacy_not_rejected": _rate(sql_runnable, "legacy_executed"),
            "legacy_db_mutated": _rate(sql_runnable, "legacy_mutated"),
            "guarded_not_rejected": _rate(sql_runnable, "guarded_executed"),
            "guarded_db_mutated": _rate(sql_runnable, "guarded_mutated"),
            "db_layer_only_not_rejected": _rate(sql_runnable, "db_layer_executed"),
            "db_layer_only_db_mutated": _rate(sql_runnable, "db_layer_mutated"),
            "false_positives": (
                sum(1 for r in sql_controls if r["parser_blocked"]), len(sql_controls)
            ),
        },
        "code": {
            "attacks": len(code_attacks),
            "legacy_attacker_wins": _rate(code_attacks, "legacy_won"),
            "guarded_attacker_wins": _rate(code_attacks, "guarded_won"),
            "runtime_only_attacker_wins": _rate(code_attacks, "runtime_only_won"),
            "controls_passing": (sum(1 for r in code_controls if r["control_ok"]), len(code_controls)),
            "control_latency_s": [r["latency_s"] for r in code_controls],
        },
        "max_output_chars": MAX_OUTPUT_CHARS,
        "backend": BACKEND,
    }
    (ROOT / "eval" / f"security_report_{BACKEND}.json").write_text(
        json.dumps({"summary": summary, "sql": sql_rows, "code": code_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def f(t):  # (hits, n) -> "h/n"
        return f"{t[0]}/{t[1]}"

    s, c = summary["sql"], summary["code"]
    print("\n=== SQL (attacks, lower is better) ===")
    print(f"parser blocks            : {f(s['parser_block_rate'])} of all {s['attacks']} attacks")
    print(f"legacy  not rejected     : {f(s['legacy_not_rejected'])}   DB mutated: {f(s['legacy_db_mutated'])}")
    print(f"guarded not rejected     : {f(s['guarded_not_rejected'])}   DB mutated: {f(s['guarded_db_mutated'])}")
    print(f"read-only conn only      : {f(s['db_layer_only_not_rejected'])}   DB mutated: {f(s['db_layer_only_db_mutated'])}")
    print(f"legit queries blocked    : {f(s['false_positives'])}")
    print("\n=== Python (attacker wins, lower is better) ===")
    print(f"legacy exec()            : {f(c['legacy_attacker_wins'])}")
    print(f"hardened (all layers)    : {f(c['guarded_attacker_wins'])}")
    print(f"runtime layer only       : {f(c['runtime_only_attacker_wins'])}")
    print(f"legit analyses passing   : {f(c['controls_passing'])}  latency(s) {c['control_latency_s']}")
    print(f"\nper-case detail -> eval/security_report_{BACKEND}.json")


if __name__ == "__main__":
    main()
