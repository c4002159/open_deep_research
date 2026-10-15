"""Human-in-the-loop approval for tool calls that expose confidential detail.

Rule (from docs/internal_kb/dataPermission.md): single-order amounts are "机密级". An
aggregate answer ("total revenue per region") is fine; a dump of individual orders needs
a human to say yes. So a ``sql_executor`` call needs approval when it is

  * a plain SELECT touching the ``sales`` table,
  * with no aggregation / GROUP BY / DISTINCT (i.e. it returns individual orders), and
  * no LIMIT, or a LIMIT above ``approval_row_limit``.

The check runs *before* any tool of the batch executes, so pausing (``interrupt``) never
leaves half-finished side effects behind and the node can be replayed safely on resume.
Every decision is appended to a JSONL audit log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

logger = logging.getLogger(__name__)

SENSITIVE_TABLES = frozenset({"sales"})


def _is_aggregated(select: exp.Select) -> bool:
    if select.args.get("group") or select.args.get("distinct"):
        return True
    for node in select.find_all(exp.AggFunc):
        # SUM(x) OVER (...) is a window function: still one output row per input row
        if node.find_ancestor(exp.Window) is not None:
            continue
        # aggregates inside a subquery do not make *this* select aggregated
        if node.find_ancestor(exp.Select) is select:
            return True
    return False


def _limit_value(select: exp.Select) -> Optional[int]:
    limit = select.args.get("limit")
    if limit is None:
        return None
    try:
        return int(limit.expression.name)
    except (AttributeError, ValueError, TypeError):
        return None  # LIMIT with an expression we cannot evaluate -> treat as unbounded


def sql_approval_reason(sql: str, row_limit: int = 10, dialect: Optional[str] = None) -> Optional[str]:
    """Why ``sql`` needs human approval, or None."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError:
        return None  # unparsable queries are rejected by the read-only guard anyway
    if not isinstance(tree, exp.Select):
        return None
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    if not tables & SENSITIVE_TABLES or _is_aggregated(tree):
        return None
    limit = _limit_value(tree)
    if limit is not None and limit <= row_limit:
        return None
    shown = "all matching rows" if limit is None else f"up to {limit} rows"
    return (f"Row-level order data (confidential per the data-permission policy): the query returns "
            f"individual orders from {', '.join(sorted(tables & SENSITIVE_TABLES))} — {shown}")


def approval_reason(tool_name: str, args: dict, row_limit: int = 10) -> Optional[str]:
    if tool_name == "sql_executor":
        return sql_approval_reason(str(args.get("query", "")), row_limit)
    return None


def sql_row_cost(sql: str, dialect: Optional[str] = None) -> Optional[int]:
    """Upper bound on individual-order rows a *bounded* row-level SELECT on a sensitive table can return.

    ``None`` for anything else (aggregates, other tables, unparsable, or no LIMIT - which needs approval anyway).
    Declared LIMITs are known before the query runs, so no result row has to be inspected.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError:
        return None
    if not isinstance(tree, exp.Select):
        return None
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    if not tables & SENSITIVE_TABLES or _is_aggregated(tree):
        return None
    return _limit_value(tree)


class RowBudget:
    """Row-level rows read *without* approval, per task (thread).

    The approval rule looks at one query's shape, so a model that always writes ``LIMIT 5`` could page through
    the whole table five rows at a time. This adds up the declared LIMITs of a task's row-level queries; once the
    total passes the per-query threshold the next one needs a human too. Charging is idempotent per tool-call id
    (the gate node is replayed when a paused task resumes) and an approval resets the task's total.
    Process-local: across several hosts each keeps its own count.
    """

    def __init__(self, max_threads: int = 2000) -> None:
        self._by_thread: "OrderedDict[str, dict[str, int]]" = OrderedDict()
        self._max, self._lock = max_threads, threading.Lock()

    def charge(self, thread_id: str, call_id: str, cost: int) -> int:
        with self._lock:
            calls = self._by_thread.setdefault(thread_id, {})
            self._by_thread.move_to_end(thread_id)
            calls.setdefault(call_id, cost)
            while len(self._by_thread) > self._max:
                self._by_thread.popitem(last=False)
            return sum(calls.values())

    def reset(self, thread_id: str) -> None:
        with self._lock:
            self._by_thread.pop(thread_id, None)


ROW_BUDGET = RowBudget()


def collect_pending(tool_calls: list[dict], *, row_limit: int, thread_id: str, budget: RowBudget = ROW_BUDGET) -> list[dict]:
    """Approval requests for a batch of tool calls: the per-query rule plus the per-task running total."""
    pending = []
    for tc in tool_calls:
        reason = approval_reason(tc["name"], tc["args"], row_limit)
        cumulative = False
        if not reason and tc["name"] == "sql_executor":
            cost = sql_row_cost(str(tc["args"].get("query", "")))
            if cost:
                total = budget.charge(thread_id, tc["id"], cost)
                if total > row_limit:
                    cumulative = True
                    reason = (f"Cumulative row-level order reads in this task: this query raises the total to up to {total} "
                              f"individual orders (more than {row_limit} without approval)")
        if reason:
            pending.append({"tool": tc["name"], "args": tc["args"], "reason": reason, "id": tc["id"], "cumulative": cumulative})
    return pending


def audit(path: str, record: dict) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}, ensure_ascii=False) + "\n")
    except OSError as e:  # never let logging break the agent, but do not hide it either
        logger.error("could not write approval audit log: %s", e)


def _normalise(decision: Any) -> dict:
    if isinstance(decision, dict):
        return {"approved": bool(decision.get("approved")), "reviewer": decision.get("reviewer", "unknown"),
                "note": decision.get("note", "")}
    return {"approved": bool(decision), "reviewer": "unknown", "note": ""}


def decide(mode: str, requests: list[dict], *, thread_id: str = "", log_path: str = "logs/approvals.jsonl",
           interrupt_fn=None) -> dict:
    """Resolve a batch of approval requests (each ``{"tool", "args", "reason"}``).

    ``interrupt_fn`` is langgraph's ``interrupt`` (injected so the logic is testable).
    A missing checkpointer makes interrupt() unusable; that fails *closed* (denied).
    """
    if mode == "auto_approve":
        result = {"approved": True, "reviewer": "auto", "note": "auto_approve mode"}
    elif mode == "deny":
        result = {"approved": False, "reviewer": "policy", "note": "approval_mode=deny"}
    else:
        payload = {"type": "approval_required", "requests": requests,
                   "instructions": "Resume with {'approved': true|false, 'reviewer': '<name>', 'note': '<why>'}"}
        try:
            result = _normalise(interrupt_fn(payload))
        except RuntimeError as e:  # GraphInterrupt is not a RuntimeError, so a real pause still propagates
            result = {"approved": False, "reviewer": "system",
                      "note": f"cannot pause for approval (no checkpointer?): {e}"}

    for r in requests:
        audit(log_path, {"thread_id": thread_id, "tool": r["tool"], "reason": r["reason"],
                         "query_sha256": hashlib.sha256(str(r["args"].get("query", "")).encode()).hexdigest()[:16],
                         "query": str(r["args"].get("query", ""))[:500], "mode": mode, **result})
    return result
