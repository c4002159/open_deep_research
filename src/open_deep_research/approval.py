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
import time
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
