"""Refund workflow: the one business write the agent may *start* but can never *finish*.

Rules come from docs/internal_kb/refundPolicy.md and are enforced here, in code, not in a prompt:

  * a refund must be requested within 30 days of the order (or claim a >72h outage, which an approver verifies)
  * the amount cannot exceed the order's revenue; one live refund per order (unique index)
  * the amount decides who must sign:  < 5,000 manager | 5,000-50,000 manager + finance | > 50,000 director
  * nobody approves their own request (separation of duties)
  * approval is atomic (BEGIN IMMEDIATE); every step lands in an append-only event table
  * an applied refund can be reversed by finance/director within 24 hours

The agent's only capabilities are ``propose`` (creates a ``pending`` row) and ``withdraw`` (its own pending row).
``decide`` and ``reverse`` are reachable only through the HTTP service, where the actor and role come from the
authenticated token. SQLite only: the write path is deliberately a small module with no "run arbitrary SQL" entry.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Optional

ROLES = ("manager", "finance", "director")
WINDOW_DAYS = 30
REVERSE_WINDOW_H = 24
REASON_MAX = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    amount REAL NOT NULL CHECK (amount > 0),
    reason TEXT NOT NULL,
    outage_claim INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('pending','applied','rejected','withdrawn','reversed')),
    required_roles TEXT NOT NULL,
    basis TEXT NOT NULL DEFAULT '[]',
    requested_by TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    decided_at TEXT,
    thread_id TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (order_id) REFERENCES sales(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_live_refund ON refunds(order_id) WHERE status IN ('pending','applied');
CREATE TABLE IF NOT EXISTS refund_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    refund_id INTEGER NOT NULL,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (refund_id) REFERENCES refunds(id)
);
"""


@dataclass
class Outcome:
    ok: bool
    message: str
    refund: Optional[dict] = None
    existing: bool = False
    code: str = ""          # not_found | forbidden | conflict | invalid  (lets the HTTP layer pick a status)


def required_roles(amount: float) -> list[str]:
    if amount < 5_000:
        return ["manager"]
    if amount <= 50_000:
        return ["manager", "finance"]
    return ["director"]


def sqlite_path(db_url: str) -> str:
    prefix = "sqlite:///"
    if not db_url or not db_url.startswith(prefix):
        raise ValueError("business writes are supported on SQLite only (sqlite:///path/to.db)")
    return db_url[len(prefix):]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def _tx(path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def ensure_schema(path: str) -> None:
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _event(conn, refund_id: int, actor: str, role: str, action: str, note: str = "") -> None:
    conn.execute("INSERT INTO refund_events (refund_id, at, actor, role, action, note) VALUES (?,?,?,?,?,?)",
                 (refund_id, _now(), actor, role, action, note))


def _shape(conn, row: sqlite3.Row, with_events: bool = False) -> dict:
    required = json.loads(row["required_roles"])
    signed = [r["role"] for r in conn.execute(
        "SELECT role FROM refund_events WHERE refund_id=? AND action='approve' ORDER BY id", (row["id"],))]
    out = {"id": row["id"], "order_id": row["order_id"], "amount": row["amount"], "reason": row["reason"],
           "outage_claim": bool(row["outage_claim"]), "status": row["status"], "required_roles": required,
           "signed_roles": signed, "missing_roles": [r for r in required if r not in signed],
           "basis": json.loads(row["basis"]), "requested_by": row["requested_by"],
           "requested_at": row["requested_at"], "decided_at": row["decided_at"], "thread_id": row["thread_id"]}
    if with_events:
        out["events"] = [dict(e) for e in conn.execute(
            "SELECT at, actor, role, action, note FROM refund_events WHERE refund_id=? ORDER BY id", (row["id"],))]
    return out


def _get_row(conn, refund_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM refunds WHERE id=?", (refund_id,)).fetchone()


# ----------------------------------------------------------------------------- agent-facing
def propose(path: str, *, order_id: int, amount: float, reason: str, outage_claim: bool, requested_by: str,
            today: date, thread_id: str = "") -> Outcome:
    """Validate a refund request and, if it passes, store it as ``pending``. Never refunds anything."""
    if not requested_by or requested_by == "unknown":
        return Outcome(False, "no authenticated requester: refund proposals can only be made on behalf of a signed-in user")
    reason = (reason or "").strip()
    if not reason:
        return Outcome(False, "a reason is required")
    if len(reason) > REASON_MAX:
        return Outcome(False, f"reason is too long ({len(reason)} > {REASON_MAX} characters)")
    try:
        amount = round(float(amount), 2)
        order_id = int(order_id)
    except (TypeError, ValueError):
        return Outcome(False, "order_id must be an integer and amount a number")
    if not math.isfinite(amount) or amount <= 0:
        return Outcome(False, "amount must be a positive number")

    try:
        with _tx(path) as conn:
            order = conn.execute("SELECT id, date, product_line, region, revenue FROM sales WHERE id=?", (order_id,)).fetchone()
            if order is None:
                return Outcome(False, f"order {order_id} does not exist; look the order up with sql_executor first")
            if amount > order["revenue"] + 1e-9:
                return Outcome(False, f"refund amount {amount:,.2f} exceeds the order's revenue {order['revenue']:,.2f}")

            live = conn.execute("SELECT * FROM refunds WHERE order_id=? AND status IN ('pending','applied')", (order_id,)).fetchone()
            if live is not None:
                return Outcome(True, f"order {order_id} already has a {live['status']} refund (#{live['id']}); nothing new was created",
                               _shape(conn, live), existing=True)

            age = (today - date.fromisoformat(order["date"])).days
            basis = []
            if age <= WINDOW_DAYS:
                basis.append(f"订单日期 {order['date']}，距业务日期 {age} 天，在 {WINDOW_DAYS} 天退款期内（退款规范 §1）")
            elif outage_claim:
                basis.append(f"订单已超 {WINDOW_DAYS} 天（{age} 天）；申请人称服务中断超过 72 小时——"
                             "该说法未经系统核实，审批人必须核实（退款规范 §1）")
            else:
                return Outcome(False, f"order {order_id} is {age} days old, beyond the {WINDOW_DAYS}-day refund window, and no "
                                      "service outage over 72 hours was claimed (refund policy §1). Do not change the facts to get "
                                      "around this; tell the user the request is not eligible.")
            roles = required_roles(amount)
            basis.append("审批层级：" + " + ".join(roles) + "（退款规范 §2：<5,000 经理；5,000–50,000 经理+财务；>50,000 财务总监）")
            basis.append("生效后对应销售的提成将在下一季度按比例扣回（退款规范 §3）")

            cur = conn.execute(
                "INSERT INTO refunds (order_id, amount, reason, outage_claim, status, required_roles, basis, requested_by,"
                " requested_at, thread_id) VALUES (?,?,?,?, 'pending', ?,?,?,?,?)",
                (order_id, amount, reason, int(outage_claim), json.dumps(roles), json.dumps(basis, ensure_ascii=False),
                 requested_by, _now(), thread_id))
            rid = cur.lastrowid
            _event(conn, rid, requested_by, "", "propose", reason)
            return Outcome(True, "proposal created", _shape(conn, _get_row(conn, rid)))
    except sqlite3.IntegrityError:        # lost a race against an identical concurrent proposal
        return propose(path, order_id=order_id, amount=amount, reason=reason, outage_claim=outage_claim,
                       requested_by=requested_by, today=today, thread_id=thread_id)


def withdraw(path: str, *, refund_id: int, user: str) -> Outcome:
    with _tx(path) as conn:
        row = _get_row(conn, int(refund_id))
        if row is None or row["requested_by"] != user:     # same answer for "missing" and "someone else's"
            return Outcome(False, f"refund {refund_id} not found")
        if row["status"] != "pending":
            return Outcome(False, f"refund {refund_id} is {row['status']}; only a pending proposal can be withdrawn")
        conn.execute("UPDATE refunds SET status='withdrawn', decided_at=? WHERE id=?", (_now(), row["id"]))
        _event(conn, row["id"], user, "", "withdraw")
        return Outcome(True, "withdrawn", _shape(conn, _get_row(conn, row["id"])))


# ----------------------------------------------------------------------------- human-facing (HTTP service only)
def decide(path: str, *, refund_id: int, user: str, role: str, approve: bool, note: str = "") -> Outcome:
    note = (note or "").strip()
    if role not in ROLES:
        return Outcome(False, "you do not hold an approver role", code="forbidden")
    if not approve and not note:
        return Outcome(False, "a reason is required when rejecting", code="invalid")
    with _tx(path) as conn:
        row = _get_row(conn, int(refund_id))
        if row is None:
            return Outcome(False, "refund not found", code="not_found")
        if row["status"] != "pending":
            return Outcome(False, f"refund is {row['status']}, not pending", code="conflict")
        if row["requested_by"] == user:
            return Outcome(False, "you cannot decide your own request (separation of duties)", code="forbidden")
        required = json.loads(row["required_roles"])
        if role not in required:
            return Outcome(False, f"this refund needs {' + '.join(required)}; your role ({role}) cannot sign it", code="forbidden")
        signed = {r["role"] for r in conn.execute(
            "SELECT role FROM refund_events WHERE refund_id=? AND action='approve'", (row["id"],))}
        if role in signed:
            return Outcome(False, f"the {role} signature is already recorded", code="conflict")
        if not approve:
            conn.execute("UPDATE refunds SET status='rejected', decided_at=? WHERE id=?", (_now(), row["id"]))
            _event(conn, row["id"], user, role, "reject", note)
        else:
            _event(conn, row["id"], user, role, "approve", note)
            if set(required) <= signed | {role}:
                conn.execute("UPDATE refunds SET status='applied', decided_at=? WHERE id=?", (_now(), row["id"]))
                _event(conn, row["id"], "system", "", "applied", "all required signatures present")
        return Outcome(True, "recorded", _shape(conn, _get_row(conn, row["id"])))


def reverse(path: str, *, refund_id: int, user: str, role: str, note: str = "", now: Optional[datetime] = None) -> Outcome:
    note = (note or "").strip()
    if role not in ("finance", "director"):
        return Outcome(False, "only finance or director can reverse an applied refund", code="forbidden")
    if not note:
        return Outcome(False, "a reason is required to reverse a refund", code="invalid")
    now = now or datetime.now(timezone.utc)
    with _tx(path) as conn:
        row = _get_row(conn, int(refund_id))
        if row is None:
            return Outcome(False, "refund not found", code="not_found")
        if row["status"] != "applied":
            return Outcome(False, f"refund is {row['status']}; only an applied refund can be reversed", code="conflict")
        decided = datetime.strptime(row["decided_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if now - decided > timedelta(hours=REVERSE_WINDOW_H):
            return Outcome(False, f"the {REVERSE_WINDOW_H}-hour reversal window has passed", code="conflict")
        conn.execute("UPDATE refunds SET status='reversed', decided_at=? WHERE id=?", (now.strftime("%Y-%m-%dT%H:%M:%SZ"), row["id"]))
        _event(conn, row["id"], user, role, "reverse", note)
        return Outcome(True, "reversed", _shape(conn, _get_row(conn, row["id"])))


# ----------------------------------------------------------------------------- reads
def get(path: str, refund_id: int) -> Optional[dict]:
    with _tx(path) as conn:
        row = _get_row(conn, int(refund_id))
        return _shape(conn, row, with_events=True) if row else None


def list_refunds(path: str, *, status: Optional[str] = None, requested_by: Optional[str] = None,
                 awaiting_role: Optional[str] = None, limit: int = 50) -> list[dict]:
    where, args = [], []
    if status:
        where.append("status=?"); args.append(status)
    if requested_by:
        where.append("requested_by=?"); args.append(requested_by)
    sql = "SELECT * FROM refunds" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
    with _tx(path) as conn:
        out = [_shape(conn, r) for r in conn.execute(sql, (*args, max(1, min(limit, 200))))]
    if awaiting_role:
        out = [r for r in out if r["status"] == "pending" and awaiting_role in r["missing_roles"]]
    return out


def audit_trail(path: str, limit: int = 100) -> list[dict]:
    with _tx(path) as conn:
        rows = conn.execute(
            "SELECT e.id, e.at, e.actor, e.role, e.action, e.note, e.refund_id, r.order_id, r.amount, r.status "
            "FROM refund_events e JOIN refunds r ON r.id=e.refund_id ORDER BY e.id DESC LIMIT ?",
            (max(1, min(limit, 500)),))
        return [dict(r) for r in rows]
