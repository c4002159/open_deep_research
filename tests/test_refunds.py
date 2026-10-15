"""Refund workflow rules, enforced in code. Real SQLite files, no mocks."""

import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from open_deep_research import refunds as R

TODAY = date(2026, 6, 10)


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "biz.db")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sales (id INTEGER PRIMARY KEY, date TEXT, product_line TEXT, region TEXT, revenue REAL, units INTEGER);
        INSERT INTO sales VALUES (1, '2026-06-01', 'AI Tools',      'North',   3000.00, 10);   -- recent, small
        INSERT INTO sales VALUES (2, '2026-06-05', 'Cloud Services','East',   20000.00, 50);   -- recent, mid
        INSERT INTO sales VALUES (3, '2026-06-08', 'Data Platform', 'West',   80000.00, 90);   -- recent, large
        INSERT INTO sales VALUES (4, '2026-01-10', 'AI Tools',      'South',   9000.00, 30);   -- old
        INSERT INTO sales VALUES (5, '2026-05-11', 'AI Tools',      'South',   1000.00,  3);   -- exactly 30 days
    """)
    con.commit(); con.close()
    R.ensure_schema(path)
    return path


def propose(db, order=1, amount=1000.0, reason="customer unhappy", outage=False, user="alice"):
    return R.propose(db, order_id=order, amount=amount, reason=reason, outage_claim=outage, requested_by=user, today=TODAY)


# ------------------------------------------------------------------ tiers
@pytest.mark.parametrize("amount,roles", [
    (0.01, ["manager"]), (4999.99, ["manager"]),
    (5000, ["manager", "finance"]), (50000, ["manager", "finance"]),
    (50000.01, ["director"]), (1e9, ["director"]),
])
def test_approval_tier_boundaries(amount, roles):
    assert R.required_roles(amount) == roles


# ------------------------------------------------------------------ propose
def test_a_valid_proposal_is_pending_with_the_right_signers_and_policy_basis(db):
    o = propose(db, order=2, amount=20000)
    assert o.ok and o.refund["status"] == "pending" and o.refund["required_roles"] == ["manager", "finance"]
    assert o.refund["missing_roles"] == ["manager", "finance"] and o.refund["requested_by"] == "alice"
    assert any("30 天" in b for b in o.refund["basis"]) and any("manager + finance" in b for b in o.refund["basis"])


def test_nothing_is_refunded_by_proposing(db):
    propose(db)
    assert R.get(db, 1)["status"] == "pending" and R.get(db, 1)["decided_at"] is None


@pytest.mark.parametrize("kwargs,fragment", [
    (dict(order=999), "does not exist"),
    (dict(amount=3000.01), "exceeds"),
    (dict(amount=0), "positive"),
    (dict(amount=-5), "positive"),
    (dict(amount=float("nan")), "positive"),
    (dict(amount=float("inf")), "positive"),
    (dict(reason="  "), "reason is required"),
    (dict(reason="x" * 301), "too long"),
])
def test_invalid_proposals_are_refused_and_create_nothing(db, kwargs, fragment):
    o = propose(db, **kwargs)
    assert not o.ok and fragment in o.message
    assert R.list_refunds(db) == []


def test_a_proposal_needs_an_authenticated_requester(db):
    for who in ("", "unknown"):
        assert not propose(db, user=who).ok


def test_refund_window_is_30_days_inclusive(db):
    assert propose(db, order=5, amount=1000).ok                # 30 days old: allowed
    assert not propose(db, order=4, amount=1000).ok            # 151 days old: refused


def test_expired_order_is_refused_with_a_message_that_discourages_workarounds(db):
    o = propose(db, order=4)
    assert not o.ok and "beyond the 30-day" in o.message and "Do not change the facts" in o.message


def test_an_outage_claim_unlocks_an_old_order_but_is_flagged_as_unverified(db):
    o = propose(db, order=4, amount=5000, outage=True)
    assert o.ok and o.refund["outage_claim"] is True
    assert any("未经系统核实" in b for b in o.refund["basis"])


def test_proposing_twice_is_idempotent(db):
    first = propose(db, order=1, amount=1000)
    again = propose(db, order=1, amount=1000)
    assert again.ok and again.existing and again.refund["id"] == first.refund["id"]
    assert len(R.list_refunds(db)) == 1


def test_concurrent_identical_proposals_create_exactly_one_refund(db):
    results = []
    ts = [threading.Thread(target=lambda: results.append(propose(db, order=2, amount=2000))) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(r.ok for r in results) and len(R.list_refunds(db)) == 1


# ------------------------------------------------------------------ decide
def test_small_refund_needs_only_a_manager(db):
    rid = propose(db, order=1, amount=1000).refund["id"]
    o = R.decide(db, refund_id=rid, user="bob", role="manager", approve=True)
    assert o.ok and o.refund["status"] == "applied" and o.refund["decided_at"]


def test_mid_refund_needs_manager_and_finance_in_any_order(db):
    rid = propose(db, order=2, amount=20000).refund["id"]
    first = R.decide(db, refund_id=rid, user="carol", role="finance", approve=True)
    assert first.refund["status"] == "pending" and first.refund["missing_roles"] == ["manager"]
    second = R.decide(db, refund_id=rid, user="bob", role="manager", approve=True)
    assert second.refund["status"] == "applied"
    actions = [e["action"] for e in R.get(db, rid)["events"]]
    assert actions == ["propose", "approve", "approve", "applied"]


def test_large_refund_needs_the_director_and_a_manager_cannot_sign_it(db):
    rid = propose(db, order=3, amount=60000).refund["id"]
    no = R.decide(db, refund_id=rid, user="bob", role="manager", approve=True)
    assert not no.ok and "director" in no.message
    assert R.decide(db, refund_id=rid, user="dave", role="director", approve=True).refund["status"] == "applied"


def test_you_cannot_decide_your_own_request(db):
    rid = propose(db, order=1, amount=1000, user="bob").refund["id"]
    o = R.decide(db, refund_id=rid, user="bob", role="manager", approve=True)
    assert not o.ok and "separation of duties" in o.message and R.get(db, rid)["status"] == "pending"


def test_a_role_cannot_sign_twice(db):
    rid = propose(db, order=2, amount=20000).refund["id"]
    R.decide(db, refund_id=rid, user="bob", role="manager", approve=True)
    again = R.decide(db, refund_id=rid, user="bob2", role="manager", approve=True)
    assert not again.ok and "already recorded" in again.message


def test_reject_needs_a_reason_and_ends_the_request(db):
    rid = propose(db, order=2, amount=20000).refund["id"]
    assert not R.decide(db, refund_id=rid, user="bob", role="manager", approve=False).ok
    o = R.decide(db, refund_id=rid, user="bob", role="manager", approve=False, note="outside policy")
    assert o.ok and o.refund["status"] == "rejected"
    assert not R.decide(db, refund_id=rid, user="carol", role="finance", approve=True).ok     # closed for good


def test_a_rejected_order_can_be_proposed_again_but_an_applied_one_cannot(db):
    rid = propose(db, order=1, amount=1000).refund["id"]
    R.decide(db, refund_id=rid, user="bob", role="manager", approve=False, note="no")
    second = propose(db, order=1, amount=1000)
    assert second.ok and not second.existing and second.refund["id"] != rid
    R.decide(db, refund_id=second.refund["id"], user="bob", role="manager", approve=True)
    third = propose(db, order=1, amount=1000)
    assert third.existing and third.refund["status"] == "applied"


def test_unknown_role_and_missing_refund(db):
    rid = propose(db).refund["id"]
    assert not R.decide(db, refund_id=rid, user="x", role="intern", approve=True).ok
    assert not R.decide(db, refund_id=404, user="bob", role="manager", approve=True).ok


def test_concurrent_final_signatures_apply_exactly_once(db):
    rid = propose(db, order=2, amount=20000).refund["id"]
    out = []
    ts = [threading.Thread(target=lambda u=u, r=r: out.append(R.decide(db, refund_id=rid, user=u, role=r, approve=True)))
          for u, r in [("bob", "manager"), ("carol", "finance")]]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(o.ok for o in out)
    ev = [e["action"] for e in R.get(db, rid)["events"]]
    assert ev.count("applied") == 1 and R.get(db, rid)["status"] == "applied"


def test_concurrent_same_role_signatures_record_only_one(db):
    rid = propose(db, order=2, amount=20000).refund["id"]
    out = []
    ts = [threading.Thread(target=lambda u=u: out.append(R.decide(db, refund_id=rid, user=u, role="manager", approve=True)))
          for u in ("bob", "bob2", "bob3")]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sum(o.ok for o in out) == 1
    assert [e["role"] for e in R.get(db, rid)["events"] if e["action"] == "approve"] == ["manager"]


# ------------------------------------------------------------------ withdraw / reverse
def test_only_the_requester_can_withdraw_and_only_while_pending(db):
    rid = propose(db).refund["id"]
    assert not R.withdraw(db, refund_id=rid, user="mallory").ok
    assert R.withdraw(db, refund_id=rid, user="alice").refund["status"] == "withdrawn"
    assert not R.withdraw(db, refund_id=rid, user="alice").ok
    assert propose(db).ok                                      # the order is free again


def test_reverse_rules(db):
    rid = propose(db, order=1, amount=1000).refund["id"]
    assert not R.reverse(db, refund_id=rid, user="carol", role="finance", note="x").ok          # not applied yet
    R.decide(db, refund_id=rid, user="bob", role="manager", approve=True)
    assert not R.reverse(db, refund_id=rid, user="bob", role="manager", note="x").ok            # wrong role
    assert not R.reverse(db, refund_id=rid, user="carol", role="finance", note="").ok           # reason required
    late = datetime.now(timezone.utc) + timedelta(hours=25)
    assert "window" in R.reverse(db, refund_id=rid, user="carol", role="finance", note="x", now=late).message
    ok = R.reverse(db, refund_id=rid, user="carol", role="finance", note="customer kept the product")
    assert ok.ok and ok.refund["status"] == "reversed"
    assert propose(db, order=1, amount=1000).ok                # a reversed refund frees the order


# ------------------------------------------------------------------ reads
def test_inbox_filters_by_the_role_that_still_has_to_sign(db):
    a = propose(db, order=2, amount=20000).refund["id"]
    b = propose(db, order=1, amount=1000).refund["id"]
    R.decide(db, refund_id=a, user="bob", role="manager", approve=True)
    assert [r["id"] for r in R.list_refunds(db, awaiting_role="manager")] == [b]
    assert [r["id"] for r in R.list_refunds(db, awaiting_role="finance")] == [a]
    assert [r["id"] for r in R.list_refunds(db, requested_by="alice", status="pending")] == [b, a]


def test_audit_trail_is_complete_and_newest_first(db):
    rid = propose(db, order=1, amount=1000).refund["id"]
    R.decide(db, refund_id=rid, user="bob", role="manager", approve=True, note="ok")
    trail = R.audit_trail(db)
    assert [e["action"] for e in trail] == ["applied", "approve", "propose"]
    assert trail[1]["actor"] == "bob" and trail[1]["note"] == "ok" and trail[2]["actor"] == "alice"


def test_schema_is_idempotent_and_non_sqlite_urls_are_refused(db):
    R.ensure_schema(db); R.ensure_schema(db)
    assert R.sqlite_path("sqlite:///x/y.db") == "x/y.db"
    with pytest.raises(ValueError):
        R.sqlite_path("postgresql://u:p@h/db")
