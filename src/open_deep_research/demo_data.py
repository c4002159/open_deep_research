"""Demo database and demo accounts for the refund workflow.

The tracked ``test.db`` stays untouched (evaluations depend on it). The demo works on a copy that also holds
the ``refunds`` tables and, optionally, a little history so the inbox and audit pages are not empty on first open.

Demo "today" is 2026-06-10: the sample orders run from January to 8 June 2026.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from open_deep_research import refunds as RF

BUSINESS_DATE = "2026-06-10"

# token -> user. Plain strings on purpose: this is demo data, and /demo/users only exists with ODR_DEMO_MODE=1.
DEMO_KEYS = {"demo-alice": "alice", "demo-bob": "bob", "demo-carol": "carol", "demo-dave": "dave", "demo-root": "root"}
DEMO_ROLES = {"bob": "manager", "carol": "finance", "dave": "director"}
DEMO_PEOPLE = {
    "alice": "销售代表 · 提需求的人",
    "bob": "区域销售经理 · 可签 5,000 元以下，或与财务双签",
    "carol": "财务经理 · 双签中的财务",
    "dave": "财务总监 · 签 50,000 元以上",
    "root": "管理员 · 可看用量看板和审计",
}


def build_demo_db(dest: str, source: str = "test.db", seed: bool = False) -> str:
    """Copy ``source`` to ``dest`` (replacing it), add the refund tables, optionally seed history."""
    src, dst = Path(source), Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"source database not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    RF.ensure_schema(str(dst))
    if seed:
        _seed(str(dst))
    return str(dst)


def _seed(db: str) -> None:
    today = date.fromisoformat(BUSINESS_DATE)

    def propose(order: int, amount: float, reason: str) -> int:
        out = RF.propose(db, order_id=order, amount=amount, reason=reason, outage_claim=False,
                         requested_by="alice", today=today, thread_id="seed")
        assert out.ok, out.message
        return out.refund["id"]

    done = propose(66, 14191.66, "客户重复下单，申请退回其中一笔")
    RF.decide(db, refund_id=done, user="bob", role="manager", approve=True, note="核对属实")
    RF.decide(db, refund_id=done, user="carol", role="finance", approve=True, note="已确认未开票")

    refused = propose(68, 19227.97, "客户觉得功能不合适")
    RF.decide(db, refund_id=refused, user="bob", role="manager", approve=False,
              note="订阅已激活，只能按剩余月份部分退款（退款规范 §1）")

    propose(67, 16952.65, "客户反馈数据平台无法接入其系统")      # left pending: shows up in the manager's inbox
