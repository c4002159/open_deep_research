"""Agent-facing business actions. The agent can *start* a refund; it has no tool that finishes one.

Identity comes from the authenticated request (``config["configurable"]["requested_by"]``, set by the service),
never from a tool argument, and never from an environment variable.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import date
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from open_deep_research import refunds
from open_deep_research.configuration import Configuration

_ready: set[str] = set()
_ready_lock = threading.Lock()


def _db_path(configurable: Configuration) -> str:
    return refunds.sqlite_path(configurable.business_db_url or configurable.internal_db_url or "")


def _prepare(path: str) -> None:
    with _ready_lock:
        if path not in _ready:
            refunds.ensure_schema(path)
            _ready.add(path)


def _today(configurable: Configuration) -> date:
    return date.fromisoformat(configurable.business_date) if configurable.business_date else date.today()


def _requester(config: Optional[RunnableConfig]) -> str:
    return str((config or {}).get("configurable", {}).get("requested_by") or "unknown")


PROPOSE_REFUND_DESCRIPTION = (
    "Create a refund PROPOSAL for one order. It does NOT refund anything: human approvers (manager, finance or "
    "director, depending on the amount) must sign it afterwards. The system checks the order, the amount, the 30-day "
    "window and duplicates and may refuse; a refusal is final, do not alter the facts to get around it. "
    "Look the order up with sql_executor first; never guess an order id or amount. Set service_outage_over_72h=true "
    "whenever the user states or reports (even second-hand) that the service was down for more than 72 hours: "
    "approvers verify that claim. Never set it otherwise."
)


@tool(description=PROPOSE_REFUND_DESCRIPTION)
async def propose_refund(order_id: int, amount: float, reason: str, service_outage_over_72h: bool = False,
                         config: RunnableConfig = None) -> str:
    """Create a pending refund proposal.

    Args:
        order_id: Id of the order in the sales table (verified against the database).
        amount: Amount to refund; at most the order's revenue.
        reason: Why the customer is being refunded: the user's own words, verbatim, in the user's language - no translation or paraphrase (max 300 characters).
        service_outage_over_72h: True whenever the user states or reports an outage of more than 72 hours (approvers verify it).
        config: Runtime configuration (injected).
    """
    configurable = Configuration.from_runnable_config(config)
    try:
        path = _db_path(configurable)
        await asyncio.to_thread(_prepare, path)
        out = await asyncio.to_thread(
            refunds.propose, path, order_id=order_id, amount=amount, reason=reason,
            outage_claim=service_outage_over_72h, requested_by=_requester(config), today=_today(configurable),
            thread_id=str((config or {}).get("configurable", {}).get("thread_id", "")))
    except ValueError as e:
        return f"Error: the refund workflow is not available ({e})"
    except Exception as e:                       # noqa: BLE001 - tell the model, never crash the graph
        return f"Error: could not create the refund proposal ({type(e).__name__}: {str(e)[:200]})"
    if not out.ok:
        return f"Proposal NOT created: {out.message}"
    r = out.refund
    head = (f"Refund proposal #{r['id']} already exists for order {r['order_id']}" if out.existing
            else f"Refund proposal #{r['id']} created for order {r['order_id']}")
    return (f"{head}: amount {r['amount']:,.2f}, status {r['status'].upper()}. "
            f"Required approvers: {' + '.join(r['required_roles'])} (still missing: {', '.join(r['missing_roles']) or 'none'}). "
            "Nothing has been refunded yet; it takes effect only after those people approve it.")


WITHDRAW_REFUND_DESCRIPTION = (
    "Withdraw the current user's own PENDING refund proposal by its id (use when the user changes their mind). "
    "It cannot touch anyone else's proposal or one that is already decided."
)


@tool(description=WITHDRAW_REFUND_DESCRIPTION)
async def withdraw_refund(refund_id: int, config: RunnableConfig = None) -> str:
    """Withdraw the requesting user's own pending refund proposal.

    Args:
        refund_id: Id of the proposal (from propose_refund or the refunds table).
        config: Runtime configuration (injected).
    """
    configurable = Configuration.from_runnable_config(config)
    try:
        path = _db_path(configurable)
        await asyncio.to_thread(_prepare, path)
        out = await asyncio.to_thread(refunds.withdraw, path, refund_id=refund_id, user=_requester(config))
    except ValueError as e:
        return f"Error: the refund workflow is not available ({e})"
    except Exception as e:                       # noqa: BLE001
        return f"Error: could not withdraw the proposal ({type(e).__name__}: {str(e)[:200]})"
    return f"Refund proposal #{refund_id} withdrawn." if out.ok else f"Not withdrawn: {out.message}"


ACTION_TOOLS = [propose_refund, withdraw_refund]
