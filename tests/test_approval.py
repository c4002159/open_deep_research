"""Human-approval gate: classification rules + the real researcher_tools node wiring (offline)."""

import asyncio
import json
import warnings

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research import deep_researcher as dr
from open_deep_research.approval import sql_approval_reason
from open_deep_research.state import ResearcherState

warnings.filterwarnings("ignore")

# ------------------------------------------------------------ classification
NEEDS = [
    "SELECT * FROM sales",
    "SELECT id, revenue FROM sales WHERE region = 'East'",
    "SELECT * FROM sales LIMIT 100",
    "SELECT s.id, p.category FROM sales s JOIN products p ON s.product_line = p.product_line",
    "SELECT id, SUM(revenue) OVER (PARTITION BY region) FROM sales",             # window, still row-level
    "WITH x AS (SELECT * FROM sales) SELECT * FROM x",
]
FREE = [
    "SELECT COUNT(*) FROM sales",
    "SELECT region, SUM(revenue) FROM sales GROUP BY region",
    "SELECT * FROM sales ORDER BY revenue DESC LIMIT 3",                          # small top-N
    "SELECT DISTINCT region FROM sales",
    "SELECT name FROM sqlite_master WHERE type = 'table'",
    "SELECT * FROM products",                                                      # not a sensitive table
    "SELECT MAX(revenue) FROM sales",
]


@pytest.mark.parametrize("sql", NEEDS)
def test_row_level_queries_need_approval(sql):
    assert sql_approval_reason(sql) is not None


@pytest.mark.parametrize("sql", FREE)
def test_aggregates_and_small_topn_do_not(sql):
    assert sql_approval_reason(sql) is None


def test_row_limit_is_configurable():
    assert sql_approval_reason("SELECT * FROM sales LIMIT 50", row_limit=100) is None
    assert sql_approval_reason("SELECT * FROM sales LIMIT 50", row_limit=10) is not None


# ------------------------------------------------------------ wiring through the real node
class Env:
    """A one-node graph around the real `researcher_tools`, with a counting fake sql_executor."""

    def __init__(self, monkeypatch, tmp_path, with_checkpointer=True):
        self.executed = []

        @tool
        async def sql_executor(query: str) -> str:
            """fake"""
            self.executed.append(query)
            return "ok-rows"

        async def fake_tools(config, researcher_type="general"):
            return [sql_executor]

        monkeypatch.setattr(dr, "get_all_tools", fake_tools)
        self.log = tmp_path / "approvals.jsonl"
        b = StateGraph(ResearcherState)
        b.add_node("researcher_tools", dr.researcher_tools)
        b.add_node("researcher", lambda s: {})
        b.add_node("compress_research", lambda s: {})
        b.add_edge(START, "researcher_tools")
        b.add_edge("researcher", END)
        b.add_edge("compress_research", END)
        self.graph = b.compile(checkpointer=InMemorySaver() if with_checkpointer else None)

    def state(self, sql):
        msg = AIMessage(content="", tool_calls=[{"name": "sql_executor", "args": {"query": sql}, "id": "c1"}])
        return {"researcher_messages": [msg], "researcher_type": "data_analyst", "research_topic": "t"}

    def cfg(self, thread, mode="interrupt"):
        return {"configurable": {"thread_id": thread, "approval_mode": mode, "approval_log_path": str(self.log)}}

    def run(self, coro):
        return asyncio.run(coro)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path)


def test_aggregate_query_runs_without_pausing(env):
    out = env.run(env.graph.ainvoke(env.state("SELECT COUNT(*) FROM sales"), env.cfg("t1")))
    assert "__interrupt__" not in out and env.executed == ["SELECT COUNT(*) FROM sales"]


def test_row_level_query_pauses_before_executing_anything(env):
    out = env.run(env.graph.ainvoke(env.state("SELECT * FROM sales"), env.cfg("t2")))
    assert out.get("__interrupt__") and env.executed == []
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "approval_required" and payload["requests"][0]["tool"] == "sql_executor"


def test_approve_executes_exactly_once(env):
    env.run(env.graph.ainvoke(env.state("SELECT * FROM sales"), env.cfg("t3")))
    out = env.run(env.graph.ainvoke(Command(resume={"approved": True, "reviewer": "alice", "note": "audit Q2"}), env.cfg("t3")))
    assert env.executed == ["SELECT * FROM sales"], "tool must run once, not once per replay"
    assert out["researcher_messages"][-1].content == "ok-rows"
    rec = json.loads(env.log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["approved"] is True and rec["reviewer"] == "alice" and rec["thread_id"] == "t3"


def test_reject_never_executes_and_stops_the_loop(env):
    env.run(env.graph.ainvoke(env.state("SELECT * FROM sales"), env.cfg("t4")))
    out = env.run(env.graph.ainvoke(Command(resume={"approved": False, "reviewer": "bob", "note": "not needed"}), env.cfg("t4")))
    assert env.executed == []
    last = out["researcher_messages"][-1].content
    assert last.startswith("[POLICY_DENIED]") and "not approved" in last
    assert json.loads(env.log.read_text(encoding="utf-8").splitlines()[-1])["approved"] is False


def test_auto_approve_mode_skips_pause_but_still_audits(env):
    out = env.run(env.graph.ainvoke(env.state("SELECT * FROM sales"), env.cfg("t5", "auto_approve")))
    assert "__interrupt__" not in out and env.executed == ["SELECT * FROM sales"]
    assert json.loads(env.log.read_text(encoding="utf-8").splitlines()[-1])["reviewer"] == "auto"


def test_deny_mode(env):
    env.run(env.graph.ainvoke(env.state("SELECT * FROM sales"), env.cfg("t6", "deny")))
    assert env.executed == []


def test_no_checkpointer_fails_closed(monkeypatch, tmp_path):
    e = Env(monkeypatch, tmp_path, with_checkpointer=False)
    e.run(e.graph.ainvoke(e.state("SELECT * FROM sales"), e.cfg("t7")))
    assert e.executed == [], "cannot pause -> must deny, never silently execute"
