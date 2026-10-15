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


# ------------------------------------------------------------------ cumulative row-level reads (LIMIT slicing)
from open_deep_research.approval import RowBudget, collect_pending, sql_row_cost  # noqa: E402


def _call(i, sql):
    return {"id": f"c{i}", "name": "sql_executor", "args": {"query": sql}}


@pytest.mark.parametrize("sql,cost", [
    ("SELECT * FROM sales WHERE id = 80 LIMIT 1", 1),
    ("SELECT * FROM sales LIMIT 5", 5),
    ("SELECT * FROM sales LIMIT 5 OFFSET 5", 5),
    ("SELECT * FROM sales", None),                              # unbounded: needs approval by the per-query rule
    ("SELECT region, SUM(revenue) FROM sales GROUP BY region", None),
    ("SELECT COUNT(*) FROM sales", None),
    ("SELECT * FROM products LIMIT 5", None),                   # not a sensitive table
    ("not sql at all", None),
])
def test_row_cost_is_the_declared_limit_of_a_bounded_row_level_query(sql, cost):
    assert sql_row_cost(sql) == cost


def test_slicing_the_table_five_rows_at_a_time_is_caught_on_the_third_slice():
    b = RowBudget()
    kw = dict(row_limit=10, thread_id="t1", budget=b)
    assert collect_pending([_call(1, "SELECT * FROM sales LIMIT 5")], **kw) == []
    assert collect_pending([_call(2, "SELECT * FROM sales LIMIT 5 OFFSET 5")], **kw) == []         # total 10: still within the limit
    third = collect_pending([_call(3, "SELECT * FROM sales LIMIT 5 OFFSET 10")], **kw)
    assert len(third) == 1 and third[0]["cumulative"] and "up to 15" in third[0]["reason"]


def test_parallel_calls_in_one_batch_are_added_up_too():
    pend = collect_pending([_call(i, "SELECT * FROM sales LIMIT 5") for i in range(3)], row_limit=10, thread_id="t", budget=RowBudget())
    assert [p["id"] for p in pend] == ["c2"]


def test_replaying_the_gate_after_a_pause_does_not_double_count():
    b = RowBudget()
    batch = [_call(1, "SELECT * FROM sales LIMIT 5"), _call(2, "SELECT * FROM sales LIMIT 5")]
    assert collect_pending(batch, row_limit=10, thread_id="t", budget=b) == []
    assert collect_pending(batch, row_limit=10, thread_id="t", budget=b) == []                     # same call ids: same total


def test_an_approval_resets_the_count_and_tasks_are_independent():
    b = RowBudget()
    big = [_call(i, "SELECT * FROM sales LIMIT 6") for i in (1, 2)]
    assert len(collect_pending(big, row_limit=10, thread_id="a", budget=b)) == 1
    assert collect_pending([_call(9, "SELECT * FROM sales LIMIT 6")], row_limit=10, thread_id="other", budget=b) == []
    b.reset("a")
    assert collect_pending([_call(10, "SELECT * FROM sales LIMIT 6")], row_limit=10, thread_id="a", budget=b) == []


def test_aggregates_and_single_order_lookups_are_never_a_nuisance():
    b = RowBudget()
    aggs = [_call(i, "SELECT region, SUM(revenue) FROM sales GROUP BY region") for i in range(30)]
    assert collect_pending(aggs, row_limit=10, thread_id="t", budget=b) == []
    lookups = [_call(100 + i, f"SELECT * FROM sales WHERE id = {i} LIMIT 1") for i in range(10)]
    assert collect_pending(lookups, row_limit=10, thread_id="t", budget=b) == []                   # ten single-order lookups
    assert len(collect_pending([_call(200, "SELECT * FROM sales WHERE id = 11 LIMIT 1")], row_limit=10, thread_id="t", budget=b)) == 1


def test_unbounded_queries_still_use_the_per_query_rule_and_do_not_charge_the_budget():
    b = RowBudget()
    pend = collect_pending([_call(1, "SELECT * FROM sales")], row_limit=10, thread_id="t", budget=b)
    assert len(pend) == 1 and not pend[0]["cumulative"] and "all matching rows" in pend[0]["reason"]
    assert b.charge("t", "x", 1) == 1


def test_the_budget_forgets_old_tasks_instead_of_growing_forever():
    b = RowBudget(max_threads=3)
    for i in range(10):
        b.charge(f"t{i}", "c", 1)
    assert len(b._by_thread) == 3


def test_slicing_through_the_real_node_pauses_the_whole_batch_before_anything_runs(env):
    calls = [{"name": "sql_executor", "args": {"query": f"SELECT * FROM sales LIMIT 5 OFFSET {5 * i}"}, "id": f"s{i}"} for i in range(3)]
    state = {"researcher_messages": [AIMessage(content="", tool_calls=calls)], "researcher_type": "data_analyst", "research_topic": "t"}
    out = env.run(env.graph.ainvoke(state, env.cfg("slice-1")))
    assert out.get("__interrupt__") and env.executed == []
    req = out["__interrupt__"][0].value["requests"]
    assert [r["id"] for r in req] == ["s2"] and req[0]["cumulative"]
    env.run(env.graph.ainvoke(Command(resume={"approved": True, "reviewer": "bob"}), env.cfg("slice-1")))
    assert len(env.executed) == 3, "each query runs exactly once after approval"
