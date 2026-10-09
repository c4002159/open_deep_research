"""Per-researcher TOTAL tool-call budget (parallel calls counted individually) through the real node."""

import asyncio
import warnings

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from open_deep_research import deep_researcher as dr
from open_deep_research.state import ResearcherState

warnings.filterwarnings("ignore")


def run_node(monkeypatch, researcher_type, n_calls, already_made=0, cfg=None):
    """Execute researcher_tools once; return (next_node, tool_calls_made)."""
    ran = []

    @tool
    async def search_knowledge_base(query: str) -> str:
        """fake"""
        ran.append(query)
        return "[来源: x.md]\nchunk"

    async def fake_tools(config, researcher_type="general"):
        return [search_knowledge_base]

    monkeypatch.setattr(dr, "get_all_tools", fake_tools)
    seen = {}
    b = StateGraph(ResearcherState)
    b.add_node("researcher_tools", dr.researcher_tools)
    b.add_node("researcher", lambda s: seen.setdefault("next", "researcher") and {})
    b.add_node("compress_research", lambda s: seen.setdefault("next", "compress_research") and {})
    b.add_edge(START, "researcher_tools")
    b.add_edge("researcher", END)
    b.add_edge("compress_research", END)
    g = b.compile()
    calls = [{"name": "search_knowledge_base", "args": {"query": f"q{i}"}, "id": f"c{i}"} for i in range(n_calls)]
    state = {"researcher_messages": [AIMessage(content="", tool_calls=calls)], "researcher_type": researcher_type,
             "research_topic": "t", "tool_calls_made": already_made, "tool_call_iterations": 1}
    out = asyncio.run(g.ainvoke(state, {"configurable": {"approval_mode": "auto_approve", **(cfg or {})}}))
    return seen["next"], out["tool_calls_made"], len(ran)


def test_parallel_calls_count_individually_and_stop_the_loop(monkeypatch):
    # 3 already made + a batch of 4 parallel searches = 7 >= budget 6 -> must go to compression, not loop again
    nxt, made, executed = run_node(monkeypatch, "knowledge_base", n_calls=4, already_made=3)
    assert nxt == "compress_research" and made == 7 and executed == 4


def test_under_budget_keeps_researching(monkeypatch):
    nxt, made, _ = run_node(monkeypatch, "knowledge_base", n_calls=2, already_made=1)
    assert nxt == "researcher" and made == 3


def test_budget_is_configurable_and_zero_means_unlimited(monkeypatch):
    nxt, _, _ = run_node(monkeypatch, "knowledge_base", n_calls=2, already_made=1, cfg={"kb_tool_call_budget": 3})
    assert nxt == "compress_research"
    nxt, _, _ = run_node(monkeypatch, "knowledge_base", n_calls=4, already_made=20, cfg={"kb_tool_call_budget": 0})
    assert nxt == "researcher"


def test_general_web_research_is_not_capped_by_this_budget(monkeypatch):
    nxt, made, _ = run_node(monkeypatch, "general", n_calls=4, already_made=30)
    assert nxt == "researcher" and made == 34


def test_data_analyst_has_its_own_budget(monkeypatch):
    nxt, _, _ = run_node(monkeypatch, "data_analyst", n_calls=3, already_made=8)       # 11 >= 10
    assert nxt == "compress_research"
