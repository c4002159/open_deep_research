"""Durability: a run that dies mid-way continues from its last checkpoint, and a pending
human approval survives a process restart. Uses a real on-disk SQLite checkpointer and a
*new connection* for the second half, which is what a restart looks like to LangGraph."""

import asyncio
import operator
import warnings
from typing import Annotated

import pytest
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

warnings.filterwarnings("ignore")


class S(TypedDict):
    log: Annotated[list, operator.add]


def build(counters, crash_once, saver):
    def node(name):
        def fn(state):
            counters[name] = counters.get(name, 0) + 1
            if name == "c" and crash_once["armed"]:
                crash_once["armed"] = False
                raise RuntimeError("simulated process crash")
            return {"log": [name]}
        return fn

    b = StateGraph(S)
    for n in "abc":
        b.add_node(n, node(n))
    b.add_edge(START, "a"); b.add_edge("a", "b"); b.add_edge("b", "c"); b.add_edge("c", END)
    return b.compile(checkpointer=saver)


def test_resume_after_crash_does_not_redo_finished_nodes(tmp_path):
    db = str(tmp_path / "ckpt.sqlite")
    cfg = {"configurable": {"thread_id": "job-1"}}
    counters, crash = {}, {"armed": True}

    async def first_life():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            with pytest.raises(RuntimeError, match="simulated"):
                await build(counters, crash, saver).ainvoke({"log": []}, cfg)

    async def second_life():  # "new process": fresh connection to the same file
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            return await build(counters, crash, saver).ainvoke(None, cfg)  # None = continue from checkpoint

    asyncio.run(first_life())
    out = asyncio.run(second_life())
    assert out["log"] == ["a", "b", "c"]
    assert counters == {"a": 1, "b": 1, "c": 2}, "a and b ran once; only the failed node c was retried"


def test_without_checkpointer_everything_reruns(tmp_path):
    """The control: no persistence -> a retry starts over and repeats side effects."""
    counters, crash = {}, {"armed": True}
    g = build(counters, crash, None)
    with pytest.raises(RuntimeError):
        asyncio.run(g.ainvoke({"log": []}))
    asyncio.run(g.ainvoke({"log": []}))
    assert counters == {"a": 2, "b": 2, "c": 2}


def test_pending_approval_survives_restart(tmp_path):
    db = str(tmp_path / "ckpt.sqlite")
    cfg = {"configurable": {"thread_id": "approval-1"}}
    executed = []

    def build_gate(saver):
        def gate(state):
            decision = interrupt({"type": "approval_required", "sql": "SELECT * FROM sales"})
            if decision["approved"]:
                executed.append("ran")
            return {"log": [f"decision={decision['approved']}"]}

        b = StateGraph(S)
        b.add_node("gate", gate)
        b.add_edge(START, "gate"); b.add_edge("gate", END)
        return b.compile(checkpointer=saver)

    async def before_restart():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            return await build_gate(saver).ainvoke({"log": []}, cfg)

    async def after_restart():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            return await build_gate(saver).ainvoke(Command(resume={"approved": True}), cfg)

    paused = asyncio.run(before_restart())
    assert paused["__interrupt__"] and executed == []
    done = asyncio.run(after_restart())
    assert done["log"] == ["decision=True"] and executed == ["ran"]
