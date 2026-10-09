"""The service runs the API, the workers and every tool call on ONE event loop. A blocking call inside an
async tool therefore freezes everything: HTTP handlers, Redis reads (spurious timeouts) and lease renewal.

Measured in the real service: the first knowledge-base search loaded the embedding model on the loop for
~13 s (a 21 s stall) and a Redis pipeline timed out. These tests slow the blocking calls down and check that
the loop keeps ticking while they run."""

import asyncio
import time

import pytest

from open_deep_research import utils

CFG = {"configurable": {"mem0_user_id": "u1"}}
BLOCK_S = 0.8          # how long the fake blocking call takes
MAX_STALL_S = 0.4      # the loop must never be unresponsive for longer than this


async def run_with_heartbeat(coro):
    """Run ``coro`` while a 20 ms ticker records the longest gap in which the loop could not run."""
    worst = 0.0
    stop = asyncio.Event()

    async def ticker():
        nonlocal worst
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            worst = max(worst, now - last)
            last = now

    t = asyncio.create_task(ticker())
    await asyncio.sleep(0.1)
    try:
        result = await coro
    finally:
        stop.set()
        await t
    return result, worst


def test_knowledge_base_search_does_not_block_the_loop(monkeypatch):
    delay = {"s": 0.0}

    def slow_vectorstore(config):
        time.sleep(delay["s"])                   # what loading the embedding model does
        return object()

    async def fake_search(*a, **k):
        return []

    monkeypatch.setattr(utils, "_get_kb_vectorstore", slow_vectorstore)
    monkeypatch.setattr(utils, "kb_search", fake_search)

    async def scenario():
        # the first call into a LangChain tool pays one-off lazy-import costs; that is not the thing under test
        await utils.search_knowledge_base.ainvoke({"query": "warm-up"}, config=CFG)
        delay["s"] = BLOCK_S
        return await run_with_heartbeat(utils.search_knowledge_base.ainvoke({"query": "x"}, config=CFG))

    out, worst = asyncio.run(scenario())
    assert "No relevant documents" in out
    assert worst < MAX_STALL_S, f"event loop stalled for {worst:.2f}s during the knowledge-base search"


class SlowMem0:
    def search(self, query, filters=None):
        time.sleep(BLOCK_S)                      # mem0's client is synchronous HTTP
        return {"results": [{"memory": 'A "big order" in the sales data is defined as having units greater than 100', "score": 0.9}]}

    def add(self, messages, user_id=None):
        time.sleep(BLOCK_S)


@pytest.fixture()
def slow_mem0(monkeypatch):
    monkeypatch.setattr(utils, "_get_mem0_client", lambda config: SlowMem0())


def test_search_memory_does_not_block_the_loop(slow_mem0):
    out, worst = asyncio.run(run_with_heartbeat(utils.search_memory.ainvoke({"query": "big order"}, config=CFG)))
    assert "big order" in out
    assert worst < MAX_STALL_S, f"event loop stalled for {worst:.2f}s during search_memory"


def test_add_memory_does_not_block_the_loop(slow_mem0):
    fact = 'A "big order" in the sales data is defined as having units greater than 100'
    out, worst = asyncio.run(run_with_heartbeat(utils.add_memory.ainvoke({"fact": fact}, config=CFG)))
    assert out.startswith("Fact saved")
    assert worst < MAX_STALL_S, f"event loop stalled for {worst:.2f}s during add_memory"


def test_the_harness_itself_detects_a_blocked_loop():
    """Guard against a test that can never fail: a deliberately blocking coroutine must be flagged."""
    async def blocker():
        time.sleep(BLOCK_S)

    _, worst = asyncio.run(run_with_heartbeat(blocker()))
    assert worst >= MAX_STALL_S


def test_mem0_client_is_built_once_per_key_and_off_the_loop(monkeypatch):
    """MemoryClient() validates the API key over HTTP (~1.2 s) every time it is constructed."""
    built = []

    class SlowCtor:
        def __init__(self, api_key=None):
            time.sleep(BLOCK_S)
            built.append(api_key)

        def search(self, query, filters=None):
            return {"results": []}

    monkeypatch.setattr(utils, "MemoryClient", SlowCtor)
    monkeypatch.setattr(utils, "_mem0_clients", {})
    cfg = {"configurable": {"mem0_user_id": "u1"}}

    async def call(key, q):
        monkeypatch.setenv("MEM0_API_KEY", key)      # Configuration reads the key from the environment first
        return await utils.search_memory.ainvoke({"query": q}, config=cfg)

    async def scenario():
        await call("k-warm", "warm-up")              # pays LangChain's one-off lazy imports with a different key
        monkeypatch.setenv("MEM0_API_KEY", "k-test")
        _, worst = await run_with_heartbeat(call("k-test", "x"))      # constructs (0.8 s) - must not stall the loop
        await call("k-test", "y")                    # second call with the same key must reuse the client
        return worst

    worst = asyncio.run(scenario())
    assert built == ["k-warm", "k-test"], "the client must be constructed once per key"
    assert worst < MAX_STALL_S, f"event loop stalled for {worst:.2f}s while constructing the mem0 client"
