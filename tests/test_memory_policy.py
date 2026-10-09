"""Memory policy. The BAD list is copied verbatim from what the agent really stored in mem0 during the
evaluation runs - including the entry that omitted the `units` column - so these are regression tests
against real failures, not invented examples."""

import asyncio
import sqlite3

import pytest

from open_deep_research import utils
from open_deep_research.memory_policy import check_fact
from open_deep_research.prompts import data_analyst_system_prompt
from open_deep_research.schema_context import Schema, UNAVAILABLE, read_schema, reset_schema_cache, schema_prompt_block

SCHEMA_IDS = {"sales", "products", "id", "date", "product_line", "region", "revenue", "units", "category", "launch_year"}

BAD = [
    # schema claims -- the first one is the poisoned fact: it leaves out `units`
    "User's sales table contains only the columns id (de-facto order id), date, product_line, region, and revenue (a single pre‑computed monetary column)",
    "The products table has columns product_line (TEXT), category (TEXT), and launch_year (INTEGER)",
    "The database contains exactly two raw tables—`sales` and `products`—and has no orders, order_items, invoices",
    "The sales table does not include columns for refund, return, currency, tax, channel, or customer.",
    # false statements about the data
    "The sales table contains exactly one transaction per date, i.e., each date from 2026‑01‑01 to 2026‑06‑08 appears in a single row",
    "The sales table has no NULL values in the units column",
    "The sales table's region column contains the categorical values East, North, South, and West",
    # data values
    "Total revenue across all rows in the sales table is $2,345,328.08 USD",
    "The sales table in the SQLite database contains 80 rows",
    "The sample standard deviation of the sales.revenue column (REAL, USD) across its 80 rows is 12,962.88 (using n‑1 denominator)",
    "In the Jan‑Jun 2026 sales dataset, the Data Platform product line recorded 28 orders with a total revenue of $826,968.89",
    "Typical monthly revenue totals in the sales database are approximately between 400,000 and 472,000 USD",
    # speculation, one-off events, tooling noise
    "Because the sales table lacks a status field, cancelled or refunded orders cannot be identified; the only proxy for refunds is filtering for negative revenue values (revenue < 0)",
    "Assistant provided an exact aggregation query: SELECT s.product_line, p.category, SUM(s.revenue) AS total_sales FROM sales s JOIN products p",
    "The tavily_search tool can fail with a DEEPSEEK_API_KEY configuration error (ChatDeepSeek validation error)",
]

GOOD = [
    ("A \"big order\" in the sales data is defined as having units greater than 100", "terminology"),
    ("sales.id serves as the de-facto order identifier in the SQLite database", "terminology"),
    ("The user said that 'sales amount' means the revenue field", "terminology"),
    ("Refund approval limits are documented in refundPolicy.md, section 2", "document_location"),
    ("销售额指的是 revenue 字段", "terminology"),
    ("用户纠正：大区指 region 字段，不是 country", "correction"),
]


@pytest.mark.parametrize("fact", BAD)
def test_real_bad_memories_are_rejected(fact):
    v = check_fact(fact, SCHEMA_IDS)
    assert not v.ok, f"should have been rejected: {fact!r}"
    assert v.message


@pytest.mark.parametrize("fact,category", GOOD)
def test_durable_knowledge_is_accepted(fact, category):
    v = check_fact(fact, SCHEMA_IDS)
    assert v.ok and v.category == category, (fact, v)


def test_structure_claim_cannot_hide_behind_a_mapping_word():
    """Regression: adding 'de-facto' to the mapping words once let the poisoned column list through."""
    v = check_fact("The sales table contains only the columns id (de-facto order id), date and revenue", SCHEMA_IDS)
    assert not v.ok and v.category == "schema"


def test_mapping_to_a_column_that_does_not_exist_is_rejected():
    v = check_fact("'order value' means the order_total field", SCHEMA_IDS)
    assert not v.ok and "order_total" in v.message
    assert check_fact("'order value' means the revenue field", SCHEMA_IDS).ok


def test_length_limits():
    assert not check_fact("too short").ok
    assert not check_fact("x " * 400).ok


# ------------------------------------------------------------------ tools
class FakeMem0:
    def __init__(self, stored=()):
        self.stored, self.added = list(stored), []

    def search(self, query, filters=None):
        return {"results": [{"memory": m, "score": 0.9} for m in self.stored]}

    def add(self, messages, user_id=None):
        self.added.append((messages, user_id))


@pytest.fixture()
def fake_mem0(monkeypatch):
    holder = {}

    def install(client):
        holder["c"] = client
        monkeypatch.setattr(utils, "_get_mem0_client", lambda config: client)
        return client

    return install


CFG = {"configurable": {"mem0_user_id": "u1"}}


def test_add_memory_refuses_bad_facts_and_never_calls_the_store(fake_mem0):
    client = fake_mem0(FakeMem0())
    for fact in BAD[:5]:
        out = asyncio.run(utils.add_memory.ainvoke({"fact": fact}, config=CFG))
        assert out.startswith("Memory not saved:") and "Only save durable knowledge" in out
    assert client.added == []


def test_add_memory_saves_a_definition_for_the_right_user(fake_mem0):
    client = fake_mem0(FakeMem0())
    out = asyncio.run(utils.add_memory.ainvoke({"fact": GOOD[0][0]}, config=CFG))
    assert out.startswith("Fact saved") and client.added[0][1] == "u1"


def test_search_memory_ignores_legacy_bad_entries(fake_mem0):
    fake_mem0(FakeMem0(stored=BAD + [GOOD[0][0], GOOD[3][0]]))
    out = asyncio.run(utils.search_memory.ainvoke({"query": "sales"}, config=CFG))
    assert GOOD[0][0] in out and GOOD[3][0] in out
    assert "only the columns" not in out and "2,345,328" not in out and "80 rows" not in out


def test_search_memory_reports_nothing_when_only_bad_entries_exist(fake_mem0):
    fake_mem0(FakeMem0(stored=BAD))
    assert "No relevant memories" in asyncio.run(utils.search_memory.ainvoke({"query": "sales"}, config=CFG))


# ------------------------------------------------------------------ verified schema
@pytest.fixture()
def db_url(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE products (product_line TEXT PRIMARY KEY, category TEXT, launch_year INTEGER);"
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, date TEXT, product_line TEXT REFERENCES products(product_line),"
        " region TEXT, revenue REAL, units INTEGER);")
    con.commit(); con.close()
    reset_schema_cache()
    yield "sqlite:///" + path.as_posix()
    reset_schema_cache()


def test_schema_is_read_from_the_database_including_every_column(db_url):
    s = read_schema(db_url)
    assert [c for c, _ in s.tables["sales"]] == ["id", "date", "product_line", "region", "revenue", "units"]
    assert s.primary_keys["sales"] == ["id"] and s.identifiers >= SCHEMA_IDS
    text = s.render()
    assert "units INTEGER" in text and "id INTEGER PK" in text and "FK: sales(product_line) -> products(product_line)" in text


def test_prompt_block_degrades_gracefully():
    assert asyncio.run(schema_prompt_block(None)) == UNAVAILABLE
    assert asyncio.run(schema_prompt_block("sqlite:////definitely/not/here/x.db")) == UNAVAILABLE


def test_data_analyst_prompt_embeds_the_schema_and_no_longer_says_trust_memory(db_url):
    block = asyncio.run(schema_prompt_block(db_url))
    prompt = data_analyst_system_prompt.format(date="2026-10-09", schema=block)
    assert "units INTEGER" in prompt and "AUTHORITATIVE" in prompt
    assert "TRUST it" not in prompt and "Memory never overrides" in prompt
    assert "{schema}" not in prompt


def test_tool_description_no_longer_contradicts_the_injected_schema():
    """The model obeys a tool description over the system prompt, so a 'go explore the schema yourself'
    sentence there silently defeated the verified-schema block (seen in a trace)."""
    d = utils.SQL_EXECUTOR_DESCRIPTION
    assert "explore available tables" not in d and "no pre-built schema" not in d
    assert "verified schema" in d and "do not use information_schema" in d
