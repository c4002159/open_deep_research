import asyncio
from types import SimpleNamespace

import pytest

from open_deep_research import kb_retrieval as kb
from open_deep_research.kb_retrieval import BM25, Corpus, Hit, rrf, search, tokenize

DOCS = {
    "a.md": "## 1. 退款条件\n订单交付后 30 天内可申请全额退款。",
    "b.md": "## 2. 提成比例\n基础产品按销售额的 5% 计提，战略产品按 8% 计提。",
    "c.md": "## 3. 发放周期\n提成每季度核算一次，次月随工资发放。",
}


class FakeStore:
    """Returns a fixed (deliberately imperfect) dense ranking, like an embedding model would."""

    def __init__(self, order):
        self.order = order

    async def asimilarity_search_with_score(self, query, k=20):
        return [(SimpleNamespace(page_content=DOCS[s], metadata={"source": s}), 0.1 * i) for i, s in enumerate(self.order[:k])]


@pytest.fixture(autouse=True)
def _corpus(monkeypatch):
    texts, sources = list(DOCS.values()), list(DOCS)
    monkeypatch.setattr(kb, "load_corpus", lambda conn, ttl=60: Corpus(texts, sources, BM25(texts)))


def run(coro):
    return asyncio.run(coro)


def test_tokenizer_handles_chinese_and_identifiers():
    t = tokenize("Security Suite 提成 5%")
    assert "security" in t and "提成" in t and "5" in t and "提" in t


def test_bm25_prefers_exact_term_match():
    docs = list(DOCS.values())
    scores = BM25(docs).scores("战略产品")
    assert scores.argmax() == 1 and scores[0] == 0


def test_rrf_rewards_agreement_over_single_list_rank():
    assert rrf([["x", "a", "b"], ["y", "a", "c"]])[0] == "a"       # a is 2nd in both lists, beats the two #1s
    assert set(rrf([["a"], ["b"]])) == {"a", "b"}


def test_hybrid_rescues_exact_term_the_embedding_ranked_low():
    store = FakeStore(["a.md", "c.md", "b.md"])                     # dense model puts the right chunk last
    vec = run(search(store, "x", "战略产品提成", k=1, mode="vector"))
    hyb = run(search(store, "x", "战略产品提成", k=1, mode="hybrid"))
    assert vec[0].source == "a.md" and hyb[0].source == "b.md"
    assert "bm25" in hyb[0].via and "vector" in hyb[0].via and hyb[0].heading.startswith("## 2")


def test_vector_mode_is_unchanged_behaviour():
    hits = run(search(FakeStore(["c.md", "a.md", "b.md"]), "x", "q", k=2, mode="vector"))
    assert [h.source for h in hits] == ["c.md", "a.md"] and all(h.via == "vector" for h in hits)


def test_falls_back_to_dense_results_when_lexical_index_is_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(kb, "load_corpus", boom)
    hits = run(search(FakeStore(["b.md", "a.md"]), "x", "q", k=2, mode="hybrid"))
    assert [h.source for h in hits] == ["b.md", "a.md"] and all(h.via == "vector" for h in hits)


def test_query_with_no_lexical_overlap_still_returns_dense_results():
    hits = run(search(FakeStore(["a.md", "b.md", "c.md"]), "x", "zzzz qqqq", k=3, mode="hybrid"))
    assert {h.source for h in hits} == {"a.md", "b.md", "c.md"}


def test_hit_heading():
    assert Hit("## 3. 发放周期\nbody", "c.md", 1, "vector").heading == "## 3. 发放周期"
    assert Hit("plain text", "c.md", 1, "vector").heading == ""
