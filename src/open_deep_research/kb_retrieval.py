"""Hybrid retrieval for the internal knowledge base: dense vectors + BM25, fused with RRF.

Why hybrid: embeddings are good at paraphrase ("客户收到货三十天以后还能退钱吗" -> 退款适用条件) but blur
exact terms, numbers and identifiers; BM25 is the opposite. Reciprocal Rank Fusion merges the two
rankings without having to calibrate their incompatible score scales.

BM25 here is an in-process index over the chunks stored in pgvector (character uni+bigrams, so no
Chinese word segmenter is needed). That is right for a knowledge base of thousands of chunks; for
hundreds of thousands, move the lexical side to Postgres full-text search or Elasticsearch.
An optional cross-encoder rerank exists but is OFF by default: in our benchmark it lifted hit@1 by
~10 points at ~4 s per query on CPU.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

COLLECTION = "internal_docs"
RERANK_MODEL = "BAAI/bge-reranker-base"


# ------------------------------------------------------------------ lexical side
def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = re.findall(r"[a-z0-9_]+", text)
    cjk = re.findall(r"[一-鿿]", text)
    return toks + cjk + [a + b for a, b in zip(cjk, cjk[1:])]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df = Counter(tok for t in self.tf for tok in t)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for tok in set(tokenize(query)):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(tok, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(rankings: list[list], k: int = 60) -> list:
    """Reciprocal Rank Fusion over lists of hashable ids; best first."""
    score: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


# ------------------------------------------------------------------ corpus (read from pgvector)
@dataclass
class Corpus:
    texts: list[str]
    sources: list[str]
    bm25: BM25
    loaded_at: float = field(default_factory=time.monotonic)

    @property
    def known_sources(self) -> set[str]:
        return set(self.sources)


_corpus_lock = threading.Lock()
_corpus: dict[str, Corpus] = {}


def load_corpus(connection: str, ttl_s: float = 60.0, collection: str = COLLECTION) -> Corpus:
    """Chunks currently stored in pgvector, with a BM25 index; cached for ``ttl_s`` seconds."""
    with _corpus_lock:
        cached = _corpus.get(connection)
        if cached and time.monotonic() - cached.loaded_at < ttl_s:
            return cached
    from sqlalchemy import create_engine, text

    engine = create_engine(connection.replace("postgresql+psycopg_async", "postgresql+psycopg"))
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT e.document, e.cmetadata->>'source' FROM langchain_pg_embedding e "
                "JOIN langchain_pg_collection c ON e.collection_id = c.uuid WHERE c.name = :n ORDER BY e.id"), {"n": collection}).fetchall()
    finally:
        engine.dispose()
    texts = [r[0] for r in rows]
    corpus = Corpus(texts, [r[1] or "unknown" for r in rows], BM25(texts))
    with _corpus_lock:
        _corpus[connection] = corpus
    return corpus


def reset_corpus_cache() -> None:
    with _corpus_lock:
        _corpus.clear()


# ------------------------------------------------------------------ search
@dataclass
class Hit:
    text: str
    source: str
    rank: int
    via: str                       # "vector", "bm25" or "vector+bm25"

    @property
    def heading(self) -> str:
        first = self.text.split("\n", 1)[0].strip()
        return first if first.startswith("#") else ""


_reranker = None
_reranker_lock = threading.Lock()


def _rerank_sync(query: str, hits: list[Hit]) -> list[Hit]:
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANK_MODEL)
    scores = _reranker.predict([(query, h.text) for h in hits])
    return [hits[i] for i in np.argsort(-np.asarray(scores))]


async def search(vectorstore, connection: str, query: str, *, k: int = 3, mode: str = "hybrid",
                 candidates: int = 20, rerank: bool = False, corpus_ttl_s: float = 60.0) -> list[Hit]:
    """Top-``k`` chunks for ``query``. ``mode`` is ``"hybrid"`` or ``"vector"`` (the previous behaviour)."""
    vec_docs = await vectorstore.asimilarity_search_with_score(query, k=candidates)
    vec_rank = [d.page_content for d, _ in vec_docs]
    meta_source = {d.page_content: d.metadata.get("source", "unknown") for d, _ in vec_docs}

    if mode != "hybrid":
        return [Hit(t, meta_source[t], i + 1, "vector") for i, t in enumerate(vec_rank[:k])]

    try:
        corpus = await asyncio.to_thread(load_corpus, connection, corpus_ttl_s)
    except Exception as e:  # noqa: BLE001 - lexical side is an enhancement; never lose the dense result
        logger.warning("BM25 corpus unavailable (%s); falling back to vector-only", e)
        return [Hit(t, meta_source[t], i + 1, "vector") for i, t in enumerate(vec_rank[:k])]

    scores = corpus.bm25.scores(query)
    order = [int(i) for i in np.argsort(-scores)[:candidates] if scores[i] > 0]
    bm25_rank = [corpus.texts[i] for i in order]
    source_of = {t: s for t, s in zip(corpus.texts, corpus.sources)} | meta_source

    fused = rrf([vec_rank, bm25_rank])
    in_vec, in_bm = set(vec_rank), set(bm25_rank)
    pool = fused[: max(k, 15) if rerank else k]
    hits = [Hit(t, source_of.get(t, "unknown"), i + 1, "+".join(x for x, s in (("vector", in_vec), ("bm25", in_bm)) if t in s))
            for i, t in enumerate(pool)]
    if rerank and len(hits) > 1:
        hits = await asyncio.to_thread(_rerank_sync, query, hits)
        hits = [Hit(h.text, h.source, i + 1, h.via + "+rerank") for i, h in enumerate(hits[:k])]
    return hits[:k]
