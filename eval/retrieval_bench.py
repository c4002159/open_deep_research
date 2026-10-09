"""LLM-free retrieval benchmark: vector vs BM25 vs hybrid (RRF) vs hybrid + rerank.

    python -m eval.retrieval_bench [--no-rerank] [--k 3]

Metrics (per query, then averaged):
  hit@k     - at least one gold section in the top-k
  recall@k  - fraction of gold sections in the top-k
  MRR@10    - reciprocal rank of the first gold section
Writes eval/retrieval_report.json. Uses the same chunking and embedding model as
the production search_knowledge_base tool, but keeps vectors in memory so it runs
without PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from eval.retrieval_bench_cases import RETRIEVAL_BENCH_CASES

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "docs" / "internal_kb"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"


# ---------------------------------------------------------------- corpus
def split_sections(text: str) -> list[str]:
    """Same rule as docs/internal_kb/build_kb.py (inline: that file runs DB code on import)."""
    return [s.strip() for s in re.split(r"\n(?=##\s)", text) if s.strip()]


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        for section in split_sections(path.read_text(encoding="utf-8")):
            m = re.match(r"##\s*(\d+)\.", section)
            chunks.append({
                "source": path.name,
                "section": int(m.group(1)) if m else 0,  # 0 = title-only chunk
                "text": section,
            })
    return chunks


# ---------------------------------------------------------------- BM25 (char uni+bigrams, no jieba needed)
def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = re.findall(r"[a-z0-9_]+", text)
    cjk = re.findall(r"[一-鿿]", text)
    toks += cjk + [a + b for a, b in zip(cjk, cjk[1:])]
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean()
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


# ---------------------------------------------------------------- retrievers
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


class Retrievers:
    def __init__(self, chunks: list[dict], use_rerank: bool):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.doc_vecs = self.embedder.encode(texts, normalize_embeddings=True)
        self.bm25 = BM25(texts)
        self.reranker = CrossEncoder(RERANK_MODEL) if use_rerank else None

    def vector(self, q: str, n: int = 10) -> list[int]:
        qv = self.embedder.encode([q], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-(self.doc_vecs @ qv))[:n]]

    def bm25_rank(self, q: str, n: int = 10) -> list[int]:
        return [int(i) for i in np.argsort(-self.bm25.scores(q))[:n]]

    def hybrid(self, q: str, n: int = 10) -> list[int]:
        return rrf([self.vector(q, 20), self.bm25_rank(q, 20)])[:n]

    def hybrid_rerank(self, q: str, n: int = 10) -> list[int]:
        cand = self.hybrid(q, 15)
        scores = self.reranker.predict([(q, self.chunks[i]["text"]) for i in cand])
        return [cand[i] for i in np.argsort(-np.asarray(scores))][:n]


# ---------------------------------------------------------------- metrics
def evaluate(name: str, fn, chunks: list[dict], k: int) -> dict:
    hits = recalls = rr = 0.0
    lat = []
    per_case = []
    for case in RETRIEVAL_BENCH_CASES:
        gold = set(case["gold"])
        t = time.perf_counter()
        ranked = fn(case["query"])
        lat.append((time.perf_counter() - t) * 1000)
        keys = [(chunks[i]["source"], chunks[i]["section"]) for i in ranked]
        found = len(gold & set(keys[:k]))
        first = next((r for r, key in enumerate(keys, 1) if key in gold), None)
        hits += found > 0
        recalls += found / len(gold)
        rr += 1 / first if first else 0
        per_case.append({"id": case["id"], "first_gold_rank": first})
    n = len(RETRIEVAL_BENCH_CASES)
    return {
        "retriever": name,
        f"hit@{k}": round(hits / n, 3),
        f"recall@{k}": round(recalls / n, 3),
        "MRR@10": round(rr / n, 3),
        "avg_latency_ms": round(sum(lat) / n, 1),
        "misses": [c["id"] for c in per_case if c["first_gold_rank"] is None or c["first_gold_rank"] > k],
    }


def validate_gold(chunks: list[dict]) -> None:
    """Every gold (file, section) must exist in the corpus - guards against typos in the case file."""
    present = {(c["source"], c["section"]) for c in chunks}
    bad = [(c["id"], g) for c in RETRIEVAL_BENCH_CASES for g in c["gold"] if g not in present]
    if bad:
        raise SystemExit(f"gold labels point at missing sections: {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    validate_gold(chunks)
    print(f"{len(chunks)} chunks from {len({c['source'] for c in chunks})} docs, {len(RETRIEVAL_BENCH_CASES)} queries")
    r = Retrievers(chunks, use_rerank=not args.no_rerank)

    runs = [("vector", r.vector), ("bm25", r.bm25_rank), ("hybrid(RRF)", r.hybrid)]
    if r.reranker is not None:
        runs.append(("hybrid+rerank", r.hybrid_rerank))
    results = [evaluate(name, fn, chunks, args.k) for name, fn in runs]

    k = args.k
    print(f"\n{'retriever':<16}{'hit@' + str(k):>8}{'recall@' + str(k):>10}{'MRR@10':>9}{'ms/query':>10}")
    for x in results:
        print(f"{x['retriever']:<16}{x[f'hit@{k}']:>8}{x[f'recall@{k}']:>10}{x['MRR@10']:>9}{x['avg_latency_ms']:>10}")
    for x in results:
        print(f"  misses[{x['retriever']}]: {x['misses']}")

    (ROOT / "eval" / "retrieval_report.json").write_text(
        json.dumps({"k": k, "n_chunks": len(chunks), "n_queries": len(RETRIEVAL_BENCH_CASES), "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
