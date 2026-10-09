"""A/B the PRODUCTION retrieval path (real PGVector + kb_retrieval.search) on the 41-query benchmark.

    python -m eval.run_kb_prod_eval [--rerank]

Unlike eval.retrieval_bench (in-memory vectors), this calls exactly what the agent's
`search_knowledge_base` tool calls, so the numbers describe what users actually get.
Needs the pgvector container and an ingested KB (docs/internal_kb/build_kb.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from eval.retrieval_bench_cases import RETRIEVAL_BENCH_CASES
from open_deep_research.configuration import Configuration
from open_deep_research.kb_retrieval import load_corpus, search
from open_deep_research.utils import _get_kb_vectorstore

OUT = Path(__file__).resolve().parent / "kb_prod_report.json"


def key_of(hit) -> tuple:
    m = re.match(r"##\s*(\d+)\.", hit.text)
    return (hit.source, int(m.group(1)) if m else 0)


async def evaluate(label, store, conn, mode, rerank, k):
    hits1 = hitsk = rec = rr = 0.0
    lat = []
    misses = []
    for case in RETRIEVAL_BENCH_CASES:
        gold = set(case["gold"])
        t = time.perf_counter()
        hits = await search(store, conn, case["query"], k=10, mode=mode, rerank=rerank)
        lat.append((time.perf_counter() - t) * 1000)
        keys = [key_of(h) for h in hits]
        first = next((r for r, kk in enumerate(keys, 1) if kk in gold), None)
        found = len(gold & set(keys[:k]))
        hits1 += (first == 1)
        hitsk += found > 0
        rec += found / len(gold)
        rr += 1 / first if first else 0
        if first is None or first > k:
            misses.append(case["id"])
    n = len(RETRIEVAL_BENCH_CASES)
    return {"config": label, "hit@1": round(hits1 / n, 3), f"hit@{k}": round(hitsk / n, 3), f"recall@{k}": round(rec / n, 3),
            "MRR@10": round(rr / n, 3), "avg_ms": round(sum(lat) / n, 1), f"misses@{k}": misses}


async def main(with_rerank: bool, k: int):
    conn = Configuration().postgres_connection_string
    store = _get_kb_vectorstore({"configurable": {}})
    corpus = load_corpus(conn)
    print(f"production KB: {len(corpus.texts)} chunks, {len(corpus.known_sources)} documents, {len(RETRIEVAL_BENCH_CASES)} queries")
    await search(store, conn, "warm up", k=1)                      # load embedding model before timing
    runs = [("vector (previous behaviour)", "vector", False), ("hybrid (new default)", "hybrid", False)]
    if with_rerank:
        runs.append(("hybrid + rerank (opt-in)", "hybrid", True))
    results = [await evaluate(l, store, conn, m, rr_, k) for l, m, rr_ in runs]
    print(f"\n{'config':<30}{'hit@1':>7}{'hit@' + str(k):>7}{'recall@' + str(k):>10}{'MRR@10':>8}{'ms/query':>10}")
    for r in results:
        print(f"{r['config']:<30}{r['hit@1']:>7}{r[f'hit@{k}']:>7}{r[f'recall@{k}']:>10}{r['MRR@10']:>8}{r['avg_ms']:>10}")
    for r in results:
        print(f"  misses@{k} [{r['config']}]: {r[f'misses@{k}']}")
    OUT.write_text(json.dumps({"k": k, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(a.rerank, a.k))
