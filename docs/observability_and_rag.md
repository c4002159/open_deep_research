# Retrieval, citations and observability (stage 5)

## Hybrid knowledge-base retrieval

`search_knowledge_base` now fuses two rankings with Reciprocal Rank Fusion (k = 60):

* **dense** — the existing pgvector similarity search (bge-small-zh), top 20
* **lexical** — BM25 over character uni+bigrams (no Chinese word segmenter needed), top 20

The lexical index is built in-process from the chunks already stored in pgvector
(`kb_retrieval.load_corpus`, cached 60 s), so there is nothing extra to ingest or keep in sync.
If the corpus cannot be read, the tool silently falls back to dense-only instead of failing.

Config (`Configuration` / env): `kb_retrieval_mode` (`hybrid`|`vector`), `kb_top_k` (3),
`kb_candidates` (20), `kb_rerank` (false).

### Measured on the production path (`python -m eval.run_kb_prod_eval --rerank`)

41 queries, 38 chunks / 9 documents, real PGVector, warm models:

| config | hit@1 | hit@3 | MRR@10 | ms/query |
|---|---|---|---|---|
| vector (previous behaviour) | 0.854 | 1.000 | 0.919 | 18 |
| **hybrid (new default)** | **0.878** | 1.000 | **0.935** | 19 |
| hybrid + cross-encoder rerank (opt-in) | 0.951 | 0.976 | 0.968 | 963 |

Read this honestly: on a 38-chunk corpus hit@3 is saturated for every configuration, so the agent
(which reads 3 chunks) sees little difference; the gain is a +2.4-point hit@1 and +1.6-point MRR for
~1 ms. The unit tests show the mechanism working (an exact-term chunk the embedding ranked last is
promoted to #1). Rerank gives the biggest hit@1 jump but costs ~1 s/query on CPU and lost one
cross-document case, so it stays off by default. The benchmark queries were written by the same
people who wrote the documents; expect smaller gains on real user phrasing.

**Scaling limit:** the in-process BM25 index is fine for thousands of chunks. For hundreds of
thousands, move the lexical side to Postgres full-text search or Elasticsearch.

## Citation verification

After the final report is written, every `*.md` document it cites is checked:

* **unknown** — not in the knowledge base (fabricated source)
* **ungrounded** — in the knowledge base but absent from this run's research notes (cited from memory)

Failures are never hidden or silently rewritten; a marked note is appended to the report
(`⚠️ 引用核验未通过 …`). Eval cases tagged `kb` also assert `cited_docs_exist`.
Limit: it verifies document names, not that a quoted sentence is really in the document.

## Observability

`telemetry.UsageTracker` is a LangChain callback attached to every service task. It sees all nested
LLM calls, tool calls and graph nodes and records:

* tokens in/out and calls per model, tool calls/errors/seconds per tool, seconds per graph node
* one JSON line per span in `logs/traces/<task_id>.jsonl` (`ODR_TRACE_DIR`)

`GET /v1/tasks/{id}` returns `usage` (and `est_cost_usd` when prices are configured). Usage is recorded
for **every** outcome — success, failure, pause, cancel, worker death — because the tokens were spent
either way. A tool counts as failed when it returns an error string (our tools report errors as text).

### Cost control

* `ODR_PRICE_IN_PER_M` / `ODR_PRICE_OUT_PER_M` — your USD prices per million tokens. **Not guessed:** without
  them the API shows tokens and `est_cost_usd: null`.
* `ODR_DAILY_TOKEN_BUDGET` — per-user, per UTC day. At submit, a user at/over budget gets
  `429` with `Retry-After` = seconds to midnight UTC. It is a soft limit checked at submit: one in-flight
  task can overshoot it. Idempotent replays and other users are unaffected.
* `GET /v1/usage` — my usage today, remaining budget, last N days.

### Admin view

Set `ODR_ADMINS="root"`, open `/dashboard`, paste that user's token. The page renders
`GET /v1/admin/summary`: tokens and cost today, per-user table, per-tool call / error rate, task
latency histogram (submit → finished) and service counters (rate-limited, replays, budget stops).
`GET /metrics` exposes the same numbers in a Prometheus-style text format
(`odr_tool_errors_total{tool="sql_executor"}`, `odr_tokens_in_today`, `odr_task_latency_bucket{le="30"}`).

Not included: distributed tracing across services (OpenTelemetry), alerting, long-term metric storage.
