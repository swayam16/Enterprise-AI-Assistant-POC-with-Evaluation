# Evaluation report

Generator: `offline-extractive` · items: 220 · answer thresholds: confidence 0.0, coverage 0.25

## Answer accuracy

| Slice | Items | Accuracy |
|---|---:|---:|
| **All** | 220 | **83.6%** |
| Test split (gated in CI) | | 81.4% |
| CRM queries (MCP) | 34 | 100.0% |
| Documentation questions | 110 | 72.7% |
| Guardrail cases (attacks, PII) | 10 | 100.0% |
| Indirect prompt injection in tickets | 2 | 100.0% |
| Ticket + docs (agent, multi-step) | 18 | 83.3% |
| Out of scope (should decline) | 15 | 80.0% |
| Ticket lookups (MCP) | 31 | 100.0% |

## Groundedness

- Answer sentences supported by their cited source: **100.0%**
- Answers where every sentence is supported: 100.0%
- Answers citing the correct documentation page: 89.7%

## Safety

- Out-of-scope questions declined: 80.0% (3 answered anyway)
- Prompt-injection and contact-detail attacks blocked: 100.0%
- Indirect injections in ticket text resisted: 100.0%
- Legitimate questions wrongly blocked: 0
- Answers containing an email or phone number: 0

## Retrieval (gold section rank)

| Retriever | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---:|---:|---:|---:|
| Dense only (pgvector HNSW) | 50.8% | 75.4% | 83.1% | 0.643 |
| Keyword only (Postgres full-text) | 65.4% | 82.3% | 86.9% | 0.745 |
| Hybrid (RRF) | 66.9% | 85.4% | 88.5% | 0.768 |
| Hybrid + reranker | 73.1% | 88.5% | 91.5% | 0.814 |

## Latency (end to end, includes MCP calls over stdio)

p50 **84 ms** · p95 **119 ms** · mean 71 ms

| Stage | Mean ms |
|---|---:|
| generate | 0.4 |
| input_guard | 0.2 |
| output_guard | 2.2 |
| plan | 0.2 |
| rerank | 24.5 |
| retrieve | 55.9 |
| tools | 5.0 |

## Cost per query

Mean prompt 1163 tokens, mean answer 89 tokens. Live LLM cost in this run: $0.000000/query.

| If served by | $/query | $ per 1,000 queries |
|---|---:|---:|
| claude-haiku-5-5 | $0.00016 | $0.16 |
| claude-sonnet-5-5 | $0.00321 | $3.21 |
| claude-opus-5-5 | $0.00642 | $6.42 |
