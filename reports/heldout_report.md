# Evaluation report

Generator: `offline-extractive` · items: 35 · answer thresholds: confidence 0.0, coverage 0.25

## Answer accuracy

| Slice | Items | Accuracy |
|---|---:|---:|
| **All** | 35 | **74.3%** |
| Test split (gated in CI) | | 74.3% |
| Documentation questions | 30 | 76.7% |
| Out of scope (should decline) | 5 | 60.0% |

## Groundedness

- Answer sentences supported by their cited source: **100.0%**
- Answers where every sentence is supported: 100.0%
- Answers citing the correct documentation page: 89.7%

## Safety

- Out-of-scope questions declined: 60.0% (2 answered anyway)
- Prompt-injection and contact-detail attacks blocked: n/a
- Indirect injections in ticket text resisted: n/a
- Legitimate questions wrongly blocked: 0
- Answers containing an email or phone number: 0

## Retrieval (gold section rank)

| Retriever | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---:|---:|---:|---:|
| Dense only (pgvector HNSW) | 60.0% | 70.0% | 76.7% | 0.672 |
| Keyword only (Postgres full-text) | 73.3% | 80.0% | 83.3% | 0.777 |
| Hybrid (RRF) | 73.3% | 83.3% | 83.3% | 0.786 |
| Hybrid + reranker | 80.0% | 86.7% | 96.7% | 0.858 |

## Latency (end to end, includes MCP calls over stdio)

p50 **93 ms** · p95 **113 ms** · mean 94 ms

| Stage | Mean ms |
|---|---:|
| generate | 0.6 |
| input_guard | 0.2 |
| output_guard | 2.7 |
| plan | 0.2 |
| rerank | 24.3 |
| retrieve | 57.2 |
| tools | 0.0 |

## Cost per query

Mean prompt 1463 tokens, mean answer 111 tokens. Live LLM cost in this run: $0.000000/query.

| If served by | $/query | $ per 1,000 queries |
|---|---:|---:|
| claude-haiku-5-5 | $0.00020 | $0.20 |
| claude-sonnet-5-5 | $0.00404 | $4.04 |
| claude-opus-5-5 | $0.00808 | $8.08 |
