# Retrieval eval — 2026-09-23

Golden set: 35 answerable + 6 control questions · corpus: 9 documents, 112 chunks (112 retrievable) · device: cuda (NVIDIA GeForce RTX 5070 Ti Laptop GPU) · LLM calls: 0 · fact-coverage ceiling (facts present in any retrievable chunk): 99.0%

Percentages. fact_cov = share of expected facts present in the top-k chunk texts (any file); ctx = the final context the synthesizer receives (reranked + parent/sibling expansion, formatted by the synthesizer). recall/MRR/nDCG use substring-derived chunk labels — see eval/README.md.

| ablation | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@5 | recall@10 | mrr@10 | ndcg@10 | source_hit@5 | source_hit@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-only | 72.7 | 85.6 | — | 49.7 | 68.5 | 58.2 | 56.6 | 94.3 | 98.6 | 21 | 23 |
| sparse-only (BM25) | 68.3 | 73.1 | — | 41.8 | 46.6 | 52.0 | 45.5 | 80.0 | 82.9 | 21 | 23 |
| hybrid RRF (no rerank) | 73.8 | 85.3 | — | 47.1 | 61.5 | 60.4 | 54.5 | 82.9 | 91.4 | 22 | 24 |
| production (hybrid + rerank) | 87.8 | 90.7 | 91.6 | 69.6 | 76.4 | 78.7 | 71.3 | 95.7 | 95.7 | 788 | 1057 |
| production + decomposition | 87.4 | 91.6 | 92.6 | 69.3 | 77.2 | 77.2 | 72.3 | 95.7 | 95.7 | 832 | 3949 |

## production (hybrid + rerank) — by query type

| type | n | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| comparative | 5 | 84.0 | 84.0 | 84.0 | 80.0 | 80.0 | 60.6 |
| financial | 8 | 90.0 | 90.0 | 90.0 | 87.5 | 62.5 | 69.8 |
| legal | 8 | 100.0 | 100.0 | 100.0 | 80.1 | 93.8 | 84.4 |
| multi_hop | 10 | 80.0 | 83.3 | 86.7 | 69.0 | 85.0 | 72.3 |
| summary | 4 | 83.3 | 100.0 | 100.0 | 61.3 | 63.3 | 59.1 |
| **all** | 35 | 87.8 | 90.7 | 91.6 | 76.4 | 78.7 | 71.3 |

## production + decomposition — by query type

| type | n | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| comparative | 5 (5 dec.) | 68.0 | 84.0 | 84.0 | 80.0 | 63.3 | 56.0 |
| financial | 8 (2 dec.) | 90.0 | 90.0 | 90.0 | 87.5 | 68.8 | 74.4 |
| legal | 8 (2 dec.) | 100.0 | 100.0 | 100.0 | 80.1 | 85.4 | 81.7 |
| multi_hop | 10 (9 dec.) | 86.7 | 86.7 | 90.0 | 71.5 | 90.0 | 76.5 |
| summary | 4 (0 dec.) | 83.3 | 100.0 | 100.0 | 61.3 | 63.3 | 59.1 |
| **all** | 35 (18 dec.) | 87.4 | 91.6 | 92.6 | 77.2 | 77.2 | 72.3 |

Decomposition, per question on fact_coverage@context: 1 improved, 0 regressed, 34 unchanged (18 of the 35 answerable questions were decomposed by Agent 1).

Changed: mh_05 66.7→100.0

## Refusal gate (Quality Assessor heuristic, no LLM)

| ablation | group | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|
| production (hybrid + rerank) | controls | 3 | 1 | 2 |
| production (hybrid + rerank) | answerable | 33 | 1 | 1 |
| production + decomposition | controls | 3 | 1 | 2 |
| production + decomposition | answerable | 34 | 1 | 0 |

Controls (production): ctrl_01 max=0.0261 refused, ctrl_02 max=0.956 admitted, ctrl_03 max=0.0525 llm_fallback, ctrl_04 max=0.3459 admitted, ctrl_05 max=0.8807 admitted, ctrl_06 max=0.0378 refused

