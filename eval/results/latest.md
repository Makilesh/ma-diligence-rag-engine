# Retrieval eval — 2026-09-23

Golden set: 35 answerable + 6 control questions · corpus: 9 documents, 112 chunks (112 retrievable) · device: cuda (NVIDIA GeForce RTX 5070 Ti Laptop GPU) · LLM calls: 0 · fact-coverage ceiling (facts present in any retrievable chunk): 99.0%

Percentages. fact_cov = share of expected facts present in the top-k chunk texts (any file); ctx = the final context the synthesizer receives (reranked + parent/sibling expansion, formatted by the synthesizer). recall/MRR/nDCG use substring-derived chunk labels — see eval/README.md.

| ablation | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@5 | recall@10 | mrr@10 | ndcg@10 | source_hit@5 | source_hit@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-only | 72.7 | 85.6 | — | 49.7 | 68.5 | 58.2 | 56.6 | 94.3 | 98.6 | 23 | 26 |
| sparse-only (BM25) | 68.3 | 73.1 | — | 41.8 | 46.6 | 52.0 | 45.5 | 80.0 | 82.9 | 23 | 26 |
| hybrid RRF (no rerank) | 73.8 | 85.3 | — | 47.1 | 61.5 | 60.4 | 54.5 | 82.9 | 91.4 | 24 | 28 |
| production (hybrid + rerank) | 81.7 | 81.7 | 85.2 | 61.2 | 62.0 | 78.1 | 65.7 | 95.7 | 95.7 | 804 | 1131 |
| production + decomposition | 84.1 | 84.1 | 87.1 | 63.1 | 64.5 | 76.7 | 67.8 | 95.7 | 95.7 | 887 | 4295 |

## production (hybrid + rerank) — by query type

| type | n | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| comparative | 5 | 68.0 | 68.0 | 70.0 | 66.7 | 80.0 | 52.7 |
| financial | 8 | 87.5 | 87.5 | 87.5 | 87.5 | 62.5 | 69.8 |
| legal | 8 | 100.0 | 100.0 | 100.0 | 58.8 | 93.8 | 80.8 |
| multi_hop | 10 | 74.3 | 74.3 | 83.3 | 58.7 | 85.0 | 65.4 |
| summary | 4 | 69.4 | 69.4 | 75.0 | 19.6 | 58.3 | 44.3 |
| **all** | 35 | 81.7 | 81.7 | 85.2 | 62.0 | 78.1 | 65.7 |

## production + decomposition — by query type

| type | n | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| comparative | 5 (5 dec.) | 68.0 | 68.0 | 70.0 | 66.7 | 63.3 | 49.2 |
| financial | 8 (2 dec.) | 87.5 | 87.5 | 87.5 | 87.5 | 68.8 | 74.4 |
| legal | 8 (2 dec.) | 100.0 | 100.0 | 100.0 | 60.6 | 85.4 | 78.5 |
| multi_hop | 10 (9 dec.) | 82.7 | 82.7 | 90.0 | 66.2 | 90.0 | 72.5 |
| summary | 4 (0 dec.) | 69.4 | 69.4 | 75.0 | 19.6 | 58.3 | 44.3 |
| **all** | 35 (18 dec.) | 84.1 | 84.1 | 87.1 | 64.5 | 76.7 | 67.8 |

Decomposition, per question on fact_coverage@context: 2 improved, 0 regressed, 33 unchanged (18 of the 35 answerable questions were decomposed by Agent 1).

Changed: mh_05 66.7→100.0, mh_06 33.3→66.7

## Refusal gate (Quality Assessor heuristic, no LLM)

| ablation | group | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|
| production (hybrid + rerank) | controls | 3 | 1 | 2 |
| production (hybrid + rerank) | answerable | 33 | 1 | 1 |
| production + decomposition | controls | 3 | 1 | 2 |
| production + decomposition | answerable | 34 | 1 | 0 |

Controls (production): ctrl_01 max=0.0261 refused, ctrl_02 max=0.956 admitted, ctrl_03 max=0.0525 llm_fallback, ctrl_04 max=0.3459 admitted, ctrl_05 max=0.8807 admitted, ctrl_06 max=0.0378 refused

