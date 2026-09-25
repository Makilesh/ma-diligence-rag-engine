# Answerability gate — dev set, 2026-09-25

eval\answerability_dev.json: 2 answerable (evidence in context), 5 unanswerable, 1 answerable-but-retrieval-missed, 0 excluded (debatable label). Adopted: `Does `passage` state the information needed to answer `question`?` over the top 3 reranked chunks, veto below 0.35. LLM calls: 0.

| label | gate | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|
| answerable | heuristic | 2 | 0 | 0 |
| answerable | + Laya | 2 | 0 | 0 |
| unanswerable | heuristic | 4 | 1 | 0 |
| unanswerable | + Laya | 2 | 0 | 3 |
| retrieval_miss | heuristic | 1 | 0 | 0 |
| retrieval_miss | + Laya | 1 | 0 | 0 |

## P1: `Does `passage` state the information needed to answer `question`?` — AUC 1.000

| veto below | false vetoes (answerable) | unanswerable caught | of which heuristic-admitted | of which ambiguous | retrieval misses vetoed |
|---|---|---|---|---|---|
| 0.25 | 0/2 | 3/5 | 2/4 | 1 | 0 |
| 0.3 | 0/2 | 3/5 | 2/4 | 1 | 0 |
| 0.325 | 0/2 | 3/5 | 2/4 | 1 | 0 |
| 0.35 | 0/2 | 3/5 | 2/4 | 1 | 0 |
| 0.375 | 0/2 | 3/5 | 2/4 | 1 | 0 |
| 0.4 | 0/2 | 3/5 | 2/4 | 1 | 0 |
| 0.425 | 0/2 | 4/5 | 3/4 | 1 | 0 |
| 0.45 | 0/2 | 4/5 | 3/4 | 1 | 0 |

Laya latency per assessment (cpu, 2 threads): p50 6634 ms, p95 8186 ms (n=8).

Lowest-scoring answerable questions: dev_a05 0.5162, dev_a03 0.6002
