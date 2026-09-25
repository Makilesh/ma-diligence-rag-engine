# Answerability gate — dev set, 2026-09-24

eval\answerability_dev.json: 26 answerable (evidence in context), 28 unanswerable, 3 answerable-but-retrieval-missed, 1 excluded (debatable label). Adopted: `Does `passage` state the information needed to answer `question`?` over the top 3 reranked chunks, veto below 0.35. LLM calls: 0.

| label | gate | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|
| answerable | heuristic | 24 | 2 | 0 |
| answerable | + Laya | 24 | 2 | 0 |
| unanswerable | heuristic | 20 | 7 | 1 |
| unanswerable | + Laya | 14 | 4 | 10 |
| retrieval_miss | heuristic | 3 | 0 | 0 |
| retrieval_miss | + Laya | 2 | 0 | 1 |

## P1: `Does `passage` state the information needed to answer `question`?` — AUC 0.799

| veto below | false vetoes (answerable) | unanswerable caught | of which heuristic-admitted | of which ambiguous | retrieval misses vetoed |
|---|---|---|---|---|---|
| 0.25 | 0/26 | 6/28 | 3/20 | 3 | 1 |
| 0.3 | 0/26 | 6/28 | 3/20 | 3 | 1 |
| 0.325 | 0/26 | 8/28 | 5/20 | 3 | 1 |
| 0.35 | 0/26 | 9/28 | 6/20 | 3 | 1 |
| 0.375 | 0/26 | 9/28 | 6/20 | 3 | 1 |
| 0.4 | 0/26 | 10/28 | 7/20 | 3 | 2 |
| 0.425 | 1/26 | 11/28 | 8/20 | 3 | 2 |
| 0.45 | 2/26 | 13/28 | 9/20 | 4 | 2 |

## P2: `Does `passage` contain the answer to `question`?` — AUC 0.778

| veto below | false vetoes (answerable) | unanswerable caught | of which heuristic-admitted | of which ambiguous | retrieval misses vetoed |
|---|---|---|---|---|---|
| 0.25 | 0/26 | 6/28 | 3/20 | 3 | 1 |
| 0.3 | 0/26 | 8/28 | 5/20 | 3 | 1 |
| 0.325 | 0/26 | 9/28 | 6/20 | 3 | 1 |
| 0.35 | 2/26 | 10/28 | 7/20 | 3 | 1 |
| 0.375 | 2/26 | 11/28 | 8/20 | 3 | 1 |
| 0.4 | 2/26 | 12/28 | 8/20 | 4 | 2 |
| 0.425 | 3/26 | 13/28 | 9/20 | 4 | 2 |
| 0.45 | 3/26 | 13/28 | 9/20 | 4 | 2 |

## P3: `Can `question` be answered using only the facts stated in `passage`?` — AUC 0.811

| veto below | false vetoes (answerable) | unanswerable caught | of which heuristic-admitted | of which ambiguous | retrieval misses vetoed |
|---|---|---|---|---|---|
| 0.25 | 0/26 | 5/28 | 3/20 | 2 | 1 |
| 0.3 | 0/26 | 6/28 | 3/20 | 3 | 1 |
| 0.325 | 0/26 | 8/28 | 5/20 | 3 | 1 |
| 0.35 | 0/26 | 9/28 | 6/20 | 3 | 1 |
| 0.375 | 1/26 | 11/28 | 7/20 | 4 | 2 |
| 0.4 | 1/26 | 11/28 | 7/20 | 4 | 2 |
| 0.425 | 3/26 | 15/28 | 10/20 | 5 | 2 |
| 0.45 | 4/26 | 18/28 | 12/20 | 6 | 2 |

Laya latency per assessment (cuda): p50 61 ms, p95 79 ms (n=57).

Lowest-scoring answerable questions: dev_a38 0.4087, dev_a40 0.4328, dev_a34 0.4581
