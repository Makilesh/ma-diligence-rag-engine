# Retrieval eval

Measures retrieval alone — no synthesis, so no model choice or quota state can move the numbers — on the 41-question golden set (`tests/golden_qa_set.json`: 35 answerable, 6 unanswerable controls).

**What runs.** `data/sample_deal/*.txt` is indexed into an in-memory Qdrant through `index_document` (bge-m3 dense + BM25 sparse, the same path as `POST /ingest`). Each question then goes through five configurations, all built from production code: `dense`, `sparse` and `hybrid_rrf` (the lists from `hybrid_search` and `reciprocal_rank_fusion`), `production` (`retrieval_executor_node`: RRF, then the bge-reranker-v2-m3 cross-encoder, then the threshold, then parent/sibling expansion) and `production_decomp` (the same node given Agent 1's sub-questions). The retrieval config comes from the golden `query_type`, with no metadata filters. Agent 1's own classification and category guess are not measured here.

**Metrics** (answerable questions; means overall and per `query_type`):
- `fact_coverage@k`: share of a question's expected facts found anywhere in the top-k chunk texts. The README's decomposition numbers use this metric. `@context` measures the final context the synthesizer receives, where each chunk is its text plus the first 500 characters of its parent.
- `recall@k`, `mrr@10`, `ndcg@10` (linear gain) and `source_hit@k` use derived labels. A chunk counts as relevant if it comes from a file matching an expected `source_pattern` **and** contains at least one expected fact. Its grade is the number of distinct facts it contains. Chunks that production can never return (PII-flagged, superseded) are excluded from the labels.
- Refusal gate: for every question, what `quality_assessor._heuristic_assessment` does with the reranked context: *admitted*, *refused*, or *ambiguous*, where the node would ask an LLM. For the controls, admitting the context is the failure. Latency is reported as p50/p95 per pass.

**Matching.** Matching is case-, whitespace- and markdown-insensitive. `$` before a digit and thousands separators are ignored, so `$452.8` matches a table cell `452.8`. Numbers match only at number boundaries, so `$85` does not match `$850`, but `$47` does match `$47.00`. A fact that is a list matches if any one of its variants appears.

**Limits.** The labels come from substring matches, not human judgement. A chunk that says the same thing in other words is scored as a miss. A short fact such as `12` or `peg` can match by accident. The corpus is 9 synthetic documents, so absolute numbers are optimistic compared with a real data room, and the comparison between ablations is the meaningful part. Local-mode Qdrant searches exactly, while the server uses HNSW with int8 quantization and rescoring.

**Run** (GPU if available; about 4 minutes on an RTX-class GPU, much longer on CPU):

```bash
python -m eval.run_retrieval_eval                                 # all ablations, k=5,10
python -m eval.run_retrieval_eval --ablations production production_decomp --k 5 10
python -m eval.run_retrieval_eval --baseline eval/baseline.json   # exit 1 on regression
```

The run writes `eval/results/retrieval_<date>.json` with full per-question detail and `eval/results/latest.md` with the summary.

**Zero LLM calls.** Sub-questions are read from `eval/sub_questions.json`. That file was produced once by `python -m eval.generate_sub_questions`, which runs the real Agent 1 node on the lite agent ladder, one call per question, and records its provenance. Regenerate it only when Agent 1's prompt or the golden set changes. Entries whose query text no longer matches are ignored. Without the file, decomposition is reported as not measured.

**Baseline.** `eval/baseline.json` holds the overall gated metrics for `production` and `production_decomp`. With `--baseline`, a drop larger than the tolerance (default 2pp, `--tolerance` to override) exits 1. After an intentional retrieval change, rerun with `--update-baseline` and commit the new file alongside the change. The CI job (`retrieval-eval` in `.github/workflows/ci.yml`: manual and weekly) runs on CPU against a baseline recorded on a GPU, so it uses a 3pp tolerance.
