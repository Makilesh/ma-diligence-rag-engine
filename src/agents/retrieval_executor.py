"""
Agent 3 — Parallel Retrieval Executor.

No LLM calls. All embedding/reranking calls via run_in_executor (non-blocking).
Executes the retrieval pipeline: embed → hybrid search → RRF fusion → rerank.
"""

import time
import numpy as np

from src.vector_db.reranker import embed_texts_async, rerank_async
from src.vector_db.hybrid_search import hybrid_search, compute_sparse_bm25
from src.vector_db.rrf_fusion import flatten_deduplicate, reciprocal_rank_fusion
from src.vector_db.parent_child_retrieval import expand_context
from src.agents.retrieval_strategy import get_retrieval_config
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Smallest share of the final context any one sub-question is guaranteed. The
# guarantee is the whole point of decomposing: without it the dominant facet
# simply crowds the others out again and nothing has been gained.
MIN_CHUNKS_PER_SUB_QUESTION = 2


async def _retrieve_for_query(
    query: str,
    config: dict,
    deal_id: str,
    metadata_filters: dict,
) -> list[dict]:
    """
    Runs one full retrieval pass for a single query string.

    Embed → BM25 → hybrid search → RRF → rerank, returning chunks scored against
    *this* query. Reranking against the query that fetched them is the reason
    this is a separate function: a passage holding only the share price scores
    poorly against "what is the implied EV/EBITDA multiple", but scores highly
    against "what is the per-share merger consideration". Scoring each candidate
    against the sub-question that went looking for it is what makes the facet
    survive to the final context.

    Args:
        query: The query or sub-question to retrieve for.
        config: Retrieval config (weights, top-k values).
        deal_id: Deal scope.
        metadata_filters: Filters, already relaxed/augmented by the caller.

    Returns:
        Chunks sorted by descending reranker score, each carrying reranker_score.
    """
    import asyncio
    from src.vector_db.reranker import get_embed_executor
    from src.vector_db.hybrid_search import fetch_chunks_by_ids

    query_vector = (await embed_texts_async([query]))[0].tolist()

    loop = asyncio.get_running_loop()
    query_sparse = await loop.run_in_executor(
        get_embed_executor(), lambda: compute_sparse_bm25(query)
    )

    dense_results, sparse_results = await hybrid_search(
        query_text=query,
        query_vector=query_vector,
        query_sparse=query_sparse,
        deal_id=deal_id,
        metadata_filters=metadata_filters,
        top_k_dense=config.get("top_k_dense", 40),
        top_k_sparse=config.get("top_k_sparse", 40),
    )

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        dense_weight=config.get("dense_weight", 0.6),
        sparse_weight=config.get("sparse_weight", 0.4),
    )

    top_chunk_ids = [cid for cid, _ in fused[: config.get("reranker_top_k", 20)]]
    chunks = await fetch_chunks_by_ids(top_chunk_ids)
    if not chunks:
        return []

    passages = []
    for c in chunks:
        parts = []
        if c.get("source_file"):
            parts.append(f"Document: {c['source_file']}")
        if c.get("section_heading"):
            parts.append(f"Section: {c['section_heading']}")
        parts.append(c.get("text", ""))
        passages.append(" | ".join(parts))

    scores = await rerank_async(query, passages)

    # Threshold of 0.0 keeps every candidate for relative sorting; absolute
    # relevance is the Quality Assessor's judgement, not this node's.
    scored = []
    for chunk, score in zip(chunks, scores):
        chunk = dict(chunk)
        chunk["reranker_score"] = float(score)
        scored.append(chunk)

    scored.sort(key=lambda c: c["reranker_score"], reverse=True)
    return scored


def _merge_by_quota(
    per_query_results: list[list[dict]],
    final_top_k: int,
) -> list[dict]:
    """
    Merges per-sub-question results so every facet is represented.

    A plain union followed by a global top-k does not work, and failing to see
    why is how the original multi-hop weakness survived: the facet with the
    strongest passages simply fills all the slots, and the other facet — the one
    decomposition was supposed to rescue — is ranked out again. Measured before
    this change: asked for an EV/EBITDA multiple, retrieval returned EBITDA
    passages and no share price, and the engine correctly reported it could not
    compute the multiple.

    So results are interleaved round-robin instead: position 1 from each
    sub-question, then position 2, and so on. The first slots therefore cover
    every facet by construction, and truncation to the context budget removes
    the tail rather than an entire facet. Duplicates keep their highest score.

    Args:
        per_query_results: One ranked list per query, best first.
        final_top_k: Total chunks to return.

    Returns:
        Merged chunks, de-duplicated by chunk_id.
    """
    merged: list[dict] = []
    seen: dict[str, dict] = {}

    depth = max((len(r) for r in per_query_results), default=0)
    for rank in range(depth):
        for results in per_query_results:
            if rank >= len(results):
                continue
            chunk = results[rank]
            key = chunk.get("chunk_id") or f"{chunk.get('source_file')}:{rank}"

            if key in seen:
                # Same passage found by another sub-question — keep the stronger
                # score so ordering reflects its best evidence of relevance.
                if chunk["reranker_score"] > seen[key]["reranker_score"]:
                    seen[key]["reranker_score"] = chunk["reranker_score"]
                continue

            seen[key] = chunk
            merged.append(chunk)
            if len(merged) >= final_top_k:
                return merged

    return merged


async def retrieval_executor_node(state: AgentState) -> dict:
    """
    LangGraph node — executes full retrieval pipeline.

    Pipeline:
    1. Get or use existing retrieval config
    2. Embed query
    3. Compute BM25 sparse vector
    4. Hybrid search (dense + sparse in parallel)
    5. RRF fusion
    6. Rerank
    7. Context expansion (parent + sibling)

    Args:
        state: Current AgentState with current_query, query_type,
               parsed_intent, extracted_filters, deal_id.

    Returns:
        Partial state dict with retrieval results.
    """
    start = time.monotonic()
    query = state["current_query"]
    deal_id = state["deal_id"]

    logger.info(
        "Agent 3: Retrieval Executor starting",
        extra={"query": query, "deal_id": deal_id},
    )

    # Step 1: Get retrieval config (deterministic — Agent 2)
    if state.get("retrieval_config"):
        config = state["retrieval_config"]
    else:
        config = get_retrieval_config(state["query_type"], state["parsed_intent"])

    # Step 2: Build filters
    # include_pii is injected here rather than living in extracted_filters, because
    # Agents 1 and 6 deliberately strip that key from anything an LLM produced. The
    # only trusted source is the authenticated request, which lands on the state root.
    include_pii = bool(state.get("include_pii", False))
    metadata_filters = {**state.get("extracted_filters", {}), "include_pii": include_pii}

    # Progressive filter relaxation.
    #
    # document_category is Agent 1's *guess* at which kind of document holds the
    # answer, applied as a hard `must` condition. When that guess is wrong the
    # answer is removed from the search space entirely and no amount of query
    # rewriting can recover it — the rewriter only changes text, never filters.
    #
    # Measured on the golden set: legal_01 ("per-share merger consideration") was
    # classified financial, so the merger agreement — the one document containing
    # the answer — was filtered out; fin_05 (credit facility terms) was classified
    # legal and excluded the financials; comp_02 (valuation methodologies) was
    # classified financial and excluded the board deck. All three refused after
    # burning both rewrite attempts against a search space that could never
    # contain the answer.
    #
    # It is also compounding two error sources: an LLM guessing a category that
    # was itself assigned by a heuristic classifier at ingestion time.
    #
    # So: keep the filter on the first attempt, where it buys precision, and drop
    # it once retrieval has already failed and we are explicitly trading precision
    # for recall. deal_id, version and PII filters are NEVER relaxed — those are
    # isolation and compliance constraints, not relevance hints.
    if state.get("rewrite_iteration", 0) >= 1:
        dropped = metadata_filters.pop("document_category", None)
        if dropped:
            logger.info(
                "Relaxing document_category filter after failed retrieval",
                extra={"dropped_category": dropped,
                       "rewrite_iteration": state.get("rewrite_iteration")},
            )
    # Step 3: Plan the retrieval passes.
    #
    # For a pointed question this is a single pass on the query itself. For a
    # multi-hop or comparative question, Agent 1 has decomposed it into atomic
    # sub-questions, and each gets its own pass. The parent query is retained as
    # the first pass so passages relevant to the question as a whole — which no
    # single sub-question asks for — are still reachable.
    import asyncio

    sub_questions = [s for s in (state.get("sub_questions") or []) if s.strip()]
    retrieval_queries = [query] + sub_questions

    final_top_k = config.get("final_top_k", 10)

    if len(retrieval_queries) > 1:
        # Widen the per-pass budget so each facet can actually field candidates,
        # then let the quota merge decide the final composition.
        per_pass_k = max(
            MIN_CHUNKS_PER_SUB_QUESTION,
            final_top_k // len(retrieval_queries) + 1,
        )
        pass_config = {**config, "final_top_k": per_pass_k}
        logger.info(
            "Agent 3: decomposed retrieval",
            extra={"sub_questions": len(sub_questions), "per_pass_k": per_pass_k},
        )
    else:
        pass_config = config

    # A sub-question must not inherit the parent's document_category.
    #
    # That filter is Agent 1's guess at which document answers the question as a
    # whole. A sub-question exists precisely because it asks for a *different*
    # fact, which usually lives somewhere else — so applying the parent's guess
    # to it re-creates the problem decomposition was meant to solve.
    #
    # Measured: "implied EV/EBITDA multiple" decomposed correctly into share
    # price, share count and FY2023 EBITDA, but every pass inherited one category
    # filter and all three returned chunks from the same document. The engine
    # then reported it could not compute the multiple — decomposition had done
    # its job and the filter had thrown the result away.
    #
    # deal_id, version and PII filters still apply to every pass: those are
    # isolation and compliance constraints, not relevance hints.
    sub_question_filters = {
        k: v for k, v in metadata_filters.items() if k != "document_category"
    }

    def _filters_for(index: int) -> dict:
        return metadata_filters if index == 0 else sub_question_filters

    # Passes are independent, so run them concurrently. The embedding and
    # reranking thread pools are bounded, so this queues rather than oversubscribes.
    per_query_results = await asyncio.gather(*[
        _retrieve_for_query(q, pass_config, deal_id, _filters_for(i))
        for i, q in enumerate(retrieval_queries)
    ])

    if len(retrieval_queries) == 1:
        reranked = per_query_results[0][:final_top_k]
    else:
        reranked = _merge_by_quota(
            [r[:pass_config["final_top_k"]] for r in per_query_results],
            final_top_k=final_top_k,
        )
        # Present the merged context best-first; the round-robin order exists to
        # guarantee coverage during selection, not to dictate reading order.
        reranked.sort(key=lambda c: c["reranker_score"], reverse=True)

    # Step 7: Context expansion
    expanded = await expand_context(
        chunks=reranked,
        include_parents=config.get("use_parent_expansion", True),
        include_siblings=config.get("use_sibling_expansion", True),
        include_pii=include_pii,
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    candidates_per_pass = [len(r) for r in per_query_results]

    logger.info(
        "Agent 3: Retrieval Executor complete",
        extra={
            "retrieval_passes": len(retrieval_queries),
            "candidates_per_pass": candidates_per_pass,
            "reranked_count": len(reranked),
            "expanded_count": len(expanded),
            "sources": len({c.get("source_file") for c in reranked}),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    # dense_results / sparse_results / fused_results are no longer meaningful as
    # single lists: with decomposition there are N of each, one per pass. The
    # per-pass candidate counts carry the same diagnostic signal without
    # pretending a single fused ranking still exists.
    return {
        "retrieval_config": config,
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "retrieval_passes": [
            {"query": q, "candidates": n}
            for q, n in zip(retrieval_queries, candidates_per_pass)
        ],
        "reranked_results": reranked,
        "expanded_context": expanded,
        "agent_trace": [
            {
                "agent": "retrieval_executor",
                "elapsed_ms": round(elapsed_ms, 2),
                "retrieval_passes": len(retrieval_queries),
                "sub_questions": sub_questions,
                "reranked_count": len(reranked),
                "sources": sorted(
                    {c.get("source_file", "") for c in reranked} - {""}
                ),
            }
        ],
    }
