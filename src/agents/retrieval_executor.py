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

    # Step 2: Embed query
    query_embeddings = await embed_texts_async([query])
    query_vector = query_embeddings[0].tolist()

    # Step 3: BM25 sparse vector
    import asyncio
    from src.vector_db.reranker import get_embed_executor
    loop = asyncio.get_running_loop()
    query_sparse = await loop.run_in_executor(
        get_embed_executor(),
        lambda: compute_sparse_bm25(query),
    )

    # Step 4: Hybrid search
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
    dense_results, sparse_results = await hybrid_search(
        query_text=query,
        query_vector=query_vector,
        query_sparse=query_sparse,
        deal_id=deal_id,
        metadata_filters=metadata_filters,
        top_k_dense=config.get("top_k_dense", 40),
        top_k_sparse=config.get("top_k_sparse", 40),
    )

    # Step 5: RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        dense_weight=config.get("dense_weight", 0.6),
        sparse_weight=config.get("sparse_weight", 0.4),
    )

    # Get top candidates for reranking
    reranker_top_k = config.get("reranker_top_k", 20)
    top_chunk_ids = [chunk_id for chunk_id, _ in fused[:reranker_top_k]]

    # Fetch full chunk payloads for reranking
    from src.vector_db.hybrid_search import fetch_chunks_by_ids
    chunks = await fetch_chunks_by_ids(top_chunk_ids)

    # Step 6: Rerank
    if chunks:
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

        # Apply threshold and sort
        # We use a threshold of 0.0 to keep all retrieved candidates for relative sorting,
        # relying on the Quality Assessor and LLM to determine absolute relevance.
        threshold = 0.0
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            s = float(score)
            if s >= threshold:
                chunk["reranker_score"] = s
                scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Limit to final_top_k
        final_top_k = config.get("final_top_k", 10)
        reranked = scored_chunks[:final_top_k]
    else:
        reranked = []

    # Step 7: Context expansion
    expanded = await expand_context(
        chunks=reranked,
        include_parents=config.get("use_parent_expansion", True),
        include_siblings=config.get("use_sibling_expansion", True),
        include_pii=include_pii,
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    logger.info(
        "Agent 3: Retrieval Executor complete",
        extra={
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused),
            "reranked_count": len(reranked),
            "expanded_count": len(expanded),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    return {
        "retrieval_config": config,
        "dense_results": [
            {"id": str(r.id), "score": r.score} for r in dense_results
        ],
        "sparse_results": [
            {"id": str(r.id), "score": r.score} for r in sparse_results
        ],
        "fused_results": [
            {"chunk_id": cid, "rrf_score": s} for cid, s in fused
        ],
        "reranked_results": reranked,
        "expanded_context": expanded,
        "agent_trace": [
            {
                "agent": "retrieval_executor",
                "elapsed_ms": round(elapsed_ms, 2),
                "dense_count": len(dense_results),
                "reranked_count": len(reranked),
            }
        ],
    }
