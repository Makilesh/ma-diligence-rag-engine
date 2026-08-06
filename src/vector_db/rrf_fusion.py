# src/vector_db/rrf_fusion.py
# Pure rank-based RRF only — no score normalization
# Implemented in Phase 3, BUILD ORDER step 9

"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Provides two functions:
  1. flatten_deduplicate() — merges multiple ranked lists (e.g., from query
     expansions) into one deduplicated list, keeping the highest-scoring
     duplicate.
  2. reciprocal_rank_fusion() — combines dense and sparse ranked lists using
     pure rank-based RRF. Raw retrieval scores are IGNORED; only rank
     positions determine the final ordering.

IMPORTANT: Do NOT add score normalization before reciprocal_rank_fusion().
Normalizing scores converts RRF into a different algorithm and reintroduces
the scale problem that RRF is designed to avoid.
"""

from __future__ import annotations

import time

from qdrant_client.models import ScoredPoint

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def flatten_deduplicate(
    result_lists: list[list[ScoredPoint]],
) -> list[ScoredPoint]:
    """
    Merges multiple ranked result lists into one deduplicated list.
    When the same chunk_id appears in multiple lists (e.g., from query expansions),
    keeps the instance with the highest individual score.
    Returns sorted by score descending.

    NOTE: This is the one place raw retrieval scores (cosine similarity for dense,
    BM25 for sparse) drive a ranking decision, which is slightly in tension with
    the "pure rank-based, no normalization" RRF principle stated elsewhere.
    This is acceptable because: (1) cosine scores ARE comparable across queries
    against the same corpus, and (2) this deduplication happens BEFORE RRF, so
    the raw-score comparison only determines which duplicate to keep, not the
    final ranking. RRF itself still uses ranks only.

    Used to merge results from: original query + N query expansions,
    before feeding the combined pool into RRF fusion.

    Args:
        result_lists: Multiple ranked lists from parallel searches.
    Returns:
        Single deduplicated list sorted by score descending.
    """
    start = time.monotonic()
    total_input = sum(len(rl) for rl in result_lists)
    logger.info(
        "flatten_deduplicate started",
        extra={
            "num_lists": len(result_lists),
            "total_input_points": total_input,
        },
    )

    seen: dict[str, ScoredPoint] = {}
    for result_list in result_lists:
        for point in result_list:
            chunk_id = point.payload["chunk_id"]
            if chunk_id not in seen or point.score > seen[chunk_id].score:
                seen[chunk_id] = point

    deduplicated = sorted(seen.values(), key=lambda p: p.score, reverse=True)

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "flatten_deduplicate completed",
        extra={
            "input_points": total_input,
            "output_points": len(deduplicated),
            "duplicates_removed": total_input - len(deduplicated),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return deduplicated


def reciprocal_rank_fusion(
    dense_results: list[ScoredPoint],
    sparse_results: list[ScoredPoint],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """
    Pure rank-based Reciprocal Rank Fusion.
    Scores from dense and sparse search are IGNORED — only rank positions matter.
    This makes RRF inherently scale-agnostic: cosine similarity (bounded [-1,1])
    and BM25 scores (unbounded) cannot distort each other because neither is used.

    The formula: rrf_score(d) = Σ weight_i / (k + rank_i(d))
    where k=60 prevents very high-ranked items from dominating absolutely.

    Do NOT add score normalization before this function — it converts RRF
    into a different algorithm and reintroduces the scale problem.

    Args:
        dense_results: Ranked list from dense vector search (already deduplicated).
        sparse_results: Ranked list from BM25 sparse search (already deduplicated).
        k: Smoothing constant. 60 is standard; increase to reduce top-rank dominance.
        dense_weight: Contribution weight for dense channel.
        sparse_weight: Contribution weight for sparse channel.
    Returns:
        List of (chunk_id, rrf_score) sorted by score descending.
    Raises:
        ValueError: If both result lists are empty.
    """
    start = time.monotonic()
    logger.info(
        "reciprocal_rank_fusion started",
        extra={
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "k": k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        },
    )

    if not dense_results and not sparse_results:
        # "Nothing matched" is a legitimate retrieval outcome, not a programming
        # error — a query can be well-formed and simply have no answer in the
        # data room, and restrictive metadata filters can empty the candidate set
        # on their own. Raising here turned that into a 500; returning empty lets
        # the Quality Assessor see zero chunks and take the refusal path, which
        # is the behaviour the pipeline is designed around.
        #
        # Found by a control question ("headcount by international office") whose
        # answer genuinely does not exist in the corpus.
        logger.info(
            "reciprocal_rank_fusion: no candidates from either retriever — "
            "returning empty for the refusal path",
        )
        return []

    scores: dict[str, float] = {}

    for rank, point in enumerate(dense_results, start=1):
        chunk_id = point.payload["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight / (k + rank)

    for rank, point in enumerate(sparse_results, start=1):
        chunk_id = point.payload["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "reciprocal_rank_fusion completed",
        extra={
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused),
            "top_score": round(fused[0][1], 6) if fused else 0.0,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return fused
