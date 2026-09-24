"""
Hybrid search module — BM25 sparse vectors and search execution.

FastEmbed BM25 sparse vector computation for Qdrant native sparse support.
The full hybrid search function is implemented here alongside the BM25 encoder.

Metadata constraints (fiscal year, document category, etc.) are passed as
Qdrant filter parameters directly into the dense and sparse search calls.
There is NO third "metadata retrieval channel" via scroll.
"""

import asyncio
import time

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from src.vector_db.constants import (
    COLLECTION_NAME,
    QUANTIZED_SEARCH_PARAMS,
)
from src.vector_db.qdrant_client import get_qdrant_client
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# BM25 Sparse Embedding — Module-level model, load once at startup
# ==============================================================================

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    """
    Lazily loads the BM25 sparse embedding model.

    Returns:
        SparseTextEmbedding model for Qdrant/bm25.
    """
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading Qdrant/bm25 sparse embedding model")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("Qdrant/bm25 sparse embedding model loaded successfully")
    return _bm25_model


def compute_sparse_bm25(text: str) -> SparseVector:
    """
    Computes BM25 sparse vector for a text string.
    FastEmbed returns embeddings as objects with .indices and .values arrays.
    These must be converted to lists before passing to Qdrant.

    NOTE: This is a synchronous CPU computation. When called from async context,
    wrap in run_in_executor using the _embed_executor pool.

    Args:
        text: Input text to encode.

    Returns:
        SparseVector with indices and values for Qdrant sparse search.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text:
        raise ValueError("Cannot compute BM25 for empty text")

    model = _get_bm25_model()
    embeddings = list(model.embed([text]))
    return SparseVector(
        indices=embeddings[0].indices.tolist(),
        values=embeddings[0].values.tolist(),
    )


def compute_sparse_bm25_batch(texts: list[str], batch_size: int = 64) -> list[SparseVector]:
    """
    Computes BM25 sparse vectors for many texts in one model call.

    Ingestion used to call compute_sparse_bm25 once per chunk, each call a
    separate executor round-trip; FastEmbed batches internally, so one call
    for the whole document is both simpler and faster.

    NOTE: Synchronous CPU work — call from a worker thread in async code.

    Args:
        texts: Non-empty texts to encode.
        batch_size: FastEmbed batch size.

    Returns:
        SparseVectors in the same order as texts.

    Raises:
        ValueError: If any text is empty.
    """
    if any(not t for t in texts):
        raise ValueError("Cannot compute BM25 for empty text")
    if not texts:
        return []

    model = _get_bm25_model()
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in model.embed(texts, batch_size=batch_size)
    ]


# ==============================================================================
# Filter Building
# ==============================================================================


VALID_DOCUMENT_CATEGORIES = frozenset({
    "financial", "legal", "board", "audit", "regulatory", "operational", "other",
})

# Payload fields a query may narrow on. Everything else in metadata_filters is
# dropped: the dict is largely LLM-authored (Agent 1 / Agent 6), and a key that
# is not on the payload (fiscal_year, currency) would silently match nothing.
ALLOWED_FILTER_KEYS = frozenset({
    "document_category",
    "is_table",
    "content_type",
    "doc_id",
})


def _valid_filter_value(key: str, value) -> bool:
    """
    Checks a whitelisted filter value has a type/value the payload can match.

    Args:
        key: Filter key (already whitelisted).
        value: Scalar or list value.

    Returns:
        True if the value should become a filter condition.
    """
    values = value if isinstance(value, list) else [value]
    if not values:
        return False
    if key == "document_category":
        return all(v in VALID_DOCUMENT_CATEGORIES for v in values)
    if key == "is_table":
        return all(v in (0, 1) and not isinstance(v, float) for v in values)
    return all(isinstance(v, str) and v for v in values)


def _build_filter(
    deal_id: str,
    metadata_filters: dict,
    include_superseded: bool = False,
) -> Filter:
    """
    Builds a Qdrant Filter from deal_id (always applied) and optional metadata.
    deal_id is mandatory — never execute a search without it.

    Default exclusions applied automatically:
    - is_current_version=1 — ALWAYS, unless the trusted `include_superseded`
      argument is set. metadata_filters cannot turn it off: that dict comes
      from the LLM, and Agent 1's schema used to echo "is_current_version": 1,
      whose mere presence skipped this condition while the whitelist then
      discarded it — superseded documents were searchable on every query.
    - contains_pii=0 (PII-flagged content excluded by default per compliance policy;
      pass include_pii=True in metadata_filters to override for authorized users.
      The retrieval executor sets that key from the authenticated request after
      discarding any LLM-supplied value.)

    Args:
        deal_id: Mandatory deal isolation filter.
        metadata_filters: Dict of payload fields to filter on (untrusted keys
                          are dropped; see ALLOWED_FILTER_KEYS).
        include_superseded: Trusted, non-LLM override to search superseded
                            versions too (e.g. an explicit version-history view).

    Returns:
        Qdrant Filter with all conditions applied.
    """
    conditions = [FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]

    if not include_superseded:
        conditions.append(
            FieldCondition(key="is_current_version", match=MatchValue(value=1))
        )

    # CRITICAL: Exclude PII-flagged content by default.
    # NOTE: Use .get() on a shallow copy — NOT .pop() on the original dict.
    # _build_filter() is called once per query expansion via asyncio.gather().
    # .pop() on the shared dict would consume include_pii on the first call.
    local_filters = dict(metadata_filters)  # shallow copy — never mutate caller's dict
    include_pii = local_filters.pop("include_pii", False)
    if not include_pii:
        conditions.append(
            FieldCondition(key="contains_pii", match=MatchValue(value=0))
        )

    dropped = []
    for key, value in local_filters.items():
        if value is None:
            continue
        if key not in ALLOWED_FILTER_KEYS or not _valid_filter_value(key, value):
            dropped.append(key)
            continue
        if key == "is_table":
            # Stored as integer 0/1; a JSON `true` would never match it.
            value = [int(v) for v in value] if isinstance(value, list) else int(value)
        if isinstance(value, list):
            conditions.append(
                FieldCondition(key=key, match=MatchAny(any=value))
            )
        else:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    if dropped:
        logger.debug("Ignored filter keys", extra={"dropped_filter_keys": dropped})

    return Filter(must=conditions)


# ==============================================================================
# Hybrid Search
# ==============================================================================


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    query_sparse: SparseVector,
    deal_id: str,
    metadata_filters: dict,
    top_k_dense: int = 40,
    top_k_sparse: int = 40,
    client: AsyncQdrantClient | None = None,
    include_superseded: bool = False,
) -> tuple[list[ScoredPoint], list[ScoredPoint]]:
    """
    Executes dense and sparse searches in parallel with metadata constraints
    applied as Qdrant filter parameters — NOT as a separate retrieval channel.

    Args:
        query_text: Original query string (used for logging).
        query_vector: Dense embedding of the query.
        query_sparse: BM25 sparse vector of the query.
        deal_id: Mandatory deal isolation filter.
        metadata_filters: Dict of payload fields to filter on
                          (e.g., {"document_category": "financial"}). Version
                          filtering is not controlled here — see include_superseded.
        top_k_dense: Number of candidates from dense search.
        top_k_sparse: Number of candidates from sparse search.
        client: AsyncQdrantClient instance.
        include_superseded: Trusted override to also search superseded
                            document versions. Never derived from LLM output.

    Returns:
        Tuple of (dense_results, sparse_results) as ScoredPoint lists.

    Raises:
        QdrantException: On connection failure.
    """
    if client is None:
        client = get_qdrant_client()

    qdrant_filter = _build_filter(
        deal_id, metadata_filters, include_superseded=include_superseded
    )

    start = time.monotonic()

    dense_task = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        query_filter=qdrant_filter,
        limit=top_k_dense,
        search_params=QUANTIZED_SEARCH_PARAMS,  # rescore=True, oversampling=2.0
        with_payload=True,
    )
    sparse_task = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_sparse,
        using="sparse",
        query_filter=qdrant_filter,
        limit=top_k_sparse,
        with_payload=True,
    )

    dense_response, sparse_response = await asyncio.gather(dense_task, sparse_task)
    dense_results = dense_response.points
    sparse_results = sparse_response.points

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Hybrid search complete",
        extra={
            "deal_id": deal_id,
            "dense_results": len(dense_results),
            "sparse_results": len(sparse_results),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    return dense_results, sparse_results


# ==============================================================================
# Fetch Chunks by IDs
# ==============================================================================


async def fetch_chunks_by_ids(
    chunk_ids: list[str],
    client: AsyncQdrantClient | None = None,
) -> list[dict]:
    """
    Fetches full chunk payloads by chunk_id from Qdrant.
    Uses client.scroll() with filter on chunk_id field.
    Returns list of payload dicts in same order as input ids.

    NOTE: scroll is acceptable here because we are doing deterministic lookup
    by chunk_id (fetching known points), not relevance-based retrieval.

    DEAL ISOLATION: This function does NOT add a deal_id filter because
    chunk_ids are globally unique (format: "{deal_id}_{doc_id}_{seq}").

    Args:
        chunk_ids: List of chunk_id strings to retrieve.
        client: AsyncQdrantClient instance. If None, uses get_qdrant_client().

    Returns:
        List of payload dicts in the same order as input chunk_ids.
        Missing IDs are silently skipped (logged as warnings).

    Raises:
        QdrantException: On connection failure after retries.
    """
    if not chunk_ids:
        return []

    if client is None:
        client = get_qdrant_client()

    start = time.monotonic()

    results = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))
            ]
        ),
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=False,
    )

    # Return in original rank order — preserves RRF ranking
    payload_map = {p.payload["chunk_id"]: p.payload for p in results[0]}
    ordered = [payload_map[cid] for cid in chunk_ids if cid in payload_map]

    if len(ordered) < len(chunk_ids):
        missing = set(chunk_ids) - set(payload_map.keys())
        logger.warning(
            f"fetch_chunks_by_ids: {len(missing)} IDs not found",
            extra={"missing_ids": list(missing)},
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "fetch_chunks_by_ids complete",
        extra={
            "requested": len(chunk_ids),
            "found": len(ordered),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    return ordered
