# src/vector_db/parent_child_retrieval.py
# Parent context expansion from manda_parent_chunks collection
# Implemented in Phase 3, BUILD ORDER step 12

"""
Parent + Sibling context expansion for the retrieval pipeline.

After reranking, the top chunks are expanded with:
  1. Parent context — fetches ~2048-token parent text blobs from the
     manda_parent_chunks collection, giving the synthesizer enough
     surrounding context to avoid hallucinating connections.
  2. Sibling context — for table chunks (is_table=1), fetches all 4
     representation siblings sharing the same table_id. Delegates to
     fetch_table_siblings() in sibling_retrieval.py.

This module sits between the reranker and the answer synthesizer in
the retrieval pipeline.
"""

from __future__ import annotations

import asyncio
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from src.vector_db.constants import (
    COLLECTION_NAME,
    PARENT_COLLECTION_NAME,
    QDRANT_MAX_RETRIES,
    QDRANT_BASE_DELAY_S,
    QDRANT_MAX_DELAY_S,
)
from src.vector_db.qdrant_client import get_qdrant_client
from src.vector_db.sibling_retrieval import fetch_table_siblings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def expand_context(
    chunks: list[dict],
    include_parents: bool = True,
    include_siblings: bool = True,
    client: AsyncQdrantClient | None = None,
    include_pii: bool = False,
    deal_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> list[dict]:
    """
    Expands reranked chunks with parent context and table siblings.

    Parent expansion: For each chunk with a parent_chunk_id, fetches the
    corresponding parent text blob from the manda_parent_chunks collection.
    Parent chunks contain ~2048 tokens of surrounding context, providing
    the synthesizer with enough context to avoid hallucinating connections.

    Sibling expansion: For table chunks (is_table=1), fetches all 4
    representation siblings sharing the same table_id. Delegates to
    fetch_table_siblings().

    Args:
        chunks: Reranked chunks from retrieval (list of payload dicts).
        include_parents: Whether to fetch parent context (from retrieval config).
        include_siblings: Whether to fetch table siblings (from retrieval config).
        client: AsyncQdrantClient. If None, uses get_qdrant_client().
        include_pii: Compliance authorization override. If False, filters out PII content.
        deal_id: Deal scope applied to parent and sibling lookups (defence in
                 depth — ids already embed the deal). When None it is derived
                 from the chunks' own deal_id payloads.
        collection_name: Child collection (for sibling lookups).
        parent_collection_name: Parent collection.
    Returns:
        Expanded list of chunk dicts. Each chunk that has a parent gets a
        "parent_text" key added. Table chunks are augmented with sibling
        representations. Original chunks are always preserved.
    Raises:
        Exception: Re-raises the last exception after max retries for Qdrant operations.
    """
    start = time.monotonic()
    logger.info(
        "expand_context started",
        extra={
            "input_chunks": len(chunks),
            "include_parents": include_parents,
            "include_siblings": include_siblings,
            "include_pii": include_pii,
        },
    )

    if client is None:
        client = get_qdrant_client()

    expanded = [dict(c) for c in chunks]  # copy dicts — prevents mutation of reranked_results

    # Parent expansion
    if include_parents:
        parent_ids = list({
            c["parent_chunk_id"] for c in chunks
            if c.get("parent_chunk_id")
        })
        if parent_ids:
            logger.info(
                "expand_context fetching parent chunks",
                extra={"parent_id_count": len(parent_ids)},
            )

            deal_scope = (
                [deal_id] if deal_id
                else sorted({c["deal_id"] for c in chunks if c.get("deal_id")})
            )
            must_conditions = [
                FieldCondition(
                    key="deal_id",
                    match=MatchAny(any=deal_scope),
                ),
                FieldCondition(
                    key="chunk_id",
                    match=MatchAny(any=parent_ids),
                ),
                FieldCondition(
                    key="is_current_version",
                    match=MatchValue(value=1),
                ),
            ]
            if not include_pii:
                must_conditions.append(
                    FieldCondition(key="contains_pii", match=MatchValue(value=0))
                )

            # Exponential backoff retry for parent scroll
            parent_map: dict[str, str] = {}
            for attempt in range(QDRANT_MAX_RETRIES):
                try:
                    op_start = time.monotonic()
                    parent_results = await client.scroll(
                        collection_name=parent_collection_name,
                        scroll_filter=Filter(must=must_conditions),
                        limit=len(parent_ids),
                        with_payload=True,
                        with_vectors=False,
                    )
                    op_elapsed_ms = (time.monotonic() - op_start) * 1000
                    parent_map = {
                        p.payload["chunk_id"]: p.payload.get("text", "")
                        for p in parent_results[0]
                    }
                    logger.info(
                        "parent chunk scroll completed",
                        extra={
                            "requested": len(parent_ids),
                            "found": len(parent_map),
                            "elapsed_ms": round(op_elapsed_ms, 2),
                            "attempt": attempt + 1,
                        },
                    )
                    break
                except Exception as e:
                    if attempt == QDRANT_MAX_RETRIES - 1:
                        logger.error(
                            "parent chunk scroll failed after max retries",
                            extra={
                                "error": str(e),
                                "attempts": QDRANT_MAX_RETRIES,
                            },
                        )
                        raise
                    delay = min(
                        QDRANT_BASE_DELAY_S * (2 ** attempt),
                        QDRANT_MAX_DELAY_S,
                    )
                    logger.warning(
                        f"parent chunk scroll attempt {attempt + 1} failed, retrying",
                        extra={"error": str(e), "delay_s": delay},
                    )
                    await asyncio.sleep(delay)

            # Attach parent text to expanded chunks
            parents_attached = 0
            for chunk in expanded:
                pid = chunk.get("parent_chunk_id")
                if pid and pid in parent_map:
                    chunk["parent_text"] = parent_map[pid]
                    parents_attached += 1

            logger.info(
                "parent text attached to chunks",
                extra={"parents_attached": parents_attached},
            )
        else:
            logger.info(
                "expand_context no parent_chunk_ids found, skipping parent expansion",
            )

    # Sibling expansion (table representations)
    if include_siblings:
        logger.info("expand_context delegating to fetch_table_siblings")
        expanded = await fetch_table_siblings(
            expanded,
            required_representations=["narrative", "row_by_row", "metrics_summary", "markdown"],
            client=client,
            include_pii=include_pii,
            deal_id=deal_id,
            collection_name=collection_name,
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "expand_context completed",
        extra={
            "input_chunks": len(chunks),
            "output_chunks": len(expanded),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return expanded
