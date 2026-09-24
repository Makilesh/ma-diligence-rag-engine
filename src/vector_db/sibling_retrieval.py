# src/vector_db/sibling_retrieval.py
# Table_id-based sibling fetch for 4-representation tables
# Implemented in Phase 3, BUILD ORDER step 12

"""
Table sibling retrieval for the 4-representation table design.

When a table chunk is retrieved (is_table=1), the retrieval pipeline may
have only found one representation (e.g., narrative). The answer synthesizer
often needs the markdown representation for exact number verification.

This module fetches all sibling representations sharing the same table_id,
using the payload index for efficient filtering. Scroll is acceptable here
because this is a deterministic lookup by table_id, not relevance-based
retrieval.
"""

from __future__ import annotations

import asyncio
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.vector_db.constants import (
    COLLECTION_NAME,
    QDRANT_MAX_RETRIES,
    QDRANT_BASE_DELAY_S,
    QDRANT_MAX_DELAY_S,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Upper bound on sibling points fetched per table_id: the 4 representations,
# plus headroom for a representation the semantic chunker had to split
# (a long row_by_row on a large sheet) and for multi-chunk verbatim tables.
MAX_SIBLINGS_PER_TABLE = 8


async def fetch_table_siblings(
    chunks: list[dict],
    required_representations: list[str],
    client: AsyncQdrantClient,
    include_pii: bool = False,
    deal_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """
    For any chunk with is_table=1, fetches all sibling representations
    sharing the same table_id. This is required because the 4-representation
    design indexes narrative/row_by_row/metrics_summary/markdown separately.

    Without this step, retrieval might return only the narrative chunk
    while the answer synthesizer needs the markdown for exact number verification.
    Uses the table_id payload index for efficient filtering.

    Args:
        chunks: Reranked chunks from retrieval (may include non-table chunks).
        required_representations: Which representations to fetch (default: all 4).
        client: AsyncQdrantClient.
        include_pii: Compliance authorization override. If False, filters out PII content.
        deal_id: Deal scope for the sibling lookup. table_ids already embed the
                 deal, but the filter is applied anyway (defence in depth); when
                 None it is taken from each chunk's own deal_id payload.
        collection_name: Collection to scroll (default: main child collection).
    Returns:
        Input chunks with table chunks expanded to include all sibling representations.
    Raises:
        Exception: Re-raises the last exception after max retries for Qdrant operations.
    """
    start = time.monotonic()
    logger.info(
        "fetch_table_siblings started",
        extra={
            "input_chunks": len(chunks),
            "required_representations": required_representations,
            "include_pii": include_pii,
        },
    )

    # table_id -> deal it belongs to (from the explicit scope, else the chunk)
    table_ids = {
        c["table_id"]: (deal_id or c.get("deal_id"))
        for c in chunks
        if c.get("is_table") == 1 and c.get("table_id")
    }
    if not table_ids:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "fetch_table_siblings completed (no table chunks)",
            extra={"elapsed_ms": round(elapsed_ms, 2)},
        )
        return chunks

    logger.info(
        "fetch_table_siblings found table_ids to expand",
        extra={"table_id_count": len(table_ids)},
    )

    # NOTE: scroll is acceptable here because we are doing deterministic lookup
    # by table_id (fetching known siblings), not relevance-based retrieval.
    # The "no scroll for retrieval" rule applies to the main search pipeline only.
    async def _scroll_for_table(tid: str, table_deal_id: str | None) -> list:
        """Scroll for siblings of a single table_id with exponential backoff retry."""
        if not table_deal_id:
            # Without a deal scope the lookup could cross deals; skip it.
            logger.warning("Skipping sibling lookup without deal_id", extra={"table_id": tid})
            return []
        must_conditions = [
            FieldCondition(key="deal_id", match=MatchValue(value=table_deal_id)),
            FieldCondition(key="table_id", match=MatchValue(value=tid)),
            FieldCondition(key="is_current_version", match=MatchValue(value=1)),
        ]
        if not include_pii:
            must_conditions.append(
                FieldCondition(key="contains_pii", match=MatchValue(value=0))
            )

        for attempt in range(QDRANT_MAX_RETRIES):
            try:
                op_start = time.monotonic()
                result = await client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=must_conditions),
                    limit=MAX_SIBLINGS_PER_TABLE,
                    with_payload=True,
                    with_vectors=False,
                )
                op_elapsed_ms = (time.monotonic() - op_start) * 1000
                logger.info(
                    "scroll for table siblings completed",
                    extra={
                        "table_id": tid,
                        "siblings_found": len(result[0]),
                        "elapsed_ms": round(op_elapsed_ms, 2),
                        "attempt": attempt + 1,
                    },
                )
                return result[0]
            except Exception as e:
                if attempt == QDRANT_MAX_RETRIES - 1:
                    logger.error(
                        "scroll for table siblings failed after max retries",
                        extra={
                            "table_id": tid,
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
                    f"scroll for table siblings attempt {attempt + 1} failed, retrying",
                    extra={
                        "table_id": tid,
                        "error": str(e),
                        "delay_s": delay,
                    },
                )
                await asyncio.sleep(delay)
        return []  # unreachable, but satisfies type checker

    # Launch all sibling scroll tasks concurrently
    sibling_tasks = [_scroll_for_table(tid, did) for tid, did in table_ids.items()]
    sibling_results = await asyncio.gather(*sibling_tasks, return_exceptions=True)

    # Merge siblings into result set, deduplicating by chunk_id
    existing_ids = {c["chunk_id"] for c in chunks}
    siblings_added = 0
    for result_batch in sibling_results:
        if isinstance(result_batch, Exception):
            logger.error(
                "sibling scroll task raised exception",
                extra={"error": str(result_batch)},
            )
            continue
        for point in result_batch:
            chunk_id = point.payload["chunk_id"]
            if chunk_id not in existing_ids:
                chunks.append(point.payload)
                existing_ids.add(chunk_id)
                siblings_added += 1

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "fetch_table_siblings completed",
        extra={
            "table_ids_expanded": len(table_ids),
            "siblings_added": siblings_added,
            "total_chunks": len(chunks),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return chunks
