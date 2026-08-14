"""
Qdrant collection creation and management.

These values CANNOT change after the collection is created.
Changing VECTOR_SIZE requires re-embedding and re-indexing the entire corpus.

Collections:
    - manda_due_diligence: Main child-chunk collection with hybrid vector support
    - manda_parent_chunks: Parent-chunk collection for context expansion
"""

import asyncio
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
)

from src.vector_db.constants import (
    VECTOR_SIZE,
    COLLECTION_NAME,
    PARENT_COLLECTION_NAME,
    QDRANT_MAX_RETRIES,
    QDRANT_BASE_DELAY_S,
    QDRANT_MAX_DELAY_S,
)
from src.vector_db.qdrant_client import get_qdrant_client
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def _retry_operation(operation, operation_name: str) -> None:
    """
    Executes a Qdrant operation with exponential backoff retry.

    Args:
        operation: Async callable to execute.
        operation_name: Human-readable name for logging.

    Raises:
        Exception: Re-raises the last exception after max retries.
    """
    for attempt in range(QDRANT_MAX_RETRIES):
        try:
            start = time.monotonic()
            await operation()
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                f"{operation_name} completed",
                extra={"elapsed_ms": round(elapsed_ms, 2), "attempt": attempt + 1},
            )
            return
        except Exception as e:
            if attempt == QDRANT_MAX_RETRIES - 1:
                logger.error(
                    f"{operation_name} failed after {QDRANT_MAX_RETRIES} attempts",
                    extra={"error": str(e)},
                )
                raise
            delay = min(
                QDRANT_BASE_DELAY_S * (2 ** attempt),
                QDRANT_MAX_DELAY_S,
            )
            logger.warning(
                f"{operation_name} attempt {attempt + 1} failed, retrying in {delay}s",
                extra={"error": str(e), "delay_s": delay},
            )
            await asyncio.sleep(delay)


async def create_collection(client: AsyncQdrantClient | None = None) -> None:
    """
    Creates the main child-chunk collection with hybrid vector support.
    HNSW parameters are explicitly set — do not rely on Qdrant defaults,
    which use ef_construct=100 (lower recall at scale than ef_construct=200).

    Args:
        client: Initialized AsyncQdrantClient instance. If None, uses singleton.

    Raises:
        QdrantException: If collection creation fails after retries.
    """
    if client is None:
        client = get_qdrant_client()

    # Check if collection already exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if COLLECTION_NAME in existing_names:
        logger.info(
            f"Collection '{COLLECTION_NAME}' already exists, skipping creation"
        )
        return

    async def _create():
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_SIZE,           # 1024 — immutable
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,                   # Connections per layer
                        ef_construct=200,       # Build-time search width
                        full_scan_threshold=10_000,
                    ),
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,    # Quantized INT8 vectors in RAM
                        )
                    ),
                    # Original float32 vectors stay on disk for rescoring
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,          # Sparse BM25 index in RAM for speed
                    )
                )
            },
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20_000,
            ),
        )

    await _retry_operation(_create, f"Create collection '{COLLECTION_NAME}'")
    logger.info(
        f"Collection '{COLLECTION_NAME}' created successfully",
        extra={
            "vector_size": VECTOR_SIZE,
            "distance": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_construct": 200,
            "quantization": "INT8",
        },
    )


async def create_parent_collection(client: AsyncQdrantClient | None = None) -> None:
    """
    Creates the parent-chunk collection for context expansion.
    Parent text blobs are large — do NOT set always_ram on payload.
    At 50k chunks × ~8KB parent text = ~400MB per deal; forcing this
    into RAM exhausts the 32GB system RAM budget at scale.
    Vectors are on disk; payload (containing parent_text) goes to disk.

    INGESTION COMPLIANCE NOTE:
    The ingestion pipeline MUST write the `contains_pii` (0 or 1) and
    `is_current_version` (0 or 1) metadata fields to both the main child-chunk
    collection payloads AND the parent-chunk collection payloads.

    Args:
        client: Initialized AsyncQdrantClient instance. If None, uses singleton.

    Raises:
        QdrantException: If collection creation fails after retries.
    """
    if client is None:
        client = get_qdrant_client()

    # Check if collection already exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if PARENT_COLLECTION_NAME in existing_names:
        logger.info(
            f"Collection '{PARENT_COLLECTION_NAME}' already exists, skipping creation"
        )
        return

    async def _create():
        await client.create_collection(
            collection_name=PARENT_COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,  # Use memory-mapped files for large collections
            ),
        )

    await _retry_operation(_create, f"Create collection '{PARENT_COLLECTION_NAME}'")
    logger.info(f"Collection '{PARENT_COLLECTION_NAME}' created successfully")


async def create_payload_indexes(client: AsyncQdrantClient | None = None) -> None:
    """
    Creates payload indexes for filtered search.

    NOTE on booleans: Qdrant does not have a dedicated BOOL PayloadSchemaType
    in all versions. Store boolean fields as integers (0/1) and index as INTEGER.

    Args:
        client: AsyncQdrantClient instance. If None, uses singleton.

    Raises:
        QdrantException: If index creation fails after retries.
    """
    if client is None:
        client = get_qdrant_client()

    # Main collection indexes
    indexes = [
        ("deal_id",             PayloadSchemaType.KEYWORD),
        ("document_category",   PayloadSchemaType.KEYWORD),
        ("file_type",           PayloadSchemaType.KEYWORD),
        ("is_table",            PayloadSchemaType.INTEGER),    # 0 or 1 — safe across versions
        ("fiscal_year",         PayloadSchemaType.KEYWORD),
        ("table_id",            PayloadSchemaType.KEYWORD),
        ("is_current_version",  PayloadSchemaType.INTEGER),    # 0 or 1
        ("currency",            PayloadSchemaType.KEYWORD),
        ("contains_pii",        PayloadSchemaType.INTEGER),    # 0 or 1
        ("content_type",        PayloadSchemaType.KEYWORD),
        # Indexed so /deals can facet distinct documents per deal without
        # scrolling the whole collection. Qdrant refuses to facet an unindexed
        # field outright, so this is a requirement of that endpoint, not a
        # performance nicety.
        ("source_file",         PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_type in indexes:
        async def _create_index(fn=field_name, ft=field_type):
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=fn,
                field_schema=ft,
            )

        await _retry_operation(
            _create_index,
            f"Create index '{field_name}' on '{COLLECTION_NAME}'",
        )

    logger.info(
        f"Created {len(indexes)} payload indexes on '{COLLECTION_NAME}'"
    )

    # Parent collection indexes (used during context expansion)
    parent_indexes = [
        ("chunk_id",            PayloadSchemaType.KEYWORD),
        ("is_current_version",  PayloadSchemaType.INTEGER),
        ("contains_pii",        PayloadSchemaType.INTEGER),
    ]

    for field_name, field_type in parent_indexes:
        async def _create_parent_index(fn=field_name, ft=field_type):
            await client.create_payload_index(
                collection_name=PARENT_COLLECTION_NAME,
                field_name=fn,
                field_schema=ft,
            )

        await _retry_operation(
            _create_parent_index,
            f"Create index '{field_name}' on '{PARENT_COLLECTION_NAME}'",
        )

    logger.info(
        f"Created {len(parent_indexes)} payload indexes on '{PARENT_COLLECTION_NAME}'"
    )


async def setup_collections(client: AsyncQdrantClient | None = None) -> None:
    """
    Complete collection setup: creates both collections and all payload indexes.
    Idempotent — safe to call on every startup.

    Args:
        client: AsyncQdrantClient instance. If None, uses singleton.
    """
    if client is None:
        client = get_qdrant_client()

    logger.info("Starting Qdrant collection setup")
    await create_collection(client)
    await create_parent_collection(client)
    await create_payload_indexes(client)
    logger.info("Qdrant collection setup complete")
