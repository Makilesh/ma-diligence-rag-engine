"""
Builds the evaluation index: the sample data room in an in-memory Qdrant.

Indexing goes through src.data_processing.ingest_pipeline.index_document with
its default models (bge-m3 dense, FastEmbed BM25 sparse) — the same path as
POST /ingest — so what is measured is what is served. Nothing touches the Docker
Qdrant or qdrant_local_db.

Retrieval code reaches Qdrant through get_qdrant_client(), a module-level
singleton. use_client() points that singleton at the in-memory client for the
duration of a run; no production code changes are needed.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.data_processing.ingest_pipeline import index_document
from src.vector_db import qdrant_client as qdrant_client_module
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "sample_deal"
GOLDEN_SET_PATH = PROJECT_ROOT / "tests" / "golden_qa_set.json"
DEAL_ID = "aurora_vertex_2024"

# Mirrors CATEGORY_OVERRIDES in run_demo.py: the three documents whose
# auto-classification is ambiguous are pinned; the rest are classified at
# ingestion exactly as an upload would be.
CATEGORY_OVERRIDES = {
    "board_deck_strategic_review_mar2024.txt": "board",
    "regulatory_and_data_privacy_memo.txt": "regulatory",
    "employment_and_retention_agreements.txt": "legal",
}


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> dict:
    """Loads tests/golden_qa_set.json."""
    return json.loads(path.read_text(encoding="utf-8"))


async def build_index(corpus_dir: Path = CORPUS_DIR) -> tuple[AsyncQdrantClient, list[dict]]:
    """
    Indexes every .txt in the corpus into a fresh in-memory Qdrant.

    Args:
        corpus_dir: Directory holding the sample data room.

    Returns:
        (client, per-document index_document results).
    """
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        documents.append(await index_document(
            str(path),
            path.name,
            DEAL_ID,
            CATEGORY_OVERRIDES.get(path.name),
            client=client,
        ))
    return client, documents


async def all_chunks(client: AsyncQdrantClient) -> list[dict]:
    """Every child-chunk payload in the index."""
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(p.payload for p in points)
        if offset is None:
            return payloads


def is_retrievable(chunk: dict) -> bool:
    """
    Whether production retrieval can ever return this chunk.

    Mirrors the non-negotiable conditions of hybrid_search._build_filter with
    include_pii=False: the deal, the current version, and no PII flag. A chunk
    failing these is unreachable by policy, so it is left out of the relevance
    labels rather than counted as a retrieval miss.
    """
    return (
        chunk.get("deal_id") == DEAL_ID
        and chunk.get("is_current_version") == 1
        and chunk.get("contains_pii") == 0
    )


@contextlib.contextmanager
def use_client(client: AsyncQdrantClient):
    """Points the get_qdrant_client() singleton at `client`, restoring it after."""
    previous = qdrant_client_module._qdrant_client
    qdrant_client_module._qdrant_client = client
    try:
        yield client
    finally:
        qdrant_client_module._qdrant_client = previous
