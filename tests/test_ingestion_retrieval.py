"""
Indexing and retrieval-side guarantees of the ingestion rework.

Runs against in-memory Qdrant with a stub embedding function — no Docker, no
bge-m3, no LLM:

- index_document is idempotent (same bytes → same points) and rolls back a
  failed upsert instead of leaving a half-indexed document
- parent-child expansion attaches parent_text; sibling expansion pulls every
  table representation; both are deal-scoped
- the version filter can no longer be switched off by Agent 1's output
- the Query Rewriter's config/filter proposals are whitelisted and clamped
- the upload route caps size, never uses the client filename as a path, and
  does not leak internal errors
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SparseVector

from src.data_processing.ingest_pipeline import index_document, mark_superseded
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME, VECTOR_SIZE
from tests.fixtures import ingestion_docs as fx

DEAL = "deal-idx"


async def _stub_embed(texts: list[str]) -> np.ndarray:
    """Deterministic unit vectors seeded by text content."""
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vec = np.random.default_rng(seed).standard_normal(VECTOR_SIZE)
        rows.append(vec / np.linalg.norm(vec))
    return np.asarray(rows, dtype=np.float32)


def _stub_sparse(texts: list[str]) -> list[SparseVector]:
    """Bag-of-words sparse vectors (hashed term ids, unit weights)."""
    out = []
    for text in texts:
        ids = sorted({zlib.crc32(w.lower().encode()) % 100_000 for w in text.split()})
        out.append(SparseVector(indices=ids, values=[1.0] * len(ids)))
    return out


async def _client() -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    return client


async def _index(client, path, deal_id=DEAL, **kwargs) -> dict:
    return await index_document(
        str(path), path.name, deal_id,
        client=client, embed_fn=_stub_embed, sparse_fn=_stub_sparse, **kwargs,
    )


async def _payloads(client, collection=COLLECTION_NAME) -> list[dict]:
    points, _ = await client.scroll(collection_name=collection, limit=10_000, with_payload=True)
    return [p.payload for p in points]


# ==============================================================================
# Idempotency and rollback
# ==============================================================================


class TestIdempotentIndexing:
    @pytest.mark.asyncio
    async def test_reupload_replaces_instead_of_duplicating(self, tmp_path):
        client = await _client()
        path = fx.write_txt(tmp_path)

        first = await _index(client, path)
        assert first["chunks_created"] > 0 and first["parent_chunks_created"] > 0
        assert first["previously_indexed"] is False

        second = await _index(client, path)
        assert second["doc_id"] == first["doc_id"]
        assert second["previously_indexed"] is True

        children = await _payloads(client)
        parents = await _payloads(client, PARENT_COLLECTION_NAME)
        assert len(children) == first["chunks_created"]
        assert len(parents) == first["parent_chunks_created"]
        assert all(p["is_current_version"] == 1 for p in children)

    @pytest.mark.asyncio
    async def test_stale_points_are_removed_on_reindex(self, tmp_path):
        client = await _client()
        path = fx.write_txt(tmp_path)
        result = await _index(client, path)

        # A leftover from an older chunking of the same document.
        from qdrant_client.models import PointStruct
        await client.upsert(COLLECTION_NAME, points=[PointStruct(
            id="00000000-0000-0000-0000-000000000001",
            vector={"dense": [0.1] * VECTOR_SIZE, "sparse": SparseVector(indices=[1], values=[1.0])},
            payload={"deal_id": DEAL, "doc_id": result["doc_id"], "chunk_id": "stale"},
        )])

        await _index(client, path)
        assert "stale" not in {p["chunk_id"] for p in await _payloads(client)}

    @pytest.mark.asyncio
    async def test_failed_upsert_rolls_back_the_document(self, tmp_path):
        client = await _client()
        path = fx.write_txt(tmp_path)
        real_upsert = client.upsert

        async def failing_upsert(collection_name, points, **kwargs):
            if collection_name == PARENT_COLLECTION_NAME:
                raise ConnectionError("qdrant went away")
            return await real_upsert(collection_name=collection_name, points=points, **kwargs)

        client.upsert = failing_upsert
        with pytest.raises(ConnectionError):
            await _index(client, path)

        # Children were written before the parent batch failed — and are gone.
        assert await _payloads(client) == []

    @pytest.mark.asyncio
    async def test_supersession_retires_old_version_but_not_itself(self, tmp_path):
        client = await _client()
        old = await _index(client, fx.write_txt(tmp_path, "v1.txt"))
        new = await _index(
            client,
            fx.write_txt(tmp_path, "v2.txt", fx.SAMPLE_TXT + "\nAmended.\n"),
            supersedes_doc_id=old["doc_id"],
        )
        await mark_superseded(DEAL, old["doc_id"], new["doc_id"], client=client)
        # Self-supersession (identical re-upload) must be a no-op.
        await mark_superseded(DEAL, new["doc_id"], new["doc_id"], client=client)

        for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
            for p in await _payloads(client, collection):
                expected = 0 if p["doc_id"] == old["doc_id"] else 1
                assert p["is_current_version"] == expected


# ==============================================================================
# Context expansion
# ==============================================================================


class TestContextExpansion:
    @pytest.mark.asyncio
    async def test_parent_text_is_attached(self, tmp_path):
        from src.vector_db.parent_child_retrieval import expand_context

        client = await _client()
        await _index(client, fx.write_txt(tmp_path))
        cap = next(p for p in await _payloads(client) if "$174 million" in p["text"])
        assert cap["parent_chunk_id"]

        expanded = await expand_context([cap], client=client, deal_id=DEAL, include_siblings=False)
        assert "$174 million" in expanded[0]["parent_text"]
        assert "Section 8.1" in expanded[0]["parent_text"]  # context beyond the child

        other_deal = await expand_context([cap], client=client, deal_id="someone-else",
                                          include_siblings=False)
        assert "parent_text" not in other_deal[0]

    @pytest.mark.asyncio
    async def test_table_siblings_are_fetched(self, tmp_path):
        from src.vector_db.parent_child_retrieval import expand_context

        client = await _client()
        await _index(client, fx.write_xlsx(tmp_path))
        payloads = await _payloads(client)
        narrative = next(p for p in payloads if p.get("table_representation") == "narrative")

        expanded = await expand_context([narrative], client=client, deal_id=DEAL, include_parents=False)
        reps = {c.get("table_representation") for c in expanded if c.get("table_id") == narrative["table_id"]}
        assert reps == {"narrative", "row_by_row", "metrics_summary", "markdown"}

        isolated = await expand_context([narrative], client=client, deal_id="someone-else",
                                        include_parents=False)
        assert len(isolated) == 1


# ==============================================================================
# Version filter regression
# ==============================================================================

# The exact metadata_filters shape Agent 1 returned on every query in
# tests/e2e_validation_results.json before the fix.
AGENT1_FILTERS = {
    "fiscal_year": "FY2023",
    "document_category": "financial",
    "is_current_version": 1,
    "currency": None,
}


def _conditions(qfilter) -> dict:
    return {c.key: c.match for c in qfilter.must}


class TestVersionFilter:
    def test_agent1_output_no_longer_disables_the_version_filter(self):
        from src.vector_db.hybrid_search import _build_filter

        conditions = _conditions(_build_filter(DEAL, {**AGENT1_FILTERS, "include_pii": False}))
        assert conditions["is_current_version"].value == 1
        assert conditions["document_category"].value == "financial"
        assert "fiscal_year" not in conditions and "currency" not in conditions

    @pytest.mark.parametrize("llm_value", [0, 1, None, "any"])
    def test_llm_cannot_choose_the_version(self, llm_value):
        from src.vector_db.hybrid_search import _build_filter

        conditions = _conditions(_build_filter(DEAL, {"is_current_version": llm_value}))
        assert conditions["is_current_version"].value == 1

    def test_only_trusted_argument_includes_superseded(self):
        from src.vector_db.hybrid_search import _build_filter

        assert "is_current_version" not in _conditions(
            _build_filter(DEAL, {}, include_superseded=True)
        )

    def test_prompt_schema_no_longer_offers_the_key(self):
        from src.llm.prompt_templates.query_intelligence import QUERY_INTELLIGENCE_SYSTEM_PROMPT

        schema = QUERY_INTELLIGENCE_SYSTEM_PROMPT.split("RULES:")[0]
        assert '"is_current_version"' not in schema
        assert '"fiscal_year": "FY2023' not in schema

    @pytest.mark.asyncio
    async def test_superseded_document_is_not_retrieved(self, tmp_path):
        from src.vector_db.hybrid_search import hybrid_search

        client = await _client()
        old = await _index(client, fx.write_txt(tmp_path, "v1.txt"), document_category="financial")
        new = await _index(
            client, fx.write_txt(tmp_path, "v2.txt", fx.SAMPLE_TXT + "\nAmended.\n"),
            document_category="financial",
        )
        await mark_superseded(DEAL, old["doc_id"], new["doc_id"], client=client)

        query = "indemnification cap"
        dense, sparse = await hybrid_search(
            query_text=query,
            query_vector=(await _stub_embed([query]))[0].tolist(),
            query_sparse=_stub_sparse([query])[0],
            deal_id=DEAL,
            metadata_filters={**AGENT1_FILTERS, "include_pii": False},
            client=client,
        )
        docs = {p.payload["doc_id"] for p in dense + sparse}
        assert docs == {new["doc_id"]}


# ==============================================================================
# Query Rewriter clamping and reranker threshold
# ==============================================================================


class TestRewriterClamping:
    @pytest.mark.asyncio
    async def test_llm_proposals_are_whitelisted_and_clamped(self, monkeypatch):
        import src.agents.query_rewriter as rw

        async def fake_call(**kwargs):
            return {
                "rewritten_query": "indemnification cap amount",
                "updated_retrieval_config": {
                    "reranker_top_k": 500,
                    "top_k_dense": -3,
                    "dense_weight": 7,
                    "reranker_threshold": "not a number",
                    "final_top_k": 12.6,
                    "use_parent_expansion": "yes",
                    "exec": "rm -rf /",
                },
                "updated_metadata_filters": {
                    "include_pii": True,
                    "is_current_version": 0,
                    "fiscal_year": "FY2023",
                    "document_category": "legal",
                },
            }

        class _Choice:
            model = "stub"
            api_key = None

        class _Tracker:
            async def get_model_for_agent(self):
                return _Choice()

        async def fake_instance(*args, **kwargs):
            return _Tracker()

        monkeypatch.setattr(rw, "call_structured_agent", fake_call)
        monkeypatch.setattr(rw.BudgetTracker, "get_instance", fake_instance)

        out = await rw.query_rewriter_node({
            "original_query": "q", "current_query": "q",
            "retrieval_config": {"reranker_top_k": 20, "use_parent_expansion": True},
            "extracted_filters": {"document_category": "financial"},
            "rewrite_iteration": 0,
        })

        config = out["retrieval_config"]
        assert config["reranker_top_k"] == 50
        assert config["top_k_dense"] == 5
        assert config["dense_weight"] == 1.0
        assert config["final_top_k"] == 13
        assert config["use_parent_expansion"] is True  # "yes" is not a bool
        assert "reranker_threshold" not in config and "exec" not in config
        assert out["extracted_filters"] == {"document_category": "legal"}

    def test_null_category_removes_the_filter(self):
        from src.agents.retrieval_strategy import apply_filter_overrides

        assert apply_filter_overrides({"document_category": "legal"}, {"document_category": None}) == {}
        assert apply_filter_overrides({}, {"document_category": "made-up"}) == {}

    def test_reranker_threshold_is_applied_with_a_floor(self):
        from src.agents.retrieval_executor import (
            MIN_CHUNKS_AFTER_THRESHOLD,
            _apply_reranker_threshold,
        )

        scored = [{"chunk_id": str(i), "reranker_score": s}
                  for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])]
        assert [c["chunk_id"] for c in _apply_reranker_threshold(scored, 0.5)] == ["0", "1", "2", "3"]
        # Never empties the context: the Quality Assessor still judges the best few.
        assert len(_apply_reranker_threshold(scored, 0.95)) == MIN_CHUNKS_AFTER_THRESHOLD


# ==============================================================================
# Upload route
# ==============================================================================


def _ingest_app(monkeypatch, index_impl=None, admin_key: str | None = None):
    """
    Minimal app around the ingest router with indexing stubbed out.

    The real security dependencies run. By default the caller is admin the way
    local development is (ENVIRONMENT=development, no ADMIN_API_KEY); passing
    admin_key configures a key the test client does not send, i.e. a public
    visitor.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api.routes.ingest as ingest
    from api.security import reset_limits

    monkeypatch.setenv("ENVIRONMENT", "development")
    if admin_key:
        monkeypatch.setenv("ADMIN_API_KEY", admin_key)
    else:
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    reset_limits()

    calls: list[dict] = []

    async def fake_index(file_path, filename, deal_id, document_category=None, **kwargs):
        calls.append({"file_path": file_path, "filename": filename})
        if index_impl:
            return await index_impl()
        return {
            "doc_id": "doc-1", "document_category": "legal", "chunks_created": 3,
            "parent_chunks_created": 1, "risk_signals": [], "warnings": [],
            "previously_indexed": False,
        }

    monkeypatch.setattr(ingest, "index_document", fake_index)
    monkeypatch.setattr(ingest, "register_document", lambda **kw: None)

    app = FastAPI()
    app.include_router(ingest.router)
    return TestClient(app), calls


class TestIngestRoute:
    def test_client_filename_never_becomes_a_path(self, monkeypatch):
        client, calls = _ingest_app(monkeypatch)
        resp = client.post(
            "/ingest",
            data={"deal_id": DEAL},
            files={"file": ("../../../etc/merger.txt", b"Section 1.1 - Terms\nText.", "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        assert calls[0]["filename"] == "merger.txt"
        assert calls[0]["file_path"].endswith("upload.txt")
        assert ".." not in calls[0]["file_path"]

    def test_dotfile_rejected(self, monkeypatch):
        client, calls = _ingest_app(monkeypatch)
        resp = client.post("/ingest", data={"deal_id": DEAL},
                           files={"file": (".env", b"SECRET=1", "text/plain")})
        assert resp.status_code == 400
        assert calls == []

    def test_oversize_upload_is_413(self, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_MB", "1")
        client, calls = _ingest_app(monkeypatch)
        resp = client.post("/ingest", data={"deal_id": DEAL},
                           files={"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")})
        assert resp.status_code == 413
        assert calls == []

    def test_xls_rejected(self, monkeypatch):
        client, _ = _ingest_app(monkeypatch)
        resp = client.post("/ingest", data={"deal_id": DEAL},
                           files={"file": ("old.xls", b"\xd0\xcf\x11\xe0", "application/vnd.ms-excel")})
        assert resp.status_code == 400

    def test_internal_errors_are_not_leaked(self, monkeypatch):
        async def explode():
            raise RuntimeError("secret path C:\\internal\\qdrant timeout")

        client, _ = _ingest_app(monkeypatch, index_impl=explode)
        resp = client.post("/ingest", data={"deal_id": DEAL},
                           files={"file": ("doc.txt", b"Some text.", "text/plain")})
        assert resp.status_code == 500
        assert "secret" not in resp.text and "internal\\" not in resp.text

    def test_public_upload_to_non_sandbox_deal_is_forbidden(self, monkeypatch):
        client, calls = _ingest_app(monkeypatch, admin_key="owner-secret")
        resp = client.post("/ingest", data={"deal_id": "aurora_vertex_2024"},
                           files={"file": ("doc.txt", b"Some text.", "text/plain")})
        assert resp.status_code == 403
        assert calls == []

    def test_admin_may_upload_to_the_demo_deal(self, monkeypatch):
        client, calls = _ingest_app(monkeypatch, admin_key="owner-secret")
        resp = client.post("/ingest", data={"deal_id": "aurora_vertex_2024"},
                           headers={"X-Admin-Key": "owner-secret"},
                           files={"file": ("doc.txt", b"Some text.", "text/plain")})
        assert resp.status_code == 200, resp.text
        assert len(calls) == 1

    def test_empty_extraction_is_not_a_success(self, monkeypatch):
        from src.data_processing.ingest_pipeline import NoExtractableContentError

        async def empty():
            raise NoExtractableContentError("No text could be extracted from the document")

        client, _ = _ingest_app(monkeypatch, index_impl=empty)
        resp = client.post("/ingest", data={"deal_id": DEAL},
                           files={"file": ("doc.txt", b"   ", "text/plain")})
        assert resp.status_code == 422
