"""
Shared pytest fixtures for M&A Due Diligence Intelligence Engine tests.
"""

import os
from dataclasses import dataclass

import pytest


# Scripts that live in tests/ but are not test modules. test_pipeline_offline.py
# matches the test_*.py glob yet defines no tests — it is a CLI validation run
# (`python tests/test_pipeline_offline.py`) — so collecting it only imports the
# ingestion stack for nothing.
collect_ignore = ["test_pipeline_offline.py"]


def pytest_collection_modifyitems(config, items):
    """
    Skips `@pytest.mark.live` tests unless RUN_LIVE_TESTS=1.

    Live tests need real services or credentials. Without this they would fail
    — not skip — on any machine that lacks them, which trains people to ignore
    red runs. CI additionally deselects them with `-m "not live and not models"`.
    """
    if os.getenv("RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes"}:
        return
    skip_live = pytest.mark.skip(reason="live test: set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip_live)


@dataclass
class MockScoredPoint:
    """Mock Qdrant ScoredPoint for testing."""
    id: int
    score: float
    payload: dict


def make_scored_point(chunk_id: str, score: float) -> MockScoredPoint:
    """Factory for mock ScoredPoint objects."""
    return MockScoredPoint(
        id=hash(chunk_id) % (2**63),
        score=score,
        payload={"chunk_id": chunk_id, "text": f"Text for {chunk_id}"},
    )


@pytest.fixture
def sample_dense_results():
    """Dense search results — 5 chunks sorted by cosine similarity."""
    return [
        make_scored_point("chunk_a", 0.95),
        make_scored_point("chunk_b", 0.88),
        make_scored_point("chunk_c", 0.82),
        make_scored_point("chunk_d", 0.75),
        make_scored_point("chunk_e", 0.68),
    ]


@pytest.fixture
def sample_sparse_results():
    """Sparse BM25 results — 5 chunks with some overlap with dense."""
    return [
        make_scored_point("chunk_c", 12.5),   # overlap with dense
        make_scored_point("chunk_f", 11.2),
        make_scored_point("chunk_a", 10.8),   # overlap with dense
        make_scored_point("chunk_g", 9.4),
        make_scored_point("chunk_h", 8.1),
    ]


@pytest.fixture
def sample_reranked_chunks():
    """Reranked chunks with scores and metadata."""
    return [
        {
            "chunk_id": "chunk_a",
            "text": "Revenue was $45.2M in FY2023.",
            "reranker_score": 0.92,
            "source_file": "financials_2023.pdf",
            "page_number": 5,
            "document_category": "financial",
        },
        {
            "chunk_id": "chunk_b",
            "text": "EBITDA margin improved to 18.5%.",
            "reranker_score": 0.85,
            "source_file": "financials_2023.pdf",
            "page_number": 7,
            "document_category": "financial",
        },
        {
            "chunk_id": "chunk_c",
            "text": "The merger agreement includes a change of control provision.",
            "reranker_score": 0.78,
            "source_file": "merger_agreement.pdf",
            "page_number": 12,
            "document_category": "legal",
        },
    ]


@pytest.fixture
def sample_agent_state():
    """Minimal valid AgentState for testing."""
    return {
        "original_query": "What was the revenue in FY2023?",
        "current_query": "What was the revenue in FY2023?",
        "query_type": "financial",
        "parsed_intent": {"requires_numerical_precision": True},
        "extracted_filters": {"fiscal_year": "FY2023"},
        "retrieval_config": {},
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "reranked_results": [],
        "expanded_context": [],
        "context_quality_score": 0.0,
        "quality_breakdown": {},
        "quality_method": "heuristic",
        "missing_aspects": [],
        "rewrite_iteration": 0,
        "rewrite_history": [],
        "agent_trace": [],
        "numerical_registry": {},
        "inconsistencies": [],
        "generated_answer": "",
        "citations": [],
        "numerical_claims": [],
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "validation_status": "passed",
        "validation_attempt": 0,
        "force_refusal": False,
        "deal_id": "test_deal_001",
        "session_id": "test_session_001",
        "total_latency_ms": 0.0,
        "status": "running",
        "error": None,
    }
