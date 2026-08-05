"""
Query routes — main ask endpoint with immutable audit logging.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException

from api.models.request_models import QueryRequest
from api.models.response_models import QueryResponse, Citation
from src.workflow.orchestrator import run_query
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint — runs the full agentic RAG pipeline.
    Logs every query to the immutable audit log.

    Args:
        request: QueryRequest with query, deal_id, optional session_id.

    Returns:
        QueryResponse with answer, citations, confidence, trace.
    """
    from api.main import get_graph

    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Application not fully initialized")

    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "Query received",
        extra={
            "deal_id": request.deal_id,
            "session_id": session_id,
            "query_length": len(request.query),
        },
    )

    try:
        result = await run_query(
            app=graph,
            query=request.query,
            deal_id=request.deal_id,
            session_id=session_id,
            include_pii=request.include_pii,
        )
        if result.get("status") == "error":
            raise ValueError(result.get("error", "Unknown pipeline error"))
    except Exception as e:
        logger.error(
            "Query pipeline failed",
            extra={"error": str(e), "deal_id": request.deal_id},
        )
        raise HTTPException(status_code=500, detail=f"Query pipeline error: {str(e)}")

    # Build response
    citations = [
        Citation(
            chunk_id=c.get("chunk_id", ""),
            source_file=c.get("source_file", ""),
            page_number=c.get("page_number"),
            section_heading=c.get("section_heading", ""),
            is_current_version=c.get("is_current_version", 1) == 1,
            content_type=c.get("content_type", "text"),
            is_redline=bool(c.get("is_redline", False)),
            superseded_by=c.get("superseded_by", ""),
        )
        for c in result.get("citations", [])
    ]

    # A refusal is any terminal state the pipeline reached without usable context —
    # either the quality gate forced it, or the rewrite loop exhausted its budget.
    is_refusal = bool(result.get("force_refusal", False)) or not result.get(
        "generated_answer", ""
    )

    response = QueryResponse(
        answer=result.get("generated_answer", ""),
        query_type=result.get("query_type", "summary"),
        confidence_score=result.get("confidence_score", 0.0),
        validation_status=result.get("validation_status", "passed"),
        citations=citations,
        hallucination_flags=result.get("hallucination_flags", []),
        total_latency_ms=result.get("total_latency_ms", 0.0),
        session_id=session_id,
        rewrite_iterations=result.get("rewrite_iteration", 0),
        agent_trace=result.get("agent_trace", []),
        is_refusal=is_refusal,
        context_quality_score=result.get("context_quality_score", 0.0),
    )

    # Immutable audit log
    logger.info(
        "AUDIT_LOG",
        extra={
            "event": "query_completed",
            "deal_id": request.deal_id,
            "session_id": session_id,
            "query": request.query,
            "query_type": response.query_type,
            "confidence_score": response.confidence_score,
            "validation_status": response.validation_status,
            "latency_ms": response.total_latency_ms,
            "citations_count": len(citations),
            "rewrite_iterations": response.rewrite_iterations,
            # Compliance: an authorized PII query must be attributable after the fact.
            "include_pii": request.include_pii,
            "is_refusal": response.is_refusal,
        },
    )

    return response


@router.get("/budget")
async def get_budget_status():
    """Returns current API budget status for all models."""
    from src.llm.budget_tracker import BudgetTracker

    try:
        tracker = await BudgetTracker.get_instance()
        return await tracker.get_budget_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget status error: {str(e)}")
