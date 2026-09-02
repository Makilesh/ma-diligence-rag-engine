"""
Query routes — main ask endpoint with immutable audit logging.

Two transports over one pipeline: `/query` returns the finished result in a
single response, `/query/stream` pushes an event per agent as the graph runs.
Both end at the same `QueryResponse`, built by one shared function so the
streamed payload can never describe a different result than the blocking one.
"""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models.request_models import QueryRequest
from api.models.response_models import QueryResponse, Citation
from src.workflow.orchestrator import run_query, stream_query
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


def _build_response(result: dict, session_id: str) -> QueryResponse:
    """
    Converts a terminal AgentState into the wire response.

    Args:
        result: Final state from the pipeline.
        session_id: Session ID this query ran under.

    Returns:
        Populated QueryResponse.
    """
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

    return QueryResponse(
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


def _audit(request: QueryRequest, response: QueryResponse, transport: str) -> None:
    """
    Writes the immutable audit record for a completed query.

    Args:
        request: The originating request.
        response: The response being returned.
        transport: "blocking" or "stream" — which endpoint served it.
    """
    logger.info(
        "AUDIT_LOG",
        extra={
            "event": "query_completed",
            "transport": transport,
            "deal_id": request.deal_id,
            "session_id": response.session_id,
            "query": request.query,
            "query_type": response.query_type,
            "confidence_score": response.confidence_score,
            "validation_status": response.validation_status,
            "latency_ms": response.total_latency_ms,
            "citations_count": len(response.citations),
            "rewrite_iterations": response.rewrite_iterations,
            # Compliance: an authorized PII query must be attributable after the fact.
            "include_pii": request.include_pii,
            "is_refusal": response.is_refusal,
        },
    )


def _require_graph():
    """Returns the compiled graph, or raises 503 if startup has not finished."""
    from api.main import get_graph

    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    return graph


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
    graph = _require_graph()
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

    response = _build_response(result, session_id)
    _audit(request, response, transport="blocking")
    return response


def _sse(event: str, payload: dict) -> str:
    """
    Formats one Server-Sent Event frame.

    `json.dumps` is what makes this safe: the spec terminates a frame at a blank
    line, so a raw newline anywhere in the payload — and answers are full of
    them — would split one event into two malformed ones. JSON escaping collapses
    every newline to `\\n` before it can reach the wire.

    Args:
        event: Event name the client listens for.
        payload: JSON-serialisable body.

    Returns:
        An SSE frame, terminated by the required blank line.
    """
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


@router.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streams pipeline progress as Server-Sent Events, then the finished result.

    The pipeline runs eight agents and takes tens of seconds, so the blocking
    endpoint leaves the client with nothing to show for most of the wait. This
    emits each agent's finding as it lands — what the query was classified as,
    how many chunks survived reranking, what the quality gate scored — and closes
    with the same QueryResponse `/query` would have returned.

    Event sequence: `start`, then one `stage` per completed agent, then exactly
    one of `result` or `error`.

    Args:
        request: QueryRequest with query, deal_id, optional session_id.

    Returns:
        StreamingResponse of `text/event-stream`.
    """
    graph = _require_graph()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "Streamed query received",
        extra={
            "deal_id": request.deal_id,
            "session_id": session_id,
            "query_length": len(request.query),
        },
    )

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event, payload in stream_query(
                app=graph,
                query=request.query,
                deal_id=request.deal_id,
                session_id=session_id,
                include_pii=request.include_pii,
            ):
                if event == "result":
                    response = _build_response(payload, session_id)
                    _audit(request, response, transport="stream")
                    yield _sse("result", response.model_dump())
                else:
                    yield _sse(event, payload)
        except Exception as e:
            # The status line is long gone by now — a 500 is not available, so the
            # only way to tell the client is an in-band error event.
            logger.error(
                "Streamed query failed",
                extra={"error": str(e), "deal_id": request.deal_id},
            )
            yield _sse("error", {"detail": str(e), "session_id": session_id})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Without this, nginx-style proxies buffer the whole response and
            # deliver every event at once at the end — which is precisely the
            # behaviour this endpoint exists to avoid.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/budget")
async def get_budget_status():
    """Returns current API budget status for all models."""
    from src.llm.budget_tracker import BudgetTracker

    try:
        tracker = await BudgetTracker.get_instance()
        return await tracker.get_budget_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget status error: {str(e)}")
