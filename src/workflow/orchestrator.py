"""
LangGraph State Machine — M&A Due Diligence Query Pipeline.

CRITICAL RULES:
- All nodes are async def. Invoke with await app.ainvoke()
- PostgresSaver checkpoints state to (deal_id, session_id)
- Edge: rewriter → executor is EXPLICIT (not conditional)
- operator.add reducer for accumulating fields (picklable)
- Do NOT use lambda reducers — breaks PostgresSaver

State Machine Topology:
    START → query_intelligence → retrieval_executor → route_to_financial_verifier
    [financial_verifier | quality_assessor] → route_after_quality_check
    [answer_synthesizer | query_rewriter → retrieval_executor | insufficient_context]
    answer_synthesizer → hallucination_validator → route_after_validation → END
"""

import time
import uuid
from contextlib import AsyncExitStack

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.state_definitions import AgentState
from src.workflow.conditional_edges import (
    route_to_financial_verifier,
    route_after_quality_check,
    route_after_validation,
)
from src.agents.query_intelligence import query_intelligence_node
from src.agents.retrieval_executor import retrieval_executor_node
from src.agents.financial_verifier import financial_verifier_node
from src.agents.quality_assessor import quality_assessor_node
from src.agents.query_rewriter import query_rewriter_node
from src.agents.answer_synthesizer import answer_synthesizer_node
from src.agents.hallucination_validator import hallucination_validator_node
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def insufficient_context_node(state: AgentState) -> dict:
    """
    Terminal node for queries where context is insufficient after max rewrites.
    Sets force_refusal=True so the synthesizer generates a refusal.

    Args:
        state: Current AgentState.

    Returns:
        State with force_refusal and generated refusal answer.
    """
    logger.info("Insufficient context — generating refusal")
    return {
        "force_refusal": True,
        "generated_answer": (
            "I was unable to find sufficient relevant information in the data room "
            "to answer this question, even after refining the search. This may mean "
            "the relevant documents haven't been uploaded yet, or the question falls "
            "outside the scope of the available materials.\n\n"
            f"Search attempts: {state.get('rewrite_iteration', 0) + 1}\n"
            f"Best quality score achieved: {state.get('context_quality_score', 0):.2f}"
        ),
        "confidence_score": 0.0,
        "validation_status": "passed",  # No validation needed for refusal
        "agent_trace": [
            {"agent": "insufficient_context", "force_refusal": True}
        ],
    }


async def retry_synthesis_node(state: AgentState) -> dict:
    """
    Re-runs synthesis after validation failure.
    Simply delegates back to answer_synthesizer_node with current state.

    Args:
        state: Current AgentState with validation feedback.

    Returns:
        Updated state from re-synthesis.
    """
    logger.info(
        "Retrying synthesis after validation failure",
        extra={"attempt": state.get("validation_attempt", 0)},
    )
    return await answer_synthesizer_node(state)


def build_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph with all nodes and edges.
    Does NOT compile — call .compile(checkpointer=...) to get a runnable.

    Returns:
        Uncompiled StateGraph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_intelligence", query_intelligence_node)
    graph.add_node("retrieval_executor", retrieval_executor_node)
    graph.add_node("financial_verifier", financial_verifier_node)
    graph.add_node("quality_assessor", quality_assessor_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("answer_synthesizer", answer_synthesizer_node)
    graph.add_node("hallucination_validator", hallucination_validator_node)
    graph.add_node("insufficient_context", insufficient_context_node)
    graph.add_node("retry_synthesis", retry_synthesis_node)

    # Set entry point
    graph.set_entry_point("query_intelligence")

    # Explicit edges
    graph.add_edge("query_intelligence", "retrieval_executor")

    # After retrieval → conditional: financial verifier or quality assessor
    graph.add_conditional_edges(
        "retrieval_executor",
        route_to_financial_verifier,
        {
            "financial_verifier": "financial_verifier",
            "quality_assessor": "quality_assessor",
        },
    )

    # After financial verifier → quality assessor (always)
    graph.add_edge("financial_verifier", "quality_assessor")

    # After quality check → conditional: synthesize, rewrite, or refuse
    graph.add_conditional_edges(
        "quality_assessor",
        route_after_quality_check,
        {
            "answer_synthesizer": "answer_synthesizer",
            "query_rewriter": "query_rewriter",
            "insufficient_context": "insufficient_context",
        },
    )

    # CRITICAL: rewriter → executor is EXPLICIT edge (self-correction loop)
    graph.add_edge("query_rewriter", "retrieval_executor")

    # After synthesis → validation
    graph.add_edge("answer_synthesizer", "hallucination_validator")

    # After validation → conditional: retry or end
    graph.add_conditional_edges(
        "hallucination_validator",
        route_after_validation,
        {
            "retry_synthesis": "retry_synthesis",
            "end": END,
        },
    )

    # Retry synthesis → validation again
    graph.add_edge("retry_synthesis", "hallucination_validator")

    # Insufficient context → end (after generating refusal)
    graph.add_edge("insufficient_context", END)

    return graph


async def get_compiled_graph(postgres_url: str):
    """
    Compiles the graph with PostgresSaver checkpointer.
    Checkpoints are keyed to (deal_id, session_id) via the config dict.

    `AsyncPostgresSaver.from_conn_string()` returns an async *context manager*,
    not a saver — the connection must stay open for as long as the graph is
    used, so it is entered into a module-level AsyncExitStack that the API
    lifespan unwinds via close_checkpointer(). Awaiting `.setup()` on the
    context manager directly raises AttributeError and silently drops every
    deployment back to MemorySaver, losing crash recovery.

    Args:
        postgres_url: PostgreSQL connection string.

    Returns:
        Compiled LangGraph application ready for ainvoke().
    """
    global _checkpointer_stack

    graph = build_graph()
    try:
        stack = AsyncExitStack()
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(postgres_url)
        )
        await checkpointer.setup()
        app = graph.compile(checkpointer=checkpointer)
        _checkpointer_stack = stack
        logger.info("LangGraph state machine compiled with PostgresSaver")
    except Exception as e:
        # Degrade rather than fail: the pipeline is fully functional without
        # durable checkpoints, it just cannot resume a session after a restart.
        logger.warning(
            f"Failed to initialize PostgresSaver: {e}. Falling back to MemorySaver."
        )
        await close_checkpointer()
        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)
        logger.info("LangGraph state machine compiled with MemorySaver")
    return app


async def close_checkpointer() -> None:
    """
    Closes the Postgres checkpointer connection, if one was opened.

    Called from the FastAPI lifespan shutdown. Safe to call when the fallback
    MemorySaver is in use — it is a no-op.
    """
    global _checkpointer_stack

    if _checkpointer_stack is None:
        return

    try:
        await _checkpointer_stack.aclose()
        logger.info("PostgresSaver checkpointer closed")
    except Exception as e:
        logger.warning(f"Error closing checkpointer: {e}")
    finally:
        _checkpointer_stack = None


async def run_query(
    app,
    query: str,
    deal_id: str,
    session_id: str | None = None,
    include_pii: bool = False,
) -> AgentState:
    """
    Executes a full query through the pipeline.

    Args:
        app: Compiled LangGraph application.
        query: User's natural language query.
        deal_id: Deal identifier for isolation.
        session_id: Optional session ID (generated if not provided).
        include_pii: Compliance override from the authenticated caller. Defaults
            to False so PII-flagged chunks stay excluded unless explicitly
            authorized. Never set from LLM output — see AgentState.include_pii.

    Returns:
        Final AgentState with answer and metadata.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    start = time.monotonic()

    initial_state: AgentState = {
        "original_query": query,
        "current_query": query,
        "query_type": "summary",  # Will be overwritten by Agent 1
        "parsed_intent": {},
        "extracted_filters": {},
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
        "deal_id": deal_id,
        "session_id": session_id,
        "include_pii": include_pii,
        "total_latency_ms": 0.0,
        "status": "running",
        "error": None,
    }

    config = {
        "configurable": {
            "thread_id": f"{deal_id}_{session_id}",
        }
    }

    logger.info(
        "Starting query pipeline",
        extra={"query": query, "deal_id": deal_id, "session_id": session_id},
    )

    try:
        # CRITICAL: Use ainvoke() — NOT invoke() which deadlocks with async nodes
        result = await app.ainvoke(initial_state, config=config)
        elapsed_ms = (time.monotonic() - start) * 1000
        result["total_latency_ms"] = round(elapsed_ms, 2)
        result["status"] = "completed"

        logger.info(
            "Query pipeline completed",
            extra={
                "session_id": session_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "validation_status": result.get("validation_status"),
                "confidence_score": result.get("confidence_score"),
            },
        )

        return result

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error(
            "Query pipeline failed",
            extra={
                "session_id": session_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": str(e),
            },
        )
        initial_state["status"] = "error"
        initial_state["error"] = str(e)
        initial_state["total_latency_ms"] = round(elapsed_ms, 2)
        return initial_state
