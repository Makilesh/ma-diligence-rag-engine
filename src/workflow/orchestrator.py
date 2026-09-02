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

# Holds the open AsyncPostgresSaver context for the process lifetime.
# None when the MemorySaver fallback is in use.
_checkpointer_stack: AsyncExitStack | None = None


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


def _build_initial_state(
    query: str,
    deal_id: str,
    session_id: str,
    include_pii: bool,
) -> AgentState:
    """
    Builds the zero-value AgentState every run starts from.

    Extracted so `run_query` and `stream_query` cannot drift apart. A field
    missing from one but not the other is not a crash — TypedDict access on an
    absent key raises only where it is read — so the two would diverge silently
    until some agent read a key the streaming path never seeded.

    Args:
        query: User's natural language question.
        deal_id: Deal identifier for retrieval isolation.
        session_id: Thread key for checkpointing.
        include_pii: Compliance override from the authenticated caller.

    Returns:
        A fully populated initial AgentState.
    """
    return {
        "original_query": query,
        "current_query": query,
        "query_type": "summary",  # Will be overwritten by Agent 1
        "parsed_intent": {},
        "extracted_filters": {},
        "sub_questions": [],
        "retrieval_config": {},
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "retrieval_passes": [],
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

    initial_state = _build_initial_state(query, deal_id, session_id, include_pii)

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


# ==============================================================================
# Streaming
# ==============================================================================
# The pipeline takes tens of seconds end to end, which is a long time to show a
# spinner. Every node already reports what it did through `agent_trace`, so the
# same information can be pushed to the client as each node lands rather than
# being withheld until the last one finishes.

# Display metadata for the graph's nodes. Keyed by node name so an unknown node
# (added later, or one of LangGraph's internal keys) degrades to a title-cased
# label rather than disappearing from the timeline.
_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "query_intelligence": (
        "Query Intelligence",
        "Classifying intent and decomposing the question",
    ),
    "retrieval_executor": (
        "Retrieval Executor",
        "Hybrid search, rank fusion and reranking",
    ),
    "financial_verifier": (
        "Financial Verifier",
        "Cross-checking figures against source tables",
    ),
    "quality_assessor": (
        "Quality Assessor",
        "Scoring whether the context can support an answer",
    ),
    "query_rewriter": ("Query Rewriter", "Reformulating after thin retrieval"),
    "answer_synthesizer": ("Answer Synthesizer", "Writing the cited answer"),
    "retry_synthesis": ("Answer Synthesizer", "Re-writing after a validation failure"),
    "hallucination_validator": (
        "Hallucination Validator",
        "Grounding every claim in retrieved text",
    ),
    "insufficient_context": ("Insufficient Context", "Declining rather than guessing"),
}

# The nodes that always run, in topological order. Sent up-front so the client
# can paint the whole timeline as pending instead of growing it a row at a time.
# Conditional nodes (verifier, rewriter, refusal) are appended only if reached.
_STAGE_ORDER: list[str] = [
    "query_intelligence",
    "retrieval_executor",
    "quality_assessor",
    "answer_synthesizer",
    "hallucination_validator",
]


def _summarize_stage(node: str, delta: dict) -> tuple[str, dict]:
    """
    Turns a node's state delta into a one-line summary and a small detail bag.

    The summary is what the client renders on the timeline row, so it is written
    to read as a finding ("decomposed into 3 sub-questions") rather than a status
    ("query_intelligence complete") — the point of showing the pipeline is that
    each step reports something the user could not have guessed.

    Args:
        node: Graph node name that just completed.
        delta: The partial state that node returned.

    Returns:
        (summary, detail) — detail holds the structured values behind the line.
    """
    # Every node appends exactly one trace entry; it carries the richest data.
    trace = (delta.get("agent_trace") or [{}])[-1]

    if node == "query_intelligence":
        subs = delta.get("sub_questions") or []
        qtype = delta.get("query_type", "summary")
        filters = delta.get("extracted_filters") or {}
        summary = f"Classified as {qtype}"
        if subs:
            plural = "s" if len(subs) > 1 else ""
            summary += f" - decomposed into {len(subs)} sub-question{plural}"
        return summary, {
            "query_type": qtype,
            "sub_questions": subs,
            "filters": {k: v for k, v in filters.items() if v},
            "model": trace.get("model", ""),
        }

    if node == "retrieval_executor":
        reranked = trace.get("reranked_count", 0)
        passes = trace.get("retrieval_passes", 0)
        sources = trace.get("sources", []) or []
        pass_plural = "es" if passes != 1 else ""
        doc_plural = "s" if len(sources) != 1 else ""
        summary = (
            f"{reranked} chunks reranked from {passes} retrieval pass{pass_plural} "
            f"across {len(sources)} document{doc_plural}"
        )
        return summary, {
            "reranked_count": reranked,
            "retrieval_passes": passes,
            "sub_questions": trace.get("sub_questions", []),
            "sources": sources,
            "elapsed_ms": trace.get("elapsed_ms", 0),
        }

    if node == "financial_verifier":
        inconsistencies = delta.get("inconsistencies") or []
        registry = delta.get("numerical_registry") or {}
        noun = "inconsistency" if len(inconsistencies) == 1 else "inconsistencies"
        summary = (
            f"{len(registry)} figures normalised - {len(inconsistencies)} {noun} flagged"
        )
        return summary, {
            "figures": len(registry),
            "inconsistencies": inconsistencies,
        }

    if node == "quality_assessor":
        score = delta.get("context_quality_score", 0.0)
        missing = delta.get("missing_aspects") or []
        method = delta.get("quality_method", "heuristic")
        summary = f"Context quality {score:.2f} ({method})"
        if missing:
            plural = "s" if len(missing) > 1 else ""
            summary += f" - {len(missing)} aspect{plural} missing"
        return summary, {
            "score": score,
            "method": method,
            "missing_aspects": missing,
            "breakdown": delta.get("quality_breakdown", {}),
        }

    if node == "query_rewriter":
        iteration = delta.get("rewrite_iteration", 0)
        return f"Reformulated the query (attempt {iteration + 1})", {
            "rewritten_query": delta.get("current_query", ""),
            "iteration": iteration,
        }

    if node in ("answer_synthesizer", "retry_synthesis"):
        citations = delta.get("citations") or []
        plural = "s" if len(citations) != 1 else ""
        return f"Answer drafted with {len(citations)} citation{plural}", {
            "citation_count": len(citations),
            "answer_length": trace.get("answer_length", 0),
            "model": trace.get("model", ""),
        }

    if node == "hallucination_validator":
        status = delta.get("validation_status", "passed")
        flags = delta.get("hallucination_flags") or []
        confidence = delta.get("confidence_score", 0.0)
        summary = f"Validation {status} - confidence {confidence:.0%}"
        if flags:
            plural = "s" if len(flags) > 1 else ""
            summary += f" - {len(flags)} claim{plural} unsupported"
        return summary, {
            "validation_status": status,
            "confidence_score": confidence,
            "hallucination_flags": flags,
        }

    if node == "insufficient_context":
        return "Refused - no passage in the data room supports an answer", {
            "force_refusal": True,
        }

    return f"{node} completed", {}


async def stream_query(
    app,
    query: str,
    deal_id: str,
    session_id: str | None = None,
    include_pii: bool = False,
):
    """
    Runs the pipeline, yielding an event per completed node then the final state.

    Uses `stream_mode=["updates", "values"]`: "updates" carries each node's own
    delta — the only place per-node output is visible — while "values" carries
    the accumulated state, whose last emission is the final result. Requesting
    both avoids re-implementing LangGraph's reducers to rebuild the end state
    from deltas, which is exactly where a hand-rolled accumulator would go wrong:
    `agent_trace` uses an `operator.add` reducer, not replacement, so naive
    dict-merging of deltas would keep only the last agent's trace entry.

    Args:
        app: Compiled LangGraph application.
        query: User's natural language query.
        deal_id: Deal identifier for isolation.
        session_id: Optional session ID (generated if not provided).
        include_pii: Compliance override from the authenticated caller.

    Yields:
        (event_name, payload) tuples: one "start", many "stage", then exactly one
        of "result" or "error".
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    start = time.monotonic()
    initial_state = _build_initial_state(query, deal_id, session_id, include_pii)
    config = {"configurable": {"thread_id": f"{deal_id}_{session_id}"}}

    logger.info(
        "Starting streamed query pipeline",
        extra={"query": query, "deal_id": deal_id, "session_id": session_id},
    )

    yield (
        "start",
        {
            "session_id": session_id,
            "planned_stages": [
                {
                    "agent": node,
                    "label": _STAGE_LABELS[node][0],
                    "description": _STAGE_LABELS[node][1],
                }
                for node in _STAGE_ORDER
            ],
        },
    )

    final_state: dict | None = None
    last_mark = start
    seq = 0

    try:
        async for mode, chunk in app.astream(
            initial_state, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                final_state = chunk
                continue

            # mode == "updates": {node_name: delta}. One key in practice, but the
            # loop keeps parallel-branch support free if the graph ever gains one.
            for node, delta in (chunk or {}).items():
                if not isinstance(delta, dict):
                    continue
                now = time.monotonic()
                summary, detail = _summarize_stage(node, delta)
                label, description = _STAGE_LABELS.get(
                    node, (node.replace("_", " ").title(), "")
                )
                seq += 1
                yield (
                    "stage",
                    {
                        "seq": seq,
                        "agent": node,
                        "label": label,
                        "description": description,
                        "summary": summary,
                        "detail": detail,
                        "duration_ms": round((now - last_mark) * 1000, 2),
                        "elapsed_ms": round((now - start) * 1000, 2),
                    },
                )
                last_mark = now

    except Exception as e:
        logger.error(
            "Streamed query pipeline failed",
            extra={"session_id": session_id, "error": str(e)},
        )
        yield ("error", {"detail": str(e), "session_id": session_id})
        return

    if final_state is None:
        yield (
            "error",
            {"detail": "Pipeline produced no state", "session_id": session_id},
        )
        return

    elapsed_ms = (time.monotonic() - start) * 1000
    final_state["total_latency_ms"] = round(elapsed_ms, 2)
    final_state["status"] = "completed"

    logger.info(
        "Streamed query pipeline completed",
        extra={
            "session_id": session_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "validation_status": final_state.get("validation_status"),
        },
    )

    yield ("result", final_state)
