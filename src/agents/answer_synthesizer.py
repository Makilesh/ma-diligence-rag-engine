"""
Agent 7 — Answer Synthesis Agent.

Model: gemini-3.5-flash (budget) / gemini-3.1-flash-lite (fallback)
Temp: 0.1 | Tokens: 3000 | JSON mode: OFF (prose answer)
"""

import json

from src.llm.litellm_wrapper import call_prose_agent
from src.llm.budget_tracker import BudgetTracker
from src.llm.prompt_templates.answer_synthesizer import (
    ANSWER_SYNTHESIZER_SYSTEM_PROMPT,
    ANSWER_SYNTHESIZER_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _format_context_for_synthesis(chunks: list[dict]) -> str:
    """
    Formats expanded context chunks for the synthesis prompt.
    Includes metadata for citation generation.

    Args:
        chunks: List of expanded context chunk dicts.

    Returns:
        Formatted string for inclusion in the synthesis prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = []
        if chunk.get("source_file"):
            meta.append(f"Source: {chunk['source_file']}")
        if chunk.get("page_number"):
            meta.append(f"Page: {chunk['page_number']}")
        if chunk.get("section_heading"):
            meta.append(f"Section: {chunk['section_heading']}")
        if chunk.get("fiscal_year"):
            meta.append(f"FY: {chunk['fiscal_year']}")
        if chunk.get("is_current_version") == 0:
            meta.append("⚠ NOT CURRENT VERSION")
        if chunk.get("content_type") == "computed_metric":
            meta.append("COMPUTED METRIC")
        if chunk.get("is_redline"):
            meta.append("REDLINE VERSION")

        meta_str = " | ".join(meta)
        text = chunk.get("text", "")
        parent_text = chunk.get("parent_text", "")

        part = f"--- Chunk {i} [{meta_str}] ---\n{text}"
        if parent_text:
            part += f"\n[Parent context]: {parent_text[:500]}"
        parts.append(part)

    return "\n\n".join(parts)


async def answer_synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph node — generates prose answer with citations.
    Populates: generated_answer, citations, numerical_claims, agent_trace.

    Uses gemini-3.5-flash for synthesis quality when budget available,
    falls back to gemini-3.1-flash-lite when exhausted.

    Args:
        state: Current AgentState with expanded_context and query.

    Returns:
        Partial state dict with answer and citations.
    """
    logger.info("Agent 7: Answer Synthesizer starting")

    # Check for forced refusal
    if state.get("force_refusal"):
        logger.info("Agent 7: Forced refusal — insufficient context")
        return {
            "generated_answer": (
                "I don't have sufficient information in the data room to answer "
                "this question accurately. The retrieved documents did not contain "
                "relevant content for your query. Please try rephrasing your "
                "question or check that the relevant documents have been uploaded."
            ),
            "citations": [],
            "numerical_claims": [],
            "confidence_score": 0.0,
            "agent_trace": [
                {"agent": "answer_synthesizer", "forced_refusal": True}
            ],
        }

    chunks = state.get("expanded_context", state.get("reranked_results", []))
    context = _format_context_for_synthesis(chunks)

    # Financial verification results
    financial_verification = ""
    if state.get("numerical_registry"):
        financial_verification = json.dumps(
            state["numerical_registry"], indent=2, default=str
        )

    inconsistencies = ""
    if state.get("inconsistencies"):
        inconsistencies = json.dumps(
            state["inconsistencies"], indent=2, default=str
        )

    user_prompt = ANSWER_SYNTHESIZER_USER_TEMPLATE.format(
        query=state["current_query"],
        query_type=state["query_type"],
        context=context,
        financial_verification=financial_verification or "N/A",
        inconsistencies=inconsistencies or "None found",
    )

    tracker = await BudgetTracker.get_instance()
    choice = await tracker.get_model_for_synthesis()
    model = choice.model

    try:
        answer = await call_prose_agent(
            system_prompt=ANSWER_SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            temperature=0.1,
            max_tokens=3000,
            api_key=choice.api_key,
        )
    except Exception as e:
        # Synthesis is the one place where an upstream failure would otherwise
        # take down the whole request. Degrade to an explicit refusal instead:
        # a reviewer who is told the engine could not answer is strictly better
        # off than one who gets a 500 and no explanation.
        logger.error(
            "Agent 7: synthesis failed, degrading to refusal",
            extra={"model": model, "error": str(e)},
        )
        return {
            "generated_answer": (
                "I could not generate an answer for this question because the "
                "language model did not return a usable response. The retrieved "
                "context may still be relevant — please retry the query."
            ),
            "citations": [],
            "numerical_claims": [],
            "confidence_score": 0.0,
            "agent_trace": [
                {
                    "agent": "answer_synthesizer",
                    "model": model,
                    "synthesis_failed": True,
                    "error": str(e),
                }
            ],
        }

    # Extract citations from the answer (basic extraction).
    #
    # KNOWN IMPRECISION: matching on source_file is document-level, not
    # passage-level. When every retrieved chunk comes from the same document,
    # all of them are returned, so the citation list can name sections the answer
    # never used. Verified against a live run: an answer citing only "SECTION 8.
    # INDEMNIFICATION" produced citations labelled SECTION 7 and SECTION 5.
    # Narrowing by section_heading does not fix it — section_heading is the
    # heading of the *chunk*, and a chunk routinely spans several sections, so
    # the heading is simply absent from the answer. A real fix means either
    # finer-grained heading metadata per chunk, or parsing the inline
    # `[file | p.N | SECTION]` markers the synthesis prompt already emits.
    #
    # The full chunk payload is carried through — not just chunk_id/source_file —
    # so the API can surface page, section, version and computed-metric provenance
    # without a second round-trip to Qdrant. Anything the ingestion pipeline does
    # not yet write (is_redline, superseded_by) simply falls back to its default.
    citations = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "source_file": c.get("source_file", ""),
            "page_number": c.get("page_number"),
            "section_heading": c.get("section_heading", "") or "",
            "is_current_version": c.get("is_current_version", 1),
            "content_type": c.get("content_type", "text") or "text",
            "is_redline": bool(c.get("is_redline", 0)),
            "superseded_by": c.get("superseded_by", "") or "",
        }
        for c in chunks
        if c.get("source_file") and c.get("source_file", "") in answer
    ]

    logger.info(
        "Agent 7: Answer Synthesizer complete",
        extra={
            "model": model,
            "answer_length": len(answer),
            "citations_count": len(citations),
        },
    )

    return {
        "generated_answer": answer,
        "citations": citations,
        "numerical_claims": state.get("inconsistencies", []),
        "agent_trace": [
            {
                "agent": "answer_synthesizer",
                "model": model,
                "answer_length": len(answer),
            }
        ],
    }
