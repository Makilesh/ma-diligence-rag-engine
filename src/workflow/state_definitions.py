"""
LangGraph state definitions for the M&A Due Diligence query pipeline.

CRITICAL RULES:
- Use Annotated[list[dict], operator.add] for accumulating fields
  (rewrite_history, agent_trace) — these accumulate across rewrite iterations.
- Do NOT use lambda reducers — they are not picklable and break PostgresSaver.
  operator.add IS picklable.
- All nodes are async def. The compiled graph must be invoked with
  await app.ainvoke() — NOT app.invoke() which deadlocks with async nodes.
"""

import operator
from typing import TypedDict, Annotated, Literal, Optional


class AgentState(TypedDict):
    """
    Central state object passed through all LangGraph nodes.
    PostgresSaver checkpoints this state keyed to (deal_id, session_id).

    Fields marked with Annotated[list, operator.add] use the add reducer:
    each node return APPENDS to the list rather than overwriting it.
    This is critical for rewrite_history and agent_trace which must
    accumulate across rewrite iterations.

    Boolean-like fields (is_table, is_current_version, contains_pii)
    are stored as integers (0/1) in Qdrant for cross-version compatibility.
    """

    # ==================== Query ====================
    original_query: str
    current_query: str
    query_type: Literal["financial", "legal", "comparative", "summary", "multi_hop"]

    # ==================== Parsed Signals ====================
    parsed_intent: dict
    extracted_filters: dict  # {fiscal_year, document_category, is_current_version, ...}
    # Atomic sub-questions for multi-hop / comparative queries, each asking for
    # one distinct fact. Distinct from parsed_intent["query_expansions"], which
    # are rephrasings of the whole question. Retrieval runs one pass per entry
    # and guarantees each a share of the final context — see retrieval_executor.
    sub_questions: list[str]

    # ==================== Retrieval ====================
    retrieval_config: dict  # From deterministic lookup — NOT from an LLM call
    dense_results: list[dict]
    sparse_results: list[dict]
    fused_results: list[dict]
    reranked_results: list[dict]
    expanded_context: list[dict]  # Parent + sibling chunks after expansion

    # ==================== Quality ====================
    context_quality_score: float
    quality_breakdown: dict
    quality_method: Literal["heuristic", "llm"]
    missing_aspects: list[str]

    # ==================== Self-Correction ====================
    # ⚠ Do NOT use lambdas here: PostgresSaver requires picklable reducers.
    #   lambda x, y: x + y is NOT picklable. operator.add IS picklable.
    rewrite_iteration: int
    rewrite_history: Annotated[list[dict], operator.add]  # accumulates — never overwrites
    agent_trace: Annotated[list[dict], operator.add]  # accumulates — never overwrites

    # ==================== Financial ====================
    numerical_registry: dict
    inconsistencies: list[dict]

    # ==================== Answer ====================
    generated_answer: str
    citations: list[dict]
    numerical_claims: list[dict]

    # ==================== Validation ====================
    confidence_score: float
    hallucination_flags: list[str]
    validation_status: Literal["passed", "warning", "failed"]
    validation_attempt: int
    force_refusal: bool  # Set by quality_assessor_node when context is insufficient

    # ==================== Session ====================
    deal_id: str
    session_id: str
    # Compliance authorization for PII-flagged content. Set ONCE from the
    # authenticated API request and never derived from LLM output — Agents 1
    # and 6 both strip `include_pii` out of extracted_filters precisely so a
    # model cannot escalate its own access. Consumed by retrieval_executor.
    include_pii: bool
    total_latency_ms: float
    status: str
    error: Optional[str]
