"""
Conditional edge functions for the LangGraph state machine.

CRITICAL: Edge functions must NOT mutate state — they only read state
and return routing strings. LangGraph's reducer/checkpointing model
merges state at superstep boundaries via node returns. Mutating state
in edge functions bypasses reducers and can break checkpoint resume.
"""

from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Quality thresholds by query type.
#
# CALIBRATED, not hand-picked. The original values (relevance 0.7–0.8,
# precision 0.5–0.8) were specified as abstract 0–1 quality targets, but they are
# compared against numbers derived from BAAI/bge-reranker-v2-m3 scores, whose
# actual distribution on this corpus nobody had measured. The result was a gate
# almost nothing could pass: `precision >= 0.8` for financial queries required
# the fifth-best chunk to score >= 0.6, and 10 of 16 golden questions were
# refused despite retrieval having surfaced a relevant chunk for 19 of 19.
#
# Re-derived by scoring all 19 golden questions with the redesigned dimensions in
# quality_assessor.py (see its calibration note). Of those 19, the 18 whose
# retrieval genuinely contained the answer scored:
#     relevance >= 0.521, precision >= 0.298, completeness >= 0.333
# and the single genuine retrieval failure (comp_02) scored 0.009 / 0.000 / 0.000.
#
# The thresholds below therefore sit in a wide empty band — nothing lands between
# 0.009 and 0.521 on relevance — so the gate still refuses the query it should
# while admitting the ones it should never have blocked. A threshold near a dense
# part of the distribution would be fragile; this one is not.
#
# Type-specific expectation lives in EXPECTED_EVIDENCE_COUNT (the completeness
# denominator), which is why these are uniform rather than a second set of
# per-type constants. See DECISIONS_LOG Decision 17.
_CALIBRATED = {"relevance": 0.30, "completeness": 0.30, "precision": 0.25}

THRESHOLDS: dict[str, dict[str, float]] = {
    "financial": dict(_CALIBRATED),
    "legal": dict(_CALIBRATED),
    "summary": dict(_CALIBRATED),
    "comparative": dict(_CALIBRATED),
    "multi_hop": dict(_CALIBRATED),
}


def route_to_financial_verifier(state: AgentState) -> str:
    """
    Routes to financial verifier if query requires numerical precision.

    Args:
        state: Current AgentState.

    Returns:
        "financial_verifier" or "quality_assessor".
    """
    if state["query_type"] == "financial" or state["parsed_intent"].get(
        "requires_numerical_precision", False
    ):
        logger.info("Routing to financial verifier")
        return "financial_verifier"
    logger.info("Skipping financial verifier, routing to quality assessor")
    return "quality_assessor"


def _meets_type_threshold(state: AgentState) -> bool:
    """
    Returns True if quality_breakdown meets ALL per-dimension thresholds
    for the current query type.

    Args:
        state: Current AgentState with quality_breakdown and query_type.

    Returns:
        True if every dimension score >= its threshold for this query type.
    """
    breakdown = state.get("quality_breakdown", {})
    thresholds = THRESHOLDS.get(state["query_type"], {})
    return all(breakdown.get(k, 0.0) >= v for k, v in thresholds.items())


def route_after_quality_check(state: AgentState) -> str:
    """
    Conditional edge function — reads state only, never mutates it.

    Routes to:
    - "answer_synthesizer": if quality is sufficient
    - "query_rewriter": if quality is insufficient and rewrites remain
    - "insufficient_context": if max rewrites exhausted (forced refusal path)

    Args:
        state: Current AgentState.

    Returns:
        Routing string for the next node.
    """
    score = state["context_quality_score"]
    iteration = state["rewrite_iteration"]

    # The thresholds read reranker relevance; an answerability veto says the
    # relevant passages still do not state what was asked, so it overrides them.
    vetoed = bool(state.get("answerability_veto", False))
    if score >= 0.3 and _meets_type_threshold(state) and not vetoed:
        logger.info(
            "Quality check passed, routing to synthesizer",
            extra={"score": score, "iteration": iteration},
        )
        return "answer_synthesizer"

    if iteration >= 2:  # Max 2 rewrites (not 3 — reduces worst-case latency)
        # force_refusal is already set by quality_assessor_node (see Agent 5).
        # Edge functions must NOT mutate state.
        logger.info(
            "Max rewrites exhausted, routing to insufficient_context",
            extra={"score": score, "iteration": iteration},
        )
        return "insufficient_context"

    logger.info(
        "Quality insufficient, routing to rewriter",
        extra={"score": score, "iteration": iteration, "answerability_veto": vetoed},
    )
    return "query_rewriter"


# Validations allowed per query: the first, plus one after a single re-synthesis.
# Each retry spends a synthesis call on the scarce reasoning-model quota, so the
# loop is bounded at one.
MAX_VALIDATION_ATTEMPTS = 2


def route_after_validation(state: AgentState) -> str:
    """
    Routes after hallucination validation.

    Routes to:
    - "retry_synthesis": if validation failed and a retry remains
    - "end": if validation passed or the retry was already used

    The validator node increments `validation_attempt` BEFORE this runs, so the
    first validation arrives here as 1. The original guard was `< 1`, which the
    real node output can never satisfy — the retry path was dead code, and its
    only test set the counter to 0 by hand. `< MAX_VALIDATION_ATTEMPTS` allows
    exactly one retry after a failed first validation.

    Args:
        state: Current AgentState.

    Returns:
        Routing string.
    """
    if (
        state["validation_status"] == "failed"
        and state.get("validation_attempt", 0) < MAX_VALIDATION_ATTEMPTS
    ):
        logger.info("Validation failed, retrying synthesis")
        return "retry_synthesis"

    logger.info(
        "Validation complete",
        extra={
            "status": state["validation_status"],
            "attempt": state.get("validation_attempt", 0),
        },
    )
    return "end"
