"""
Agent 5 — Quality Assessor Agent.

Primary: heuristics (no LLM)
Fallback: the agent ladder's current model, only for the narrow ambiguous band

When heuristic signals are clear (high reranker scores, good coverage),
skip the LLM entirely. Only invoke the LLM when quality is borderline
and heuristics disagree.
"""

import json

import numpy as np

from src.llm.litellm_wrapper import call_structured_agent
from src.llm.budget_tracker import BudgetTracker
from src.llm.prompt_templates.quality_assessor import (
    QUALITY_ASSESSOR_SYSTEM_PROMPT,
    QUALITY_ASSESSOR_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# Calibration constants
#
# These are measured, not chosen. tests/golden_qa_set.json provides 19 questions
# with expected facts; labelling every reranked chunk relevant/irrelevant by
# whether it contains an expected fact yields a ground-truth score distribution
# for BAAI/bge-reranker-v2-m3 on this corpus:
#
#   relevant chunks   (n=49):  median 0.241, max 0.998
#   irrelevant chunks (n=143): median 0.006, p75 0.111
#
# A floor sweep over that labelled data put peak F1 at 0.10, so that is the
# boundary between "usable evidence" and noise. See DECISIONS_LOG Decision 17.
# ==============================================================================

# Score at or above which a chunk counts as usable evidence (peak-F1 from sweep).
RELEVANCE_FLOOR = 0.10

# Below this, the best chunk is noise and no amount of context will help.
CONFIDENT_REFUSAL_CEILING = 0.05

# Above this, the heuristic is confident enough to skip the LLM assessor.
CONFIDENT_PASS_FLOOR = 0.30

# How many usable chunks a fully-answered query of each type needs. Taken from
# the observed relevant-chunk counts per type in the same labelled run: pointed
# lookups (financial, legal) are satisfied by a couple of chunks, whereas
# summary/comparative/multi-hop genuinely need breadth. This denominator is where
# query-type expectations live — which is why the gate thresholds themselves are
# uniform rather than a second set of hand-tuned per-type numbers.
EXPECTED_EVIDENCE_COUNT: dict[str, int] = {
    "financial": 2,
    "legal": 2,
    "comparative": 3,
    "summary": 3,
    "multi_hop": 3,
}


def _heuristic_assessment(state: AgentState) -> dict | None:
    """
    Attempt heuristic quality assessment without LLM.
    Returns assessment dict if confident, None if LLM fallback needed.

    Scores the *usable* evidence rather than the whole retrieved set. This is the
    key correction over the original mean/min-of-top-5 design: a cross-encoder is
    a per-pair relevance classifier, so its outputs are sharply bimodal — on this
    corpus a genuinely relevant chunk scores 0.24–0.99 while noise sits near
    0.006. Averaging across the whole top-k therefore drags quality down in
    proportion to how many candidates were retrieved, so improving recall made
    the context look *worse*, and requiring `min(top_5) >= 0.2` demanded that the
    fifth-best chunk be excellent — a bar that peaked score distributions almost
    never clear. Measured effect: two golden queries scored 0.638 and 0.645 with
    genuinely good context and were still refused.

    Dimensions:
      relevance    — max score. Is there strong evidence at all? One decisive
                     chunk is enough to answer a pointed question.
      precision    — mean score of the chunks above RELEVANCE_FLOOR, i.e. the
                     quality of the evidence that would actually be used. Noise
                     is excluded rather than allowed to dilute the signal.
      completeness — how much usable evidence there is relative to what this
                     query type needs (EXPECTED_EVIDENCE_COUNT).

    Args:
        state: Current AgentState with reranked_results.

    Returns:
        Assessment dict if heuristic is confident, None otherwise.
    """
    chunks = state.get("reranked_results", [])

    if not chunks:
        return {
            "context_quality_score": 0.0,
            "quality_breakdown": {"relevance": 0.0, "completeness": 0.0, "precision": 0.0},
            "missing_aspects": ["No chunks retrieved"],
            "quality_method": "heuristic",
            "force_refusal": True,
        }

    scores = [float(c.get("reranker_score", 0.0)) for c in chunks]
    usable = [s for s in scores if s >= RELEVANCE_FLOOR]

    relevance = max(scores) if scores else 0.0

    # Clearly nothing worth answering from — refuse without spending an LLM call.
    if relevance < CONFIDENT_REFUSAL_CEILING:
        return {
            "context_quality_score": round(relevance, 3),
            "quality_breakdown": {
                "relevance": round(relevance, 3),
                "completeness": 0.0,
                "precision": 0.0,
            },
            "missing_aspects": ["No retrieved chunk is relevant to the query"],
            "quality_method": "heuristic",
            "force_refusal": True,
        }

    expected = EXPECTED_EVIDENCE_COUNT.get(state.get("query_type", ""), 2)
    precision = float(np.mean(usable)) if usable else 0.0
    completeness = min(len(usable) / expected, 1.0)
    overall = relevance * 0.4 + completeness * 0.3 + precision * 0.3

    # Confident enough to skip the LLM assessor entirely.
    if relevance >= CONFIDENT_PASS_FLOOR and usable:
        missing = []
        if completeness < 1.0:
            missing.append(
                f"Only {len(usable)} usable passage(s); "
                f"{expected} expected for a {state.get('query_type', 'general')} query"
            )
        return {
            "context_quality_score": round(overall, 3),
            "quality_breakdown": {
                "relevance": round(relevance, 3),
                "completeness": round(completeness, 3),
                "precision": round(precision, 3),
            },
            "missing_aspects": missing,
            "quality_method": "heuristic",
            "force_refusal": False,
        }

    # Genuinely ambiguous band (best chunk scores 0.05–0.30) — ask the LLM.
    return None


async def quality_assessor_node(state: AgentState) -> dict:
    """
    LangGraph node — assesses retrieved context quality.
    Populates: context_quality_score, quality_breakdown, quality_method,
    missing_aspects, force_refusal, agent_trace.

    Args:
        state: Current AgentState with reranked_results.

    Returns:
        Partial state dict with quality assessment.
    """
    logger.info("Agent 5: Quality Assessor starting")

    # Try heuristics first (~60% of queries)
    heuristic_result = _heuristic_assessment(state)
    if heuristic_result is not None:
        logger.info(
            "Agent 5: Heuristic assessment sufficient",
            extra={
                "score": heuristic_result["context_quality_score"],
                "method": "heuristic",
            },
        )
        return {
            **heuristic_result,
            "agent_trace": [
                {
                    "agent": "quality_assessor",
                    "method": "heuristic",
                    "score": heuristic_result["context_quality_score"],
                }
            ],
        }

    # Fallback to LLM for ambiguous cases
    chunks = state.get("reranked_results", [])
    scores = [c.get("reranker_score", 0.0) for c in chunks[:5]]

    tracker = await BudgetTracker.get_instance()
    choice = await tracker.get_model_for_agent()
    model = choice.model

    chunks_summary = "\n".join(
        f"Chunk {i+1} (score={c.get('reranker_score', 0):.2f}, "
        f"source={c.get('source_file', 'unknown')}, "
        f"category={c.get('document_category', 'unknown')}): "
        f"{c.get('text', '')[:1200]}..."
        for i, c in enumerate(chunks[:10])
    )

    user_prompt = QUALITY_ASSESSOR_USER_TEMPLATE.format(
        query=state["current_query"],
        query_type=state["query_type"],
        num_chunks=len(chunks),
        chunks_summary=chunks_summary,
        min_score=f"{min(scores):.3f}" if scores else "N/A",
        max_score=f"{max(scores):.3f}" if scores else "N/A",
        mean_score=f"{float(np.mean(scores)):.3f}" if scores else "N/A",
    )

    result = await call_structured_agent(
        system_prompt=QUALITY_ASSESSOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
        max_tokens=600,
        api_key=choice.api_key,
    )

    logger.info(
        "Agent 5: LLM assessment complete",
        extra={
            "score": result.get("context_quality_score", 0),
            "method": "llm",
            "model": model,
        },
    )

    return {
        "context_quality_score": result.get("context_quality_score", 0.0),
        "quality_breakdown": result.get("quality_breakdown", {}),
        "quality_method": "llm",
        "missing_aspects": result.get("missing_aspects", []),
        "force_refusal": result.get("force_refusal", False),
        "agent_trace": [
            {
                "agent": "quality_assessor",
                "method": "llm",
                "model": model,
                "score": result.get("context_quality_score", 0),
            }
        ],
    }
