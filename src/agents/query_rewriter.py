"""
Agent 6 — Query Rewriter Agent.

Model: gemini-3.1-flash-lite | Temp: 0.2 | Tokens: 600
JSON mode: response_format={"type": "json_object"}
Max iterations: 2

CRITICAL: JSON output key → state key mapping.
The rewriter's JSON output uses updated_retrieval_config and updated_metadata_filters,
but AgentState fields are retrieval_config and extracted_filters.
This node MUST map these correctly when returning state.
"""

import json

from src.llm.litellm_wrapper import call_structured_agent
from src.llm.budget_tracker import BudgetTracker
from src.llm.prompt_templates.query_rewriter import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    QUERY_REWRITER_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def query_rewriter_node(state: AgentState) -> dict:
    """
    LangGraph node — rewrites query to improve retrieval quality.

    CRITICAL: The rewriter feeds back into the retrieval executor.
    Edge is: rewriter → executor (explicit, not conditional).
    This creates the self-correction loop.

    Args:
        state: Current AgentState with quality assessment results.

    Returns:
        Partial state dict with rewritten query and updated config.
    """
    iteration = state.get("rewrite_iteration", 0) + 1

    logger.info(
        "Agent 6: Query Rewriter starting",
        extra={"iteration": iteration, "max_iterations": 2},
    )

    tracker = await BudgetTracker.get_instance()
    choice = await tracker.get_model_for_agent()
    model = choice.model

    user_prompt = QUERY_REWRITER_USER_TEMPLATE.format(
        original_query=state["original_query"],
        current_query=state["current_query"],
        missing_aspects=json.dumps(state.get("missing_aspects", []), indent=2),
        quality_score=state.get("context_quality_score", 0.0),
        quality_breakdown=json.dumps(state.get("quality_breakdown", {}), indent=2),
        retrieval_config=json.dumps(state.get("retrieval_config", {}), indent=2),
        iteration=iteration,
    )

    result = await call_structured_agent(
        system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=0.2,  # Slightly higher temp for creative rewrites
        max_tokens=600,
        api_key=choice.api_key,
    )

    rewritten_query = result.get("rewritten_query", state["current_query"])

    # Map JSON output keys → AgentState keys
    updated_config = state.get("retrieval_config", {}).copy()
    if result.get("updated_retrieval_config"):
        updated_config.update(result["updated_retrieval_config"])

    updated_filters = state.get("extracted_filters", {}).copy()
    if result.get("updated_metadata_filters"):
        updated_filters.update(result["updated_metadata_filters"])
        # Compliance: never allow rewriter to set include_pii
        updated_filters.pop("include_pii", None)

    logger.info(
        "Agent 6: Query Rewriter complete",
        extra={
            "iteration": iteration,
            "original_query": state["original_query"],
            "rewritten_query": rewritten_query,
        },
    )

    return {
        "current_query": rewritten_query,
        "rewrite_iteration": iteration,
        "retrieval_config": updated_config,
        "extracted_filters": updated_filters,
        "rewrite_history": [
            {
                "iteration": iteration,
                "rewritten_query": rewritten_query,
                "reasoning": result.get("rewrite_reasoning", ""),
                "alternatives": result.get("alternative_formulations", []),
            }
        ],
        "agent_trace": [
            {
                "agent": "query_rewriter",
                "model": model,
                "iteration": iteration,
                "rewritten_query": rewritten_query,
            }
        ],
    }
