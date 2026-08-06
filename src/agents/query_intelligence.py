"""
Agent 1 — Query Intelligence Agent.

Parses natural language M&A queries into structured intent signals.
These signals drive the entire downstream pipeline: retrieval config
selection, filter construction, and answer formatting.

Model: gemini-3.1-flash-lite | Temp: 0.0 | Tokens: 800
JSON mode: response_format={"type": "json_object"}
"""

from src.llm.litellm_wrapper import call_structured_agent
from src.llm.budget_tracker import BudgetTracker
from src.llm.prompt_templates.query_intelligence import (
    QUERY_INTELLIGENCE_SYSTEM_PROMPT,
    QUERY_INTELLIGENCE_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def query_intelligence_node(state: AgentState) -> dict:
    """
    LangGraph node — parses query into structured intent signals.
    Populates: query_type, parsed_intent, extracted_filters,
    current_query, rewrite_iteration, agent_trace.

    Args:
        state: Current AgentState with original_query and deal_id.

    Returns:
        Partial state dict to merge — includes query_type, parsed_intent,
        extracted_filters, current_query.
    """
    query = state["original_query"]
    deal_id = state["deal_id"]

    logger.info(
        "Agent 1: Query Intelligence starting",
        extra={"query": query, "deal_id": deal_id},
    )

    tracker = await BudgetTracker.get_instance()
    choice = await tracker.get_model_for_agent()
    model = choice.model

    user_prompt = QUERY_INTELLIGENCE_USER_TEMPLATE.format(
        query=query,
        deal_id=deal_id,
    )

    result = await call_structured_agent(
        system_prompt=QUERY_INTELLIGENCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
        max_tokens=800,
        api_key=choice.api_key,
    )

    # Validate query_type
    valid_types = {"financial", "legal", "comparative", "summary", "multi_hop"}
    query_type = result.get("query_type", "summary")
    if query_type not in valid_types:
        logger.warning(
            f"Agent 1 returned invalid query_type '{query_type}', defaulting to 'summary'"
        )
        query_type = "summary"

    # Extract metadata filters — NEVER include include_pii (compliance constraint)
    extracted_filters = result.get("metadata_filters", {})
    extracted_filters.pop("include_pii", None)  # Compliance enforcement

    logger.info(
        "Agent 1: Query Intelligence complete",
        extra={
            "query_type": query_type,
            "num_expansions": len(result.get("query_expansions", [])),
        },
    )

    return {
        "query_type": query_type,
        "parsed_intent": result,
        "extracted_filters": extracted_filters,
        "current_query": result.get("reformulated_query", query),
        "rewrite_iteration": 0,
        "agent_trace": [
            {
                "agent": "query_intelligence",
                "model": model,
                "input_query": query,
                "output": result,
            }
        ],
    }
