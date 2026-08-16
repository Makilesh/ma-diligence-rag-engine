"""
Agent 1 — Query Intelligence Agent.

Parses natural language M&A queries into structured intent signals.
These signals drive the entire downstream pipeline: retrieval config
selection, filter construction, and answer formatting.

Model: selected per call from the agent ladder (src/llm/model_registry.py) | Temp: 0.0 | Tokens: 800
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

# Query types the prompt is told to decompose. Retained for the prompt contract
# and for tests, but deliberately NOT used to veto sub-questions the model
# returns under another classification — see the comment at the validation site.
DECOMPOSABLE_QUERY_TYPES = {"multi_hop", "comparative"}

# Ceiling on sub-questions. Each one costs a full retrieval pass (embed, hybrid
# search, rerank) and takes a share of the final context budget, so an
# over-eager decomposition would both slow the query and starve each facet.
MAX_SUB_QUESTIONS = 4


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

    # Decomposition is honoured whenever the model returns sub-questions, and is
    # NOT additionally gated on query_type.
    #
    # It was, and that silently discarded the decomposition on the very query
    # decomposition was built for. "What is the implied EV/EBITDA multiple on
    # adjusted rather than reported EBITDA?" needs the share price, the share
    # count and the EBITDA figure — three facts in three documents. Agent 1
    # decomposed it correctly and then classified it `financial` rather than
    # `multi_hop`, so the veto threw the sub-questions away and the engine
    # answered about the offer price instead.
    #
    # Emitting sub-questions IS the model's judgment that the answer spans
    # several facts. Classification is a separate judgment, made for routing.
    # Letting one veto the other means the feature works only when two
    # independent guesses agree, and fails invisibly when they do not. The
    # prompt decides when to decompose; MAX_SUB_QUESTIONS bounds the cost.
    sub_questions = result.get("sub_questions") or []
    if not isinstance(sub_questions, list):
        sub_questions = []
    sub_questions = [
        s.strip() for s in sub_questions
        if isinstance(s, str) and s.strip()
    ][:MAX_SUB_QUESTIONS]

    if sub_questions:
        logger.info(
            "Agent 1: query decomposed into sub-questions",
            extra={"query_type": query_type, "count": len(sub_questions)},
        )

    return {
        "query_type": query_type,
        "parsed_intent": result,
        "extracted_filters": extracted_filters,
        "sub_questions": sub_questions,
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
