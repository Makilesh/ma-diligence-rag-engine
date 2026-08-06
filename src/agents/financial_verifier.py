"""
Agent 4 — Financial Verification Agent.

Model: Qwen2.5:14b local via Ollama | Temp: 0.0 | Tokens: 1500
JSON mode: response_format={"type": "json_object"}
Only triggered when: query_type == "financial" OR requires_numerical_precision == True
"""

import json

from src.llm.litellm_wrapper import call_verification_agent, active_verification_model
from src.llm.prompt_templates.financial_verifier import (
    FINANCIAL_VERIFIER_SYSTEM_PROMPT,
    FINANCIAL_VERIFIER_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def financial_verifier_node(state: AgentState) -> dict:
    """
    LangGraph node — cross-checks financial data across chunks.
    Populates: numerical_registry, inconsistencies, agent_trace.

    Only runs for financial queries or when numerical precision is required.
    Uses local Ollama model to avoid consuming API budget on verification.

    Args:
        state: Current AgentState with reranked_results and current_query.

    Returns:
        Partial state dict with financial verification results.
    """
    query = state["current_query"]
    chunks = state.get("expanded_context", state.get("reranked_results", []))

    logger.info(
        "Agent 4: Financial Verifier starting",
        extra={"num_chunks": len(chunks)},
    )

    # Extract financial chunks and numerical values
    financial_chunks = []
    numerical_values = []

    for chunk in chunks:
        if chunk.get("is_table") or chunk.get("content_type") in (
            "table_narrative",
            "table_row_by_row",
            "table_metrics_summary",
            "table_markdown",
            "computed_metric",
        ):
            financial_chunks.append(chunk)
            # Extract any embedded numerical values
            if "normalized_value" in chunk:
                numerical_values.append({
                    "metric": chunk.get("metric_name", "unknown"),
                    "raw_value": chunk.get("raw_value"),
                    "normalized_value": chunk.get("normalized_value"),
                    "currency": chunk.get("currency", "USD"),
                    "scale_factor": chunk.get("scale_factor", 1),
                    "source": chunk.get("source_file", "unknown"),
                    "fiscal_year": chunk.get("fiscal_year"),
                })

    if not financial_chunks:
        logger.info("Agent 4: No financial chunks to verify, skipping")
        return {
            "numerical_registry": {},
            "inconsistencies": [],
            "agent_trace": [
                {"agent": "financial_verifier", "skipped": True, "reason": "no_financial_chunks"}
            ],
        }

    user_prompt = FINANCIAL_VERIFIER_USER_TEMPLATE.format(
        query=query,
        financial_chunks=json.dumps(
            [
                {k: v for k, v in c.items() if k != "text" or len(str(v)) < 500}
                for c in financial_chunks[:10]  # Limit context size
            ],
            indent=2,
            default=str,
        ),
        numerical_values=json.dumps(numerical_values, indent=2, default=str),
    )

    result = await call_verification_agent(
        system_prompt=FINANCIAL_VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=2000,
    )

    logger.info(
        "Agent 4: Financial Verifier complete",
        extra={
            "inconsistencies_found": len(result.get("inconsistencies", [])),
            "metrics_verified": len(result.get("computed_metrics_verified", [])),
        },
    )

    return {
        "numerical_registry": result.get("numerical_registry", {}),
        "inconsistencies": result.get("inconsistencies", []),
        "agent_trace": [
            {
                "agent": "financial_verifier",
                "model": active_verification_model(),
                "inconsistencies": len(result.get("inconsistencies", [])),
            }
        ],
    }
