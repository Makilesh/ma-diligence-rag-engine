"""
Agent 4 — Financial Verification Agent.

Deterministic: no LLM call. Only triggered when query_type == "financial" OR
requires_numerical_precision == True.

This agent used to hand up to ten table chunks to the verification LLM and
return whatever "numerical_registry" and "inconsistencies" it produced. Three
things were wrong with that:

- The chunk filter `{k: v … if k != "text" or len(v) < 500}` dropped the text
  of every table chunk of 500+ characters — which is most tables — so the model
  was usually asked to cross-check figures it was never shown.
- The `numerical_values` list it was given was always empty (retrieved chunks
  carry no top-level `normalized_value`), and the NumericalRegistry built for
  exactly this job was imported nowhere.
- Whatever the model returned was passed to the synthesizer as "Inconsistencies
  Found", i.e. presented as verified fact. A model inventing a discrepancy is
  worse than one missing it: the answer then asserts it with a citation.

Now the registry is built from the chunks themselves (structured table rows
where ingestion wrote them, fixed-width tables in text otherwise) and a
disagreement is reported only when two sources state the same labelled metric
for the same period with values that differ beyond rounding. That is a
measurement, so it can be shown as one — and it costs no quota.
"""

import time

from src.utils.numerical_registry import NumericalRegistry
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def financial_verifier_node(state: AgentState) -> dict:
    """
    LangGraph node — cross-checks financial figures across retrieved documents.
    Populates: numerical_registry, inconsistencies, agent_trace.

    Args:
        state: Current AgentState with expanded_context / reranked_results.

    Returns:
        Partial state dict with financial verification results.
    """
    start = time.monotonic()
    chunks = state.get("expanded_context") or state.get("reranked_results") or []

    logger.info(
        "Agent 4: Financial Verifier starting",
        extra={"num_chunks": len(chunks)},
    )

    registry = NumericalRegistry.from_chunks(chunks)

    if len(registry) == 0:
        logger.info("Agent 4: No labelled figures found in context, skipping")
        return {
            "numerical_registry": {},
            "inconsistencies": [],
            "agent_trace": [
                {
                    "agent": "financial_verifier",
                    "skipped": True,
                    "reason": "no_labelled_figures",
                    "method": "deterministic",
                }
            ],
        }

    inconsistencies = registry.find_inconsistencies()
    cross_checked = registry.cross_checked_count()
    sources = sorted({c.get("source_file", "") for c in chunks if c.get("source_file")})
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)

    logger.info(
        "Agent 4: Financial Verifier complete",
        extra={
            "figures": len(registry),
            "cross_checked": cross_checked,
            "inconsistencies_found": len(inconsistencies),
            "elapsed_ms": elapsed_ms,
        },
    )

    return {
        "numerical_registry": registry.to_dict(),
        "inconsistencies": inconsistencies,
        "agent_trace": [
            {
                "agent": "financial_verifier",
                "method": "deterministic",
                "model": "none (deterministic registry)",
                "figures": len(registry),
                "cross_checked": cross_checked,
                "inconsistencies": len(inconsistencies),
                "sources": sources,
                "elapsed_ms": elapsed_ms,
            }
        ],
    }
