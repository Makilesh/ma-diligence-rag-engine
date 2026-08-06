"""
Agent 8 — Hallucination Validation Agent.

Model: Qwen2.5:14b local via Ollama | Temp: 0.0 | Tokens: 1000
JSON mode: response_format={"type": "json_object"}
"""

import json

from src.llm.litellm_wrapper import call_verification_agent, active_verification_model
from src.llm.prompt_templates.hallucination_validator import (
    HALLUCINATION_VALIDATOR_SYSTEM_PROMPT,
    HALLUCINATION_VALIDATOR_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def hallucination_validator_node(state: AgentState) -> dict:
    """
    LangGraph node — validates answer for hallucinations.
    Populates: confidence_score, hallucination_flags, validation_status,
    validation_attempt, agent_trace.

    Uses local Ollama model to avoid consuming API budget on validation.

    Args:
        state: Current AgentState with generated_answer and context.

    Returns:
        Partial state dict with validation results.
    """
    logger.info("Agent 8: Hallucination Validator starting")

    answer = state.get("generated_answer", "")
    chunks = state.get("expanded_context", state.get("reranked_results", []))

    # Skip validation for forced refusals
    if state.get("force_refusal") or not answer:
        logger.info("Agent 8: Skipping validation — refusal or empty answer")
        return {
            "confidence_score": 0.0,
            "hallucination_flags": [],
            "validation_status": "passed",
            "validation_attempt": state.get("validation_attempt", 0) + 1,
            "agent_trace": [
                {"agent": "hallucination_validator", "skipped": True}
            ],
        }

    # Format context for validation.
    #
    # The validator MUST see the same evidence the synthesizer wrote from.
    # This previously truncated each chunk to 500 characters while the
    # synthesizer received the full chunk plus its parent context — so the
    # validator was asked "is this claim supported?" while being shown roughly a
    # sixth of the supporting text. It duly reported unsupported claims that were
    # perfectly grounded: on the golden set it flagged "revenue was $452.8M for
    # FY2023 and $387.1M for FY2022" as a hallucination on an answer that scored
    # 100% fact recall, because the figures sat past the cutoff. Ten of nineteen
    # answers were marked failed this way, which trains a reviewer to ignore the
    # validator — the worst possible outcome for a safety control.
    parts = []
    for i, c in enumerate(chunks):
        body = c.get("text", "")
        parent = c.get("parent_text", "")
        entry = f"[Chunk {i + 1} | {c.get('source_file', 'unknown')}]: {body}"
        if parent:
            entry += f"\n[Parent context]: {parent}"
        parts.append(entry)
    context_text = "\n\n".join(parts)

    user_prompt = HALLUCINATION_VALIDATOR_USER_TEMPLATE.format(
        query=state["current_query"],
        answer=answer,
        context=context_text,
        citations=json.dumps(state.get("citations", []), indent=2, default=str),
    )

    result = await call_verification_agent(
        system_prompt=HALLUCINATION_VALIDATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=2000,
    )

    validation_status = result.get("validation_status", "warning")
    confidence = result.get("confidence_score", 0.5)
    flags = result.get("hallucination_flags", [])

    logger.info(
        "Agent 8: Hallucination Validator complete",
        extra={
            "validation_status": validation_status,
            "confidence_score": confidence,
            "hallucination_flags": len(flags),
        },
    )

    return {
        "confidence_score": confidence,
        "hallucination_flags": flags,
        "validation_status": validation_status,
        "validation_attempt": state.get("validation_attempt", 0) + 1,
        "agent_trace": [
            {
                "agent": "hallucination_validator",
                "model": active_verification_model(),
                "validation_status": validation_status,
                "confidence_score": confidence,
                "flags_count": len(flags),
            }
        ],
    }
