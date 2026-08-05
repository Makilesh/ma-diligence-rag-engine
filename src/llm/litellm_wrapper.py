"""
LiteLLM wrapper for structured agent calls.

Enforces JSON mode via response_format parameter for agents that expect
structured JSON output (Agents 1, 5, 6, 7, 8).
Includes retry on JSON parse failure (max 3 attempts).

All agents that return JSON MUST use call_structured_agent().
Agent 7 (Answer Synthesizer) returns prose and uses call_prose_agent().
"""

import asyncio
import json

import litellm

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Retry policy for prose calls. Matches the 3-attempt budget the structured
# agent already used, with linear backoff so a transient upstream 503 gets a
# meaningful gap before the next attempt.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


async def call_structured_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> dict:
    """
    Wrapper for all agents that return JSON.
    Enforces JSON mode via response_format parameter.
    Includes retry on JSON parse failure (max 3 attempts).

    Args:
        system_prompt: System-level instructions for the agent.
        user_prompt: User query / context for the agent.
        model: LiteLLM model string (e.g., "gemini/gemini-3.1-flash-lite").
        temperature: Sampling temperature (default 0.0 for determinism).
        max_tokens: Maximum output tokens.

    Returns:
        Parsed JSON dict from the agent response.

    Raises:
        ValueError: If agent returns invalid JSON after 3 attempts.
        litellm.exceptions.APIError: On API communication failure.
    """
    for attempt in range(3):
        try:
            logger.info(
                "Calling structured agent",
                extra={
                    "model": model,
                    "attempt": attempt + 1,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},  # Enforces JSON mode
            }
            if model.startswith("ollama/"):
                kwargs["num_ctx"] = 8192  # Expand context window for local Ollama to prevent truncation

            response = await litellm.acompletion(**kwargs)

            raw = response.choices[0].message.content
            # Strip accidental markdown fences before parsing
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(clean)

            logger.info(
                "Structured agent call successful",
                extra={
                    "model": model,
                    "attempt": attempt + 1,
                    "response_keys": list(parsed.keys()) if isinstance(parsed, dict) else "non-dict",
                },
            )
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(
                "Agent returned invalid JSON",
                extra={
                    "model": model,
                    "attempt": attempt + 1,
                    "error": str(e),
                    "raw_preview": raw[:200] if raw else "empty",
                },
            )
            if attempt == 2:
                raise ValueError(
                    f"Agent returned invalid JSON after 3 attempts: {e}\nRaw: {raw}"
                )
            continue

        except Exception as e:
            logger.error(
                "Structured agent call failed",
                extra={
                    "model": model,
                    "attempt": attempt + 1,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            if attempt == 2:
                raise
            continue

    # Should never reach here due to raises above, but satisfy type checker
    raise ValueError("Unexpected: exhausted all retry attempts without raising")


async def call_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> str:
    """
    Wrapper for agents that return prose (not JSON).
    Used by Agent 7 (Answer Synthesizer) which returns natural language answers.
    Does NOT enforce JSON mode.

    Args:
        system_prompt: System-level instructions for the agent.
        user_prompt: User query / context for the agent.
        model: LiteLLM model string.
        temperature: Sampling temperature (default 0.1 for slight variety).
        max_tokens: Maximum output tokens (default 3000 for long answers).

    Returns:
        Raw string response from the agent. Never None.

    Raises:
        RuntimeError: If every attempt fails or returns an empty completion.

    Retries exist because both failure modes were observed on the golden-set run:
    a transient upstream 503 ("model is currently experiencing high demand") and,
    more insidiously, a 200 response whose content was empty. The empty case used
    to be returned as None and propagate until something did `x in None`, failing
    the whole request with an opaque TypeError. Both are now retried, and an
    exhausted retry raises a typed error the synthesizer converts into a refusal —
    a refusal is a far better outcome for the caller than a 500.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            "Calling prose agent",
            extra={
                "model": model,
                "attempt": attempt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content

            if content and content.strip():
                logger.info(
                    "Prose agent call successful",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "response_length": len(content),
                    },
                )
                return content

            last_error = RuntimeError("empty completion")
            logger.warning(
                "Prose agent returned an empty completion",
                extra={"model": model, "attempt": attempt},
            )

        except Exception as e:
            last_error = e
            logger.warning(
                "Prose agent call failed",
                extra={"model": model, "attempt": attempt, "error": str(e)},
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Prose agent failed after {MAX_RETRIES} attempts "
        f"(model={model}): {last_error}"
    )


async def call_local_agent(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> dict:
    """
    Wrapper for local Ollama agents (Agents 4 and 8).
    Always uses the local Qwen2.5:14b model, no budget consumption.
    Enforces JSON mode.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User query / context.
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum output tokens (default 1500).

    Returns:
        Parsed JSON dict from the local model.

    Raises:
        ValueError: If model returns invalid JSON after 3 attempts.
    """
    return await call_structured_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="ollama/qwen2.5:14b",
        temperature=temperature,
        max_tokens=max_tokens,
    )
