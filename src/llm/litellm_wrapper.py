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
import os

import litellm

from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Local fallback for the verification agents. The cloud model is no longer named
# here — it comes from BudgetTracker's agent ladder, so verification routes and
# rotates across keys like every other cloud call instead of pinning one model.
LOCAL_VERIFICATION_MODEL = LOCAL_MODEL

# Retry policy for prose calls. Matches the 3-attempt budget the structured
# agent already used, with linear backoff so a transient upstream 503 gets a
# meaningful gap before the next attempt.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# Ladder rungs to try before giving up. Bounded so a provider-wide outage
# cannot walk the whole ladder on every request.
MAX_LADDER_FALLBACKS = 6

# Rate limits are enforced over a rolling minute, so a 429 needs a longer pause
# than a generic transport error before the next attempt is worth making.
RATE_LIMIT_BACKOFF_SECONDS = 20.0

# Per-request ceilings. LiteLLM defaults to 600s, which is not a timeout so much
# as an absence of one: when Gemini returned 503 "experiencing high demand", each
# synthesis call hung for ~125 seconds before the provider dropped it, and the
# ladder could not descend until it did. Three of those in a row is a query that
# looks hung to the user while a perfectly good fallback model sits unused.
#
# The values are generous against observed behaviour rather than tight: healthy
# structured calls finish in 1-3s and healthy synthesis in 10-30s, so these only
# fire when something is actually wrong. Exceeding them raises, which the retry
# loop already treats as a transport error — so the effect is to reach the next
# rung sooner, never to lose a call that would have succeeded.
STRUCTURED_TIMEOUT_SECONDS = 60.0
PROSE_TIMEOUT_SECONDS = 120.0


def is_quota_error(exc: Exception) -> bool:
    """
    True when the provider refused the call for rate or quota reasons.

    Distinguishing this from a generic failure is what lets a caller move down
    the model ladder instead of retrying a model the provider will keep refusing
    all day. A per-minute limit and a per-day limit both surface as 429; the
    caller treats them the same because in either case another rung is a better
    bet than waiting.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "ratelimit" in name
        or "429" in text
        or "resource_exhausted" in text
        or "quota" in text
    )


def is_service_unavailable(exc: Exception) -> bool:
    """
    True when the provider says the model itself is temporarily unavailable.

    Worth separating from both quota and transport errors because the right
    response is different. A 429 means *this key* is spent, so another key on
    the same model is the best next move. A 503 — "This model is currently
    experiencing high demand" — means the model is down provider-wide, so every
    key will fail the same way and only another *model* helps.

    Observed live: gemini-3.6-flash returned 503 on roughly a third of synthesis
    calls while gemini-3.5-flash answered every request in ~1.2s. Treating the
    503 as a generic transport error meant three retries with backoff against a
    model that was not going to answer, before anything else was tried.

    Deliberately NOT treated as quota exhaustion: marking the slot spent would
    retire that model for the rest of the day over a blip that typically clears
    in minutes.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "serviceunavailable" in name
        or "overloaded" in name
        or "503" in text
        or "unavailable" in text
        or "experiencing high demand" in text
    )


def is_auth_error(exc: Exception) -> bool:
    """
    True when the provider rejected the credential itself.

    Distinct from a quota refusal in an important way: an exhausted key recovers
    at the next daily reset, whereas a rejected credential is unusable for every
    model until someone fixes the configuration. The caller therefore retires the
    whole key rather than one (key, model) slot.

    Worth handling rather than letting it fail the request: a run with four
    configured keys lost 16 of 35 questions because one key was invalid — one
    entry in GEMINI_API_KEYS had been split by a stray comma into two fragments —
    and every request routed to it died instead of moving to a working key.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "authentication" in name
        or "api key not valid" in text
        or "api_key_invalid" in text
        or "access_token_type_unsupported" in text
        or "unauthenticated" in text
        or "permission_denied" in text
    )


async def call_structured_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    api_key: str | None = None,
) -> dict:
    """
    Wrapper for all agents that return JSON.
    Enforces JSON mode via response_format parameter.
    Includes retry on JSON parse failure (max 3 attempts).

    Args:
        system_prompt: System-level instructions for the agent.
        user_prompt: User query / context for the agent.
        model: LiteLLM model string (e.g., "gemini/gemini-3.5-flash-lite").
        temperature: Sampling temperature (default 0.0 for determinism).
        max_tokens: Maximum output tokens.
        api_key: Credential for this specific call. Required when multiple keys
            are configured — see the note at the kwargs assembly below.

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
                "timeout": STRUCTURED_TIMEOUT_SECONDS,
            }
            if model.startswith("ollama/"):
                kwargs["num_ctx"] = 8192  # Expand context window for local Ollama to prevent truncation
            if api_key:
                # Explicit per-call credential. With key rotation the key is
                # chosen per request, so relying on the ambient GEMINI_API_KEY
                # env var would send every call to key 1 while the tracker
                # debited whichever key it thought it had picked.
                kwargs["api_key"] = api_key

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
            # Back off before retrying. Retrying a 429 or a connection drop with
            # no delay just reproduces the same failure — the original loop did
            # exactly that, so a rate-limited call burned all three attempts
            # inside a few milliseconds and still failed. Rate limits get a
            # longer wait than other transport errors because the window they
            # are enforcing is measured in seconds.
            is_rate_limit = "ratelimit" in type(e).__name__.lower() or "429" in str(e)
            delay = RATE_LIMIT_BACKOFF_SECONDS if is_rate_limit else RETRY_BACKOFF_SECONDS
            await asyncio.sleep(delay * (attempt + 1))
            continue

    # Should never reach here due to raises above, but satisfy type checker
    raise ValueError("Unexpected: exhausted all retry attempts without raising")


async def call_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    api_key: str | None = None,
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
            prose_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": PROSE_TIMEOUT_SECONDS,
            }
            if api_key:
                prose_kwargs["api_key"] = api_key
            response = await litellm.acompletion(**prose_kwargs)
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


async def call_verification_agent(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> dict:
    """
    Wrapper for the verification agents (Agent 4 Financial Verifier,
    Agent 8 Hallucination Validator). Enforces JSON mode.

    Model routing is a deployment choice, not an architectural one, so it is
    controlled by VERIFICATION_BACKEND:

      "cloud" (default) — Gemini. Higher-quality structured reasoning and no
          local GPU requirement, which also means the project runs from a clone
          plus an API key: no 9GB model pull, no CUDA.
      "local"           — Ollama / Qwen2.5:14b. Keeps verification off the
          metered API entirely and out of the daily quota, at the cost of a
          12GB-VRAM machine. This was the original design: verification is the
          highest-volume agent traffic, so pushing it to a local model was how
          the pipeline stayed inside the free Gemini tier.

    Both paths are live; the constant only decides the default. If the cloud
    call fails for any reason, the local model is tried before giving up, so a
    quota exhaustion degrades rather than fails.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User query / context.
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum output tokens (default 1500).

    Returns:
        Parsed JSON dict from the verification model.

    Raises:
        ValueError: If every configured model returns invalid JSON.
    """
    backend = os.getenv("VERIFICATION_BACKEND", "cloud").strip().lower()

    if backend == "local":
        return await call_structured_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=LOCAL_VERIFICATION_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Route through the budget tracker rather than calling Gemini directly.
    # Verification is now the highest-volume cloud traffic in the pipeline (two
    # calls per query), so it has to take a rate-limiter slot and debit the daily
    # quota like every other cloud call — otherwise it would silently blow
    # through the 15 RPM free-tier limit. get_model_for_agent() already returns
    # the local model when the quota is spent, which gives cloud-first routing
    # with automatic local fallback for free.
    from src.llm.budget_tracker import BudgetTracker  # deferred: avoids import cycle

    tracker = await BudgetTracker.get_instance()

    # Descend the ladder on quota refusals rather than failing. The tracker's
    # counters can lag the provider's (restarts reset the in-memory fallback,
    # other processes share the key), so a rung it offers may already be closed.
    last_error: Exception | None = None
    choice = None

    for _ in range(MAX_LADDER_FALLBACKS):
        choice = await tracker.get_model_for_agent()
        try:
            return await call_structured_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=choice.model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=choice.api_key,
            )
        except Exception as e:
            last_error = e
            if choice.key_index >= 0 and is_auth_error(e):
                tracker.mark_key_unusable(choice.key_index)
                continue
            if is_quota_error(e) and choice.key_index >= 0:
                await tracker.mark_slot_exhausted(choice.key_index, choice.model)
                logger.warning(
                    "Verification rung refused by provider, descending the ladder",
                    extra={"model": choice.model, "key_index": choice.key_index},
                )
                continue
            if is_service_unavailable(e) and choice.key_index >= 0:
                # The model is down for everyone, so rotating keys is pointless.
                # Skip the whole model for this request without debiting quota —
                # this clears in minutes and should not cost the day's capacity.
                tracker.skip_model_for_request(choice.model)
                logger.warning(
                    "Verification rung unavailable provider-side, trying another model",
                    extra={"model": choice.model},
                )
                continue
            break

    if choice is not None and choice.model == LOCAL_VERIFICATION_MODEL:
        raise last_error if last_error else RuntimeError("verification failed")

    if True:
        e = last_error
        logger.warning(
            "Cloud verification failed, falling back to local model",
            extra={"error": str(e), "fallback": LOCAL_VERIFICATION_MODEL},
        )
        return await call_structured_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=LOCAL_VERIFICATION_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Backwards-compatible alias — the verification agents were originally local-only.
call_local_agent = call_verification_agent


def active_verification_model() -> str:
    """
    Returns the model string the verification agents will try first.

    Used for the agent trace so the UI reports the model actually in use rather
    than a hardcoded name that silently goes stale when the backend changes.
    """
    backend = os.getenv("VERIFICATION_BACKEND", "cloud").strip().lower()
    if backend == "local":
        return LOCAL_VERIFICATION_MODEL
    # First rung of the agent ladder — what the selector will try first.
    return AGENT_LADDER[0]
