"""
LiteLLM wrapper for structured agent calls.

Enforces JSON mode via response_format parameter for agents that expect
structured JSON output (Agents 1, 5, 6, 7, 8).
Includes retry on JSON parse failure (max 3 attempts).

All agents that return JSON MUST use call_structured_agent().
Agent 7 (Answer Synthesizer) returns prose and uses call_prose_agent().
"""

import asyncio
import contextvars
import json
import os
import time

import litellm

from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Observability (optional, free) ────────────────────────────────────────────
# Per-request context attached to every LLM call's metadata: which deal/session
# the call served. Set once per query by the orchestrator; contextvars propagate
# into LangGraph's node tasks, so no agent has to thread it through.
_trace_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_trace_context", default=None
)

# The model that actually answered the most recent verification call in this
# context. See active_verification_model().
_last_verification_model: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "last_verification_model", default=None
)


def set_trace_context(**fields) -> None:
    """
    Records request-scoped metadata (deal_id, session_id) for LLM call tracing.

    Args:
        **fields: Values to attach to every subsequent call in this context.
    """
    _trace_context.set({k: v for k, v in fields.items() if v is not None})


def _langfuse_enabled() -> bool:
    """True when both Langfuse keys are configured."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _configure_tracing() -> bool:
    """
    Enables LiteLLM's Langfuse callbacks when LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are set; a no-op otherwise.

    Langfuse is deliberately NOT a hard dependency (`pip install langfuse` to
    use it; LANGFUSE_HOST selects a self-hosted or regional instance). If the
    keys are set but the package is missing, tracing is skipped with a warning
    rather than failing every LLM call at the callback.

    Returns:
        True when tracing was enabled.
    """
    if not _langfuse_enabled():
        return False
    try:
        import langfuse  # noqa: F401  (presence check only)
    except ImportError:
        logger.warning(
            "LANGFUSE_* keys are set but the langfuse package is not installed; "
            "tracing disabled (pip install langfuse)"
        )
        return False
    for hook in (litellm.success_callback, litellm.failure_callback):
        if "langfuse" not in hook:
            hook.append("langfuse")
    logger.info("LLM tracing enabled (Langfuse via LiteLLM callbacks)")
    return True


_TRACING = _configure_tracing()


def _call_metadata(agent: str | None) -> dict | None:
    """
    LiteLLM `metadata` for a call — only when tracing is on, so untraced calls
    send exactly the kwargs they always did.

    Args:
        agent: Pipeline agent making the call.

    Returns:
        Metadata dict, or None.
    """
    if not _TRACING:
        return None
    ctx = _trace_context.get() or {}
    tags = [t for t in (agent, ctx.get("deal_id")) if t]
    return {
        "generation_name": agent or "llm_call",
        "trace_name": "manda-query",
        "session_id": ctx.get("session_id"),
        "tags": tags,
        "trace_metadata": {**ctx, "agent": agent},
    }


def _log_usage(response, model: str, agent: str | None, started: float) -> None:
    """
    Logs token usage and latency for one completed call.

    Free observability even without Langfuse: the structured log line is enough
    to see which agent spends the quota. Tolerates responses without `usage`
    (test doubles, some local backends).
    """
    usage = getattr(response, "usage", None)

    def _get(name: str):
        if usage is None:
            return None
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        return value

    logger.info(
        "LLM call usage",
        extra={
            "model": model,
            "agent": agent,
            "prompt_tokens": _get("prompt_tokens"),
            "completion_tokens": _get("completion_tokens"),
            "total_tokens": _get("total_tokens"),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            **{k: v for k, v in (_trace_context.get() or {}).items() if k == "deal_id"},
        },
    )

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


def is_model_unavailable_for_key(exc: Exception) -> bool:
    """
    True when this credential specifically cannot use this model.

    Distinct from every other failure mode here, and the distinction is not
    hypothetical. `gemini-2.5-flash` and `gemini-2.5-flash-lite` return
    404 "This model is no longer available to new users" on two of five
    configured keys and answer normally on the other two — Google grandfathers
    older keys when a model is closed to new sign-ups.

    So availability is a property of the (key, model) pair, not of the model.
    A 429 retires the pair for the day, a 503 retires the model for everyone
    briefly, an auth error retires the key entirely — none of those describe
    "this key may never use this model", which is permanent and affects one
    slot.
    """
    text = str(exc).lower()
    if "404" not in text and "notfound" not in type(exc).__name__.lower():
        return False
    return (
        "no longer available" in text
        or "not found for api version" in text
        or "is not supported for generatecontent" in text
    )


def is_timeout_error(exc: Exception) -> bool:
    """
    True when the request exceeded its own deadline.

    Treated exactly like a 503, and for the same reason: a model that just
    consumed the full timeout without answering is not more likely to answer on
    the next identical attempt, and every retry costs another full timeout.

    Measured: `gemini-3.7-flash` began timing out consistently, and because a
    timeout matched none of the other classifiers it fell through to the generic
    retry path — three attempts at 120 seconds each. The caller's 300-second
    budget expired before the ladder was ever allowed to try a healthy model, so
    the query returned nothing at all rather than a slightly worse answer.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "timeout passed" in text


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
    agent: str | None = None,
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
        agent: Calling agent's name, for usage logs and optional tracing.

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
            metadata = _call_metadata(agent)
            if metadata:
                kwargs["metadata"] = metadata

            started = time.monotonic()
            response = await litellm.acompletion(**kwargs)
            _log_usage(response, model, agent, started)

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
    agent: str | None = None,
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
        api_key: Credential for this specific call.
        agent: Calling agent's name, for usage logs and optional tracing.

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
            metadata = _call_metadata(agent)
            if metadata:
                prose_kwargs["metadata"] = metadata
            started = time.monotonic()
            response = await litellm.acompletion(**prose_kwargs)
            _log_usage(response, model, agent, started)
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

            # A 503 means the model is down for everyone, so retrying *this*
            # model is time spent waiting for the same answer. Surface it
            # immediately and let the caller's ladder pick a different model.
            #
            # Measured: gemini-3.7-flash returning 503 cost three attempts and
            # ~48 seconds of backoff before the ladder was allowed to move to
            # gemini-3.6-flash, which answered on the first try.
            if is_service_unavailable(e) or is_timeout_error(e):
                raise

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Prose agent failed after {MAX_RETRIES} attempts "
        f"(model={model}): {last_error}"
    )


async def call_verification_agent_with_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    agent: str | None = None,
) -> tuple[dict, str]:
    """
    Verification call that also reports which model actually answered.

    Model routing is a deployment choice, not an architectural one, so it is
    controlled by VERIFICATION_BACKEND:

      "cloud" (default) — Gemini via the budget tracker's agent ladder. Higher-
          quality structured reasoning and no local GPU requirement.
      "local"           — Ollama / Qwen2.5:14b. Keeps verification off the
          metered API entirely, at the cost of a 12GB-VRAM machine.

    If the cloud ladder fails for any reason, the local model is tried before
    giving up, so a quota exhaustion degrades rather than fails.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User query / context.
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum output tokens (default 1500).
        agent: Calling agent's name, for usage logs and optional tracing.

    Returns:
        (parsed JSON dict, model string that served the call).

    Raises:
        ValueError: If every configured model returns invalid JSON.
        Exception: The last transport error when even the local fallback fails.
    """
    backend = os.getenv("VERIFICATION_BACKEND", "cloud").strip().lower()

    if backend == "local":
        result = await call_structured_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=LOCAL_VERIFICATION_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            agent=agent,
        )
        _last_verification_model.set(LOCAL_VERIFICATION_MODEL)
        return result, LOCAL_VERIFICATION_MODEL

    # Route through the budget tracker rather than calling Gemini directly, so
    # verification takes a rate-limiter slot and debits the daily quota like
    # every other cloud call. get_model_for_agent() already returns the local
    # model when the quota is spent.
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
            result = await call_structured_agent(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=choice.model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=choice.api_key,
                agent=agent,
            )
            _last_verification_model.set(choice.model)
            return result, choice.model
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
            if is_model_unavailable_for_key(e) and choice.key_index >= 0:
                # This key may never use this model. Retire the pair, not the
                # key and not the model — both are still fine elsewhere.
                tracker.mark_slot_unavailable(choice.key_index, choice.model)
                logger.warning(
                    "Model not available for this credential; retiring the slot",
                    extra={"model": choice.model, "key_index": choice.key_index},
                )
                continue
            if (is_service_unavailable(e) or is_timeout_error(e)) and choice.key_index >= 0:
                # The model is down or unresponsive for everyone, so rotating
                # keys is pointless. Skip the whole model for this request
                # without debiting quota — this clears in minutes.
                tracker.skip_model_for_request(choice.model)
                logger.warning(
                    "Verification rung unavailable provider-side, trying another model",
                    extra={"model": choice.model},
                )
                continue
            break

    if choice is not None and choice.model == LOCAL_VERIFICATION_MODEL:
        raise last_error if last_error else RuntimeError("verification failed")

    logger.warning(
        "Cloud verification failed, falling back to local model",
        extra={"error": str(last_error), "fallback": LOCAL_VERIFICATION_MODEL},
    )
    result = await call_structured_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=LOCAL_VERIFICATION_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        agent=agent,
    )
    _last_verification_model.set(LOCAL_VERIFICATION_MODEL)
    return result, LOCAL_VERIFICATION_MODEL


async def call_verification_agent(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    agent: str | None = None,
) -> dict:
    """
    Wrapper for the verification LLM calls. Enforces JSON mode.

    Thin form of call_verification_agent_with_model for callers that only need
    the result; the serving model is still recorded for
    active_verification_model().

    Args:
        system_prompt: System-level instructions.
        user_prompt: User query / context.
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum output tokens (default 1500).
        agent: Calling agent's name, for usage logs and optional tracing.

    Returns:
        Parsed JSON dict from the verification model.
    """
    result, _model = await call_verification_agent_with_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        agent=agent,
    )
    return result


# Backwards-compatible alias — the verification agents were originally local-only.
call_local_agent = call_verification_agent


def active_verification_model() -> str:
    """
    The model that served the most recent verification call in this context,
    or — before any call — the model the verification path will try first.

    This used to return AGENT_LADDER[0] unconditionally, so the trace named the
    top rung even when quota exhaustion or a 503 had sent the call down the
    ladder or to the local model. The served model is now recorded by
    call_verification_agent_with_model; prefer the model that function returns
    over calling this.
    """
    served = _last_verification_model.get()
    if served:
        return served
    backend = os.getenv("VERIFICATION_BACKEND", "cloud").strip().lower()
    if backend == "local":
        return LOCAL_VERIFICATION_MODEL
    return AGENT_LADDER[0]


# ─── Streaming ────────────────────────────────────────────────────────────────


class StreamInterrupted(RuntimeError):
    """
    A streamed completion failed after tokens had already been emitted.

    Distinct from a failure before the first token: the client has partial text
    on screen, so the caller must tell it to discard that draft before falling
    back to another model.
    """

    def __init__(self, message: str, emitted_chars: int, cause: Exception):
        super().__init__(message)
        self.emitted_chars = emitted_chars
        self.cause = cause


class _UsageHolder:
    """Response-shaped holder so _log_usage can read a streamed call's usage."""

    def __init__(self, usage):
        self.usage = usage


async def stream_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    on_token,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    api_key: str | None = None,
    agent: str | None = None,
) -> str:
    """
    Prose completion streamed token by token, returning the full text.

    Failure semantics mirror call_prose_agent so the synthesizer's ladder logic
    is unchanged:

    - fails BEFORE the first token: a 503 / timeout / quota / auth error is
      re-raised for the ladder to classify; any other error falls back to the
      non-streamed call_prose_agent on the same model (which retries), and its
      whole answer is emitted as one token.
    - fails AFTER tokens were emitted: raises StreamInterrupted, and the caller
      emits an answer reset before trying elsewhere.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User prompt.
        model: LiteLLM model string.
        on_token: Callable receiving each text delta.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        api_key: Credential for this call.
        agent: Calling agent's name.

    Returns:
        The complete generated text. Never empty.

    Raises:
        StreamInterrupted: The stream broke after emitting text.
        RuntimeError: Empty completion.
        Exception: Provider errors the ladder classifies (see above).
    """
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": PROSE_TIMEOUT_SECONDS,
        "stream": True,
    }
    if api_key:
        kwargs["api_key"] = api_key
    metadata = _call_metadata(agent)
    if metadata:
        kwargs["metadata"] = metadata

    parts: list[str] = []
    emitted = 0
    started = time.monotonic()
    usage = None
    try:
        stream = await litellm.acompletion(**kwargs)
        async for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                parts.append(delta)
                emitted += len(delta)
                on_token(delta)
    except Exception as e:
        if emitted:
            raise StreamInterrupted(
                f"stream from {model} failed after {emitted} chars: {e}", emitted, e
            ) from e
        if (
            is_service_unavailable(e)
            or is_timeout_error(e)
            or is_quota_error(e)
            or is_auth_error(e)
            or is_model_unavailable_for_key(e)
        ):
            raise
        logger.warning(
            "Streaming failed before the first token; retrying without streaming",
            extra={"model": model, "error": str(e)},
        )
        text = await call_prose_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            agent=agent,
        )
        on_token(text)
        return text

    _log_usage(_UsageHolder(usage), model, agent, started)

    text = "".join(parts)
    if not text.strip():
        raise RuntimeError(f"empty streamed completion (model={model})")
    return text
