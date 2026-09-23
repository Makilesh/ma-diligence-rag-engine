"""
Guardrails for running the API as an unauthenticated public demo.

The deployed backend spends the owner's free-tier Gemini quota on every query,
and anyone who finds the URL can script it. Nothing here is user authentication
— there are no accounts — it is cost and availability control:

  * An optional admin key (`ADMIN_API_KEY`, sent as `X-Admin-Key`) that lifts
    every limit and is the only way to touch non-sandbox data.
  * Per-client rate limits, keyed on the client IP.
  * A process-wide cap on concurrent pipeline runs, which fails fast instead of
    queueing: on a 2 vCPU Space a queued request is a request that will time out
    at the proxy anyway, after holding a connection the whole time.
  * A global daily query cap — the last line of defence when per-IP limits are
    evaded by rotating addresses.
  * A per-request correlation id, so a generic error shown to a visitor can be
    matched to the full exception in the server log.

Everything is in-process and in-memory. That matches the deployment — one
uvicorn worker on one Space — and a restart resetting the counters is harmless:
Gemini's own quotas are the hard ceiling, these only keep one visitor from
spending all of it.

Configuration is read from the environment at call time rather than at import:
`.env` is loaded as a side effect of importing litellm, which may happen after
this module is imported, and tests need to change limits per case.
"""

import hmac
import itertools
import math
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ADMIN_HEADER = "X-Admin-Key"
REQUEST_ID_HEADER = "X-Request-ID"

# A pipeline slot older than this is assumed leaked and reclaimed. The streaming
# endpoint releases its slot from a generator `finally`, which never runs if the
# client disconnects before the first byte is sent; without a reaper, each such
# disconnect would permanently shrink capacity until the Space restarted. Set
# well above the slowest real query (minutes on CPU) so a live run is never
# reclaimed out from under itself.
_SLOT_STALE_SECONDS = 15 * 60


# ==============================================================================
# Configuration
# ==============================================================================


def _env_int(name: str, default: int) -> int:
    """Reads a non-negative integer env var, falling back on junk values."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"Ignoring non-integer {name}={raw!r}; using {default}")
        return default


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def is_development() -> bool:
    """True only when ENVIRONMENT is explicitly `development`.

    Unset counts as production. The Space has no `.env`, so a missing value
    must fail closed rather than hand every visitor admin rights.
    """
    return os.getenv("ENVIRONMENT", "").strip().lower() == "development"


# ==============================================================================
# Caller identity
# ==============================================================================


def is_admin(request: Request) -> bool:
    """
    Decides whether the caller may bypass limits and touch non-sandbox data.

    With ADMIN_API_KEY set, only a matching `X-Admin-Key` header qualifies. With
    it unset, every caller is admin in development (so the local UI, Streamlit
    and the eval harness keep working unchanged) and nobody is otherwise.

    Args:
        request: Incoming request.

    Returns:
        True if the caller has admin rights.
    """
    configured = os.getenv("ADMIN_API_KEY", "")
    if configured:
        presented = request.headers.get(ADMIN_HEADER, "")
        # Constant-time compare: the key is the only secret guarding deletes.
        return bool(presented) and hmac.compare_digest(
            presented.encode("utf-8"), configured.encode("utf-8")
        )
    return is_development()


def client_ip(request: Request) -> str:
    """
    Returns the address rate limits are keyed on.

    Behind the Hugging Face proxy every request arrives from the proxy's own
    address, so without X-Forwarded-For all visitors would share one bucket.
    The header is only honoured when TRUST_PROXY_HEADERS says a proxy is in
    front: without one, any client could set it and pick its own bucket.

    Args:
        request: Incoming request.

    Returns:
        Client IP string, or "unknown".
    """
    if _env_flag("TRUST_PROXY_HEADERS"):
        first_hop = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if first_hop:
            return first_hop[:64]
    return request.client.host if request.client else "unknown"


def get_request_id(request: Request) -> str:
    """Returns the correlation id RequestIdMiddleware assigned, minting one if absent."""
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
    return request_id


@dataclass(frozen=True)
class ClientContext:
    """Who is calling, resolved once per request by the rate-limit dependency."""

    ip: str
    is_admin: bool
    request_id: str


def client_context(request: Request) -> ClientContext:
    """Builds the ClientContext for a request."""
    return ClientContext(
        ip=client_ip(request),
        is_admin=is_admin(request),
        request_id=get_request_id(request),
    )


async def require_admin(request: Request) -> None:
    """
    FastAPI dependency: rejects non-admin callers with 403.

    Args:
        request: Incoming request.

    Raises:
        HTTPException: 403 when the caller is not admin.
    """
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail=f"This operation requires the {ADMIN_HEADER} header.",
        )


def public_error_detail(operation: str, request_id: str) -> str:
    """
    The client-facing text for an internal failure.

    Exception strings carry file paths, Qdrant URLs and provider error bodies —
    none of which a public caller should see. The request id is what lets the
    owner find the real error in the log.

    Args:
        operation: What failed, e.g. "Query".
        request_id: Correlation id for this request.

    Returns:
        A generic message naming the request id.
    """
    return f"{operation} failed due to an internal error (request id: {request_id})."


# ==============================================================================
# Per-client rate limiting
# ==============================================================================

# Bucket name -> (per-minute env var, default, per-day env var, default).
# Upload defaults are lower than query defaults: ingestion does not touch Gemini
# but embeds on the Space's CPU, and a few large PDFs can pin it for minutes.
_BUCKETS: dict[str, tuple[str, int, str, int]] = {
    "query": ("RATE_LIMIT_QUERY_PER_MINUTE", 5, "RATE_LIMIT_QUERY_PER_DAY", 50),
    "ingest": ("RATE_LIMIT_INGEST_PER_MINUTE", 3, "RATE_LIMIT_INGEST_PER_DAY", 20),
}

_MINUTE = 60.0
_DAY = 24 * 60 * 60.0

# Bound on tracked clients before a full prune. Stale keys are otherwise only
# dropped when the same client returns, so a scan from many addresses would grow
# the table without limit.
_MAX_TRACKED_KEYS = 10_000


class _SlidingWindowLimiter:
    """Sliding-window counter per key, holding one timestamp per admitted hit."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    def reset(self) -> None:
        self._hits.clear()

    def hit(self, key: str, per_minute: int, per_day: int, now: float) -> float | None:
        """
        Records a hit unless it would exceed a limit.

        Args:
            key: Client/bucket key.
            per_minute: Allowed hits in any 60s window (0 = unlimited).
            per_day: Allowed hits in any 24h window (0 = unlimited).
            now: Current monotonic time.

        Returns:
            None if admitted, else seconds until the caller may retry.
        """
        if len(self._hits) > _MAX_TRACKED_KEYS:
            self._prune(now)

        window = self._hits.setdefault(key, deque())
        while window and now - window[0] >= _DAY:
            window.popleft()

        if per_day and len(window) >= per_day:
            return _DAY - (now - window[0])

        if per_minute:
            recent = [t for t in window if now - t < _MINUTE]
            if len(recent) >= per_minute:
                return _MINUTE - (now - recent[0])

        window.append(now)
        return None

    def _prune(self, now: float) -> None:
        for key in [k for k, w in self._hits.items() if not w or now - w[-1] >= _DAY]:
            del self._hits[key]


_limiter = _SlidingWindowLimiter()


def rate_limit(bucket: str):
    """
    Builds a FastAPI dependency enforcing one bucket's per-client limits.

    The dependency returns the ClientContext so the route can reuse the
    admin/IP/request-id resolution rather than repeating it.

    Args:
        bucket: Key into _BUCKETS ("query" or "ingest").

    Returns:
        An async dependency callable.
    """
    minute_var, minute_default, day_var, day_default = _BUCKETS[bucket]

    async def dependency(request: Request) -> ClientContext:
        ctx = client_context(request)
        if ctx.is_admin:
            return ctx

        retry_after = _limiter.hit(
            f"{bucket}:{ctx.ip}",
            per_minute=_env_int(minute_var, minute_default),
            per_day=_env_int(day_var, day_default),
            now=time.monotonic(),
        )
        if retry_after is not None:
            logger.warning(
                "Rate limit exceeded",
                extra={"bucket": bucket, "client_ip": ctx.ip, "request_id": ctx.request_id},
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for this demo. Please wait and try again.",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )
        return ctx

    dependency.__name__ = f"{bucket}_rate_limit"
    return dependency


query_rate_limit = rate_limit("query")
ingest_rate_limit = rate_limit("ingest")


# ==============================================================================
# Pipeline concurrency + global daily cap
# ==============================================================================


class PipelineSlot:
    """A held pipeline slot. `release()` is idempotent, so it can be wired to
    more than one cleanup path (generator `finally` and a background task)."""

    def __init__(self, gate: "_PipelineGate", token: int) -> None:
        self._gate = gate
        self._token = token

    def release(self) -> None:
        self._gate._active.pop(self._token, None)

    def __enter__(self) -> "PipelineSlot":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class _PipelineGate:
    """Counts in-flight pipeline runs and pipeline runs per UTC day."""

    def __init__(self) -> None:
        self._active: dict[int, float] = {}
        self._tokens = itertools.count()
        self._day: str = ""
        self._day_count = 0

    def reset(self) -> None:
        self._active.clear()
        self._day, self._day_count = "", 0

    @property
    def in_flight(self) -> int:
        return len(self._active)

    def acquire(self, ctx: ClientContext, count_toward_daily_cap: bool = True) -> PipelineSlot:
        """
        Takes a slot or raises immediately.

        Admin runs are never rejected, but they still occupy a slot and count
        toward the daily total — they spend the same CPU and the same quota.

        Args:
            ctx: Caller context from the rate-limit dependency.
            count_toward_daily_cap: False for work that does not spend Gemini
                quota (ingestion), which should hold a slot but not a query.

        Returns:
            A PipelineSlot to release when the run ends.

        Raises:
            HTTPException: 429 when the daily cap is spent, 503 when every
                slot is busy.
        """
        now = time.monotonic()
        for token, started in list(self._active.items()):
            if now - started > _SLOT_STALE_SECONDS:
                logger.warning("Reclaiming stale pipeline slot", extra={"slot": token})
                self._active.pop(token, None)

        utc_now = datetime.now(timezone.utc)
        today = utc_now.date().isoformat()
        if today != self._day:
            self._day, self._day_count = today, 0

        if not ctx.is_admin:
            daily_cap = _env_int("DAILY_QUERY_CAP", 300)
            if count_toward_daily_cap and daily_cap and self._day_count >= daily_cap:
                midnight = datetime.combine(
                    utc_now.date() + timedelta(days=1), datetime.min.time(), timezone.utc
                )
                logger.warning(
                    "Global daily query cap reached",
                    extra={"cap": daily_cap, "request_id": ctx.request_id},
                )
                raise HTTPException(
                    status_code=429,
                    detail="The demo's daily query budget is spent. It resets at 00:00 UTC.",
                    headers={"Retry-After": str(int((midnight - utc_now).total_seconds()) + 1)},
                )

            max_concurrent = max(1, _env_int("MAX_CONCURRENT_PIPELINES", 2))
            if len(self._active) >= max_concurrent:
                raise HTTPException(
                    status_code=503,
                    detail="The engine is busy with other questions. Please retry shortly.",
                    headers={"Retry-After": "15"},
                )

        token = next(self._tokens)
        self._active[token] = now
        if count_toward_daily_cap:
            self._day_count += 1
        return PipelineSlot(self, token)


_gate = _PipelineGate()


def acquire_pipeline_slot(ctx: ClientContext, count_toward_daily_cap: bool = True) -> PipelineSlot:
    """Module-level entry point to the shared gate; see `_PipelineGate.acquire`."""
    return _gate.acquire(ctx, count_toward_daily_cap=count_toward_daily_cap)


def reset_limits() -> None:
    """Clears every counter. For tests; nothing in production calls it."""
    _limiter.reset()
    _gate.reset()


# ==============================================================================
# Sandbox deal guard (for the ingestion route)
# ==============================================================================


async def require_sandbox_deal(deal_id: str, request: Request) -> None:
    """
    Allows a write into `deal_id` only if the caller may write there.

    Admin may write anywhere. Everyone else may only write into a live sandbox
    deal — otherwise any visitor could inject documents into the curated demo
    corpus, or into a permanent deal, and those uploads would never be swept.

    Args:
        deal_id: Target deal.
        request: Incoming request (for the admin check).

    Raises:
        HTTPException: 422 for a malformed id, 403 for a non-sandbox target,
            410 for an expired sandbox.
    """
    # Imported here: deals.py imports this module at load time.
    from api.models.request_models import DEAL_ID_REGEX
    from api.routes.deals import is_sandbox_id, sandbox_is_expired

    if not DEAL_ID_REGEX.fullmatch(deal_id or ""):
        raise HTTPException(status_code=422, detail="Invalid deal_id.")
    if is_admin(request):
        return
    if not is_sandbox_id(deal_id):
        raise HTTPException(
            status_code=403,
            detail="Uploads are only accepted into a temporary sandbox deal.",
        )
    if sandbox_is_expired(deal_id):
        raise HTTPException(status_code=410, detail="This sandbox deal has expired.")


# ==============================================================================
# Correlation id middleware
# ==============================================================================


class RequestIdMiddleware:
    """
    Assigns every HTTP request an id, exposed as `request.state.request_id`
    and echoed in the `X-Request-ID` response header.

    Pure ASGI rather than `@app.middleware("http")`: BaseHTTPMiddleware wraps
    the response body, which interferes with the SSE endpoint's streaming.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Always minted server-side: echoing a client-supplied id into logs would
        # let a caller forge correlation with someone else's request.
        request_id = secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        await self.app(scope, receive, send_with_id)
