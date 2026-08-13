"""
Budget tracker — Postgres-backed daily API call tracking with per-model RPM rate limiting.

The BudgetTracker must be:
- Persistent across restarts — stored in Postgres, not in-memory
- Per (API key, model) — quotas are enforced per key and per model, so
  accounting at any coarser granularity mis-counts
- Thread-safe — uses DB transactions for increment
- UTC-midnight resetting — checked on every call
- Exposed in the UI — Streamlit sidebar shows live budget status

INITIALIZATION:
    BudgetTracker uses asyncpg.create_pool() which is async — it CANNOT
    be called in __init__. Use the async factory classmethod:
        tracker = await BudgetTracker.get_instance(postgres_url)

RATE LIMITERS:
    asyncio.Lock() requires a running event loop. Class-level attributes
    are evaluated at import time — BEFORE any event loop is running.
    Rate limiters are created lazily on first use via _get_rate_limiter().
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import asyncpg

from src.llm.model_registry import (
    AGENT_LADDER,
    LOCAL_MODEL,
    SYNTHESIS_LADDER,
    is_local,
    rpm_for,
    usable_rpd,
)
from src.llm.rate_limiter import RateLimiter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class ModelChoice:
    """
    A resolved routing decision: which model, billed to which API key.

    Model selection returns this rather than a bare model string because with
    key rotation the key is no longer implicit — the same model may be served
    by different keys on consecutive calls, and the caller has to pass the
    matching credential or the quota accounting is a fiction.
    """

    model: str
    api_key: str | None  # None for the local model, which needs no credential
    key_index: int       # -1 for local; otherwise position in the key list


@dataclass
class ModelBudget:
    """Current budget state for a single model."""

    model_key: str
    daily_limit: int
    used_today: int
    reset_date: str  # ISO date string UTC


class BudgetTracker:
    """
    Singleton that persists daily API call counts to Postgres.
    Call get_model_for_agent() before every LiteLLM call to get
    the appropriate model string given current budget state.

    Schema (auto-created on startup):
        CREATE TABLE IF NOT EXISTS api_budget (
            model_key TEXT PRIMARY KEY,
            used_today INTEGER DEFAULT 0,
            reset_date DATE DEFAULT CURRENT_DATE
        );
    """

    _instance: "BudgetTracker | None" = None

    # Quota accounting is per (API key, model) — a "slot".
    #
    # Provider quotas attach to the key, so N keys multiply capacity, and they
    # attach per model, so spend on one model does not reduce another. Anything
    # coarser mis-counts: the previous per-bucket accounting allowed 960
    # requests/day against a single model's 500 RPD cap because two buckets each
    # believed they owned a separate 480.
    #
    # Limits come from model_registry.MODEL_LIMITS, never from literals here —
    # one table, so allocation errors are visible side by side rather than
    # scattered across three constants.
    SLOT_SEPARATOR = "::"

    @staticmethod
    def _load_api_keys() -> list[str]:
        """
        Collects Gemini API keys from the environment, in priority order.

        Accepts either form, because both are common:
            GEMINI_API_KEYS=key1,key2,key3
            GEMINI_API_KEY_1=key1  GEMINI_API_KEY_2=key2   (numbered, any count)
            GEMINI_API_KEY=key1                             (single, legacy)

        Order is meaningful: the selector drains key 1 on a given model before
        touching key 2, so the first key should be the one you most want spent.

        Returns:
            De-duplicated list of keys, preserving order. Empty if none set,
            which makes the pipeline run entirely on the local model.
        """
        keys: list[str] = []

        multi = os.getenv("GEMINI_API_KEYS", "")
        keys.extend(k.strip() for k in multi.split(",") if k.strip())

        i = 1
        while True:
            numbered = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
            if not numbered:
                break
            keys.append(numbered)
            i += 1

        single = os.getenv("GEMINI_API_KEY", "").strip()
        if single:
            keys.append(single)

        seen: set[str] = set()
        ordered = [k for k in keys if not (k in seen or seen.add(k))]
        return ordered

    def _slot(self, key_index: int, model: str) -> str:
        """Composite accounting identifier for one (API key, model) pair."""
        return f"{key_index}{self.SLOT_SEPARATOR}{model}"

    def _slot_model(self, slot: str) -> str:
        """Extracts the model from a slot identifier."""
        _, _, model = slot.partition(self.SLOT_SEPARATOR)
        return model or slot

    def _slot_limit(self, slot: str) -> int:
        """Daily allowance for a slot, from the registry plus safety margin."""
        return usable_rpd(self._slot_model(slot))

    def _all_slots(self) -> list[str]:
        """Every (key, model) pair this process may spend against."""
        models = [m for m in dict.fromkeys(SYNTHESIS_LADDER + AGENT_LADDER)
                  if not is_local(m)]
        return [self._slot(i, m)
                for i in range(len(self._api_keys))
                for m in models]

    # DO NOT instantiate RateLimiter at class definition time.
    # asyncio.Lock() inside RateLimiter.__init__ will fail before event loop starts.
    _rate_limiters: dict[str, RateLimiter] = {}  # populated lazily on first use
    _instance_lock: asyncio.Lock | None = None  # lazily created — see get_instance()

    @classmethod
    async def get_instance(cls, postgres_url: str | None = None) -> "BudgetTracker":
        """
        Async factory — the ONLY way to create a BudgetTracker.
        asyncpg.create_pool() is async and cannot be called in __init__.

        Uses double-checked locking to prevent race condition where two
        concurrent startup requests both see _instance is None and create
        duplicate connection pools.

        Args:
            postgres_url: PostgreSQL connection string. Required on first call.

        Returns:
            Singleton BudgetTracker instance with initialized connection pool.

        Raises:
            ValueError: If postgres_url is None on first call.
            asyncpg.PostgresError: If database connection fails.
        """
        if cls._instance is not None:
            return cls._instance

        # Lazy lock creation — safe because this is called inside a running event loop
        if cls._instance_lock is None:
            cls._instance_lock = asyncio.Lock()

        async with cls._instance_lock:
            if cls._instance is None:
                if postgres_url is None:
                    raise ValueError(
                        "postgres_url is required on first BudgetTracker initialization"
                    )
                logger.info("Initializing BudgetTracker singleton")
                instance = cls.__new__(cls)
                # Must be set before _ensure_schema(), which enumerates slots.
                instance._api_keys = cls._load_api_keys()
                instance._rate_limiters = {}
                logger.info(
                    "Gemini API keys loaded",
                    extra={"key_count": len(instance._api_keys)},
                )
                if not instance._api_keys:
                    logger.warning(
                        "No Gemini API key configured — every agent will run on "
                        "the local model. Set GEMINI_API_KEYS or GEMINI_API_KEY."
                    )
                try:
                    instance._pool = await asyncpg.create_pool(
                        postgres_url, min_size=2, max_size=5
                    )
                    instance._is_mock = False
                    await instance._ensure_schema()
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize Postgres pool: {e}. Falling back to In-Memory Budget Tracking."
                    )
                    instance._is_mock = True
                    instance._mock_budgets = {}
                cls._instance = instance
                logger.info("BudgetTracker initialized successfully")
        return cls._instance

    async def _ensure_schema(self) -> None:
        """
        Creates the api_budget table if it doesn't exist.
        Idempotent — safe to call on every startup.

        Raises:
            asyncpg.PostgresError: If schema creation fails.
        """
        if getattr(self, "_is_mock", False):
            return
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_budget (
                    model_key TEXT PRIMARY KEY,
                    used_today INTEGER DEFAULT 0,
                    reset_date DATE DEFAULT CURRENT_DATE
                )
            """)
            # Ensure rows exist for each model
            for model_key in self._all_slots():
                await conn.execute(
                    """
                    INSERT INTO api_budget (model_key, used_today, reset_date)
                    VALUES ($1, 0, CURRENT_DATE)
                    ON CONFLICT (model_key) DO NOTHING
                """,
                    model_key,
                )
        logger.info("Budget schema verified")

    def _get_rate_limiter(self, slot: str) -> RateLimiter:
        """
        Lazily creates RateLimiter instances, one per (API key, model) slot.

        The granularity is the point. RPM is metered by the provider per key and
        per model, so a limiter any coarser under-counts and one any finer
        over-counts. An earlier version keyed limiters by budget bucket, and
        because two buckets resolved to the same model their separate 5 and 15
        RPM allowances permitted 20 requests/minute against a single 15 RPM
        quota. Keying by slot makes the client-side ceiling structurally match
        the provider's.

        Lazily constructed because asyncio.Lock() inside RateLimiter.__init__
        requires a running event loop, which does not exist at import time.

        Args:
            slot: "<key_index>::<model>" identifier.

        Returns:
            RateLimiter dedicated to that key/model pair.
        """
        if slot not in self._rate_limiters:
            self._rate_limiters[slot] = RateLimiter(
                rpm_limit=rpm_for(self._slot_model(slot))
            )
        return self._rate_limiters[slot]

    async def _select_from_ladder(self, ladder: list[str], purpose: str) -> ModelChoice:
        """
        Walks a model ladder and the configured keys to find capacity.

        Order is deliberate: for each model, every key is tried before dropping
        to the next model down. Keys multiply capacity at a given quality level,
        so exhausting all of them on gemini-3.6-flash before falling back to
        gemini-3.5-flash is what "keep the best for the best" actually means. The
        opposite order — cycling models within a key — would degrade answer
        quality while premium quota sat unused on another key.

        Two passes:
          1. Non-blocking. Take the first slot with both daily budget and a free
             RPM window. This is where key rotation pays off — a saturated key is
             skipped instantly rather than slept on.
          2. Blocking. If every slot is momentarily rate-limited but some still
             have daily budget, wait on the best of them. Waiting a few seconds
             for the good model beats permanently degrading to the local one.

        Falls back to the local model only when no cloud slot has daily budget
        left, which is the genuine end of the quota.

        Args:
            ladder: Models in preference order.
            purpose: "synthesis" or "agent", for logging.

        Returns:
            ModelChoice naming the model and the key to bill it to.
        """
        # Pass 1 — immediate capacity anywhere.
        for model in ladder:
            if is_local(model):
                break
            for key_index, api_key in enumerate(self._api_keys):
                slot = self._slot(key_index, model)
                if not await self._budget_available(slot):
                    continue
                if not await self._get_rate_limiter(slot).try_acquire():
                    continue
                if await self._try_consume(slot):
                    return ModelChoice(model=model, api_key=api_key, key_index=key_index)

        # Pass 2 — everything is rate-limited; wait rather than degrade quality.
        for model in ladder:
            if is_local(model):
                break
            for key_index, api_key in enumerate(self._api_keys):
                slot = self._slot(key_index, model)
                if not await self._budget_available(slot):
                    continue
                logger.info(
                    "All slots rate-limited; waiting for capacity",
                    extra={"purpose": purpose, "model": model, "key_index": key_index},
                )
                await self._get_rate_limiter(slot).acquire()
                if await self._try_consume(slot):
                    return ModelChoice(model=model, api_key=api_key, key_index=key_index)

        logger.warning(
            "All cloud quota exhausted, falling back to local model",
            extra={"purpose": purpose, "keys_configured": len(self._api_keys)},
        )
        return ModelChoice(model=LOCAL_MODEL, api_key=None, key_index=-1)

    async def get_model_for_synthesis(self) -> ModelChoice:
        """
        Picks the best available model for answer synthesis.

        Synthesis is one call per query and the only agent whose reasoning
        quality is visible in the output, so it gets the reasoning ladder —
        each rung worth ~20 requests/day per key — and only spills to a Lite
        model once those are spent.

        CRITICAL: the rate-limiter slot is taken BEFORE the daily budget is
        debited. If the task is cancelled while waiting, the budget is not
        charged for a call that was never sent.

        Returns:
            ModelChoice with the model and the API key to bill it to.
        """
        return await self._select_from_ladder(SYNTHESIS_LADDER, "synthesis")

    async def get_model_for_agent(self) -> ModelChoice:
        """
        Picks a model for the high-volume agent calls.

        Agents spend ~4 calls per query on classification, rewriting, assessment
        and validation — structured work that does not need frontier reasoning.
        The ladder is ordered by daily capacity rather than capability, and
        deliberately excludes the reasoning models: at 20 RPD each they would be
        drained by five queries and then be unavailable to synthesis, which is
        where that quality actually matters.

        Returns:
            ModelChoice with the model and the API key to bill it to.
        """
        return await self._select_from_ladder(AGENT_LADDER, "agent")

    async def mark_slot_exhausted(self, key_index: int, model: str) -> None:
        """
        Records that the provider has refused further calls on this (key, model).

        Local counters drift from the provider's for reasons that cannot be
        engineered away: the process restarts and the in-memory fallback resets
        to zero while the provider's daily counter keeps running, another process
        shares the key, or someone uses the key by hand. When that happens the
        tracker believes it has quota it does not have, and every call to that
        model returns 429.

        Observed cost of not having this: a golden-set run opened with 19
        consecutive synthesis failures — exactly gemini-3.6-flash's daily
        allowance — because the previous run had already spent it. The gate had
        passed the context with scores up to 0.945; the answers were lost purely
        to a stale counter, and the run only recovered once the local counter
        also hit its limit and the ladder moved on by itself.

        Burning the local counter makes the provider's 429 authoritative, so the
        next selection skips this rung immediately.

        Args:
            key_index: Position of the API key, or -1 for local (ignored).
            model: Upstream model the provider refused.
        """
        if key_index < 0 or is_local(model):
            return

        slot = self._slot(key_index, model)
        limit = self._slot_limit(slot)
        today = datetime.now(timezone.utc).date().isoformat()

        logger.warning(
            "Provider refused further calls; marking slot exhausted for today",
            extra={"model": model, "key_index": key_index},
        )

        if getattr(self, "_is_mock", False):
            self._mock_budgets[slot] = {"used_today": limit, "reset_date": today}
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE api_budget SET used_today = $1, reset_date = $2::date "
                    "WHERE model_key = $3",
                    limit, today, slot,
                )
        except Exception as e:
            # Never let bookkeeping failure break the caller's fallback path.
            logger.error("Failed to persist exhausted slot",
                         extra={"slot": slot, "error": str(e)})

    async def get_budget_status(self) -> dict:
        """
        Returns current budget status, aggregated per model across all keys.

        The UI cares about "how much gemini-3.6-flash is left", not about which
        key it sits on, so per-key slots are summed. Per-key detail stays in
        `keys`, so an exhausted individual key is still diagnosable.

        Returns:
            Dict of model -> {used, limit, remaining, reset_date, keys}.
        """
        status: dict[str, dict] = {}

        for slot in self._all_slots():
            model = self._slot_model(slot)
            budget = await self._load_budget(slot)
            limit = self._slot_limit(slot)

            entry = status.setdefault(
                model,
                {"used": 0, "limit": 0, "remaining": 0,
                 "reset_date": budget.reset_date, "keys": []},
            )
            entry["used"] += budget.used_today
            entry["limit"] += limit
            entry["remaining"] += max(0, limit - budget.used_today)
            entry["keys"].append(
                {"key_index": int(slot.split(self.SLOT_SEPARATOR)[0]),
                 "used": budget.used_today, "limit": limit}
            )

        return status

    async def _budget_available(self, model_key: str) -> bool:
        """
        Read-only check for UI display.
        Do NOT use for consumption decisions — use _try_consume() instead.

        Args:
            model_key: Model identifier.

        Returns:
            True if budget is available for this model today.
        """
        budget = await self._load_budget(model_key)
        today = datetime.now(timezone.utc).date().isoformat()
        if budget.reset_date != today:
            return True
        return budget.used_today < self._slot_limit(model_key)

    async def _try_consume(self, model_key: str) -> bool:
        """
        Atomic check-and-increment: returns True and increments counter if budget
        remains, returns False without incrementing if exhausted.

        CRITICAL: This replaces the old _budget_available() + _increment() pattern
        which had a TOCTOU race: between checking and incrementing, a concurrent
        request could pass the same check, causing both to increment and overshoot
        the daily limit. The single conditional UPDATE is atomic at the DB level.

        Args:
            model_key: Model identifier.

        Returns:
            True if budget was consumed, False if exhausted.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if getattr(self, "_is_mock", False):
            if model_key not in self._mock_budgets:
                self._mock_budgets[model_key] = {"used_today": 0, "reset_date": today}
            b = self._mock_budgets[model_key]
            if b["reset_date"] < today:
                b["used_today"] = 0
                b["reset_date"] = today
            if b["used_today"] < self._slot_limit(model_key):
                b["used_today"] += 1
                logger.info(
                    f"Budget consumed (local mock): {model_key} ({b['used_today']}/{self._slot_limit(model_key)})"
                )
                return True
            logger.warning(f"Budget exhausted (local mock) for model {model_key}")
            return False

        async with self._pool.acquire() as conn:
            # Reset if new day (idempotent)
            await conn.execute(
                "UPDATE api_budget SET used_today = 0, reset_date = $1::date "
                "WHERE model_key = $2 AND reset_date < $1::date",
                today,
                model_key,
            )
            # Atomic conditional increment — prevents TOCTOU race
            result = await conn.execute(
                "UPDATE api_budget SET used_today = used_today + 1 "
                "WHERE model_key = $1 AND used_today < $2",
                model_key,
                self._slot_limit(model_key),
            )
            # result is "UPDATE N" where N is rows affected
            consumed = result == "UPDATE 1"

            if consumed:
                logger.info(
                    "Budget consumed",
                    extra={"model_key": model_key},
                )
            else:
                logger.warning(
                    "Budget exhausted for model",
                    extra={"model_key": model_key},
                )

            return consumed

    async def _load_budget(self, model_key: str) -> ModelBudget:
        """
        Loads current budget state from Postgres.

        Args:
            model_key: Model identifier.

        Returns:
            ModelBudget dataclass with used_today and reset_date.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if getattr(self, "_is_mock", False):
            if model_key not in self._mock_budgets:
                self._mock_budgets[model_key] = {"used_today": 0, "reset_date": today}
            b = self._mock_budgets[model_key]
            return ModelBudget(
                model_key=model_key,
                daily_limit=self._slot_limit(model_key),
                used_today=b["used_today"],
                reset_date=b["reset_date"],
            )

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT model_key, used_today, reset_date FROM api_budget WHERE model_key = $1",
                model_key,
            )
            if row is None:
                # First use — insert default
                await conn.execute(
                    "INSERT INTO api_budget (model_key, used_today, reset_date) "
                    "VALUES ($1, 0, CURRENT_DATE)",
                    model_key,
                )
                return ModelBudget(
                    model_key=model_key,
                    daily_limit=self._slot_limit(model_key),
                    used_today=0,
                    reset_date=datetime.now(timezone.utc).date().isoformat(),
                )
            return ModelBudget(
                model_key=row["model_key"],
                daily_limit=self._slot_limit(model_key),
                used_today=row["used_today"],
                reset_date=(
                    row["reset_date"].isoformat()
                    if hasattr(row["reset_date"], "isoformat")
                    else str(row["reset_date"])
                ),
            )

    @classmethod
    async def close(cls) -> None:
        """
        Closes the connection pool. Call on application shutdown.

        Raises:
            RuntimeError: If no instance exists.
        """
        if cls._instance is not None:
            if not getattr(cls._instance, "_is_mock", False) and hasattr(cls._instance, "_pool"):
                await cls._instance._pool.close()
                logger.info("BudgetTracker connection pool closed")
            cls._instance = None
