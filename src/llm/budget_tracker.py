"""
Budget tracker — Postgres-backed daily API call tracking with per-model RPM rate limiting.

The BudgetTracker must be:
- Persistent across restarts — stored in Postgres, not in-memory
- Per-model — tracks synthesis_primary and agent_workhorse separately
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
from dataclasses import dataclass
from datetime import datetime, timezone

import asyncpg

from src.llm.rate_limiter import RateLimiter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


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

    DAILY_LIMITS: dict[str, int] = {
        "synthesis_primary": 480,  # gemini-3.1-flash-lite: leave 20 RPD buffer from 500
        "agent_workhorse": 480,  # gemini-3.1-flash-lite: leave 20 RPD buffer from 500
    }

    RPM_LIMITS: dict[str, int] = {
        "synthesis_primary": 5,
        "agent_workhorse": 15,
    }

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
            for model_key in self.DAILY_LIMITS:
                await conn.execute(
                    """
                    INSERT INTO api_budget (model_key, used_today, reset_date)
                    VALUES ($1, 0, CURRENT_DATE)
                    ON CONFLICT (model_key) DO NOTHING
                """,
                    model_key,
                )
        logger.info("Budget schema verified")

    def _get_rate_limiter(self, model_key: str) -> RateLimiter:
        """
        Lazily creates RateLimiter instances on first use.
        This ensures asyncio.Lock() is called inside a running event loop,
        not at class definition / import time.

        Args:
            model_key: Model identifier (e.g., "synthesis_primary").

        Returns:
            RateLimiter instance for the specified model.
        """
        if model_key not in self._rate_limiters:
            self._rate_limiters[model_key] = RateLimiter(
                rpm_limit=self.RPM_LIMITS[model_key]
            )
        return self._rate_limiters[model_key]

    # Synthesis model. config/litellm_config.yaml has specified
    # gemini-3.1-flash-lite since the quota decision recorded there ("swapped
    # from 3.5-flash to avoid daily quota limit, 20 RPD"), but this method kept
    # returning 3.5-flash — the config and the code had drifted apart, and the
    # code won. The golden-set run made the cost concrete: 3.5-flash produced one
    # upstream 503 and two empty completions across 19 queries. Aligning with the
    # documented decision removes the drift and the quota exposure in one move.
    SYNTHESIS_MODEL = "gemini/gemini-3.1-flash-lite"

    async def get_model_for_synthesis(self) -> str:
        """
        Returns the synthesis model string if daily budget remains,
        otherwise falls back to the agent workhorse model.

        CRITICAL: Acquire the rate limiter slot BEFORE consuming the daily budget.
        If the task is cancelled while sleeping in acquire(), the budget is NOT
        debited for a call that was never sent.

        Returns:
            LiteLLM model string for synthesis.
        """
        # Read-only check to avoid rate-limiter waiting if budget is already exhausted
        if not await self._budget_available("synthesis_primary"):
            logger.info("Synthesis budget exhausted, falling back to agent model")
            return await self.get_model_for_agent()

        await self._get_rate_limiter("synthesis_primary").acquire()
        if await self._try_consume("synthesis_primary"):
            return self.SYNTHESIS_MODEL
        return await self.get_model_for_agent()

    async def get_model_for_agent(self) -> str:
        """
        Returns gemini-3.1-flash-lite if budget remains, else local Ollama fallback.

        CRITICAL: Acquire the rate limiter slot BEFORE consuming the daily budget.

        Returns:
            LiteLLM model string for agent calls.
        """
        # Read-only check to avoid rate-limiter waiting if budget is already exhausted
        if not await self._budget_available("agent_workhorse"):
            logger.info("Agent budget exhausted, falling back to Ollama")
            return "ollama/qwen2.5:14b"

        await self._get_rate_limiter("agent_workhorse").acquire()
        if await self._try_consume("agent_workhorse"):
            return "gemini/gemini-3.1-flash-lite"  # ⚠ VERIFY STRING
        return "ollama/qwen2.5:14b"

    async def get_budget_status(self) -> dict:
        """
        Returns current budget status for all models.
        Used by Streamlit sidebar for live display.

        Returns:
            Dict with model_key -> {used, limit, remaining, reset_date}.
        """
        status = {}
        for model_key in self.DAILY_LIMITS:
            budget = await self._load_budget(model_key)
            status[model_key] = {
                "used": budget.used_today,
                "limit": self.DAILY_LIMITS[model_key],
                "remaining": self.DAILY_LIMITS[model_key] - budget.used_today,
                "reset_date": budget.reset_date,
            }
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
        return budget.used_today < self.DAILY_LIMITS[model_key]

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
            if b["used_today"] < self.DAILY_LIMITS[model_key]:
                b["used_today"] += 1
                logger.info(
                    f"Budget consumed (local mock): {model_key} ({b['used_today']}/{self.DAILY_LIMITS[model_key]})"
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
                self.DAILY_LIMITS[model_key],
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
                daily_limit=self.DAILY_LIMITS[model_key],
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
                    daily_limit=self.DAILY_LIMITS[model_key],
                    used_today=0,
                    reset_date=datetime.now(timezone.utc).date().isoformat(),
                )
            return ModelBudget(
                model_key=row["model_key"],
                daily_limit=self.DAILY_LIMITS[model_key],
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
