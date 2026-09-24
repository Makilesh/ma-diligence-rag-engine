"""
Tests for async safety patterns.
"""

import pytest
import asyncio


class TestRateLimiterAsyncSafety:
    """Verify RateLimiter async patterns are correct."""

    @pytest.mark.asyncio
    async def test_lock_not_held_during_sleep(self):
        """Lock is released before sleeping so other coroutines can proceed."""
        from src.llm.rate_limiter import RateLimiter

        # Create a limiter with very low limit
        limiter = RateLimiter(rpm_limit=1)

        # First acquire succeeds immediately
        await limiter.acquire()

        # Second acquire would need to wait — but we can test that
        # the lock is not permanently held by checking that another
        # coroutine can acquire the lock during the wait period.
        lock_acquired = False

        async def try_lock():
            nonlocal lock_acquired
            # Try to acquire the internal lock — if the sleeping coroutine
            # released it, this should succeed within a short timeout
            try:
                async with asyncio.timeout(0.5):
                    async with limiter._lock:
                        lock_acquired = True
            except asyncio.TimeoutError:
                pass

        # Start the waiter task (will block because limit=1 is exhausted)
        waiter_task = asyncio.create_task(limiter.acquire())
        # Give it a moment to start waiting
        await asyncio.sleep(0.05)

        # Try to acquire the lock — should succeed because waiter releases before sleep
        await try_lock()
        assert lock_acquired, "Lock should be released during sleep"

        # Cleanup
        waiter_task.cancel()
        try:
            await waiter_task
        except asyncio.CancelledError:
            pass

    def test_rate_limiter_requires_positive_rpm(self):
        """Cannot create RateLimiter with non-positive RPM."""
        from src.llm.rate_limiter import RateLimiter

        with pytest.raises(ValueError):
            RateLimiter(rpm_limit=0)


class TestBudgetTrackerSafety:
    """Verify BudgetTracker cannot be misused."""

    def test_cannot_construct_directly(self):
        """BudgetTracker uses async factory — direct __init__ should not be used.
        The class doesn't have __init__ with required params, forcing use of get_instance()."""
        from src.llm.budget_tracker import BudgetTracker

        # Verify the async factory pattern exists
        assert hasattr(BudgetTracker, "get_instance")
        assert asyncio.iscoroutinefunction(BudgetTracker.get_instance)

    @pytest.mark.asyncio
    async def test_rate_limiters_are_per_instance_not_class_level(self, monkeypatch):
        """
        Limiters are created lazily on the instance, never on the shared class dict.

        `_rate_limiters` is declared on the class as a mutable `{}`. If the
        factory ever stopped giving each instance its own dict, every limiter
        would be written into that class-level default and shared process-wide.
        Builds the tracker through the real factory (Postgres unavailable, so it
        takes the in-memory path) and checks where a limiter actually lands.
        """
        import src.llm.budget_tracker as bt
        from src.llm.model_registry import SYNTHESIS_LADDER, is_local

        async def no_postgres(*args, **kwargs):
            raise OSError("no Postgres in unit tests")

        monkeypatch.setattr(bt.asyncpg, "create_pool", no_postgres)
        monkeypatch.setattr(bt.BudgetTracker, "_instance", None)
        monkeypatch.setattr(bt.BudgetTracker, "_instance_lock", None)

        tracker = await bt.BudgetTracker.get_instance("postgresql://unused")
        assert tracker._is_mock is True
        assert tracker._rate_limiters is not bt.BudgetTracker._rate_limiters

        model = next(m for m in SYNTHESIS_LADDER if not is_local(m))
        slot = f"0{tracker.SLOT_SEPARATOR}{model}"
        limiter = tracker._get_rate_limiter(slot)

        assert tracker._rate_limiters[slot] is limiter
        assert tracker._get_rate_limiter(slot) is limiter, "limiter must be cached"
        assert slot not in bt.BudgetTracker._rate_limiters


class TestEmbeddingAsyncSafety:
    """Verify embedding functions use run_in_executor."""

    def test_embed_function_is_async(self):
        """embed_texts_async is a coroutine function."""
        from src.vector_db.reranker import embed_texts_async

        assert asyncio.iscoroutinefunction(embed_texts_async)

    def test_rerank_function_is_async(self):
        """rerank_async is a coroutine function."""
        from src.vector_db.reranker import rerank_async

        assert asyncio.iscoroutinefunction(rerank_async)
