"""
Tests for sliding window rate limiter.
"""

import asyncio
import time
import pytest


@pytest.mark.asyncio
async def test_acquire_under_limit():
    """Acquire returns immediately when under RPM limit."""
    from src.llm.rate_limiter import RateLimiter

    limiter = RateLimiter(rpm_limit=10)
    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed < 0.1  # Should be near-instant
    assert limiter.current_usage == 1


@pytest.mark.asyncio
async def test_acquire_multiple_under_limit():
    """Multiple acquires under limit all return fast."""
    from src.llm.rate_limiter import RateLimiter

    limiter = RateLimiter(rpm_limit=5)
    for _ in range(5):
        await limiter.acquire()

    assert limiter.current_usage == 5


@pytest.mark.asyncio
async def test_current_usage_property():
    """current_usage reflects calls within the window."""
    from src.llm.rate_limiter import RateLimiter

    limiter = RateLimiter(rpm_limit=10)
    assert limiter.current_usage == 0

    await limiter.acquire()
    await limiter.acquire()
    assert limiter.current_usage == 2


def test_invalid_rpm_limit():
    """Non-positive rpm_limit raises ValueError."""
    from src.llm.rate_limiter import RateLimiter

    with pytest.raises(ValueError, match="positive"):
        RateLimiter(rpm_limit=0)

    with pytest.raises(ValueError, match="positive"):
        RateLimiter(rpm_limit=-5)


@pytest.mark.asyncio
async def test_concurrent_acquires():
    """Multiple concurrent acquires don't exceed limit."""
    from src.llm.rate_limiter import RateLimiter

    limiter = RateLimiter(rpm_limit=3)

    # Launch 3 concurrent acquires — all should succeed quickly
    tasks = [limiter.acquire() for _ in range(3)]
    await asyncio.gather(*tasks)

    assert limiter.current_usage == 3


class TestSharedModelRateLimiter:
    """
    Buckets that spend against the same upstream model must share one limiter.

    Separate limiters per budget bucket (5 RPM + 15 RPM) allowed 20 requests per
    minute against a single 15 RPM provider quota once both buckets resolved to
    the same Gemini model, producing RateLimitError under load.
    """

    def test_buckets_on_same_model_share_a_limiter(self):
        from src.llm.budget_tracker import BudgetTracker

        tracker = BudgetTracker.__new__(BudgetTracker)
        tracker._rate_limiters = {}

        synth = tracker._get_rate_limiter("synthesis_primary")
        agent = tracker._get_rate_limiter("agent_workhorse")

        assert BudgetTracker.BUCKET_MODELS["synthesis_primary"] == \
               BudgetTracker.BUCKET_MODELS["agent_workhorse"]
        assert synth is agent, "same upstream model must mean one shared limiter"

    def test_limiter_uses_model_rpm_not_bucket_rpm(self):
        from src.llm.budget_tracker import BudgetTracker

        tracker = BudgetTracker.__new__(BudgetTracker)
        tracker._rate_limiters = {}

        limiter = tracker._get_rate_limiter("synthesis_primary")
        model = BudgetTracker.BUCKET_MODELS["synthesis_primary"]

        assert limiter.rpm_limit == BudgetTracker.MODEL_RPM_LIMITS[model]
