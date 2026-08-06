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


class TestModelRegistry:
    """
    The registry is the single source of truth for provider quotas.

    Both quota bugs in this codebase were arithmetic errors over limits that
    lived in three different places. These tests keep the routing ladders
    honest against the one table that now holds them.
    """

    def test_every_laddered_model_has_declared_limits(self):
        """A model with no registry entry silently gets default quotas."""
        from src.llm.model_registry import (
            AGENT_LADDER, SYNTHESIS_LADDER, MODEL_LIMITS,
        )

        for model in set(SYNTHESIS_LADDER + AGENT_LADDER):
            assert model in MODEL_LIMITS, (
                f"{model} is routed to but has no declared quota, so its spend "
                f"is metered against guessed defaults."
            )

    def test_slot_allowance_never_exceeds_provider_cap(self):
        """
        Per-slot daily allowance must fit inside the provider's RPD.

        This is the structural replacement for the old per-bucket allocation,
        where two buckets each claiming 480 allowed 960 against a 500 cap.
        """
        from src.llm.model_registry import MODEL_LIMITS, usable_rpd

        for model, limits in MODEL_LIMITS.items():
            assert usable_rpd(model) <= limits.rpd, (
                f"{model}: allowance {usable_rpd(model)} exceeds cap {limits.rpd}"
            )

    def test_agent_ladder_excludes_scarce_reasoning_models(self):
        """
        Agent traffic must never land on a 20 RPD reasoning model.

        Agents spend ~4 calls per query, so a 20 RPD model would be drained by
        five queries — and would then be unavailable to synthesis, which is the
        one place reasoning quality reaches the user. This is the "keep the best
        for the best" guarantee, in executable form.
        """
        from src.llm.model_registry import AGENT_LADDER, MODEL_LIMITS, is_local

        for model in AGENT_LADDER:
            if is_local(model):
                continue
            limits = MODEL_LIMITS[model]
            assert not limits.reasoning, (
                f"{model} is a reasoning model but sits on the agent ladder; "
                f"its {limits.rpd} RPD would be consumed by agent traffic."
            )

    def test_synthesis_ladder_is_ordered_by_capability_then_volume(self):
        """Reasoning models must all precede Lite models on the synthesis ladder."""
        from src.llm.model_registry import SYNTHESIS_LADDER, MODEL_LIMITS, is_local

        cloud = [m for m in SYNTHESIS_LADDER if not is_local(m)]
        reasoning_flags = [MODEL_LIMITS[m].reasoning for m in cloud]
        assert reasoning_flags == sorted(reasoning_flags, reverse=True), (
            "Synthesis ladder must exhaust reasoning models before Lite ones"
        )

    def test_local_model_is_the_last_resort_on_both_ladders(self):
        """Cloud capacity must be tried before degrading to the local model."""
        from src.llm.model_registry import AGENT_LADDER, SYNTHESIS_LADDER, LOCAL_MODEL

        for ladder in (AGENT_LADDER, SYNTHESIS_LADDER):
            assert ladder[-1] == LOCAL_MODEL
            assert ladder.count(LOCAL_MODEL) == 1


class TestKeyRotation:
    """Multiple API keys must multiply capacity, not just sit unused."""

    @staticmethod
    def _tracker(keys: list[str]):
        from src.llm.budget_tracker import BudgetTracker

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = keys
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}
        return t

    def test_slots_cover_every_key_and_model(self):
        """Each key gets independent accounting for each cloud model."""
        from src.llm.model_registry import (
            AGENT_LADDER, SYNTHESIS_LADDER, is_local,
        )

        t = self._tracker(["k1", "k2", "k3"])
        slots = t._all_slots()
        cloud_models = {m for m in SYNTHESIS_LADDER + AGENT_LADDER if not is_local(m)}

        assert len(slots) == 3 * len(cloud_models)
        assert len(set(slots)) == len(slots), "slot identifiers must be unique"

    def test_different_keys_get_separate_rate_limiters(self):
        """
        RPM is metered per key, so two keys on one model have separate windows.

        Sharing a limiter across keys would throttle to a single key's quota and
        waste the capacity the extra keys were added for.
        """
        t = self._tracker(["k1", "k2"])
        model = "gemini/gemini-3.5-flash-lite"

        a = t._get_rate_limiter(t._slot(0, model))
        b = t._get_rate_limiter(t._slot(1, model))
        assert a is not b

    def test_same_key_and_model_share_one_limiter(self):
        """Conversely, one key+model pair must not get two independent windows."""
        t = self._tracker(["k1"])
        model = "gemini/gemini-3.5-flash-lite"

        assert t._get_rate_limiter(t._slot(0, model)) is t._get_rate_limiter(
            t._slot(0, model)
        )

    def test_limiter_rpm_comes_from_the_registry(self):
        """Client-side RPM must equal the provider's, per model."""
        from src.llm.model_registry import MODEL_LIMITS

        t = self._tracker(["k1"])
        for model in ("gemini/gemini-3.5-flash-lite", "gemini/gemini-3.6-flash"):
            limiter = t._get_rate_limiter(t._slot(0, model))
            assert limiter.rpm_limit == MODEL_LIMITS[model].rpm

    @pytest.mark.asyncio
    async def test_exhausting_one_key_rotates_before_downgrading_model(self):
        """
        The core routing guarantee.

        When key 1's quota on the best model is spent, the next request must go
        to key 2 on that SAME model — not to a weaker model on key 1. Cycling
        models within a key would degrade answer quality while premium quota sat
        unused on another key.
        """
        from src.llm.model_registry import SYNTHESIS_LADDER, usable_rpd

        t = self._tracker(["k1", "k2"])
        best = SYNTHESIS_LADDER[0]

        # Drain key 0 on the best model only.
        for _ in range(usable_rpd(best)):
            await t._try_consume(t._slot(0, best))

        choice = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")

        assert choice.model == best, "must stay on the best model while a key remains"
        assert choice.key_index == 1, "must rotate to the second key"
        assert choice.api_key == "k2"

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_downgrades_to_next_model(self):
        """Only when every key is spent on a model does quality step down."""
        from src.llm.model_registry import SYNTHESIS_LADDER, usable_rpd

        t = self._tracker(["k1", "k2"])
        best = SYNTHESIS_LADDER[0]

        for key_index in (0, 1):
            for _ in range(usable_rpd(best)):
                await t._try_consume(t._slot(key_index, best))

        choice = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")
        assert choice.model == SYNTHESIS_LADDER[1]

    @pytest.mark.asyncio
    async def test_no_keys_configured_falls_back_to_local(self):
        """With no credentials the pipeline still runs, entirely on Ollama."""
        from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL

        t = self._tracker([])
        choice = await t._select_from_ladder(AGENT_LADDER, "agent")

        assert choice.model == LOCAL_MODEL
        assert choice.api_key is None
        assert choice.key_index == -1
