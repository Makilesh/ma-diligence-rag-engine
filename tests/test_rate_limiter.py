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


class TestQuotaRefusalHandling:
    """
    A provider 429 must move the request down the ladder, not fail it.

    Local counters drift from the provider's - a restart resets the in-memory
    fallback while the provider's daily counter keeps running, and other
    processes may share the key. When that happens the tracker offers a rung the
    provider has already closed. Measured cost of not handling it: a golden-set
    run lost its first 19 answers to a model whose daily quota was already spent,
    with context quality scores as high as 0.945.
    """

    def test_recognises_provider_quota_errors(self):
        from src.llm.litellm_wrapper import is_quota_error

        class RateLimitError(Exception):
            pass

        assert is_quota_error(RateLimitError("boom"))
        assert is_quota_error(Exception("429 Too Many Requests"))
        assert is_quota_error(Exception("RESOURCE_EXHAUSTED"))
        assert is_quota_error(Exception(
            'quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"'
        ))

    def test_ignores_unrelated_errors(self):
        """A parse error must not be mistaken for exhausted quota."""
        from src.llm.litellm_wrapper import is_quota_error

        assert not is_quota_error(ValueError("invalid JSON"))
        assert not is_quota_error(ConnectionError("connection reset"))

    @staticmethod
    def _tracker(keys):
        from src.llm.budget_tracker import BudgetTracker

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = keys
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}
        return t

    @pytest.mark.asyncio
    async def test_marking_exhausted_skips_that_rung(self):
        """After a refusal, selection must not return the same rung again."""
        from src.llm.model_registry import SYNTHESIS_LADDER

        t = self._tracker(["k1"])
        first = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")

        await t.mark_slot_exhausted(first.key_index, first.model)

        second = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")
        assert second.model != first.model, (
            "a rung the provider refused must not be offered again"
        )

    @pytest.mark.asyncio
    async def test_marking_exhausted_prefers_another_key_first(self):
        """With a spare key, a refusal should rotate before downgrading."""
        from src.llm.model_registry import SYNTHESIS_LADDER

        t = self._tracker(["k1", "k2"])
        first = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")
        await t.mark_slot_exhausted(first.key_index, first.model)

        second = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")
        assert second.model == first.model
        assert second.key_index != first.key_index

    @pytest.mark.asyncio
    async def test_marking_local_is_a_noop(self):
        """The local model has no provider quota to exhaust."""
        from src.llm.model_registry import LOCAL_MODEL

        t = self._tracker([])
        await t.mark_slot_exhausted(-1, LOCAL_MODEL)
        assert t._mock_budgets == {}


class TestInvalidKeyHandling:
    """
    A rejected credential must retire the key, not fail the request.

    Distinct from quota exhaustion: an exhausted key recovers at the daily
    reset, whereas a bad key is unusable for every model until someone fixes
    the config. Measured cost of not handling it: a run with four configured
    keys lost 16 of 35 questions to one invalid key.
    """

    def test_recognises_credential_rejections(self):
        from src.llm.litellm_wrapper import is_auth_error

        class AuthenticationError(Exception):
            pass

        assert is_auth_error(AuthenticationError("bad key"))
        assert is_auth_error(Exception("API key not valid"))
        assert is_auth_error(Exception('reason: "ACCESS_TOKEN_TYPE_UNSUPPORTED"'))
        assert is_auth_error(Exception("PERMISSION_DENIED"))

    def test_auth_and_quota_errors_are_distinguished(self):
        """Confusing the two would retire a key that merely ran out for the day."""
        from src.llm.litellm_wrapper import is_auth_error, is_quota_error

        quota = Exception("429 RESOURCE_EXHAUSTED")
        auth = Exception("API key not valid")

        assert is_quota_error(quota) and not is_auth_error(quota)
        assert is_auth_error(auth) and not is_quota_error(auth)

    @staticmethod
    def _tracker(keys):
        from src.llm.budget_tracker import BudgetTracker

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = list(keys)
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}
        return t

    @pytest.mark.asyncio
    async def test_retired_key_is_never_selected_again(self):
        from src.llm.model_registry import SYNTHESIS_LADDER

        t = self._tracker(["bad", "good"])
        first = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")
        assert first.key_index == 0

        t.mark_key_unusable(first.key_index)

        for _ in range(5):
            choice = await t._select_from_ladder(SYNTHESIS_LADDER, "synthesis")
            assert choice.key_index != 0, "retired key must not be offered again"

    @pytest.mark.asyncio
    async def test_retiring_preserves_indices_of_other_keys(self):
        """
        Blanking rather than removing keeps key_index stable.

        In-flight requests already hold a key_index; compacting the list would
        silently repoint those at a different credential.
        """
        t = self._tracker(["bad", "good"])
        t.mark_key_unusable(0)

        assert len(t._api_keys) == 2
        assert t._api_keys[1] == "good"

    @pytest.mark.asyncio
    async def test_all_keys_retired_falls_back_to_local(self):
        from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL

        t = self._tracker(["bad1", "bad2"])
        t.mark_key_unusable(0)
        t.mark_key_unusable(1)

        choice = await t._select_from_ladder(AGENT_LADDER, "agent")
        assert choice.model == LOCAL_MODEL

    def test_retiring_out_of_range_index_is_safe(self):
        t = self._tracker(["k"])
        t.mark_key_unusable(-1)
        t.mark_key_unusable(99)
        assert t._api_keys == ["k"]


class TestMalformedKeyRejection:
    """Fragments from a comma-split or truncated paste must not enter rotation."""

    def test_short_fragments_are_dropped(self, monkeypatch):
        from src.llm.budget_tracker import BudgetTracker

        valid = "A" * 40
        monkeypatch.setenv("GEMINI_API_KEYS", f"{valid},v_WED6ccc2rA")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        keys = BudgetTracker._load_api_keys()
        assert keys == [valid]

    def test_wellformed_keys_are_kept(self, monkeypatch):
        from src.llm.budget_tracker import BudgetTracker

        a, b = "A" * 39, "B" * 53
        monkeypatch.setenv("GEMINI_API_KEYS", f"{a},{b}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        assert BudgetTracker._load_api_keys() == [a, b]


class TestDateBindingToPostgres:
    """
    Anything bound to a DATE column must be a `date`, never an ISO string.

    asyncpg encodes parameters using the type it infers from the statement, so
    `reset_date = $1::date` demands a `datetime.date` and rejects a str with
    "'str' object has no attribute 'toordinal'". The server-side `::date` cast
    does not help — encoding happens first.

    This went unnoticed for a long time for two compounding reasons: the
    long-lived database still held reset_date as text from an older schema, so
    string binding worked there, and every other test in this file sets
    `_is_mock = True`, which skips SQL entirely. The result was a tracker that
    passed its whole suite and then failed *every query* against a database
    created fresh from the current schema. These tests exercise the SQL path
    with a recording connection so the binding types are actually asserted.
    """

    class _Conn:
        def __init__(self):
            self.calls: list[tuple[str, tuple]] = []

        async def execute(self, query, *args):
            self.calls.append((query, args))
            return "UPDATE 1"

        async def fetchrow(self, query, *args):
            self.calls.append((query, args))
            return None

    class _Acquire:
        def __init__(self, conn):
            self._conn = conn

        async def __aenter__(self):
            return self._conn

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def __init__(self, conn):
            self._conn = conn

        def acquire(self):
            return TestDateBindingToPostgres._Acquire(self._conn)

    @classmethod
    def _tracker(cls):
        from src.llm.budget_tracker import BudgetTracker

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = ["K" * 40]
        t._rate_limiters = {}
        t._is_mock = False
        conn = cls._Conn()
        t._pool = cls._Pool(conn)
        return t, conn

    @staticmethod
    def _date_params(calls):
        """
        Arguments bound into a `::date` statement, split by how they encode.

        Other parameters in the same statement (model_key, a counter) are
        legitimately str and int, so the check cannot simply reject every string
        — it looks for values that carry a date, and asserts none of them
        arrives as text.
        """
        import re
        from datetime import date

        iso = re.compile(r"\d{4}-\d{2}-\d{2}")
        dates, date_strings = [], []
        for query, args in calls:
            if "::date" not in query:
                continue
            for a in args:
                if isinstance(a, date):
                    dates.append(a)
                elif isinstance(a, str) and iso.fullmatch(a):
                    date_strings.append(a)
        return dates, date_strings

    @pytest.mark.asyncio
    async def test_consume_binds_a_date_object(self):
        t, conn = self._tracker()
        slot = t._all_slots()[0]

        await t._try_consume(slot)

        dates, date_strings = self._date_params(conn.calls)
        assert not date_strings, (
            f"ISO strings bound to a DATE column: {date_strings!r} — asyncpg "
            "raises \"'str' object has no attribute 'toordinal'\""
        )
        assert dates, "expected the daily-reset UPDATE to bind a date parameter"

    @pytest.mark.asyncio
    async def test_marking_exhausted_binds_a_date_object(self):
        from src.llm.model_registry import SYNTHESIS_LADDER, is_local

        t, conn = self._tracker()
        model = next(m for m in SYNTHESIS_LADDER if not is_local(m))

        await t.mark_slot_exhausted(0, model)

        dates, date_strings = self._date_params(conn.calls)
        assert not date_strings, (
            f"ISO strings bound to a DATE column: {date_strings!r}"
        )
        assert dates, "expected the exhaustion UPDATE to bind a date parameter"

    def test_utc_today_returns_a_date_not_a_string(self):
        from datetime import date

        from src.llm.budget_tracker import _utc_today, _utc_today_iso

        assert isinstance(_utc_today(), date)
        assert isinstance(_utc_today_iso(), str)
        assert _utc_today_iso() == _utc_today().isoformat()
