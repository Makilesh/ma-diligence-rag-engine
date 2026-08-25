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


class TestRequestTimeouts:
    """
    Every upstream call must carry an explicit timeout.

    LiteLLM's default is 600s, which in practice is no timeout at all. When
    Gemini returned 503 "This model is currently experiencing high demand", each
    synthesis call hung for ~125 seconds before the provider dropped it, and the
    model ladder could not descend to a healthy rung until it did. A third of
    calls failing that way turned a 25-second query into a two-minute one while
    a working fallback model sat idle.

    These assert the kwargs actually reach litellm, because the omission is
    invisible: nothing errors, calls simply take as long as the provider takes.
    """

    @staticmethod
    def _capture(monkeypatch):
        """Replaces litellm.acompletion with a recorder; returns the call list."""
        import src.llm.litellm_wrapper as wrapper

        calls: list[dict] = []

        class _Message:
            content = '{"ok": true}'

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _Response()

        monkeypatch.setattr(wrapper.litellm, "acompletion", fake_acompletion)
        return calls

    @pytest.mark.asyncio
    async def test_structured_agent_sends_a_timeout(self, monkeypatch):
        from src.llm.litellm_wrapper import (
            STRUCTURED_TIMEOUT_SECONDS,
            call_structured_agent,
        )

        calls = self._capture(monkeypatch)
        await call_structured_agent(
            model="gemini/test",
            system_prompt="s",
            user_prompt="u",
            api_key="k",
        )

        assert calls, "no upstream call was made"
        assert calls[0].get("timeout") == STRUCTURED_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_prose_agent_sends_a_timeout(self, monkeypatch):
        from src.llm.litellm_wrapper import PROSE_TIMEOUT_SECONDS, call_prose_agent

        calls = self._capture(monkeypatch)
        await call_prose_agent(
            model="gemini/test",
            system_prompt="s",
            user_prompt="u",
            api_key="k",
        )

        assert calls, "no upstream call was made"
        assert calls[0].get("timeout") == PROSE_TIMEOUT_SECONDS

    def test_timeouts_leave_headroom_over_healthy_latency(self):
        """
        The ceilings must not clip calls that would have succeeded.

        Healthy structured calls finish in 1-3s and healthy synthesis in 10-30s
        (measured across the 41-question evaluation). A timeout tight enough to
        cut those would trade a slow answer for no answer, which is the wrong
        direction for a tool whose whole premise is not guessing.
        """
        from src.llm.litellm_wrapper import (
            PROSE_TIMEOUT_SECONDS,
            STRUCTURED_TIMEOUT_SECONDS,
        )

        assert STRUCTURED_TIMEOUT_SECONDS >= 30
        assert PROSE_TIMEOUT_SECONDS >= 60
        assert PROSE_TIMEOUT_SECONDS > STRUCTURED_TIMEOUT_SECONDS


class TestServiceUnavailableHandling:
    """
    A 503 is not a quota refusal and must not be treated as one.

    Observed live: gemini-3.6-flash returned "This model is currently
    experiencing high demand" on roughly a third of synthesis calls while
    gemini-3.5-flash answered every request in ~1.2s. Two distinct mistakes are
    possible here and both are expensive:

      * treating it as a transport error retries the same dead model three times
        with backoff before anything else is tried;
      * treating it as a quota refusal debits the slot and retires the best
        synthesis model for the rest of the day over a blip that clears in
        minutes.

    The right response is neither: skip that MODEL briefly, across all keys,
    without touching quota.
    """

    @staticmethod
    def _service_error():
        class ServiceUnavailableError(Exception):
            pass

        return ServiceUnavailableError(
            'litellm.ServiceUnavailableError: GeminiException - {"error": '
            '{"code": 503, "message": "This model is currently experiencing '
            'high demand.", "status": "UNAVAILABLE"}}'
        )

    def test_recognises_provider_unavailability(self):
        from src.llm.litellm_wrapper import is_service_unavailable

        assert is_service_unavailable(self._service_error())

    def test_not_confused_with_quota_or_auth(self):
        """The three classifications must stay disjoint on a real 503."""
        from src.llm.litellm_wrapper import (
            is_auth_error,
            is_quota_error,
            is_service_unavailable,
        )

        err = self._service_error()
        assert is_service_unavailable(err)
        assert not is_quota_error(err), (
            "a 503 classified as quota would debit the slot and retire the "
            "model for the rest of the day"
        )
        assert not is_auth_error(err)

    def test_quota_errors_are_not_treated_as_unavailability(self):
        from src.llm.litellm_wrapper import is_quota_error, is_service_unavailable

        class RateLimitError(Exception):
            pass

        err = RateLimitError("litellm.RateLimitError: 429 RESOURCE_EXHAUSTED")
        assert is_quota_error(err)
        assert not is_service_unavailable(err)

    def test_cooldown_removes_the_model_from_selection(self):
        from src.llm.budget_tracker import BudgetTracker
        from src.llm.model_registry import SYNTHESIS_LADDER, is_local

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = ["k" * 40]
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}

        model = next(m for m in SYNTHESIS_LADDER if not is_local(m))
        assert model not in t._models_in_cooldown()

        t.skip_model_for_request(model)
        assert model in t._models_in_cooldown()

    def test_cooldown_expires(self, monkeypatch):
        """A model must return to service once the spike passes."""
        import src.llm.budget_tracker as bt
        from src.llm.model_registry import SYNTHESIS_LADDER, is_local

        t = bt.BudgetTracker.__new__(bt.BudgetTracker)
        t._api_keys = ["k" * 40]
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}

        model = next(m for m in SYNTHESIS_LADDER if not is_local(m))
        t.skip_model_for_request(model)
        assert model in t._models_in_cooldown()

        # Jump past the cooldown window rather than sleeping through it.
        t._model_cooldowns[model] -= bt.SERVICE_COOLDOWN_SECONDS + 1
        assert model not in t._models_in_cooldown()

    def test_cooldown_does_not_debit_quota(self):
        """
        The whole point: a transient outage must not cost the day's capacity.
        """
        from src.llm.budget_tracker import BudgetTracker
        from src.llm.model_registry import SYNTHESIS_LADDER, is_local

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = ["k" * 40]
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}

        model = next(m for m in SYNTHESIS_LADDER if not is_local(m))
        t.skip_model_for_request(model)

        assert t._mock_budgets == {}, (
            "skipping an unavailable model must not touch budget accounting"
        )

    def test_local_model_is_never_put_in_cooldown(self):
        """The local model is the last resort; it must always remain selectable."""
        from src.llm.budget_tracker import BudgetTracker
        from src.llm.model_registry import LOCAL_MODEL

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = []
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}

        t.skip_model_for_request(LOCAL_MODEL)
        assert LOCAL_MODEL not in t._models_in_cooldown()


class TestBudgetStatusReporting:
    """
    The reported budget must agree with the routing decision.

    The daily reset is applied lazily, inside `_try_consume`, on a slot's next
    call. `_budget_available` already accounts for that — a stale reset_date
    means a full allowance — but `get_budget_status` read `used_today` raw, so
    after a quota reset the sidebar reported "26/95 left" while the router
    considered all 95 available. A status panel that contradicts the thing it
    is reporting on is worse than no panel.
    """

    @staticmethod
    def _tracker(used_today: int, reset_date: str):
        from src.llm.budget_tracker import BudgetTracker, ModelBudget

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = ["k" * 40]
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}

        async def fake_load(slot):
            return ModelBudget(
                model_key=slot,
                daily_limit=t._slot_limit(slot),
                used_today=used_today,
                reset_date=reset_date,
            )

        t._load_budget = fake_load
        return t

    @pytest.mark.asyncio
    async def test_todays_usage_is_reported(self):
        from src.llm.budget_tracker import _utc_today_iso

        t = self._tracker(used_today=5, reset_date=_utc_today_iso())
        status = await t.get_budget_status()

        assert status, "no models reported"
        entry = next(iter(status.values()))
        assert entry["used"] == 5

    @pytest.mark.asyncio
    async def test_stale_counter_reports_as_unused(self):
        """Yesterday's spend must not be shown as today's."""
        from src.llm.budget_tracker import _utc_today_iso

        t = self._tracker(used_today=19, reset_date="2000-01-01")
        status = await t.get_budget_status()

        entry = next(iter(status.values()))
        assert entry["used"] == 0, (
            "a counter from a previous day was reported as today's usage"
        )
        assert entry["remaining"] == entry["limit"]
        assert entry["reset_date"] == _utc_today_iso()

    @pytest.mark.asyncio
    async def test_display_agrees_with_availability(self):
        """
        The invariant that was violated: if the router says capacity exists,
        the panel must not say it is spent.
        """
        t = self._tracker(used_today=19, reset_date="2000-01-01")
        slot = t._all_slots()[0]

        assert await t._budget_available(slot) is True
        status = await t.get_budget_status()
        assert next(iter(status.values()))["remaining"] > 0


class TestLadderModelsExist:
    """
    Every rung must be a model the provider actually serves.

    Laddered models that returned 404 "not found ... or is not supported for
    generateContent": `gemini-3-flash`, `gemini-2.5-flash` and
    `gemini-2.5-flash-lite`. Half the fallback ladder was unreachable, so once
    the working rungs were spent or failing the router descended into models
    that could only 404 — burning an attempt and a timeout on each.

    The three failed for two different reasons, and the distinction matters:

      * `gemini-3-flash` was a WRONG NAME. The console calls it "Gemini 3
        Flash"; the servable id is `gemini-3-flash-preview`. It works, and is
        back on the ladder under its real id.
      * `gemini-2.5-flash` and `gemini-2.5-flash-lite` are genuinely
        unreachable on this key. Both appear in the rate-limit console *and* in
        the ListModels catalogue, and both 404 on every generateContent call.
        A catalogue entry is not an availability guarantee.

    This cannot assert live availability without a network call, so it pins the
    ladders to the set that was probed end-to-end. Changing that set should mean
    re-probing, not editing this list until it passes.
    """

    PROBED_AVAILABLE = {
        # Available on at least one configured key. gemini-2.5-flash and
        # gemini-2.5-flash-lite answer on 2 of 5 and return 404 "no longer
        # available to new users" on the rest; the router retires those slots
        # individually rather than dropping the model.
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-3.7-flash",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.5-flash",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
    }

    PROBED_MISSING = {
        "gemini/gemini-3-flash",   # wrong id — the real one is -preview
        "gemini/gemini-2-flash",   # console shows 0/0/0: no free-tier access
        "gemini/gemini-2.5-pro",   # console shows 0/0/0
        "gemini/gemini-3.1-pro",   # console shows 0/0/0
    }

    def test_no_ladder_contains_a_known_missing_model(self):
        from src.llm.model_registry import AGENT_LADDER, SYNTHESIS_LADDER

        for ladder_name, ladder in (("synthesis", SYNTHESIS_LADDER),
                                    ("agent", AGENT_LADDER)):
            offenders = self.PROBED_MISSING.intersection(ladder)
            assert not offenders, (
                f"{ladder_name} ladder routes to {sorted(offenders)}, which the "
                f"provider returns 404 for"
            )

    def test_registry_declares_no_known_missing_model(self):
        from src.llm.model_registry import MODEL_LIMITS

        offenders = self.PROBED_MISSING.intersection(MODEL_LIMITS)
        assert not offenders, (
            f"MODEL_LIMITS declares quotas for nonexistent models: {sorted(offenders)}"
        )

    def test_every_cloud_rung_was_probed_available(self):
        from src.llm.model_registry import AGENT_LADDER, SYNTHESIS_LADDER, is_local

        for model in set(SYNTHESIS_LADDER + AGENT_LADDER):
            if is_local(model):
                continue
            assert model in self.PROBED_AVAILABLE, (
                f"{model} is on a ladder but has not been probed against the "
                f"provider — add it to PROBED_AVAILABLE only after confirming it "
                f"answers, not because the name looks plausible"
            )

    def test_each_ladder_still_has_a_working_cloud_rung(self):
        """Removing the phantoms must not leave a ladder that is local-only."""
        from src.llm.model_registry import AGENT_LADDER, SYNTHESIS_LADDER, is_local

        for name, ladder in (("synthesis", SYNTHESIS_LADDER), ("agent", AGENT_LADDER)):
            cloud = [m for m in ladder if not is_local(m)]
            assert len(cloud) >= 2, (
                f"{name} ladder has {len(cloud)} cloud rungs; it needs a real "
                f"fallback, not just a top choice and the local model"
            )

    def test_console_quotas_are_recorded_faithfully(self):
        """
        Declared limits must match the provider console, not a convenient guess.

        Taken from the Google AI Studio rate-limit page: the reasoning tier is
        5 RPM / 250K TPM / 20 RPD and the volume tier is 15 RPM / 250K TPM /
        500 RPD. These numbers drive every routing and budget decision, so a
        drift here silently mis-meters the whole pipeline.
        """
        from src.llm.model_registry import MODEL_LIMITS

        reasoning = {
            "gemini/gemini-3.7-flash", "gemini/gemini-3.6-flash",
            "gemini/gemini-3.5-flash", "gemini/gemini-3-flash-preview",
        }
        volume = {"gemini/gemini-3.5-flash-lite", "gemini/gemini-3.1-flash-lite"}

        for model in reasoning:
            limits = MODEL_LIMITS[model]
            assert (limits.rpm, limits.tpm, limits.rpd) == (5, 250_000, 20), model
            assert limits.reasoning is True, model

        for model in volume:
            limits = MODEL_LIMITS[model]
            assert (limits.rpm, limits.tpm, limits.rpd) == (15, 250_000, 500), model
            assert limits.reasoning is False, model

    def test_newest_reasoning_model_leads_synthesis(self):
        """
        The synthesis ladder must start at the newest reasoning model.

        gemini-3.7-flash was released after this ladder was written and sat
        unused while synthesis ran on 3.6. Nothing fails when a new model is
        missed — the pipeline just quietly stops using the best thing available.
        """
        from src.llm.model_registry import SYNTHESIS_LADDER

        assert SYNTHESIS_LADDER[0] == "gemini/gemini-3.7-flash"


class TestUnavailableModelIsNotRetried:
    """
    A 503 must reach the ladder immediately, not after three backoffs.

    Retrying the same model is only useful when the failure is local or
    transient to the request. "This model is currently experiencing high demand"
    is neither — it is true for every key and every retry. Measured live:
    gemini-3.7-flash cost three attempts and ~48s of backoff before the ladder
    was allowed to try gemini-3.6-flash, which answered first time.
    """

    @pytest.mark.asyncio
    async def test_prose_agent_raises_on_first_503(self, monkeypatch):
        import src.llm.litellm_wrapper as wrapper

        calls = {"n": 0}

        class ServiceUnavailableError(Exception):
            pass

        async def always_503(**kwargs):
            calls["n"] += 1
            raise ServiceUnavailableError(
                '503 "This model is currently experiencing high demand."'
            )

        monkeypatch.setattr(wrapper.litellm, "acompletion", always_503)

        with pytest.raises(Exception):
            await wrapper.call_prose_agent(
                model="gemini/test", system_prompt="s", user_prompt="u", api_key="k"
            )

        assert calls["n"] == 1, (
            f"made {calls['n']} attempts against a model the provider said is "
            f"unavailable; it should bail after one so the ladder can descend"
        )

    @pytest.mark.asyncio
    async def test_ordinary_transport_errors_still_retry(self, monkeypatch):
        """The fast bail must not disable retries for genuinely transient faults."""
        import src.llm.litellm_wrapper as wrapper

        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            raise ConnectionError("connection reset by peer")

        monkeypatch.setattr(wrapper.litellm, "acompletion", flaky)
        monkeypatch.setattr(wrapper, "RETRY_BACKOFF_SECONDS", 0.0)

        with pytest.raises(Exception):
            await wrapper.call_prose_agent(
                model="gemini/test", system_prompt="s", user_prompt="u", api_key="k"
            )

        assert calls["n"] == wrapper.MAX_RETRIES


class TestPerKeyModelAvailability:
    """
    A model can exist for one credential and not another.

    `gemini-2.5-flash` and `gemini-2.5-flash-lite` are closed to new sign-ups:
    on two of five configured keys they answer normally, and on the other three
    they return 404 "This model is no longer available to new users". Google
    grandfathers older keys.

    Every other failure mode retires the wrong thing here. A 429 retires the
    pair for a day, a 503 retires the model for everyone briefly, an auth error
    retires the key entirely — none of them mean "this key may never use this
    model", which is permanent and scoped to one slot.
    """

    @staticmethod
    def _revoked():
        class NotFoundError(Exception):
            pass

        return NotFoundError(
            'litellm.NotFoundError: GeminiException - {"error": {"code": 404, '
            '"message": "This model models/gemini-2.5-flash is no longer '
            'available to new users", "status": "NOT_FOUND"}}'
        )

    def test_recognises_a_per_key_revocation(self):
        from src.llm.litellm_wrapper import is_model_unavailable_for_key

        assert is_model_unavailable_for_key(self._revoked())

    def test_distinct_from_quota_service_and_auth(self):
        from src.llm.litellm_wrapper import (
            is_auth_error, is_model_unavailable_for_key,
            is_quota_error, is_service_unavailable,
        )

        err = self._revoked()
        assert is_model_unavailable_for_key(err)
        assert not is_quota_error(err), "would retire the pair for only a day"
        assert not is_auth_error(err), "would retire a perfectly good key"
        assert not is_service_unavailable(err), (
            "would retire the model for every key, including the ones it works on"
        )

    def test_a_healthy_503_is_not_a_revocation(self):
        from src.llm.litellm_wrapper import is_model_unavailable_for_key

        class ServiceUnavailableError(Exception):
            pass

        assert not is_model_unavailable_for_key(
            ServiceUnavailableError('503 "currently experiencing high demand"')
        )

    def _tracker(self):
        from src.llm.budget_tracker import BudgetTracker

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = ["k" * 40, "j" * 40]
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}
        return t

    def test_retiring_a_slot_leaves_the_other_key_usable(self):
        """The whole point: one key losing a model must not cost the other one."""
        t = self._tracker()
        model = "gemini/gemini-2.5-flash"

        t.mark_slot_unavailable(0, model)

        assert t._slot_is_unavailable(t._slot(0, model))
        assert not t._slot_is_unavailable(t._slot(1, model))

    def test_retiring_a_slot_leaves_other_models_on_that_key_usable(self):
        t = self._tracker()

        t.mark_slot_unavailable(0, "gemini/gemini-2.5-flash")

        assert not t._slot_is_unavailable(t._slot(0, "gemini/gemini-3.6-flash"))

    @pytest.mark.asyncio
    async def test_selection_skips_a_retired_slot(self):
        from src.llm.model_registry import SYNTHESIS_LADDER

        t = self._tracker()
        top = SYNTHESIS_LADDER[0]

        # Retire the top model on both keys; selection must move down the ladder.
        t.mark_slot_unavailable(0, top)
        t.mark_slot_unavailable(1, top)

        choice = await t.get_model_for_synthesis()
        assert choice.model != top

    def test_local_model_is_never_retired(self):
        from src.llm.model_registry import LOCAL_MODEL

        t = self._tracker()
        t.mark_slot_unavailable(0, LOCAL_MODEL)
        assert not t._slot_is_unavailable(t._slot(0, LOCAL_MODEL))


class TestTimeoutIsTreatedAsUnavailability:
    """
    A model that burned its full deadline must not be retried.

    `gemini-3.7-flash` began timing out consistently. A timeout matched none of
    the classifiers, so it fell through to the generic retry path: three
    attempts at 120 seconds each. The caller's 300-second budget expired before
    the ladder ever reached a healthy model, so the query returned nothing at
    all rather than an answer from the next rung down.

    Retrying a model that just consumed 120 seconds and produced nothing is the
    single most expensive thing this pipeline can do.
    """

    @staticmethod
    def _timeout():
        class Timeout(Exception):
            pass

        return Timeout("litellm.Timeout: Connection timed out. Timeout passed=120.0")

    def test_recognises_a_timeout(self):
        from src.llm.litellm_wrapper import is_timeout_error

        assert is_timeout_error(self._timeout())
        assert is_timeout_error(TimeoutError("timed out"))

    def test_not_confused_with_other_failures(self):
        from src.llm.litellm_wrapper import (
            is_auth_error, is_model_unavailable_for_key,
            is_quota_error, is_timeout_error,
        )

        err = self._timeout()
        assert not is_quota_error(err), "would retire the slot for the whole day"
        assert not is_auth_error(err), "would retire a working key"
        assert not is_model_unavailable_for_key(err), "would retire the slot forever"

        class RateLimitError(Exception):
            pass

        assert not is_timeout_error(RateLimitError("429 RESOURCE_EXHAUSTED"))

    @pytest.mark.asyncio
    async def test_prose_agent_does_not_retry_a_timeout(self, monkeypatch):
        import src.llm.litellm_wrapper as wrapper

        calls = {"n": 0}

        class Timeout(Exception):
            pass

        async def always_timeout(**kwargs):
            calls["n"] += 1
            raise Timeout("litellm.Timeout: Connection timed out. Timeout passed=120.0")

        monkeypatch.setattr(wrapper.litellm, "acompletion", always_timeout)

        with pytest.raises(Exception):
            await wrapper.call_prose_agent(
                model="gemini/test", system_prompt="s", user_prompt="u", api_key="k"
            )

        assert calls["n"] == 1, (
            f"made {calls['n']} attempts; at {wrapper.PROSE_TIMEOUT_SECONDS}s each "
            f"that is {calls['n'] * wrapper.PROSE_TIMEOUT_SECONDS}s spent before the "
            f"ladder may try a healthy model"
        )

    def test_prose_timeout_budget_fits_a_single_request(self):
        """One timeout must leave room to try another rung inside a 300s budget."""
        from src.llm.litellm_wrapper import PROSE_TIMEOUT_SECONDS

        assert PROSE_TIMEOUT_SECONDS * 2 < 300, (
            "a single timeout plus one fallback attempt must fit inside the "
            "client's request budget"
        )


class TestEscalatingModelCooldown:
    """
    Repeated failure on one model must get progressively cheaper.

    A fixed cooldown is wrong in both directions. Too long and a model that
    recovers from a brief 503 spike sits stranded. Too short and a persistently
    sick model is re-tried on every query: when `gemini-3.7-flash` began timing
    out, a flat 90s meant each query paid the full 120s timeout before
    descending, putting a 41-question evaluation on course for 95 minutes
    instead of 30 — and spending a scarce 20-RPD slot each time to learn
    nothing.
    """

    @staticmethod
    def _tracker():
        from src.llm.budget_tracker import BudgetTracker

        t = BudgetTracker.__new__(BudgetTracker)
        t._api_keys = ["k" * 40]
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}
        return t

    def test_cooldown_doubles_on_consecutive_failures(self):
        import time

        from src.llm.budget_tracker import SERVICE_COOLDOWN_SECONDS

        t = self._tracker()
        model = "gemini/gemini-3.7-flash"

        seen = []
        for _ in range(4):
            t.skip_model_for_request(model)
            seen.append(round(t._model_cooldowns[model] - time.monotonic()))

        base = SERVICE_COOLDOWN_SECONDS
        assert seen[0] == round(base)
        for i in range(1, len(seen)):
            assert seen[i] >= seen[i - 1] * 2 - 2, f"not doubling: {seen}"

    def test_cooldown_is_capped(self):
        import time

        from src.llm.budget_tracker import MAX_SERVICE_COOLDOWN_SECONDS

        t = self._tracker()
        model = "gemini/gemini-3.7-flash"
        for _ in range(20):
            t.skip_model_for_request(model)

        remaining = t._model_cooldowns[model] - time.monotonic()
        assert remaining <= MAX_SERVICE_COOLDOWN_SECONDS + 1, (
            "an unbounded backoff would retire a model for the rest of the day"
        )

    def test_one_success_clears_the_streak(self):
        """A recovered model must return to rotation immediately, not in 30 minutes."""
        import time

        from src.llm.budget_tracker import SERVICE_COOLDOWN_SECONDS

        t = self._tracker()
        model = "gemini/gemini-3.7-flash"
        for _ in range(5):
            t.skip_model_for_request(model)
        assert model in t._models_in_cooldown()

        t.note_model_healthy(model)
        assert model not in t._models_in_cooldown()

        # And the next failure starts from the base delay, not the escalated one.
        t.skip_model_for_request(model)
        assert round(t._model_cooldowns[model] - time.monotonic()) == round(
            SERVICE_COOLDOWN_SECONDS
        )

    def test_failure_streaks_are_tracked_per_model(self):
        """One sick model must not push a healthy one into backoff."""
        import time

        from src.llm.budget_tracker import SERVICE_COOLDOWN_SECONDS

        t = self._tracker()
        sick, healthy = "gemini/gemini-3.7-flash", "gemini/gemini-3.6-flash"

        for _ in range(4):
            t.skip_model_for_request(sick)
        t.skip_model_for_request(healthy)

        assert round(t._model_cooldowns[healthy] - time.monotonic()) == round(
            SERVICE_COOLDOWN_SECONDS
        )
        assert t._model_cooldowns[sick] > t._model_cooldowns[healthy]

    def test_note_healthy_on_an_unknown_model_is_safe(self):
        t = self._tracker()
        t.note_model_healthy("gemini/never-seen")  # must not raise
