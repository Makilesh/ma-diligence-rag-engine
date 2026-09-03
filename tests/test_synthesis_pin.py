"""
Tests for SYNTHESIS_MODEL_PIN.

The pin exists because recall on the golden set is dominated by which model
answered, not by retrieval. Measured inside a single run — one index, identical
retrieval — mean recall was 100% on gemini-3.6-flash, 93% on
gemini-3-flash-preview, 66% on gemini-3.5-flash and 50% on gemini-2.5-flash.
Comparing two retrieval configurations across runs therefore compares their
synthesis quota states unless the model is held fixed.

The property worth testing is the refusal to substitute. A pin that silently
fell back would still produce a results file, and nothing in that file would
reveal that the comparison it was made for had been invalidated.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm.budget_tracker import BudgetTracker, ModelChoice
from src.llm.model_registry import SYNTHESIS_LADDER


@pytest.fixture
def tracker():
    """A BudgetTracker instance with no I/O behind it."""
    return BudgetTracker.__new__(BudgetTracker)


@pytest.fixture(autouse=True)
def _clear_pin(monkeypatch):
    monkeypatch.delenv("SYNTHESIS_MODEL_PIN", raising=False)


class TestUnpinned:
    """Absent the env var, behaviour must be exactly the ladder as before."""

    @pytest.mark.asyncio
    async def test_uses_full_ladder(self, tracker):
        chosen = ModelChoice(
            model="gemini/gemini-3.6-flash", api_key="k", key_index=0
        )
        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=chosen)
        ) as select:
            result = await tracker.get_model_for_synthesis()

        select.assert_awaited_once_with(SYNTHESIS_LADDER, "synthesis")
        assert result.model == "gemini/gemini-3.6-flash"

    @pytest.mark.asyncio
    async def test_empty_pin_is_treated_as_unset(self, tracker, monkeypatch):
        """An env var set to whitespace is a deployment slip, not a pin."""
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", "   ")
        chosen = ModelChoice(model="gemini/gemini-3.5-flash", api_key="k", key_index=0)

        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=chosen)
        ) as select:
            await tracker.get_model_for_synthesis()

        select.assert_awaited_once_with(SYNTHESIS_LADDER, "synthesis")


class TestPinned:
    @pytest.mark.asyncio
    async def test_restricts_the_ladder_to_one_model(self, tracker, monkeypatch):
        pin = "gemini/gemini-3.6-flash"
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", pin)
        chosen = ModelChoice(model=pin, api_key="k", key_index=0)

        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=chosen)
        ) as select:
            result = await tracker.get_model_for_synthesis()

        select.assert_awaited_once_with([pin], "synthesis(pinned)")
        assert result.model == pin

    @pytest.mark.asyncio
    async def test_raises_rather_than_substituting_when_quota_is_spent(
        self, tracker, monkeypatch
    ):
        """
        The load-bearing test.

        `_select_from_ladder` falls back to the local model when no cloud slot
        has budget — correct in production, fatal here. A pinned run that
        answered on a different model would produce a results file that looks
        valid and silently invalidates the comparison it was made for.
        """
        pin = "gemini/gemini-3.6-flash"
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", pin)
        fallback = ModelChoice(model="ollama/qwen2.5:14b", api_key=None, key_index=-1)

        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=fallback)
        ), patch.object(
            tracker, "_model_has_daily_budget", new=AsyncMock(return_value=False)
        ):
            with pytest.raises(RuntimeError, match="no daily quota left"):
                await tracker.get_model_for_synthesis()

    @pytest.mark.asyncio
    async def test_waits_out_a_transient_cooldown(self, tracker, monkeypatch):
        """
        A 503 must not end a pinned run.

        The reasoning rungs 503 several times an hour on this provider, and the
        cooldown escalates to 720s. Failing on the first one would make the pin
        unusable for exactly the long measurement runs it exists to serve — so
        while daily quota remains, the pin waits and retries.
        """
        pin = "gemini/gemini-3.6-flash"
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", pin)

        cooled = ModelChoice(model="ollama/qwen2.5:14b", api_key=None, key_index=-1)
        recovered = ModelChoice(model=pin, api_key="k", key_index=0)

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        with patch.object(
            tracker,
            "_select_from_ladder",
            new=AsyncMock(side_effect=[cooled, recovered]),
        ), patch.object(
            tracker, "_model_has_daily_budget", new=AsyncMock(return_value=True)
        ), patch("src.llm.budget_tracker.asyncio.sleep", new=fake_sleep):
            result = await tracker.get_model_for_synthesis()

        assert result.model == pin, "must return the pinned model, never a substitute"
        assert slept, "must wait for the cooldown rather than failing immediately"

    @pytest.mark.asyncio
    async def test_gives_up_after_the_wait_budget(self, tracker, monkeypatch):
        """
        Waiting is bounded. A model that stays down is reported, not waited on
        forever, so a stuck run cannot masquerade as a slow one.
        """
        pin = "gemini/gemini-3.6-flash"
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", pin)
        monkeypatch.setattr("src.llm.budget_tracker.PIN_MAX_WAIT_SECONDS", 0.0)

        cooled = ModelChoice(model="ollama/qwen2.5:14b", api_key=None, key_index=-1)

        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=cooled)
        ), patch.object(
            tracker, "_model_has_daily_budget", new=AsyncMock(return_value=True)
        ):
            with pytest.raises(RuntimeError, match="provider-side unavailable"):
                await tracker.get_model_for_synthesis()

    @pytest.mark.asyncio
    async def test_unknown_model_warns_but_still_pins(self, tracker, monkeypatch):
        """
        A pin outside the ladder is honoured, with a warning.

        Refusing it outright would block the legitimate case of measuring against
        a model that is not on the production ladder at all.
        """
        pin = "gemini/some-unlisted-model"
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", pin)
        chosen = ModelChoice(model=pin, api_key="k", key_index=0)

        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=chosen)
        ) as select:
            with patch("src.llm.budget_tracker.logger") as log:
                result = await tracker.get_model_for_synthesis()

        assert result.model == pin
        select.assert_awaited_once_with([pin], "synthesis(pinned)")
        assert log.warning.called

    @pytest.mark.asyncio
    async def test_agent_ladder_is_unaffected(self, tracker, monkeypatch):
        """
        The pin covers synthesis only.

        Agent calls run on the volume ladder by design, and pinning them to a
        reasoning model would drain in five queries the quota synthesis needs.
        """
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", "gemini/gemini-3.6-flash")
        chosen = ModelChoice(
            model="gemini/gemini-3.5-flash-lite", api_key="k", key_index=0
        )

        with patch.object(
            tracker, "_select_from_ladder", new=AsyncMock(return_value=chosen)
        ) as select:
            await tracker.get_model_for_agent()

        ladder_arg = select.await_args.args[0]
        assert "gemini/gemini-3.6-flash" not in ladder_arg
