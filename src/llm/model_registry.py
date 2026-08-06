"""
Provider model registry — the single source of truth for model capabilities
and the quotas the provider actually enforces.

WHY THIS EXISTS
---------------
Limits used to be scattered as literals across budget_tracker (RPM here, RPD
there, a model string in a third place). That is how the two quota bugs in this
codebase happened: separate rate limiters summing past a shared model's RPM, and
per-bucket daily allowances summing to 960 against a 500 RPD cap. Both were
arithmetic errors over numbers nobody could see side by side. Keeping every
limit in one table makes those errors visible and testable.

THE SHAPE OF THE FREE TIER
--------------------------
The numbers below are the free-tier quotas and they are lopsided in a way that
dictates the whole routing strategy:

    model                    RPM     TPM      RPD
    gemini-3.6-flash           5    250K       20
    gemini-3.5-flash           5    250K       20
    gemini-3-flash             5    250K       20
    gemini-2.5-flash           5    250K       20
    gemini-3.5-flash-lite     15    250K      500     <- the only one with volume
    gemini-2.5-flash-lite     10    250K       20

Exactly one model can sustain real traffic. Every reasoning-grade model is
capped at 20 requests/day. A pipeline that spends ~4 agent calls and 1 synthesis
call per query therefore cannot put agent traffic on a reasoning model at all —
five queries would exhaust it. This is why routing is split into two ladders:

  * AGENT_LADDER   — high volume, low reasoning demand (query classification,
                     rewriting, quality assessment, validation). Must sit on the
                     500 RPD lite model.
  * SYNTHESIS_LADDER — one call per query, and the only place where reasoning
                     quality shows up in the output. Gets the good models, and
                     spills to lite when their 20 RPD each is spent.

With a single key that is ~80 premium syntheses/day (4 reasoning models x 20)
before quality degrades, and ~500 agent calls. Adding keys multiplies both,
which is what BudgetTracker's key rotation is for.

TPM IS DECLARED BUT NOT ENFORCED
--------------------------------
Tokens-per-minute is recorded here for completeness, but nothing meters it yet:
enforcing it needs per-call token accounting that the pipeline does not do. At
current traffic RPM binds long before TPM (15 requests/min against a 250K TPM
ceiling implies ~16K tokens per request before TPM could bind, well above the
~8K these prompts run at), so this is a real but presently inactive gap. It is
recorded rather than left implicit so it does not become a third quota surprise.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLimits:
    """Provider-enforced quotas for a single upstream model."""

    rpm: int          # requests per minute
    tpm: int          # tokens per minute — declared, not currently enforced
    rpd: int          # requests per day
    reasoning: bool   # True for full Flash models, False for Lite variants


# Provider free-tier quotas, exactly as reported by the Google AI Studio console.
# Change these ONLY to match the provider — never to make an allocation fit.
MODEL_LIMITS: dict[str, ModelLimits] = {
    "gemini/gemini-3.6-flash":      ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-3.5-flash":      ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-3-flash":        ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-2.5-flash":      ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-3.5-flash-lite": ModelLimits(rpm=15, tpm=250_000, rpd=500, reasoning=False),
    "gemini/gemini-2.5-flash-lite": ModelLimits(rpm=10, tpm=250_000, rpd=20,  reasoning=False),
    # Local fallback. Not provider-metered; the numbers just mean "effectively
    # unlimited" so the same accounting code path works without special-casing.
    "ollama/qwen2.5:14b":           ModelLimits(rpm=1000, tpm=10**9, rpd=10**6, reasoning=True),
}

LOCAL_MODEL = "ollama/qwen2.5:14b"

# Fraction of each provider quota we allow ourselves to spend. The remainder
# absorbs clock skew between our day boundary and the provider's, and in-flight
# requests that were counted upstream but not yet locally.
SAFETY_MARGIN = 0.95

# Synthesis: one call per query, and the only agent whose reasoning quality is
# visible in the answer. Ordered best-first; each rung is worth ~20 requests/day
# per key, then it falls through. The lite model is last so synthesis degrades
# gracefully rather than failing once the reasoning models are spent.
SYNTHESIS_LADDER: list[str] = [
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3-flash",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-3.5-flash-lite",
    LOCAL_MODEL,
]

# Agents: ~4 calls per query doing classification, rewriting, assessment and
# validation — structured tasks that do not need frontier reasoning. Volume is
# the binding constraint, so this ladder is ordered by RPD, not by capability.
# Reasoning models are deliberately absent: at 20 RPD they would be drained by
# five queries and then be unavailable to synthesis, which actually needs them.
AGENT_LADDER: list[str] = [
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-2.5-flash-lite",
    LOCAL_MODEL,
]


def usable_rpd(model: str) -> int:
    """Daily request allowance after the safety margin."""
    limits = MODEL_LIMITS.get(model)
    if limits is None:
        return 0
    return max(1, int(limits.rpd * SAFETY_MARGIN))


def rpm_for(model: str) -> int:
    """Requests-per-minute ceiling for a model."""
    limits = MODEL_LIMITS.get(model)
    return limits.rpm if limits else 5


def is_local(model: str) -> bool:
    """True for models that cost nothing and bypass key rotation."""
    return model.startswith("ollama/")
