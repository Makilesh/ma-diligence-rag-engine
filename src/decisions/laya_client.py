"""
Laya decision-model client — one lazily loaded model per process.

Laya (convaiinnovations/laya, Apache-2.0) answers typed questions about a piece
of text in a single encoder forward pass: `noul` returns P(true), `choice` a
distribution over named options, `score` a position on an ordinal scale. It
generates no text, so it cannot hallucinate an answer — only be wrong about a
probability, which is measurable.

The default checkpoint is `typed-decisions`, the authors' fine-tune. The base
checkpoints are near chance zero-shot per the model card, so they are not used.
Inputs are truncated at 1024 tokens: pose questions over a chunk or a
(question, passage) pair, never a whole document.

Every caller must keep a non-Laya path. `LAYA_ENABLED=0`, a missing package or a
failed download all raise LayaUnavailable, and the caller falls back to the
rule or heuristic that ran before Laya existed.

Env:
    LAYA_ENABLED   "0" disables every Laya decision (default on).
    LAYA_MODEL     Router checkpoint key (default "typed-decisions").
    LAYA_DEVICE    "cuda" / "cpu"; default picks CUDA when available.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_MODEL = "typed-decisions"
# Requests per forward pass. 16 keeps a CPU host responsive between batches
# while still amortising the encoder over many (state, questions) pairs.
BATCH_SIZE = 16

_router = None
_load_failed = False
_load_lock = threading.Lock()
# The Router is not documented as thread-safe; to_thread callers share it.
_predict_lock = threading.Lock()


class LayaUnavailable(RuntimeError):
    """Raised when Laya is disabled or cannot be loaded; callers fall back."""


def laya_enabled() -> bool:
    """True unless LAYA_ENABLED is set to a falsy value."""
    return os.getenv("LAYA_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def laya_model() -> str:
    """The Router checkpoint key in use."""
    return os.getenv("LAYA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _device() -> str:
    configured = os.getenv("LAYA_DEVICE", "").strip()
    if configured:
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_router():
    """
    Loads the Laya Router once per process (thread-safe, lazy).

    A load failure is remembered, so a missing package or an offline host costs
    one attempt per process rather than one download attempt per query.

    Returns:
        The laya.Router.

    Raises:
        LayaUnavailable: If disabled, not installed, or failed to load.
    """
    global _router, _load_failed

    if not laya_enabled():
        raise LayaUnavailable("LAYA_ENABLED=0")
    if _router is not None:
        return _router
    if _load_failed:
        raise LayaUnavailable("Laya failed to load earlier in this process")

    with _load_lock:
        if _router is not None:
            return _router
        try:
            from laya import Router

            device = _device()
            logger.info(f"Loading Laya ({laya_model()}) on {device.upper()}")
            # max_loaded=1: only the one checkpoint is ever requested, and each
            # resident checkpoint is ~1.7GB.
            _router = Router(device=device, max_loaded=1)
            return _router
        except Exception as e:
            _load_failed = True
            raise LayaUnavailable(f"Laya could not be loaded: {e}") from e


def decide_batch(requests: list[tuple[dict, dict]]) -> list[dict[str, dict[str, Any]]]:
    """
    Answers typed questions for many states synchronously. Call via adecide_batch.

    Args:
        requests: (state, questions) pairs. `state` maps field names to text;
            `questions` maps a name to a Laya question dict whose instructions
            refer to state fields in backticks.

    Returns:
        One {question_name: answer} dict per request, in order. A `noul`
        answer carries "noul" (P(true)); a `choice` answer "choice",
        "probabilities" and "confidence".

    Raises:
        LayaUnavailable: If Laya is disabled or cannot be loaded, or inference
            fails.
    """
    if not requests:
        return []
    router = _get_router()
    batch = [{"state": s, "questions": q, "model": laya_model()} for s, q in requests]
    try:
        with _predict_lock:
            outputs = router.predict_batch(batch, batch_size=BATCH_SIZE)
    except Exception as e:
        raise LayaUnavailable(f"Laya inference failed: {e}") from e
    return [out.get("answers", {}) for out in outputs]


async def adecide_batch(requests: list[tuple[dict, dict]]) -> list[dict[str, dict[str, Any]]]:
    """
    Answers typed questions off the event loop.

    Args:
        requests: (state, questions) pairs — see decide_batch.

    Returns:
        One {question_name: answer} dict per request, in order.

    Raises:
        LayaUnavailable: If Laya is disabled or cannot be loaded.
    """
    if not requests:
        return []
    return await asyncio.to_thread(decide_batch, requests)


def noul(answer: dict[str, Any]) -> float:
    """P(true) from a `noul` answer."""
    return float(answer.get("noul", 0.0))


def loaded_model() -> str | None:
    """Checkpoint key if Laya has been loaded in this process, else None."""
    return laya_model() if _router is not None else None


async def warm_laya() -> float | None:
    """
    Loads and exercises Laya so the first query or upload does not pay for it.

    Mirrors warm_models / warm_nli_model: failures are logged and swallowed — a
    warmup must never stop the API from starting.

    Returns:
        Seconds taken, or None when disabled or failed.
    """
    if not laya_enabled():
        return None
    start = time.monotonic()
    probe = {"passage": "The Company is not party to any pending litigation."}
    question = {
        "q": {"type": "noul", "instructions": "Does `passage` disclose pending litigation?"}
    }
    try:
        await adecide_batch([(probe, question)])
    except LayaUnavailable as e:
        logger.warning(f"Laya warmup failed: {e}")
        return None
    elapsed = round(time.monotonic() - start, 2)
    logger.info("Laya warmed", extra={"model": laya_model(), "seconds": elapsed})
    return elapsed
