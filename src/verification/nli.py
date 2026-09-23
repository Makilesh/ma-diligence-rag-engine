"""
Local natural-language-inference scoring for claim verification.

A small cross-encoder NLI model (DeBERTa-v3 fine-tuned on SNLI + MultiNLI)
scores whether an evidence passage entails, contradicts, or is neutral to a
claim. It replaces the per-query LLM judge for the common case: it runs on the
same machine as the embedding and reranker models, costs no API quota, and is
deterministic for a given input.

Model choice (env NLI_MODEL overrides):
    GPU: cross-encoder/nli-deberta-v3-small   (~140M params)
    CPU: cross-encoder/nli-deberta-v3-xsmall  (~70M params) — roughly half the
         latency on a CPU-only host such as a free Hugging Face Space, at a
         small accuracy cost.
Set VERIFY_NLI=0 to disable it entirely on hosts that cannot spare the memory;
verification then falls back to numeric grounding plus the LLM judge.

CRITICAL: CrossEncoder.predict() is synchronous and CPU/GPU bound. Every call
from async code goes through asyncio.to_thread so it never blocks the event
loop, and model loading happens on that worker thread too.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_GPU_DEFAULT = "cross-encoder/nli-deberta-v3-small"
_CPU_DEFAULT = "cross-encoder/nli-deberta-v3-xsmall"

# Premise windows are one to three sentences, capped well inside the model's
# 512-token limit (see split_premises for why short windows).
PREMISE_MAX_WORDS = 90

_RULE_LINE = re.compile(r"^\s*[=\-_*~]{4,}\s*$")
_COLUMN_GAP = re.compile(r"\S\s{2,}\S")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'$])")
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16

_model = None
_model_name: str | None = None
_label_index: dict[str, int] = {}
_load_lock = threading.Lock()
# One inference at a time: the reranker and embedder share the device, and
# concurrent predict() calls on one GPU only contend for the same memory.
_predict_lock = threading.Lock()
_load_failed = False


@dataclass(frozen=True)
class NLIScore:
    """Softmax probabilities for one (premise, hypothesis) pair."""

    entailment: float
    contradiction: float
    neutral: float


def nli_enabled() -> bool:
    """True unless VERIFY_NLI is set to a falsy value."""
    return os.getenv("VERIFY_NLI", "1").strip().lower() not in ("0", "false", "no", "off")


def nli_model_name() -> str:
    """
    The NLI model this process uses: NLI_MODEL if set, else a device default.

    Returns:
        Hugging Face model id.
    """
    configured = os.getenv("NLI_MODEL", "").strip()
    if configured:
        return configured
    try:
        import torch

        return _GPU_DEFAULT if torch.cuda.is_available() else _CPU_DEFAULT
    except Exception:
        return _CPU_DEFAULT


def _get_model():
    """
    Loads the NLI cross-encoder once per process (thread-safe, lazy).

    The label order is read from the model config rather than assumed: the
    cross-encoder NLI checkpoints use {0: contradiction, 1: entailment,
    2: neutral}, which differs from the MNLI convention most people remember,
    and a silently swapped label would invert every verdict.

    Returns:
        The loaded CrossEncoder.

    Raises:
        RuntimeError: If the model cannot be loaded (cached after first failure,
            so a missing model is not re-downloaded on every query).
    """
    global _model, _model_name, _label_index, _load_failed

    if _model is not None:
        return _model
    if _load_failed:
        raise RuntimeError("NLI model failed to load earlier in this process")

    with _load_lock:
        if _model is not None:
            return _model
        name = nli_model_name()
        try:
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading NLI model {name} to {device.upper()}")
            model = CrossEncoder(name, max_length=MAX_SEQ_LENGTH, device=device)

            id2label = getattr(model.config, "id2label", None) or {}
            labels = {str(v).lower(): int(k) for k, v in id2label.items()}
            required = {"entailment", "contradiction", "neutral"}
            if not required.issubset(labels):
                raise RuntimeError(
                    f"NLI model {name} labels {sorted(labels)} lack {sorted(required)}"
                )
            _label_index = labels
            _model = model
            _model_name = name
            logger.info(f"NLI model {name} loaded on {device.upper()}", extra={"labels": labels})
            return _model
        except Exception:
            _load_failed = True
            raise


def loaded_model_name() -> str | None:
    """Name of the model actually loaded, or None if not loaded yet."""
    return _model_name


def score_pairs(pairs: list[tuple[str, str]]) -> list[NLIScore]:
    """
    Scores (premise, hypothesis) pairs synchronously. Call via ascore_pairs.

    Args:
        pairs: (evidence passage, claim) tuples.

    Returns:
        One NLIScore per pair, in order.
    """
    if not pairs:
        return []
    model = _get_model()
    with _predict_lock:
        probs = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            apply_softmax=True,
            show_progress_bar=False,
        )
    e, c, n = (
        _label_index["entailment"],
        _label_index["contradiction"],
        _label_index["neutral"],
    )
    return [
        NLIScore(entailment=float(p[e]), contradiction=float(p[c]), neutral=float(p[n]))
        for p in probs
    ]


async def ascore_pairs(pairs: list[tuple[str, str]]) -> list[NLIScore]:
    """
    Scores pairs off the event loop.

    Args:
        pairs: (evidence passage, claim) tuples.

    Returns:
        One NLIScore per pair, in order.
    """
    return await asyncio.to_thread(score_pairs, pairs)


async def warm_nli_model() -> float | None:
    """
    Loads and exercises the NLI model so the first query does not pay for it.

    Mirrors src.vector_db.reranker.warm_models: one real forward pass, failures
    logged and swallowed — a warmup must never stop the API from starting.

    Returns:
        Seconds taken, or None when disabled or failed.
    """
    if not nli_enabled():
        return None
    start = time.monotonic()
    probe = "Aurora Technologies reported total revenue of $452.8 million in FY2023."
    try:
        await ascore_pairs([(probe, probe)])
    except Exception as e:
        logger.warning(f"NLI model warmup failed: {e}")
        return None
    elapsed = round(time.monotonic() - start, 2)
    logger.info("NLI model warmed", extra={"model": _model_name, "seconds": elapsed})
    return elapsed


def _premise_units(text: str) -> list[str]:
    """
    Splits evidence text into sentence-sized units.

    Data-room text is hard-wrapped prose interleaved with fixed-width tables, so
    neither newlines nor full stops alone mark a unit. Paragraphs (blank-line
    separated) whose lines look tabular — a gap of 2+ spaces between columns —
    keep one unit per line; other paragraphs are unwrapped and sentence-split.
    Separator rules ("=====", "-----") are dropped.
    """
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        lines = [ln for ln in para.splitlines() if ln.strip() and not _RULE_LINE.match(ln)]
        if not lines:
            continue
        tabular = sum(1 for ln in lines if len(_COLUMN_GAP.findall(ln.strip())) >= 1)
        if tabular >= max(1, len(lines) // 2):
            units.extend(re.sub(r"\s+", " ", ln).strip() for ln in lines)
            continue
        joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
        units.extend(s.strip() for s in _SENTENCE_END.split(joined) if s.strip())
    return units


def split_premises(text: str, max_words: int = PREMISE_MAX_WORDS) -> list[str]:
    """
    Builds short premise windows from evidence text.

    Long premises make small NLI models unreliable: measured on the sample data
    room, 180-word windows produced confident "contradiction" for claims whose
    exact supporting sentence sat inside the window, because the same window
    also held a neighbouring clause about a different jurisdiction or period.
    Scoring against one to three consecutive sentences (the standard
    sentence-level approach, e.g. SummaC) and taking the best window removes
    most of that noise. Windows of 2-3 units cover claims that fuse adjacent
    sentences.

    Args:
        text: Evidence text.
        max_words: Upper bound on window length.

    Returns:
        Windows, each 1-3 consecutive units within max_words.
    """
    units = _premise_units(text)
    windows: list[str] = []
    for i in range(len(units)):
        words = 0
        for k in range(i, min(i + 3, len(units))):
            words += len(units[k].split())
            if words > max_words and k > i:
                break
            windows.append(" ".join(units[i: k + 1]))
    return windows
