"""
Embedding and reranking model management with async wrappers.

CRITICAL: SentenceTransformer.encode() and CrossEncoder.predict() are synchronous
CPU/GPU operations. Calling them directly inside async functions blocks the entire
asyncio event loop — every other coroutine is frozen until encoding completes.

All calls MUST go through the async wrappers which use run_in_executor with
SEPARATE ThreadPoolExecutor pools to prevent embed/rerank from starving each other.

Uses asyncio.get_running_loop() — NOT get_event_loop() which is deprecated in 3.10+
and broken in 3.12.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# Module-level executors — separate pools prevent embed/rerank from starving
# each other. With a single executor of max_workers=2, one request uses both
# threads (embed + rerank), blocking any concurrent request entirely.
# ==============================================================================
_embed_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")
_rerank_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank")

# ==============================================================================
# Model singletons — lazily loaded on first use
# ==============================================================================
_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None


def _get_embedding_model() -> SentenceTransformer:
    """
    Lazily loads the BAAI/bge-m3 embedding model on first use.
    Always resident in VRAM (every query uses it).

    Returns:
        SentenceTransformer model for BAAI/bge-m3 on CUDA.
    """
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BAAI/bge-m3 embedding model to {device.upper()}")
        _embedding_model = SentenceTransformer(
            "BAAI/bge-m3",
            device=device,
        )
        logger.info(f"BAAI/bge-m3 embedding model loaded successfully on {device.upper()}")
    return _embedding_model


def _get_reranker_model() -> CrossEncoder:
    """
    Lazily loads the BAAI/bge-reranker-v2-m3 cross-encoder on first use.
    Always resident in VRAM (every query uses it).

    CRITICAL: bge-reranker-v2-m3 outputs RAW LOGITS (unbounded) by default.
    Without sigmoid normalization, scores can be negative or >1, which silently
    breaks every reranker_threshold (0.25-0.8) and the Quality Assessor's
    "mean reranker score of top-5" heuristic.
    activation_fct=torch.nn.Sigmoid() normalizes output to [0, 1].

    Returns:
        CrossEncoder model with sigmoid activation for [0,1] normalized scores.
    """
    global _reranker_model
    if _reranker_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BAAI/bge-reranker-v2-m3 to {device.upper()}")
        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=1024,
            device=device,
            default_activation_function=torch.nn.Sigmoid(),  # ← MANDATORY for [0,1] scores
        )
        logger.info(f"BAAI/bge-reranker-v2-m3 loaded successfully on {device.upper()} (sigmoid activated)")
    return _reranker_model


# ==============================================================================
# Async wrappers — NEVER call model.encode() or model.predict() directly
# in async context. Always use these wrappers.
# ==============================================================================


async def embed_texts_async(texts: list[str]) -> np.ndarray:
    """
    Non-blocking wrapper for synchronous SentenceTransformer.encode().
    Always use this in async context — never call model.encode() directly.

    Returns np.ndarray (not list[list[float]]). Callers must call .tolist()
    when passing to Qdrant or other APIs that expect plain Python lists.

    Args:
        texts: List of text strings to embed.

    Returns:
        np.ndarray of shape (len(texts), 1024) with L2-normalized embeddings.

    Raises:
        RuntimeError: If no running event loop is available.
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed empty text list")

    model = _get_embedding_model()
    loop = asyncio.get_running_loop()  # ← NOT get_event_loop()

    logger.info("Embedding texts asynchronously", extra={"num_texts": len(texts)})

    result = await loop.run_in_executor(
        _embed_executor,
        lambda: model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ),
    )

    logger.info(
        "Embedding complete",
        extra={"num_texts": len(texts), "output_shape": list(result.shape)},
    )
    return result


async def rerank_async(query: str, passages: list[str]) -> np.ndarray:
    """
    Non-blocking wrapper for synchronous cross-encoder reranking.

    Returns np.ndarray (not list[float]). CrossEncoder.predict() returns
    numpy array. Callers must convert individual scores: float(score).

    Expected score range after sigmoid: [0.0, 1.0]
    Scores > 0.5 indicate positive relevance. All reranker_threshold values
    in RETRIEVAL_CONFIGS (0.25-0.8) and Quality Assessor heuristics assume
    this normalized range.

    Args:
        query: The search query string.
        passages: List of passage texts to score against the query.

    Returns:
        np.ndarray of shape (len(passages),) with sigmoid-normalized scores in [0, 1].

    Raises:
        RuntimeError: If no running event loop is available.
        ValueError: If passages is empty.
    """
    if not passages:
        raise ValueError("Cannot rerank empty passage list")

    model = _get_reranker_model()
    loop = asyncio.get_running_loop()  # ← NOT get_event_loop()
    pairs = [[query, p] for p in passages]

    logger.info("Reranking passages asynchronously", extra={"num_passages": len(passages)})

    result = await loop.run_in_executor(
        _rerank_executor,
        lambda: model.predict(pairs),
    )

    logger.info(
        "Reranking complete",
        extra={
            "num_passages": len(passages),
            "min_score": float(np.min(result)) if len(result) > 0 else None,
            "max_score": float(np.max(result)) if len(result) > 0 else None,
        },
    )
    return result


def get_embed_executor() -> ThreadPoolExecutor:
    """
    Returns the module-level embedding executor for use by other modules
    that need to wrap synchronous embedding-related operations (e.g., BM25).

    Returns:
        ThreadPoolExecutor dedicated to embedding operations.
    """
    return _embed_executor


async def warm_models() -> dict[str, float]:
    """
    Loads and exercises every local model so the first query does not pay for it.

    All three models load lazily on first use: BAAI/bge-m3 for dense embeddings,
    bge-reranker-v2-m3 for cross-encoding, and FastEmbed's BM25 for sparse
    vectors. Together that is several gigabytes of weights, so the first query
    after a restart was tens of seconds slower than every query after it —
    fine for a batch evaluation, bad in front of a live audience where the first
    question is the one being watched.

    Loading is not enough on its own; each model is also run once, because the
    first forward pass allocates buffers and triggers kernel selection that a
    bare constructor call does not.

    Failures are logged and swallowed. A warmup is an optimisation, and it must
    never be the reason the API refuses to start.

    Returns:
        Mapping of model name to seconds taken. Missing keys indicate a failure.
    """
    import asyncio
    import time

    from src.vector_db.hybrid_search import compute_sparse_bm25

    timings: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    probe = "Aurora Technologies reported total revenue of $452.8 million in FY2023."

    async def _timed(name: str, fn) -> None:
        start = time.monotonic()
        try:
            await loop.run_in_executor(_embed_executor, fn)
            timings[name] = round(time.monotonic() - start, 2)
        except Exception as e:
            logger.warning(f"Model warmup failed for {name}: {e}")

    await _timed("dense_embedding", lambda: _get_embedding_model().encode([probe]))
    await _timed("reranker", lambda: _get_reranker_model().predict([(probe, probe)]))
    await _timed("sparse_bm25", lambda: compute_sparse_bm25(probe))

    logger.info("Local model warmup complete", extra={"seconds": timings})
    return timings
