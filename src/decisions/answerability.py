"""
Answerability of retrieved context — a Laya check behind the Quality Assessor.

The context gate in quality_assessor.py scores the reranker, and a cross-encoder
measures RELEVANCE: a passage about FY2023 revenue is highly relevant to a
question about FY2024 revenue, so an on-topic question whose figure is simply
not in the data room sails through the gate and spends a reasoning-model
synthesis call on an answer that ends up declining. This module asks the other
question — does any passage STATE what was asked — with one Laya `noul` per
(facet, passage) pair.

Design, and why (chosen on eval/answerability_dev.json; tests/golden_qa_set.json
is the held-out test set — see eval/README.md and eval/results/):

  * Facets are the user's question plus Agent 1's sub-questions. A facet's
    score is the max P over the passages; the question's score is the max over
    facets. The gate only VETOES — "Laya is confident that no facet is stated
    anywhere" — because zero-shot Laya separates answerable from unanswerable
    context only moderately (dev AUC 0.80): it is trustworthy at the bottom of
    its range and not above it. "All facets covered" was not adopted: several
    decomposed golden questions have a sub-question that is legitimately
    unanswered (e.g. "if applicable" facets), so it would refuse good context.
  * Only the TOP_K_PASSAGES best-reranked chunks are scored. Laya's P is noisy
    on off-topic passages (an unrelated litigation chunk often scores ~0.55 for
    any question), so scoring the whole reranked list lets noise mask a real
    "no". The top 3 keep dev false vetoes at zero with the widest margin while
    still covering evidence ranked 2nd or 3rd.
  * Child chunks, not parent sections: parents run ~2k tokens and Laya
    truncates at 1024, so the part of a parent that answers could be cut off.
  * VETO_THRESHOLD sits below the lowest answerable question on the dev set
    (0.409), not at it, so one unlucky phrasing does not flip a good answer.
    On the golden test set the lowest answerable question scored 0.392 — the
    margin is real but thin, which is why this only ever vetoes.

Measured (golden, held out): controls refused 2 -> 4 of 6 (ctrl_02 vetoed,
ctrl_03 refused without the LLM call), answerables admitted unchanged.

Env:
    LAYA_GATE   "0" disables the answerability check (default on). LAYA_ENABLED=0
                disables it too, along with every other Laya decision.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from src.decisions.laya_client import adecide_batch, noul
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Chosen on the dev set from three phrasings; this one had the best dev AUC and
# was the only one with zero false vetoes at every threshold tried.
ANSWERABILITY_INSTRUCTION = "Does `passage` state the information needed to answer `question`?"

TOP_K_PASSAGES = 3

# Below this, no facet is stated in the best passages. Dev: the lowest-scoring
# answerable question scored 0.409, and 0.35 still refuses 9 of 28 unanswerable
# ones (6 of the 20 that the reranker heuristic admits).
VETO_THRESHOLD = 0.35

# Agent 1 emits at most a handful of sub-questions; the cap bounds CPU time on
# the deployed host (facets x TOP_K_PASSAGES forward passes per assessment).
MAX_FACETS = 5


def laya_gate_enabled() -> bool:
    """True unless LAYA_GATE is set to a falsy value."""
    return os.getenv("LAYA_GATE", "1").strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Answerability:
    """
    Laya's view of whether the retrieved context answers the question.

    Attributes:
        facet_scores: facet text -> max P(stated) over the scored passages.
        score: max over facets; the number the veto compares.
        vetoed: score < threshold — no facet is stated in the best passages.
        unanswered: facets scoring below the threshold, for missing_aspects.
        passages_scored: number of passages each facet was scored against.
        latency_ms: wall time of the Laya call.
    """

    facet_scores: dict[str, float]
    score: float
    vetoed: bool
    unanswered: list[str] = field(default_factory=list)
    passages_scored: int = 0
    latency_ms: float = 0.0

    def as_breakdown(self) -> dict:
        """Compact, JSON-safe form for quality_breakdown and the agent trace."""
        return {
            "answerability": round(self.score, 3),
            "answerability_facets": {f: round(p, 3) for f, p in self.facet_scores.items()},
            "answerability_veto": self.vetoed,
        }


def answerability_facets(question: str, sub_questions: list[str] | None) -> list[str]:
    """
    The question plus its distinct sub-questions, capped at MAX_FACETS.

    Args:
        question: The user's question (original, not a rewrite: the gate judges
            whether what was ASKED is answered).
        sub_questions: Agent 1's decomposition, possibly empty.

    Returns:
        Ordered, de-duplicated facet list, question first.
    """
    facets: list[str] = []
    for text in [question, *(sub_questions or [])]:
        text = (text or "").strip()
        if text and text not in facets:
            facets.append(text)
    return facets[:MAX_FACETS]


def top_passages(chunks: list[dict], k: int = TOP_K_PASSAGES) -> list[str]:
    """
    Texts of the k highest-reranked chunks that have text.

    Args:
        chunks: reranked_results payloads.
        k: How many to keep.

    Returns:
        Up to k chunk texts, best first.
    """
    ranked = sorted(
        (c for c in chunks if (c.get("text") or "").strip()),
        key=lambda c: float(c.get("reranker_score", 0.0)),
        reverse=True,
    )
    return [c["text"] for c in ranked[:k]]


def aggregate(
    facet_passage_scores: dict[str, list[float]], threshold: float = VETO_THRESHOLD
) -> tuple[dict[str, float], float, bool, list[str]]:
    """
    Reduces per-(facet, passage) probabilities to the gate verdict.

    A facet is as answered as its best passage; the question is as answered as
    its best facet. The veto fires only when even that best pair is below the
    threshold.

    Args:
        facet_passage_scores: facet -> P(stated) for each scored passage.
        threshold: Veto threshold.

    Returns:
        (facet_scores, score, vetoed, unanswered_facets).
    """
    facet_scores = {f: max(ps) if ps else 0.0 for f, ps in facet_passage_scores.items()}
    score = max(facet_scores.values()) if facet_scores else 0.0
    unanswered = [f for f, p in facet_scores.items() if p < threshold]
    return facet_scores, score, bool(facet_scores) and score < threshold, unanswered


async def assess_answerability(
    question: str,
    sub_questions: list[str] | None,
    chunks: list[dict],
    threshold: float = VETO_THRESHOLD,
) -> Answerability | None:
    """
    Scores whether the best retrieved passages state what the question asks.

    Args:
        question: The user's question.
        sub_questions: Agent 1's sub-questions.
        chunks: The reranked chunks the gate is judging.
        threshold: Veto threshold (the eval harness sweeps it).

    Returns:
        An Answerability, or None when there is nothing to score.

    Raises:
        LayaUnavailable: Laya disabled or unloadable — callers fall back to the
            heuristic alone.
    """
    facets = answerability_facets(question, sub_questions)
    passages = top_passages(chunks)
    if not facets or not passages:
        return None

    question_spec = {"stated": {"type": "noul", "instructions": ANSWERABILITY_INSTRUCTION}}
    requests = [
        ({"question": facet, "passage": passage}, question_spec)
        for facet in facets
        for passage in passages
    ]
    start = time.perf_counter()
    answers = await adecide_batch(requests)
    latency_ms = (time.perf_counter() - start) * 1000

    per_facet: dict[str, list[float]] = {f: [] for f in facets}
    for (state, _), answer in zip(requests, answers):
        per_facet[state["question"]].append(noul(answer.get("stated", {})))

    facet_scores, score, vetoed, unanswered = aggregate(per_facet, threshold)
    return Answerability(
        facet_scores=facet_scores,
        score=score,
        vetoed=vetoed,
        unanswered=unanswered,
        passages_scored=len(passages),
        latency_ms=round(latency_ms, 1),
    )
