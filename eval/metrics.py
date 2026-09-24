"""
Retrieval metrics over substring-labelled relevance. Pure functions, no models.

The golden set (tests/golden_qa_set.json) has no chunk labels, and chunk ids
change whenever chunking does, so relevance is derived rather than stored:

    a chunk is relevant to a question when it comes from a file matching one of
    the question's expected source_patterns AND contains at least one of its
    expected facts; its graded relevance is how many distinct facts it contains.

Fact coverage@k — the fraction of a question's expected facts present anywhere
in the top-k texts, from any file — is the headline metric and the one the
README's decomposition numbers were reported in.

A fact is either a string or a list of equivalent surface forms (any one
counts), exactly as tests/run_end_to_end_validation.py treats them.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, Sequence

# Markdown emphasis never carries meaning for matching (see the note on
# normalise_for_matching in tests/run_end_to_end_validation.py).
_MARKDOWN_EMPHASIS = re.compile(r"[*_`]+")
_DASHES = re.compile(r"[‐-―−]")
# "9,065,750" -> "9065750", so "$9,065,750" and "9065750" compare equal and a
# fact "$150" cannot hide inside "$150,000".
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# "$452.8" and a table cell "452.8" state the same figure.
_CURRENCY_BEFORE_DIGIT = re.compile(r"\$\s*(?=\d)")


def normalise(text: str | None) -> str:
    """
    Normalises text for fact matching.

    NFKC (non-breaking spaces, full-width digits), unicode dashes to "-",
    markdown emphasis removed, lower-cased, thousands separators and a "$"
    before a digit dropped, whitespace collapsed.

    Args:
        text: Raw text (chunk, answer or fact).

    Returns:
        Normalised text.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _DASHES.sub("-", text)
    text = _MARKDOWN_EMPHASIS.sub("", text).lower()
    text = _THOUSANDS_SEPARATOR.sub("", text)
    text = _CURRENCY_BEFORE_DIGIT.sub("", text)
    return " ".join(text.split())


def _variant_pattern(variant: str) -> re.Pattern | None:
    """
    Compiles one normalised surface form into a match pattern.

    Plain substring semantics, except at a numeric edge: a fact that starts with
    a digit may not be preceded by a digit or decimal point, and one that ends
    with a digit may not continue into more digits — only into trailing zero
    decimals. So "$85" matches "$85 million" and "$47" matches "$47.00", but
    "$85" does not match "$850", "$1.85" or "$85.5".

    Args:
        variant: One raw surface form of a fact.

    Returns:
        Compiled pattern, or None for an empty variant.
    """
    norm = normalise(variant)
    if not norm:
        return None
    pattern = re.escape(norm)
    if norm[0].isdigit():
        pattern = r"(?<![\d.])" + pattern
    if norm[-1].isdigit():
        pattern += r"(?:\.0+)?(?!\.?\d)"
    return re.compile(pattern)


def fact_variants(fact: str | Sequence[str]) -> list[str]:
    """Returns a fact's accepted surface forms as a list of strings."""
    return [fact] if isinstance(fact, str) else [str(v) for v in fact]


def fact_label(fact: str | Sequence[str]) -> str:
    """Stable display label for a fact (variants joined with ' | ')."""
    return " | ".join(fact_variants(fact))


def fact_in_normalised(fact: str | Sequence[str], normalised_text: str) -> bool:
    """
    True if any surface form of the fact occurs in already-normalised text.

    Args:
        fact: Expected fact (string or list of variants).
        normalised_text: Output of normalise().

    Returns:
        Whether the fact is present.
    """
    for variant in fact_variants(fact):
        pattern = _variant_pattern(variant)
        if pattern is not None and pattern.search(normalised_text):
            return True
    return False


def fact_present(fact: str | Sequence[str], text: str) -> bool:
    """True if the fact (in any accepted form) occurs in the raw text."""
    return fact_in_normalised(fact, normalise(text))


def facts_in_text(facts: Sequence, text: str) -> set[int]:
    """
    Indices of the expected facts present in one text.

    Args:
        facts: The question's expected_answer_contains.
        text: Chunk text.

    Returns:
        Set of fact indices found.
    """
    norm = normalise(text)
    return {i for i, fact in enumerate(facts) if fact_in_normalised(fact, norm)}


def source_matches(source_file: str | None, patterns: Iterable[str]) -> bool:
    """True if the source file name contains any expected source_pattern."""
    name = (source_file or "").lower()
    return any(p.lower() in name for p in patterns if p)


def relevance_grade(text: str, source_file: str | None, facts: Sequence,
                    source_patterns: Sequence[str]) -> int:
    """
    Graded relevance of one chunk to one question.

    Args:
        text: Chunk text.
        source_file: The chunk's source_file payload field.
        facts: The question's expected facts.
        source_patterns: The question's expected_citations source_patterns.

    Returns:
        Number of distinct expected facts in the chunk if it comes from an
        expected source, else 0.
    """
    if not facts or not source_matches(source_file, source_patterns):
        return 0
    return len(facts_in_text(facts, text))


def build_qrels(chunks: Iterable[dict], facts: Sequence,
                source_patterns: Sequence[str]) -> dict[str, int]:
    """
    Relevance labels for one question over the whole index.

    Args:
        chunks: Every retrievable chunk payload (needs chunk_id, text, source_file).
        facts: Expected facts.
        source_patterns: Expected source patterns.

    Returns:
        chunk_id -> grade, for chunks with grade > 0 only.
    """
    qrels = {}
    for chunk in chunks:
        grade = relevance_grade(chunk.get("text", ""), chunk.get("source_file"),
                                facts, source_patterns)
        if grade > 0:
            qrels[chunk["chunk_id"]] = grade
    return qrels


# ==============================================================================
# Ranked-list metrics. `ranked_ids` is best-first; `qrels` maps id -> grade > 0.
# Each returns None when the metric is undefined for the question (no relevant
# chunk exists), so aggregates average only over questions where it means
# something rather than counting an impossible question as a zero.
# ==============================================================================


def recall_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float | None:
    """Fraction of all relevant chunks that appear in the top k."""
    if not qrels:
        return None
    hits = {cid for cid in ranked_ids[:k] if cid in qrels}
    return len(hits) / len(qrels)


def mrr_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float | None:
    """Reciprocal rank of the first relevant chunk within the top k (0 if none)."""
    if not qrels:
        return None
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in qrels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float | None:
    """
    nDCG@k with linear gain (gain = number of facts the chunk contains).

    The ideal ordering is the k highest grades over the whole index. A chunk id
    repeated in the ranking earns its gain only once.
    """
    if not qrels:
        return None
    seen: set[str] = set()
    dcg = 0.0
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in seen:
            continue
        seen.add(cid)
        dcg += qrels.get(cid, 0) / math.log2(rank + 1)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else None


def fact_coverage(texts: Iterable[str], facts: Sequence) -> tuple[float | None, list[int]]:
    """
    Fraction of expected facts present anywhere in the given texts.

    Args:
        texts: Retrieved texts (top-k chunks, or the final context).
        facts: Expected facts.

    Returns:
        (coverage or None when there are no facts, sorted indices of facts found).
    """
    if not facts:
        return None, []
    found: set[int] = set()
    for text in texts:
        found |= facts_in_text(facts, text)
        if len(found) == len(facts):
            break
    return len(found) / len(facts), sorted(found)


def source_hit_at_k(sources: Sequence[str | None], source_patterns: Sequence[str],
                    k: int) -> float | None:
    """Fraction of distinct expected source patterns hit by the top-k sources."""
    patterns = sorted({p.lower() for p in source_patterns if p})
    if not patterns:
        return None
    top = [(s or "").lower() for s in sources[:k]]
    return sum(any(p in s for s in top) for p in patterns) / len(patterns)


# ==============================================================================
# Aggregation
# ==============================================================================


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-None values, or None if there are none."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """
    Linear-interpolated percentile (q in [0, 100]), None for an empty input.
    """
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    return vals[low] + (vals[high] - vals[low]) * (pos - low)


def compare_to_baseline(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """
    Lists gated metrics that dropped by more than the tolerance.

    Both dicts are {ablation: {metric: value}}. Only metrics present in the
    baseline are gated. A baseline metric that the current run did not produce
    (ablation skipped, value None) is a failure too: a gate that silently stops
    measuring is not a gate.

    Args:
        current: Aggregates from this run.
        baseline: Aggregates recorded in eval/baseline.json.
        tolerance: Allowed absolute drop (0.02 = 2 percentage points).

    Returns:
        Human-readable failure lines; empty means the gate passes.
    """
    failures = []
    for ablation, metrics in baseline.items():
        for metric, base_value in metrics.items():
            if base_value is None:
                continue
            value = (current.get(ablation) or {}).get(metric)
            if value is None:
                failures.append(f"{ablation}.{metric}: not measured (baseline {base_value:.4f})")
            elif value < base_value - tolerance - 1e-9:
                failures.append(
                    f"{ablation}.{metric}: {value:.4f} < baseline {base_value:.4f} "
                    f"- tolerance {tolerance:.4f}"
                )
    return failures
