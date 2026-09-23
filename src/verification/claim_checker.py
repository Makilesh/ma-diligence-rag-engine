"""
Claim-level answer verification.

Pipeline, cheapest first:

1. Split the answer into atomic claims (src/verification/claims.py).
2. Ground every figure deterministically (numeric_grounding.py).
3. Score each claim against its evidence with a local NLI model (nli.py).
   Evidence is the chunks the claim cites, plus the best-matching windows of
   the top retrieved chunks, so a correct claim with a sloppy citation is not
   failed for the citation alone.
4. Only claims neither step could decide go to an LLM judge — batched into ONE
   call, and only when at least one such claim exists. The common case makes
   zero LLM calls; the previous validator made one on every query.

Per-claim decision (first matching rule wins):

    any figure unsupported                         -> unsupported   (numeric)
    NLI entailment >= ENTAIL_THRESHOLD             -> supported     (nli)
    every figure grounded/derived, no strong
      NLI contradiction                            -> supported     (numeric)
    NLI contradiction >= CONTRADICTION_THRESHOLD
      and entailment < 0.1, figures not all
      grounded                                     -> contradicted  (nli)
    NLI clearly neutral (entailment below
      CLEAR_NEUTRAL_MAX_ENTAIL, contradiction low) -> unsupported   (nli)
    otherwise                                      -> LLM judge, or "unverified"
                                                      when no judge is available

Answer-level outcome:

    failed   any contradicted claim, or any figure not found in the context
    warning  any non-numeric unsupported claim, unverifiable calculation, or
             a verification step that was unavailable
    passed   otherwise

Confidence (shown to users) is COMPUTED, not self-reported by a model:

    claim_support   = (supported + 0.5 * unverified) / checked claims
    numeric_support = (grounded + derived + 0.5 * derived_unverified) / figures
    confidence      = claim_support                      if no figures
                    = (claim_support + numeric_support)/2 otherwise

An answer with no checkable claims (a pure decline) scores 0.0 — there is
nothing it asserts that could be verified — and passes.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from src.utils.logger import setup_logger
from src.verification import nli
from src.verification.claims import Claim, resolve_cited_chunks, split_claims
from src.verification.numeric_grounding import (
    ContextIndex,
    ground_texts,
    summarize_numeric,
)

logger = setup_logger(__name__)

# Thresholds on the NLI softmax. Entailment >= 0.5 means entailment is the
# model's most likely label by a margin; contradiction is held to a much higher
# bar because a false "contradicted" fails the answer and triggers a re-synthesis
# on a scarce reasoning-model quota.
ENTAIL_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.9
CLEAR_NEUTRAL_MAX_ENTAIL = 0.02
CLEAR_NEUTRAL_MAX_CONTRADICTION = 0.5
# Share of a claim's content tokens a premise window must contain before its
# contradiction score is believed (see the note where it is applied).
CONTRADICTION_MIN_OVERLAP = 0.5

# Evidence selection per claim: best windows (by lexical overlap) from the
# chunks the claim cites, plus from the top retrieved chunks.
CITED_WINDOWS_PER_CLAIM = 2
OTHER_WINDOWS_PER_CLAIM = 2
FALLBACK_TOP_K = 6

# Bounds on work per answer.
MAX_CLAIMS = 40
MAX_JUDGE_CLAIMS = 15
MAX_JUDGE_DOCUMENTS = 8

Scorer = Callable[[list[tuple[str, str]]], Awaitable[list[nli.NLIScore]]]
# judge(query, [{"id", "claim"}], [chunk dicts]) -> ({id: {"status", "reason"}}, model)
Judge = Callable[[str, list[dict], list[dict]], Awaitable[tuple[dict[int, dict], str]]]

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are",
    "has", "had", "have", "its", "which", "into", "than", "their", "also", "been",
    "under", "over", "per", "all", "any", "not", "but", "such", "these", "those",
    "fy", "company", "total",
}
_TOKEN = re.compile(r"[a-z][a-z\-]{2,}|\d+(?:\.\d+)?")


@dataclass
class VerificationReport:
    """Everything the validator node needs to populate state."""

    claim_checks: list[dict] = field(default_factory=list)
    numeric_checks: list[dict] = field(default_factory=list)
    validation_status: str = "passed"
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    status_counts: dict = field(default_factory=dict)
    method_counts: dict = field(default_factory=dict)
    numeric_counts: dict = field(default_factory=dict)
    nli_model: str | None = None
    llm_model: str | None = None
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def _tokens(text: str) -> set[str]:
    """Content tokens for lexical evidence ranking (numbers de-comma'd)."""
    lowered = (text or "").lower().replace(",", "")
    return {t for t in _TOKEN.findall(lowered) if t not in _STOPWORDS}


_LABEL_PREFIX = re.compile(r"^([^:]{2,60}):\s+(.+)$")


def _nli_hypothesis(claim_text: str) -> str:
    """
    Rewrites a claim into the plain-sentence form NLI models were trained on.

    Synthesis answers lean on "Label: statement" bullets. Measured on the sample
    data room, the label alone flipped verdicts: "Reportability & Valuation: The
    transaction is reportable under the HSR Act" scored 0.986 CONTRADICTION
    against "The transaction is reportable under the Hart-Scott-Rodino Antitrust
    Improvements Act." When the statement after the label is a sentence in its
    own right, the label is dropped; when it is a fragment ("36 months
    post-closing") the pair is joined with "is" so it still reads as a claim.

    Args:
        claim_text: Clean claim text.

    Returns:
        Hypothesis text for NLI scoring.
    """
    m = _LABEL_PREFIX.match(claim_text.strip())
    if not m or len(m.group(1).split()) > 6:
        return claim_text
    label, rest = m.group(1).strip(), m.group(2).strip().rstrip(" .")
    if len(rest.split()) >= 5 and rest[:1].isupper():
        return rest
    return f"{label} is {rest}"


def _chunk_windows(chunk: dict) -> list[str]:
    """Premise windows for a chunk: its text, plus its parent when distinct."""
    text = chunk.get("text", "") or ""
    parent = chunk.get("parent_text", "") or ""
    body = text if not parent or parent in text else f"{text}\n{parent}"
    return nli.split_premises(body)


def _best_windows(
    claim_tokens: set[str],
    chunk_ids: list[int],
    windows: dict[int, list[str]],
    limit: int,
) -> list[tuple[int, str, float]]:
    """
    Picks the windows most likely to hold a claim's evidence, by token overlap.

    Lexical preselection keeps NLI cost at a few pairs per claim instead of
    every window of every chunk — the difference between milliseconds and many
    seconds on a CPU-only host.

    Returns:
        (chunk index, window text, overlap) tuples, best first; zero-overlap
        windows are dropped.
    """
    scored = []
    for ci in chunk_ids:
        for w in windows.get(ci, []):
            overlap = len(claim_tokens & _tokens(w)) / max(len(claim_tokens), 1)
            if overlap > 0:
                scored.append((ci, w, overlap))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:limit]


def _snippet(window: str, claim_tokens: set[str], width: int = 280) -> str:
    """The part of a window around its first claim-token hit."""
    lowered = window.lower()
    hit = min(
        (lowered.find(t) for t in claim_tokens if len(t) > 3 and lowered.find(t) >= 0),
        default=0,
    )
    start = max(0, hit - 60)
    return window[start: start + width].strip()


def _evidence_fields(chunk: dict | None) -> dict:
    """Source metadata for a claim check's evidence."""
    if not chunk:
        return {"evidence_source": "", "evidence_page": None, "evidence_section": ""}
    return {
        "evidence_source": chunk.get("source_file", "") or "",
        "evidence_page": chunk.get("page_number") or None,
        "evidence_section": chunk.get("section_heading", "") or "",
    }


def _short(text: str, limit: int = 200) -> str:
    """Truncates claim text for flags."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def verify_answer(
    answer: str,
    chunks: list[dict],
    query: str,
    *,
    scorer: Scorer | None = None,
    judge: Judge | None = None,
    use_nli: bool | None = None,
) -> VerificationReport:
    """
    Verifies an answer claim by claim against its retrieved context.

    Args:
        answer: Generated answer (markdown with inline citation markers).
        chunks: Retrieved context the answer was written from.
        query: The user's original question (for the judge prompt).
        scorer: NLI scorer; defaults to the local model. Injected in tests.
        judge: LLM judge for undecided claims; None disables LLM judging.
        use_nli: Force NLI on/off; defaults to VERIFY_NLI.

    Returns:
        VerificationReport.
    """
    start = time.monotonic()
    report = VerificationReport()

    claims: list[Claim] = split_claims(answer)
    if len(claims) > MAX_CLAIMS:
        report.notes.append(f"{len(claims) - MAX_CLAIMS} claims beyond the first {MAX_CLAIMS} not checked")
        claims = claims[:MAX_CLAIMS]

    if not claims:
        report.notes.append("no verifiable claims")
        report.elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return report

    # ── 1. Numeric grounding ────────────────────────────────────────────────
    index = ContextIndex(chunks)
    numeric = ground_texts([c.text for c in claims], index)
    report.numeric_checks = [n.to_dict() for group in numeric for n in group]
    report.numeric_counts = summarize_numeric(report.numeric_checks)

    # ── 2. NLI ──────────────────────────────────────────────────────────────
    if use_nli is None:
        use_nli = nli.nli_enabled()
    if scorer is None and use_nli:
        scorer = nli.ascore_pairs

    windows = {ci: _chunk_windows(c) for ci, c in enumerate(chunks)}
    top_k = list(range(min(len(chunks), FALLBACK_TOP_K)))

    # (claim idx, chunk idx, cited, window, lexical overlap)
    pair_owner: list[tuple[int, int, bool, str, float]] = []
    pairs: list[tuple[str, str]] = []
    cited_by_claim: dict[int, list[int]] = {}
    claim_tokens: dict[int, set[str]] = {}

    for claim in claims:
        toks = _tokens(claim.text)
        claim_tokens[claim.index] = toks
        cited = resolve_cited_chunks(claim.markers, chunks)
        cited_by_claim[claim.index] = cited
        others = [ci for ci in top_k if ci not in cited]
        chosen = _best_windows(toks, cited, windows, CITED_WINDOWS_PER_CLAIM)
        chosen += _best_windows(
            toks, others, windows,
            OTHER_WINDOWS_PER_CLAIM + (0 if cited else 1),
        )
        hypothesis = _nli_hypothesis(claim.text)
        for ci, w, overlap in chosen:
            pair_owner.append((claim.index, ci, ci in cited, w, overlap))
            pairs.append((w, hypothesis))

    nli_scores: list[nli.NLIScore] = []
    nli_available = False
    if use_nli and scorer is not None and pairs:
        try:
            nli_scores = await scorer(pairs)
            nli_available = True
            report.nli_model = nli.loaded_model_name() or nli.nli_model_name()
        except Exception as e:
            logger.warning("NLI scoring unavailable; falling back", extra={"error": str(e)})
            report.notes.append(f"NLI unavailable: {e}")
    elif not use_nli:
        report.notes.append("NLI disabled (VERIFY_NLI=0)")

    # Best evidence per claim.
    best: dict[int, dict] = {}
    for (claim_idx, ci, cited, window, overlap), score in zip(pair_owner, nli_scores):
        entry = best.setdefault(claim_idx, {
            "entail": 0.0, "entail_chunk": None, "entail_window": "",
            "contra": 0.0, "contra_chunk": None, "contra_window": "",
        })
        if score.entailment > entry["entail"]:
            entry.update(entail=score.entailment, entail_chunk=ci, entail_window=window)
        # A contradiction only counts from a window that is plainly about the
        # same thing. Small NLI models return confident "contradiction" for a
        # premise that merely sits near the topic — measured on the sample data
        # room, "Aurora has not sought a waiver" scored 0.997 contradiction
        # against the one-line document title "AURORA TECHNOLOGIES INC.".
        if overlap >= CONTRADICTION_MIN_OVERLAP and score.contradiction > entry["contra"]:
            entry.update(contra=score.contradiction, contra_chunk=ci, contra_window=window)

    # ── 3. Decide ───────────────────────────────────────────────────────────
    checks: list[dict] = []
    undecided: list[int] = []

    for claim in claims:
        nums = numeric[claim.index]
        num_dicts = [
            {k: v for k, v in n.to_dict().items() if k in ("raw", "status", "formula", "evidence_source") and v}
            for n in nums
        ]
        unsupported_nums = [n.raw for n in nums if n.status == "unsupported"]
        all_grounded = bool(nums) and all(n.status in ("grounded", "derived") for n in nums)
        ev = best.get(claim.index) if nli_available else None
        cited = cited_by_claim[claim.index]
        contra = ev["contra"] if ev else 0.0
        entail = ev["entail"] if ev else 0.0

        check = {
            "claim": claim.text,
            "status": "unverified",
            "method": "none",
            "score": None,
            "numbers": num_dicts,
            "cited": bool(cited),
            "reason": "",
            "evidence": "",
            **_evidence_fields(None),
        }
        toks = claim_tokens[claim.index]

        if unsupported_nums:
            check.update(
                status="unsupported",
                method="numeric",
                reason="figure(s) not found in the retrieved documents: " + ", ".join(unsupported_nums),
            )
        elif ev and entail >= ENTAIL_THRESHOLD:
            chunk = chunks[ev["entail_chunk"]]
            check.update(
                status="supported",
                method="nli",
                score=round(entail, 3),
                evidence=_snippet(ev["entail_window"], toks),
                **_evidence_fields(chunk),
            )
        elif all_grounded:
            # Exact figure matches outrank the NLI model on figure-dense claims.
            # Measured on the 35 RESULTS.md answers: of the claims whose every
            # figure was grounded yet NLI scored >= 0.9 contradiction, the large
            # majority were table rows and "Label: $X" lines the model misreads
            # ("Minimum Fixed Charge Coverage Ratio: >= 1.25x"). The score is
            # kept on the check so a reviewer can still see the disagreement.
            first = next((n for n in nums if n.evidence_source), None)
            check.update(
                status="supported",
                method="numeric",
                score=round(entail, 3) if ev else None,
                reason="every figure appears in, or is computed from, the retrieved documents",
                evidence=first.evidence_snippet if first else "",
                evidence_source=first.evidence_source if first else "",
            )
            if contra >= CONTRADICTION_THRESHOLD:
                check["nli_contradiction"] = round(contra, 3)
        elif ev and contra >= CONTRADICTION_THRESHOLD and entail < 0.1:
            chunk = chunks[ev["contra_chunk"]]
            check.update(
                status="contradicted",
                method="nli",
                score=round(contra, 3),
                evidence=_snippet(ev["contra_window"], toks),
                **_evidence_fields(chunk),
            )
        elif (
            ev
            and entail < CLEAR_NEUTRAL_MAX_ENTAIL
            and contra < CLEAR_NEUTRAL_MAX_CONTRADICTION
            and not nums
        ):
            check.update(
                status="unsupported",
                method="nli",
                score=round(entail, 3),
                reason="no retrieved passage entails this statement",
            )
        else:
            undecided.append(claim.index)
            check["score"] = round(entail, 3) if ev else None
            if nli_available and ev is None:
                check["reason"] = "no retrieved passage shares vocabulary with this claim"
        checks.append(check)

    # ── 4. LLM judge for what is left ───────────────────────────────────────
    if undecided:
        to_judge = undecided[:MAX_JUDGE_CLAIMS]
        if len(undecided) > MAX_JUDGE_CLAIMS:
            report.notes.append(
                f"{len(undecided) - MAX_JUDGE_CLAIMS} undecided claims beyond the judge cap left unverified"
            )
        if judge is None:
            report.notes.append("validation unavailable: no LLM judge configured for undecided claims")
        else:
            doc_ids: list[int] = []
            for idx in to_judge:
                for ci in cited_by_claim[idx]:
                    if ci not in doc_ids:
                        doc_ids.append(ci)
                ev = best.get(idx)
                if ev and ev["entail_chunk"] is not None and ev["entail_chunk"] not in doc_ids:
                    doc_ids.append(ev["entail_chunk"])
            for ci in top_k:
                if len(doc_ids) >= MAX_JUDGE_DOCUMENTS:
                    break
                if ci not in doc_ids:
                    doc_ids.append(ci)
            documents = [chunks[ci] for ci in sorted(doc_ids[:MAX_JUDGE_DOCUMENTS])]
            payload = [{"id": i + 1, "claim": claims[idx].text} for i, idx in enumerate(to_judge)]

            try:
                verdicts, model = await judge(query, payload, documents)
                report.llm_calls = 1
                report.llm_model = model
                for i, idx in enumerate(to_judge):
                    verdict = verdicts.get(i + 1) or {}
                    status = str(verdict.get("status", "")).lower()
                    if status not in ("supported", "contradicted", "unsupported"):
                        continue
                    doc_no = verdict.get("document")
                    doc = None
                    try:
                        if doc_no:
                            doc = documents[int(doc_no) - 1]
                    except (ValueError, IndexError, TypeError):
                        doc = None
                    checks[idx].update(
                        status=status,
                        method="llm",
                        reason=str(verdict.get("reason", ""))[:300],
                        **_evidence_fields(doc),
                    )
            except Exception as e:
                logger.warning("LLM judge unavailable", extra={"error": str(e)})
                report.notes.append(f"validation unavailable: LLM judge failed ({type(e).__name__}: {e})")

    # ── 5. Aggregate ────────────────────────────────────────────────────────
    report.claim_checks = checks
    status_counts = Counter(c["status"] for c in checks)
    method_counts = Counter(c["method"] for c in checks if c["method"] != "none")
    report.status_counts = {
        s: status_counts.get(s, 0)
        for s in ("supported", "contradicted", "unsupported", "unverified")
    }
    report.method_counts = dict(method_counts)

    n = len(checks)
    claim_support = (status_counts["supported"] + 0.5 * status_counts["unverified"]) / n
    num = report.numeric_counts
    num_total = sum(num.values())
    if num_total:
        numeric_support = (
            num["grounded"] + num["derived"] + 0.5 * num["derived_unverified"]
        ) / num_total
        report.confidence = round((claim_support + numeric_support) / 2, 3)
    else:
        report.confidence = round(claim_support, 3)

    flags: list[str] = []
    for c in checks:
        if c["status"] == "contradicted":
            src = c["evidence_source"] or "the retrieved documents"
            flags.append(f"Contradicted by {src}: {_short(c['claim'])}")
        elif c["status"] == "unsupported" and c["method"] == "numeric":
            flags.append(f"Figure not found in the documents — {_short(c['claim'])}")
        elif c["status"] == "unsupported":
            flags.append(f"Not supported by the retrieved documents: {_short(c['claim'])}")
    unverifiable_calcs = [
        n["raw"] for n in report.numeric_checks if n["status"] == "derived_unverified"
    ]
    if unverifiable_calcs:
        flags.append(
            "Calculation not reproducible from the documents: "
            + ", ".join(dict.fromkeys(unverifiable_calcs))
        )
    unavailable = [note for note in report.notes if note.startswith("validation unavailable")]
    if unavailable and status_counts["unverified"]:
        flags.append(
            f"validation unavailable: {status_counts['unverified']} claim(s) could not be verified"
        )
    report.flags = flags

    numeric_failure = any(
        c["status"] == "unsupported" and c["method"] == "numeric" for c in checks
    )
    if status_counts["contradicted"] or numeric_failure:
        report.validation_status = "failed"
    elif (
        status_counts["unsupported"]
        or status_counts["unverified"]
        or unverifiable_calcs
    ):
        report.validation_status = "warning"
    else:
        report.validation_status = "passed"

    report.elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    return report


def revision_feedback(claim_checks: list[dict], flags: list[str] | None = None) -> list[str]:
    """
    Builds the "fix these" list fed back into a re-synthesis.

    Args:
        claim_checks: claim_checks from a failed validation.
        flags: hallucination_flags, used when claim_checks is empty.

    Returns:
        One line per contradicted or unsupported claim, with the reason.
    """
    lines = []
    for c in claim_checks or []:
        if c.get("status") not in ("contradicted", "unsupported"):
            continue
        reason = c.get("reason") or (
            f"contradicted by {c.get('evidence_source')}" if c.get("status") == "contradicted"
            else "not supported by the documents"
        )
        lines.append(f"\"{_short(c.get('claim', ''), 300)}\" — {reason}")
    if not lines and flags:
        lines = [f for f in flags if not f.startswith("validation unavailable")]
    return lines
