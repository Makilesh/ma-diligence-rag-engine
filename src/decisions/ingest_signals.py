"""
Ingest-time Laya decisions: risk signals, PII and document category.

The regex detectors in src/data_processing flag keywords, not meaning: "no
pending or threatened litigation" is a litigation signal to them, and an
invoice number shaped like an SSN is PII — which matters, because PII-flagged
chunks are excluded from retrieval, so a false positive silently deletes
evidence. Laya reads the chunk and answers typed questions about it instead.

Cost is bounded by asking every ingest question of a chunk in ONE request (the
risk categories and PII share a (state, questions) pair), and by batching every
chunk of a document into as few decide_batch calls as the row bound allows.
Laya encodes one sequence per question, so the number of questions is the
cost: on CPU the risk task defaults to "confirm" mode, which asks only about
the categories the regex proposed for that chunk.

Every phrasing and threshold below was chosen on the DEV split of
eval/decisions/ and measured on TEST — see eval/decisions/results.md. The
alternatives that were tried stay in the *_PHRASINGS tables so the evaluation
runner and the pipeline share one definition of each question.

Env (each also requires LAYA_ENABLED, see laya_client):
    LAYA_RISK       Laya risk signals (default on: +precision and +recall on TEST).
    LAYA_RISK_MODE  "union" (Laya over every category, regex as a confirmed
                    backstop), "confirm" (Laya only confirms regex matches) or
                    "auto" (default: union on CUDA, confirm on CPU).
    LAYA_PII        Laya PII judgement (default OFF: it lost precision on TEST
                    and excluded a chunk holding golden-set facts).
    LAYA_CATEGORY   Laya document classification (default on).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.decisions.laya_client import LayaUnavailable, decide_batch, noul
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# Feature flags
# ==============================================================================

# Default per task — on only where the evaluation showed a win on TEST.
_FLAG_DEFAULTS = {
    "LAYA_RISK": "1",
    "LAYA_PII": "0",
    "LAYA_CATEGORY": "1",
}
RISK_MODES = ("union", "confirm")


def _flag(name: str) -> bool:
    value = os.getenv(name, _FLAG_DEFAULTS[name]).strip().lower()
    return value not in ("0", "false", "no", "off", "")


def laya_risk_enabled() -> bool:
    """True when Laya should judge risk signals at ingest."""
    return _flag("LAYA_RISK")


def laya_pii_enabled() -> bool:
    """True when Laya should judge PII at ingest."""
    return _flag("LAYA_PII")


def laya_category_enabled() -> bool:
    """True when Laya should classify documents at ingest."""
    return _flag("LAYA_CATEGORY")


def _uses_cuda() -> bool:
    """Mirrors laya_client's device choice without loading the model."""
    configured = os.getenv("LAYA_DEVICE", "").strip().lower()
    if configured:
        return configured.startswith("cuda")
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def risk_mode() -> str:
    """
    Which risk design runs: "union" or "confirm".

    Union asks all ten categories of every chunk (~10 sequences per chunk);
    confirm asks only the categories the regex matched (~0.3 per chunk on the
    sample deal). Union measured better on TEST, but on a 2-core CPU host ten
    sequences per chunk is minutes per document, so "auto" picks by device.

    Returns:
        "union" or "confirm".
    """
    configured = os.getenv("LAYA_RISK_MODE", "auto").strip().lower()
    if configured in RISK_MODES:
        return configured
    return "union" if _uses_cuda() else "confirm"


# ==============================================================================
# Questions
# ==============================================================================

# Keys must match RISK_PATTERNS in risk_signal_extractor.py (a test enforces it).
RISK_DESCRIPTIONS: dict[str, str] = {
    "change_of_control": (
        "a change of control provision, such as a consent requirement, termination "
        "right, repayment, acceleration or payment triggered by a change of control, "
        "merger or sale of the company"
    ),
    "material_adverse_change": (
        "a material adverse change or material adverse effect (MAC or MAE) clause, "
        "definition or closing condition"
    ),
    "litigation": (
        "pending or threatened litigation, lawsuits, arbitration or legal claims "
        "against the company"
    ),
    "regulatory_risk": (
        "a regulatory risk, such as a required antitrust or merger-control approval, "
        "a government investigation, an enforcement action, a fine or a compliance "
        "violation"
    ),
    "financial_distress": (
        "financial distress, such as doubt about the company continuing as a going "
        "concern, a loan default, a covenant breach, insolvency or a shortage of cash"
    ),
    "environmental_liability": (
        "an environmental liability, such as contamination, remediation costs or "
        "environmental claims"
    ),
    "key_person_dependency": (
        "a dependency on a key person, where the business relies on a specific "
        "executive, founder or employee who might leave"
    ),
    "ip_risk": (
        "an intellectual property risk, such as infringement claims, patent challenges "
        "or expiry, open-source licence violations or gaps in IP ownership"
    ),
    "customer_concentration": (
        "customer concentration, where a large share of revenue comes from one or a "
        "few customers"
    ),
    "indemnification": (
        "indemnification provisions, such as indemnity obligations, caps, deductibles, "
        "baskets, escrows or survival periods"
    ),
}

# Phrasings tried on DEV. {desc} is a RISK_DESCRIPTIONS entry.
RISK_PHRASINGS: dict[str, str] = {
    "disclose": "Does `passage` disclose {desc}?",
    "disclose_negation": (
        "Does `passage` disclose {desc}? Answer no if `passage` only says there is none."
    ),
    "reviewer": (
        "Would an M&A due-diligence reviewer flag `passage` because it discloses {desc}?"
    ),
}
RISK_PHRASING = "disclose"

PII_PHRASINGS: dict[str, dict[str, str]] = {
    # One question over every kind of personal data.
    "single": {
        "pii": (
            "Does `passage` contain private personal information about a specific "
            "individual, such as a social security number, date of birth, home address, "
            "personal phone or email, bank account, health information or the "
            "individual's salary?"
        ),
    },
    # Framed around a named person, to steer away from company data.
    "named_person": {
        "pii": (
            "Does `passage` reveal private details about a specific named person, such "
            "as their government ID number, date of birth, home address, personal contact "
            "details, bank account, health or pay?"
        ),
    },
    # Identifiers and individual pay asked separately; the flag is their max.
    "split": {
        "pii_identifier": (
            "Does `passage` contain private personal information about a specific "
            "individual, such as a social security, passport or driver's licence number, "
            "a date of birth, a home address, a personal phone number or email, a bank "
            "account or health information?"
        ),
        "pii_compensation": (
            "Does `passage` state the salary, bonus, severance or other pay of a specific "
            "named individual?"
        ),
    },
}
PII_PHRASING = "single"

CATEGORY_CRITERIA: dict[str, str] = {
    "financial": "financial statements, accounts, quality of earnings, forecasts or valuation",
    "legal": "contracts, agreements, disclosure schedules, litigation or IP schedules",
    "board": "board minutes, board decks, resolutions or management presentations",
    "audit": "auditor reports, internal audit or control assessments",
    "regulatory": "regulatory approvals, antitrust filings, permits or privacy compliance memos",
    "operational": "HR, IT, supply chain, facilities, customers or operations reviews",
    "other": "none of the other options fits",
}
CATEGORY_PHRASINGS: dict[str, str] = {
    "kind": "What kind of due-diligence document is `document`, given its `filename`?",
    "kind_content": "Based on its content, what kind of due-diligence document is `document`?",
}
CATEGORY_PHRASING = "kind"
# Characters of the document opening passed to Laya (inputs truncate at 1024 tokens).
CATEGORY_SAMPLE_CHARS = 1500


def risk_questions(phrasing: str = RISK_PHRASING) -> dict[str, dict]:
    """
    Builds one `noul` question per risk category.

    Args:
        phrasing: Key into RISK_PHRASINGS.

    Returns:
        {"risk:<category>": question} for every category.
    """
    template = RISK_PHRASINGS[phrasing]
    return {
        f"risk:{category}": {"type": "noul", "instructions": template.format(desc=desc)}
        for category, desc in RISK_DESCRIPTIONS.items()
    }


def pii_questions(phrasing: str = PII_PHRASING) -> dict[str, dict]:
    """
    Builds the PII `noul` question(s).

    Args:
        phrasing: Key into PII_PHRASINGS.

    Returns:
        {"pii:<name>": question}.
    """
    return {
        f"pii:{name}": {"type": "noul", "instructions": text}
        for name, text in PII_PHRASINGS[phrasing].items()
    }


def category_question(phrasing: str = CATEGORY_PHRASING) -> dict[str, dict]:
    """
    Builds the document-category `choice` question.

    Args:
        phrasing: Key into CATEGORY_PHRASINGS.

    Returns:
        {"category": question}.
    """
    return {
        "category": {
            "type": "choice",
            "instructions": CATEGORY_PHRASINGS[phrasing],
            "criteria": dict(CATEGORY_CRITERIA),
        }
    }


# ==============================================================================
# Thresholds (chosen on DEV — see eval/decisions/results.md)
# ==============================================================================

# Risk: a category is flagged when Laya's P(true) reaches RISK_THRESHOLD, or when
# the regex matched and Laya's P(true) reaches RISK_CONFIRM_THRESHOLD. Laya is
# near-uninformative on its own below ~0.6 (TEST precision 0.52 at 0.55), so a
# lone Laya flag needs a high bar while a regex match only needs Laya not to
# disagree.
RISK_THRESHOLD = 0.675
RISK_CONFIRM_THRESHOLD = 0.375

# PII (only when LAYA_PII=1; not adopted): strong regex identifiers always
# flag; otherwise Laya's P(true) must reach PII_CONFIRM_THRESHOLD for a regex
# hit and PII_THRESHOLD on its own. DEV chose the same value for both.
PII_THRESHOLD = 0.25
PII_CONFIRM_THRESHOLD = 0.25

# Category: Laya's choice is used whenever it answers. A confidence floor tied
# with it on DEV, so the simpler rule was kept (TEST: 0.955 vs 0.864). The rules
# remain for files with no text sample (.xlsx) and when Laya is unavailable.


# ==============================================================================
# Scoring
# ==============================================================================

# Upper bound on (chunk, question) sequences per decide_batch call.
MAX_ROWS_PER_CALL = 48


@dataclass
class ChunkScores:
    """Laya probabilities for one chunk (empty dicts when a task is off)."""
    risk: dict[str, float] = field(default_factory=dict)
    pii: dict[str, float] = field(default_factory=dict)

    @property
    def pii_probability(self) -> float | None:
        """Max P(true) over the PII questions, or None when PII was not asked."""
        return max(self.pii.values()) if self.pii else None


def score_chunks(
    texts: Sequence[str],
    *,
    risk_categories: Sequence[Iterable[str] | None] | None = None,
    pii: bool = False,
    risk_phrasing: str = RISK_PHRASING,
    pii_phrasing: str = PII_PHRASING,
) -> list[ChunkScores]:
    """
    Asks each chunk its ingest questions — one request per chunk, batched.

    Synchronous and model-bound: call it from a worker thread (extract_and_chunk
    already runs in one), never on the event loop.

    Args:
        texts: Chunk texts.
        risk_categories: Per chunk, the risk categories to ask about (None =
            all ten). Omit the argument to ask no risk questions at all.
        pii: Also ask the PII question(s) of every chunk.
        risk_phrasing: Key into RISK_PHRASINGS.
        pii_phrasing: Key into PII_PHRASINGS.

    Returns:
        One ChunkScores per text, in order; a chunk with no questions gets an
        empty ChunkScores and costs nothing.

    Raises:
        LayaUnavailable: If Laya is disabled, cannot load, or inference fails —
            callers fall back to the regex detectors.
    """
    all_risk = risk_questions(risk_phrasing)
    pii_qs = pii_questions(pii_phrasing) if pii else {}

    requests: list[tuple[int, dict]] = []
    for i in range(len(texts)):
        questions: dict[str, dict] = {}
        if risk_categories is not None:
            wanted = risk_categories[i]
            if wanted is None:
                questions.update(all_risk)
            else:
                questions.update({f"risk:{c}": all_risk[f"risk:{c}"] for c in wanted})
        questions.update(pii_qs)
        if questions:
            requests.append((i, questions))

    scores = [ChunkScores() for _ in texts]
    if not requests:
        return scores

    # Laya encodes one sequence per (state, question), while laya_client batches
    # by request: 16 chunks x 11 questions is 176 sequences in one forward pass,
    # which exhausted memory and crashed a CPU run. Calls are cut so that no
    # call exceeds MAX_ROWS_PER_CALL sequences (a single request always fits).
    batches: list[list[tuple[int, dict]]] = [[]]
    rows = 0
    for item in requests:
        if batches[-1] and rows + len(item[1]) > MAX_ROWS_PER_CALL:
            batches.append([])
            rows = 0
        batches[-1].append(item)
        rows += len(item[1])

    for batch in batches:
        answers = decide_batch([({"passage": texts[i]}, qs) for i, qs in batch])
        for (i, _), answer in zip(batch, answers):
            _fill(scores[i], answer)
    return scores


def _fill(chunk: ChunkScores, answer: dict) -> None:
    """Copies one request's noul answers into a ChunkScores."""
    for name, value in answer.items():
        kind, _, key = name.partition(":")
        if kind == "risk":
            chunk.risk[key] = round(noul(value), 4)
        elif kind == "pii":
            chunk.pii[key] = round(noul(value), 4)


def category_state(filename: str, content_sample: str, phrasing: str = CATEGORY_PHRASING) -> dict:
    """
    The Laya state for classifying one document.

    The filename is included only when the phrasing refers to it: a state field
    the instructions never mention is still read by the encoder.

    Args:
        filename: Original filename.
        content_sample: Opening text of the document.
        phrasing: Key into CATEGORY_PHRASINGS.

    Returns:
        State dict.
    """
    state = {"document": (content_sample or "").strip()[:CATEGORY_SAMPLE_CHARS]}
    if "`filename`" in CATEGORY_PHRASINGS[phrasing]:
        state = {"filename": filename, **state}
    return state


def classify_document(
    filename: str,
    content_sample: str,
    phrasing: str = CATEGORY_PHRASING,
) -> tuple[str, float]:
    """
    Classifies a document from its filename and opening text.

    Args:
        filename: Original filename.
        content_sample: Opening text of the document.
        phrasing: Key into CATEGORY_PHRASINGS.

    Returns:
        (category, confidence).

    Raises:
        LayaUnavailable: If Laya is disabled or unavailable, or the sample is
            empty (there is nothing for the model to read).
    """
    state = category_state(filename, content_sample, phrasing)
    if not state["document"]:
        raise LayaUnavailable("No content to classify")
    answer = decide_batch([(state, category_question(phrasing))])[0]["category"]
    choice = answer.get("choice")
    if choice not in CATEGORY_CRITERIA:
        raise LayaUnavailable(f"Unexpected category choice: {choice!r}")
    confidence = float(answer.get("probabilities", {}).get(choice, answer.get("confidence", 0.0)))
    return choice, round(confidence, 4)
