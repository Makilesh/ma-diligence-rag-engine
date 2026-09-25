# src/data_processing/risk_signal_extractor.py
"""
Risk signal extractor for M&A due diligence documents.

Detects risk-relevant signals in document text:
- Change of control provisions
- Material adverse change/effect (MAC/MAE) clauses
- Litigation mentions and pending lawsuits
- Regulatory risk indicators
- Financial distress signals
- Environmental liability references
- Key person dependencies

Risk signals are stored in the risk_signals field (list[str]) of the
Qdrant payload for downstream use by the risk dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.decisions import ingest_signals
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Risk signal patterns ────────────────────────────────────────────────────

RISK_PATTERNS: dict[str, list[re.Pattern]] = {
    "change_of_control": [
        re.compile(r'(?i)\bchange\s+of\s+control\b'),
        re.compile(r'(?i)\bchange\s+in\s+control\b'),
        re.compile(r'(?i)\bcontrol\s+(?:shall\s+)?change\b'),
        re.compile(r'(?i)\btransfer\s+of\s+(?:controlling|majority)\s+interest\b'),
        re.compile(r'(?i)\bacquisition\s+of\s+(?:all|substantially\s+all)\b'),
        re.compile(r'(?i)\bmerger\s+or\s+consolidation\b'),
    ],
    "material_adverse_change": [
        re.compile(r'(?i)\bmaterial\s+adverse\s+(?:change|effect|event|impact)\b'),
        # Case-sensitive on purpose: (?i) made "Mac" laptops and "Mac Duffie" MAC clauses.
        re.compile(r'\b(?:MAC|MAE)\b'),
        re.compile(r'(?i)\bmaterial(?:ly)?\s+adverse\b'),
        re.compile(r'(?i)\badverse(?:ly)?\s+affect(?:s|ed|ing)?\s+(?:the\s+)?(?:business|operations|financial\s+condition)\b'),
    ],
    "litigation": [
        re.compile(r'(?i)\b(?:pending|threatened|ongoing)\s+(?:litigation|lawsuit|legal\s+action|claim)\b'),
        re.compile(r'(?i)\b(?:plaintiff|defendant|court\s+order|injunction|judgment)\b'),
        re.compile(r'(?i)\b(?:class\s+action|arbitration|mediation|settlement)\b'),
        re.compile(r'(?i)\bsuit\s+(?:filed|pending|brought)\b'),
        re.compile(r'(?i)\blegal\s+(?:proceeding|dispute|action)\b'),
    ],
    "regulatory_risk": [
        re.compile(r'(?i)\b(?:regulatory|government)\s+(?:investigation|inquiry|enforcement|action)\b'),
        re.compile(r'(?i)\b(?:consent\s+decree|cease\s+and\s+desist|warning\s+letter)\b'),
        re.compile(r'(?i)\b(?:non-?compliance|violation|breach)\s+(?:of|with)\s+(?:law|regulation|statute)\b'),
        re.compile(r'(?i)\b(?:fine|penalty|sanction)\s+(?:imposed|assessed|levied)\b'),
        re.compile(r'(?i)\bantitrust\s+(?:review|approval|clearance|concern)\b'),
    ],
    "financial_distress": [
        re.compile(r'(?i)\bgoing\s+concern\b'),
        re.compile(r'(?i)\b(?:bankruptcy|insolvency|liquidation|receivership)\b'),
        # A bare "default" matched "by default" and "default settings"; only
        # credit-default phrasings are distress.
        re.compile(
            r'(?i)\b(?:(?:event|notice)\s+of\s+default|payment\s+default|in\s+default|defaulted'
            r'|covenant\s+(?:violation|breach))\b'
        ),
        re.compile(r'(?i)\b(?:debt\s+restructuring|forbearance|workout)\b'),
        re.compile(r'(?i)\b(?:qualified|adverse)\s+(?:audit\s+)?opinion\b'),
        re.compile(r'(?i)\bmaterial\s+weakness\b'),
    ],
    "environmental_liability": [
        re.compile(r'(?i)\b(?:environmental|hazardous|toxic)\s+(?:liability|contamination|cleanup|remediation)\b'),
        re.compile(r'(?i)\b(?:superfund|cercla|brownfield)\b'),
        re.compile(r'(?i)\b(?:asbestos|lead\s+paint|pcb|petroleum)\s+(?:contamination|exposure)\b'),
        re.compile(r'(?i)\benvironmental\s+(?:assessment|audit|investigation)\b'),
    ],
    "key_person_dependency": [
        re.compile(r'(?i)\bkey\s+(?:person|man|employee|executive)\s+(?:clause|provision|risk|dependency)\b'),
        re.compile(r'(?i)\b(?:founder|ceo|cto|cfo)\s+(?:departure|retention|non-?compete)\b'),
        re.compile(r'(?i)\b(?:critical|essential)\s+(?:personnel|talent|employee)\b'),
    ],
    "ip_risk": [
        # `expir\b` could never match ("expiry", "expires" continue the word).
        re.compile(r'(?i)\b(?:patent|trademark|copyright)s?\s+(?:infringement|challenge|dispute|expir\w*)'),
        re.compile(r'(?i)\b(?:intellectual\s+property|ip)\s+(?:risk|challenge|litigation|dispute)\b'),
        re.compile(r'(?i)\b(?:trade\s+secret|proprietary)\s+(?:misappropriation|theft|disclosure)\b'),
    ],
    "customer_concentration": [
        re.compile(r'(?i)\b(?:customer|client|revenue)\s+concentration\b'),
        # `represent\b` never matched "represents"; the verb stem needs \w*.
        re.compile(r'(?i)\b(?:single|major|largest)\s+customer\s+(?:represent|account|compris)\w*'),
        # A customer (or top-N customers) tied to a share of revenue in the same
        # sentence: "the largest customer, Northstar, represents 12.0% of revenue".
        # This replaces a bare "top N customer" match, which also fired on
        # "the top 5 customer contracts were renegotiated".
        re.compile(r'(?i)\bcustomers?\b[^.]{0,80}?\b\d{1,3}(?:\.\d+)?%\s+of\s+(?:total\s+|annual\s+)?(?:\w+\s+)?(?:revenue|sales)\b'),
    ],
    "indemnification": [
        re.compile(r'(?i)\bindemnif(?:y|ication|ied|ies)\b'),
        re.compile(r'(?i)\b(?:hold\s+harmless|indemnity\s+cap|basket|deductible|escrow)\b'),
        re.compile(r'(?i)\b(?:survival\s+period|limitation\s+of\s+liability)\b'),
    ],
}


@dataclass
class RiskSignalResult:
    """
    Result of risk signal extraction.

    Each signal_details entry carries signal_type, match_count, sample_matches,
    source ("regex", "laya" or "regex+laya") and confidence (Laya's P(true), or
    None when Laya did not judge it). `rejected` lists regex matches Laya
    overruled, kept for audit rather than silently dropped.
    """
    signals: list[str]
    signal_details: list[dict]
    signal_count: int
    rejected: list[dict] = field(default_factory=list)


class RiskSignalExtractor:
    """
    Extracts risk-relevant signals from M&A document text.

    Detects:
    - Change of control provisions
    - Material adverse change/effect (MAC/MAE) clauses
    - Litigation mentions and pending lawsuits
    - Regulatory risk indicators
    - Financial distress signals
    - Environmental liability references
    - Key person dependencies
    - IP risk
    - Customer concentration risk
    - Indemnification provisions

    Risk signals are stored as list[str] in the risk_signals payload field
    and displayed on the risk dashboard.
    """

    def __init__(self) -> None:
        """Initialize the risk signal extractor."""
        self._patterns = RISK_PATTERNS

    def extract(
        self,
        text: str,
        file_name: str = "",
        document_category: str = "",
        laya_scores: dict[str, float] | None = None,
    ) -> RiskSignalResult:
        """
        Extract risk signals from document text.

        Args:
            text: Document or chunk text to scan.
            file_name: Original filename for context.
            document_category: Document category for context-aware detection.
            laya_scores: Laya P(true) per category for this text (see
                ingest_signals.score_chunks). None runs the regex alone.

        Returns:
            RiskSignalResult with detected signals and details.

        Raises:
            ValueError: If text is None.
        """
        if text is None:
            raise ValueError("Cannot extract risk signals from None text")

        logger.info(
            "Extracting risk signals",
            extra={
                "file_name": file_name,
                "text_length": len(text),
                "document_category": document_category,
            },
        )

        signals: list[str] = []
        signal_details: list[dict] = []

        for signal_type, patterns in self._patterns.items():
            matches_found: list[str] = []

            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    matches_found.extend(matches)

            if matches_found:
                signals.append(signal_type)
                signal_details.append({
                    "signal_type": signal_type,
                    "match_count": len(matches_found),
                    "sample_matches": matches_found[:3],  # First 3 matches as samples
                    "source": "regex",
                    "confidence": None,
                })

        result = RiskSignalResult(
            signals=signals,
            signal_details=signal_details,
            signal_count=len(signals),
        )
        if laya_scores is not None:
            result = self.apply_laya(result, laya_scores)

        if result.signals:
            logger.info(
                "Risk signals detected",
                extra={
                    "file_name": file_name,
                    "signals": result.signals,
                    "signal_count": result.signal_count,
                    "rejected": [r["signal_type"] for r in result.rejected],
                },
            )
        else:
            logger.debug(
                "No risk signals detected",
                extra={"file_name": file_name},
            )

        return result

    def apply_laya(
        self,
        regex_result: RiskSignalResult,
        laya_scores: dict[str, float],
    ) -> RiskSignalResult:
        """
        Combines a regex result with Laya's per-category probabilities.

        A category is flagged when Laya alone is confident (P >= RISK_THRESHOLD),
        or when the regex matched and Laya does not disagree (P >=
        RISK_CONFIRM_THRESHOLD). The regex keeps recall on phrasings Laya scores
        timidly; Laya removes the keyword hits that mean the opposite ("no
        pending litigation", "Mac laptops"). A regex match with no Laya score
        (the category was not asked) is kept as a plain regex signal.

        Args:
            regex_result: Output of extract() without laya_scores.
            laya_scores: {category: P(true)} — all categories in "union" mode,
                only the regex-matched ones in "confirm" mode.

        Returns:
            New RiskSignalResult; overruled regex matches go to `rejected`.
        """
        regex_details = {d["signal_type"]: d for d in regex_result.signal_details}
        signals: list[str] = []
        details: list[dict] = []
        rejected: list[dict] = []

        for signal_type in self._patterns:
            p = laya_scores.get(signal_type)
            regex_detail = regex_details.get(signal_type)
            if p is None:
                if regex_detail:
                    signals.append(signal_type)
                    details.append(dict(regex_detail))
                continue
            confidence = round(p, 4)
            if regex_detail and p >= ingest_signals.RISK_CONFIRM_THRESHOLD:
                signals.append(signal_type)
                details.append({**regex_detail, "source": "regex+laya", "confidence": confidence})
            elif p >= ingest_signals.RISK_THRESHOLD:
                signals.append(signal_type)
                details.append({
                    "signal_type": signal_type,
                    "match_count": 1,  # one chunk judged; Laya gives no match spans
                    "sample_matches": [],
                    "source": "laya",
                    "confidence": confidence,
                })
            elif regex_detail:
                rejected.append({
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "sample_matches": regex_detail["sample_matches"],
                })

        return RiskSignalResult(
            signals=signals,
            signal_details=details,
            signal_count=len(signals),
            rejected=rejected,
        )

    def extract_from_chunks(
        self,
        chunks: list[dict],
        file_name: str = "",
    ) -> dict[int, RiskSignalResult]:
        """
        Extract risk signals from multiple chunks.

        Args:
            chunks: List of chunk dicts with 'text' key.
            file_name: Source filename for logging.

        Returns:
            Dict mapping chunk index to RiskSignalResult.
        """
        results: dict[int, RiskSignalResult] = {}

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if text:
                result = self.extract(text, file_name=file_name)
                if result.signals:
                    results[idx] = result

        logger.info(
            "Batch risk signal extraction complete",
            extra={
                "file_name": file_name,
                "num_chunks": len(chunks),
                "chunks_with_signals": len(results),
            },
        )
        return results

    def build_risk_metadata(self, result: RiskSignalResult) -> dict:
        """
        Build risk-related payload fields for Qdrant ingestion.

        Args:
            result: RiskSignalResult from extract().

        Returns:
            Dict with risk_signals list for payload.
        """
        return {
            "risk_signals": result.signals,
        }
