# src/data_processing/pii_detector.py
"""
PII (Personally Identifiable Information) detector for M&A documents.

Flags HR/comp documents, SSNs, salary data, and other PII.
Sets contains_pii=1 in metadata.

PII-flagged content is excluded from retrieval by default per compliance
policy. The _build_filter function in hybrid_search.py applies
contains_pii=0 filter unless explicitly overridden for authorized users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.decisions import ingest_signals
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── PII detection patterns ──────────────────────────────────────────────────

# Social Security Numbers: XXX-XX-XXXX, or a bare/spaced 9-digit number only when
# an SSN label precedes it. A bare 9-digit run on its own is far more often an
# invoice, order or account number ("Invoice 123456789") than an SSN.
SSN_PATTERN = re.compile(
    r'\b\d{3}-\d{2}-\d{4}\b'
    r'|(?i:\b(?:ssn|social\s+security(?:\s+(?:number|no\.?|#))?)\s*[:#\-]?\s*)'
    r'\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
)

# Email addresses
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Phone numbers (US format)
PHONE_PATTERN = re.compile(
    r'\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)?[2-9]\d{2}[-.\s]?\d{4}\b'
)

# Date of birth patterns
DOB_PATTERN = re.compile(
    r'(?i)\b(?:date\s+of\s+birth|dob|born\s+on|birth\s*date)\s*[:\-]?\s*'
    r'(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})'
)

# Salary / compensation data
# The amount must carry a currency sign: with `\$?` optional, "Transaction Bonus
# 2024" and "Commission 5" were salary data.
SALARY_PATTERNS = [
    re.compile(r'(?i)\b(?:salary|base\s+pay|compensation|annual\s+pay)\s*[:\-]?\s*\$\s*[\d,]+'),
    re.compile(r'(?i)\b(?:bonus|incentive|commission|stock\s+option|equity\s+grant)\s*[:\-]?\s*\$\s*[\d,]+'),
    re.compile(r'(?i)\b(?:hourly\s+rate|pay\s+rate|wage)\s*[:\-]?\s*\$\s*[\d,.]+'),
]

# Bank account / routing numbers
BANK_PATTERN = re.compile(
    r'(?i)\b(?:account\s*(?:number|#|no\.?)|routing\s*(?:number|#|no\.?))\s*[:\-]?\s*\d{6,}'
)

# Passport numbers
PASSPORT_PATTERN = re.compile(
    r'(?i)\b(?:passport\s*(?:number|#|no\.?))\s*[:\-]?\s*[A-Z0-9]{6,}'
)

# Tax ID / EIN
TAX_ID_PATTERN = re.compile(
    r'(?i)\b(?:tax\s*id|ein|employer\s+identification)\s*[:\-]?\s*\d{2}[-]?\d{7}\b'
)

# ─── HR/Comp document filename indicators ────────────────────────────────────
HR_FILENAME_PATTERNS = [
    re.compile(r'(?i)(employee|personnel|staff)[-_\s]*(list|roster|record|data)'),
    re.compile(r'(?i)(salary|compensation|pay[-_\s]*roll|benefit)'),
    re.compile(r'(?i)(hr|human[-_\s]*resource)[-_\s]*(report|data|record)'),
    re.compile(r'(?i)(performance[-_\s]*review|appraisal|evaluation)'),
    re.compile(r'(?i)(offer[-_\s]*letter|employment[-_\s]*contract)'),
    re.compile(r'(?i)(background[-_\s]*check|drug[-_\s]*test)'),
    re.compile(r'(?i)(w[-_]?[249]|1099|i[-_]?9|tax[-_\s]*form)'),
    re.compile(r'(?i)(social[-_\s]*security|ssn)'),
]

# ─── PII keyword indicators in content ───────────────────────────────────────
PII_CONTENT_KEYWORDS = [
    "social security number", "ssn", "date of birth", "home address",
    "personal phone", "personal email", "emergency contact",
    "marital status", "dependents", "medical history",
    "disability status", "veteran status", "ethnic",
    "driver's license", "drivers license", "national id",
    "bank account", "direct deposit", "routing number",
]


@dataclass
class PIIDetectionResult:
    """Result of PII detection on a document or chunk."""
    contains_pii: int  # 0 or 1 — integer for Qdrant payload
    pii_types: list[str]
    pii_count: int
    is_hr_document: bool
    confidence: float  # 0.0 to 1.0
    # Laya's P(PII) when it was consulted (LAYA_PII=1), else None.
    laya_probability: float | None = None
    source: str = "regex"  # "regex" or "regex+laya"


class PIIDetector:
    """
    Detects personally identifiable information in M&A documents.

    Flags HR/comp documents, SSNs, salary data, and other PII.
    Sets contains_pii=1 in metadata for Qdrant payload.

    PII-flagged content is excluded from retrieval by default.
    The compliance filter in hybrid_search.py applies contains_pii=0
    unless explicitly overridden for authorized users.
    """

    # Structured identifiers whose regex match is specific enough to trust on
    # its own. Every other type (salary wording, bank/tax-id labels, keyword
    # counts) matches business text too, so a Laya check can overrule it.
    STRONG_TYPES = frozenset({"ssn", "date_of_birth", "passport"})

    def __init__(self, sensitivity: str = "high") -> None:
        """
        Initialize the PII detector.

        Args:
            sensitivity: Detection sensitivity level ("low", "medium", "high").
                "high" — flag on any PII indicator (recommended for M&A compliance).
                "medium" — flag on 2+ indicators or strong signals (SSN, salary).
                "low" — flag only on explicit SSN or bulk PII data.
        """
        self.sensitivity = sensitivity

    def detect(
        self,
        text: str,
        file_name: str = "",
        laya_probability: float | None = None,
    ) -> PIIDetectionResult:
        """
        Detect PII in text content and filename.

        Args:
            text: Document or chunk text content to scan.
            file_name: Original filename for HR/comp document detection.
            laya_probability: Laya's P(PII) for this text, when LAYA_PII is on.
                A strong identifier (STRONG_TYPES) or HR filename always flags;
                any other regex hit needs P >= PII_CONFIRM_THRESHOLD, and Laya
                alone flags at P >= PII_THRESHOLD. None runs the regex alone.

        Returns:
            PIIDetectionResult with contains_pii flag and details.

        Raises:
            ValueError: If text is None.
        """
        if text is None:
            raise ValueError("Cannot detect PII in None text")

        logger.info(
            "Running PII detection",
            extra={"file_name": file_name, "text_length": len(text)},
        )

        pii_types: list[str] = []
        pii_count = 0

        # Check if HR/compensation document by filename
        is_hr = self._check_hr_filename(file_name)
        if is_hr:
            pii_types.append("hr_document")

        # Scan content for PII patterns
        content_pii = self._scan_content(text)
        pii_types.extend(content_pii["types"])
        pii_count += content_pii["count"]

        # Determine if document contains PII based on sensitivity
        contains_pii = self._evaluate(
            is_hr=is_hr,
            pii_types=pii_types,
            pii_count=pii_count,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(is_hr, pii_types, pii_count)

        source = "regex"
        if laya_probability is not None:
            source = "regex+laya"
            # Excluding real PII is a compliance feature, so structured
            # identifiers are never second-guessed; Laya only arbitrates the
            # keyword-level hits and may add what the regex cannot see.
            strong = is_hr or bool(set(pii_types) & self.STRONG_TYPES)
            contains_pii = (
                strong
                or (contains_pii and laya_probability >= ingest_signals.PII_CONFIRM_THRESHOLD)
                or laya_probability >= ingest_signals.PII_THRESHOLD
            )

        result = PIIDetectionResult(
            contains_pii=1 if contains_pii else 0,
            pii_types=sorted(set(pii_types)),
            pii_count=pii_count,
            is_hr_document=is_hr,
            confidence=confidence,
            laya_probability=laya_probability,
            source=source,
        )

        logger.info(
            "PII detection complete",
            extra={
                "file_name": file_name,
                "contains_pii": result.contains_pii,
                "pii_types": result.pii_types,
                "pii_count": result.pii_count,
            },
        )

        return result

    def _check_hr_filename(self, file_name: str) -> bool:
        """
        Check if filename indicates an HR/compensation document.

        Args:
            file_name: Document filename.

        Returns:
            True if the filename matches HR document patterns.
        """
        if not file_name:
            return False

        return any(pattern.search(file_name) for pattern in HR_FILENAME_PATTERNS)

    def _scan_content(self, text: str) -> dict:
        """
        Scan text content for PII patterns.

        Args:
            text: Text to scan.

        Returns:
            Dict with 'types' (list[str]) and 'count' (int).
        """
        types: list[str] = []
        count = 0

        # SSN detection
        ssn_matches = SSN_PATTERN.findall(text)
        if ssn_matches:
            # Filter out likely false positives (e.g., phone number fragments)
            real_ssns = [
                m for m in ssn_matches
                if self._validate_ssn(m)
            ]
            if real_ssns:
                types.append("ssn")
                count += len(real_ssns)

        # Email detection
        email_matches = EMAIL_PATTERN.findall(text)
        # Exclude corporate/company emails, focus on personal
        personal_emails = [
            e for e in email_matches
            if any(domain in e.lower() for domain in [
                "gmail", "yahoo", "hotmail", "outlook", "aol", "icloud",
                "proton", "mail.com",
            ])
        ]
        if personal_emails:
            types.append("personal_email")
            count += len(personal_emails)

        # DOB detection
        dob_matches = DOB_PATTERN.findall(text)
        if dob_matches:
            types.append("date_of_birth")
            count += len(dob_matches)

        # Salary/compensation detection
        for pattern in SALARY_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                types.append("salary_data")
                count += len(matches)
                break  # Don't double-count salary patterns

        # Bank account detection
        bank_matches = BANK_PATTERN.findall(text)
        if bank_matches:
            types.append("bank_account")
            count += len(bank_matches)

        # Passport detection
        passport_matches = PASSPORT_PATTERN.findall(text)
        if passport_matches:
            types.append("passport")
            count += len(passport_matches)

        # Tax ID detection
        tax_matches = TAX_ID_PATTERN.findall(text)
        if tax_matches:
            types.append("tax_id")
            count += len(tax_matches)

        # Keyword-based detection
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in PII_CONTENT_KEYWORDS if kw in text_lower)
        if keyword_hits >= 3:
            types.append("pii_keywords")
            count += keyword_hits

        return {"types": types, "count": count}

    def _validate_ssn(self, candidate: str) -> bool:
        """
        Validate that a candidate string is likely an SSN, not a false positive.

        Args:
            candidate: Candidate SSN string.

        Returns:
            True if it's likely a real SSN.
        """
        # The labelled branch of SSN_PATTERN includes the "SSN:" prefix.
        digits = re.sub(r'\D', '', candidate)
        if len(digits) != 9:
            return False

        # SSN area number cannot be 000, 666, or 900-999
        area = int(digits[:3])
        if area == 0 or area == 666 or area >= 900:
            return False

        # Group number cannot be 00
        group = int(digits[3:5])
        if group == 0:
            return False

        # Serial number cannot be 0000
        serial = int(digits[5:])
        if serial == 0:
            return False

        return True

    def _evaluate(
        self,
        is_hr: bool,
        pii_types: list[str],
        pii_count: int,
    ) -> bool:
        """
        Evaluate whether the document should be flagged as containing PII.

        Args:
            is_hr: Whether the document is an HR document.
            pii_types: Detected PII type labels.
            pii_count: Total number of PII instances found.

        Returns:
            True if the document should be flagged.
        """
        if self.sensitivity == "high":
            # Flag on any indicator
            return is_hr or len(pii_types) > 0

        elif self.sensitivity == "medium":
            # Flag on strong signals or multiple indicators
            strong_signals = {"ssn", "salary_data", "bank_account", "passport"}
            has_strong = any(t in strong_signals for t in pii_types)
            return is_hr or has_strong or len(pii_types) >= 2

        else:  # low
            # Flag only on explicit SSN or bulk PII
            return "ssn" in pii_types or pii_count >= 5

    def _calculate_confidence(
        self,
        is_hr: bool,
        pii_types: list[str],
        pii_count: int,
    ) -> float:
        """
        Calculate confidence score for PII detection.

        Args:
            is_hr: Whether the document is an HR document.
            pii_types: Detected PII type labels.
            pii_count: Total PII instance count.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        confidence = 0.0

        if is_hr:
            confidence += 0.4

        strong_signals = {"ssn", "salary_data", "bank_account", "passport"}
        strong_count = sum(1 for t in pii_types if t in strong_signals)
        confidence += min(strong_count * 0.25, 0.5)

        # Moderate signals
        moderate_count = len(set(pii_types) - strong_signals - {"hr_document", "pii_keywords"})
        confidence += min(moderate_count * 0.1, 0.2)

        return min(confidence, 1.0)

    def build_pii_metadata(self, result: PIIDetectionResult) -> dict:
        """
        Build PII-related payload fields for Qdrant ingestion.

        Args:
            result: PIIDetectionResult from detect().

        Returns:
            Dict with contains_pii and related metadata.
        """
        return {
            "contains_pii": result.contains_pii,
        }
