# src/data_processing/document_classifier.py
"""
Document classifier for M&A due diligence documents.

Classifies documents into one of 7 categories based on filename patterns,
content keywords, and structural heuristics:
    financial | legal | board | audit | regulatory | operational | other

classify() is the rule path. classify_with_source() — what ingestion calls —
asks the Laya decision model first (LAYA_CATEGORY) and falls back to the rules
when Laya is off, unavailable, or there is no text sample.

The classification is stored as document_category in the Qdrant payload
and used as a filter parameter during retrieval.
"""

from __future__ import annotations

import re
from typing import Literal

from src.decisions import ingest_signals
from src.decisions.laya_client import LayaUnavailable, laya_enabled
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DocumentCategory = Literal[
    "financial", "legal", "board", "audit", "regulatory", "operational", "other"
]

# ─── Filename-based classification patterns ──────────────────────────────────
# Matched against the filename with underscores turned into spaces (see
# _classify_by_filename). Every alternative starts at a word boundary, and short
# acronyms end at one too: unanchored, `it` matched inside "quality" and
# "credit", and `p\s*&?\s*l` inside "employment", which filed a quality of
# earnings report as operational and an employment schedule as financial.
FILENAME_PATTERNS: dict[DocumentCategory, list[str]] = {
    "financial": [
        r'(?i)\b(income.?statement|balance.?sheet|cash.?flow|p\s*&\s*l\b|pnl\b|profit.?loss)',
        r'(?i)\b(financials?\b|financial.?statement|cap.?table|capitalization|revenue|budget)',
        r'(?i)\b(forecast|projection|valuation|dcf\b|model|ebitda)',
        r'(?i)\b(10-?[kq]\b|annual.?report|quarterly.?report)',
        r'(?i)\b(quality.?of.?earnings|qoe\b|earnings)',
    ],
    "legal": [
        r'(?i)\b(contract|agreement|amendment|addendum|mou\b|memorandum)',
        r'(?i)\b(merger|acquisition|purchase|sale\b|asset.?purchase)',
        r'(?i)\b(nda\b|non.?disclosure|confidential|indemnif)',
        r'(?i)\b(term.?sheet|loi\b|letter.?of.?intent|definitive)',
        r'(?i)\b(license|lease|employment.?agreement|ip.?assign)',
        r'(?i)\b(representation|warrant|covenant|escrow)',
        r'(?i)\b(litigation|patent|trademark|intellectual.?property|ip\b)',
    ],
    "board": [
        r'(?i)\b(board|director|presentation|deck\b|slides?\b|pptx\b)',
        r'(?i)\b(committee|governance|meeting.?minute|minutes\b|resolution)',
        r'(?i)\b(strategy|overview|executive.?summary)',
    ],
    "audit": [
        r'(?i)\b(audit|auditor|sox\b|internal.?control)',
        r'(?i)\b(compliance|accounting|gaap\b|ifrs\b)',
        r'(?i)\b(review|assessment|finding|observation)',
    ],
    "regulatory": [
        r'(?i)\b(regulatory|regulation|filing|permit|license)',
        r'(?i)\b(sec|fda|epa|osha|ftc|doj|hsr)\b|\bantitrust',
        r'(?i)\b(compliance.?report|consent|decree|enforcement)',
    ],
    "operational": [
        r'(?i)\b(operational|operation|process|procedure|workflow)',
        r'(?i)\b(hr\b|human.?resource|employee|headcount|org.?chart)',
        r'(?i)\b(it\b|technology|system|infrastructure|cybersecurity)',
        r'(?i)\b(supply.?chain|vendor|customer|inventory)',
        r'(?i)\b(insurance|real.?estate|property|facility)',
    ],
}

# ─── Content-based classification keywords ───────────────────────────────────
CONTENT_KEYWORDS: dict[DocumentCategory, list[str]] = {
    "financial": [
        "revenue", "ebitda", "net income", "total assets", "total liabilities",
        "cash and cash equivalents", "depreciation", "amortization",
        "earnings per share", "operating income", "gross profit",
        "accounts receivable", "accounts payable", "working capital",
        "fiscal year", "fy20", "budget", "forecast",
    ],
    "legal": [
        "whereas", "now therefore", "in witness whereof", "shall mean",
        "representations and warranties", "indemnification", "covenants",
        "conditions precedent", "termination", "governing law",
        "material adverse", "change of control", "non-compete",
        "intellectual property", "confidential information",
    ],
    "board": [
        "board of directors", "meeting minutes", "resolution",
        "approved unanimously", "strategic plan", "management discussion",
        "key performance indicator", "market overview",
    ],
    "audit": [
        "audit opinion", "material weakness", "significant deficiency",
        "internal controls", "going concern", "fair value",
        "auditor report", "unqualified opinion", "qualified opinion",
    ],
    "regulatory": [
        "regulatory approval", "compliance requirement", "filing deadline",
        "permit", "antitrust", "hart-scott-rodino", "sec filing",
        "environmental compliance", "data protection",
    ],
    "operational": [
        "standard operating procedure", "organizational chart",
        "employee handbook", "supply chain", "vendor management",
        "information technology", "cybersecurity assessment",
        "facilities management", "headcount",
    ],
}


class DocumentClassifier:
    """
    Classifies M&A due diligence documents into standardized categories.

    Classification strategy (in priority order):
    1. Filename pattern matching — fast, reliable for well-named files
    2. Content keyword analysis — catches generic filenames
    3. File extension heuristics — fallback for ambiguous cases

    Categories:
        financial | legal | board | audit | regulatory | operational | other
    """

    def __init__(self) -> None:
        """Initialize the classifier with compiled patterns."""
        self._filename_patterns: dict[DocumentCategory, list[re.Pattern]] = {
            category: [re.compile(p) for p in patterns]
            for category, patterns in FILENAME_PATTERNS.items()
        }

    def classify(
        self,
        file_name: str,
        file_type: str,
        content_sample: str = "",
    ) -> DocumentCategory:
        """
        Classify a document into a category.

        Args:
            file_name: Original filename (e.g., "Q3_Financial_Statement.xlsx").
            file_type: File extension without dot (e.g., "pdf", "docx", "xlsx").
            content_sample: Optional first ~2000 chars of content for keyword matching.

        Returns:
            DocumentCategory string.
        """
        logger.info(
            "Classifying document",
            extra={"file_name": file_name, "file_type": file_type},
        )

        # Strategy 1: Filename pattern matching
        category = self._classify_by_filename(file_name)
        if category != "other":
            logger.info(
                "Classified by filename",
                extra={"file_name": file_name, "category": category},
            )
            return category

        # Strategy 2: Content keyword analysis
        if content_sample:
            category = self._classify_by_content(content_sample)
            if category != "other":
                logger.info(
                    "Classified by content",
                    extra={"file_name": file_name, "category": category},
                )
                return category

        # Strategy 3: File extension heuristics
        category = self._classify_by_extension(file_type, file_name)
        logger.info(
            "Classified by extension/fallback",
            extra={"file_name": file_name, "category": category},
        )
        return category

    def classify_with_source(
        self,
        file_name: str,
        file_type: str,
        content_sample: str = "",
    ) -> tuple[DocumentCategory, str, float | None]:
        """
        Classifies with Laya when enabled, falling back to classify()'s rules.

        Laya reads the filename and the opening text, so a generically named
        upload ("doc_0147.pdf") is filed by what it says rather than by an
        extension guess. On the evaluation set it beat the (fixed) rules on
        TEST, 0.955 vs 0.773 — see eval/decisions/results.md.

        Args:
            file_name: Original filename.
            file_type: File extension without dot.
            content_sample: First ~2000 chars of content.

        Returns:
            (category, source, confidence): source is "laya" or "rules";
            confidence is Laya's probability for its choice, None for rules.
        """
        # laya_enabled() first: with Laya off globally the attempt would only
        # log an "unavailable" warning per upload.
        if laya_enabled() and ingest_signals.laya_category_enabled() and content_sample.strip():
            try:
                category, confidence = ingest_signals.classify_document(file_name, content_sample)
                logger.info(
                    "Classified by Laya",
                    extra={"file_name": file_name, "category": category, "confidence": confidence},
                )
                return category, "laya", confidence  # type: ignore[return-value]
            except LayaUnavailable as e:
                logger.warning(
                    "Laya classification unavailable; using rules",
                    extra={"file_name": file_name, "error": str(e)},
                )
        return self.classify(file_name, file_type, content_sample), "rules", None

    def _classify_by_filename(self, file_name: str) -> DocumentCategory:
        """
        Classify by matching filename against known patterns.

        Args:
            file_name: Document filename.

        Returns:
            Matched category or "other".
        """
        scores: dict[DocumentCategory, int] = {}
        # "_" is a word character, so without this \b never fires inside
        # "quality_of_earnings_report" and every pattern would need its own guard.
        name = file_name.replace("_", " ")

        for category, patterns in self._filename_patterns.items():
            score = sum(1 for p in patterns if p.search(name))
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]
        return "other"

    def _classify_by_content(self, content: str) -> DocumentCategory:
        """
        Classify by counting keyword matches in content.

        Args:
            content: Text sample from the document (first ~2000 chars).

        Returns:
            Category with the most keyword matches, or "other" if no matches.
        """
        content_lower = content.lower()
        scores: dict[DocumentCategory, int] = {}

        for category, keywords in CONTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scores[category] = score

        if scores:
            best = max(scores, key=scores.get)  # type: ignore[arg-type]
            # Require at least 2 keyword matches for content-based classification
            if scores[best] >= 2:
                return best
        return "other"

    def _classify_by_extension(
        self,
        file_type: str,
        file_name: str,
    ) -> DocumentCategory:
        """
        Fallback classification based on file extension.

        Args:
            file_type: File extension (e.g., "xlsx", "pptx").
            file_name: Filename for additional context.

        Returns:
            Best-guess category based on extension.
        """
        extension_hints: dict[str, DocumentCategory] = {
            "xlsx": "financial",
            "xls": "financial",
            "csv": "financial",
            "pptx": "board",
            "ppt": "board",
        }
        return extension_hints.get(file_type.lower(), "other")

    def classify_batch(
        self,
        documents: list[dict],
    ) -> list[tuple[str, DocumentCategory]]:
        """
        Classify multiple documents at once.

        Args:
            documents: List of dicts with keys: file_name, file_type, content_sample (optional).

        Returns:
            List of (file_name, category) tuples.
        """
        results: list[tuple[str, DocumentCategory]] = []

        for doc in documents:
            category = self.classify(
                file_name=doc["file_name"],
                file_type=doc["file_type"],
                content_sample=doc.get("content_sample", ""),
            )
            results.append((doc["file_name"], category))

        logger.info(
            "Batch classification complete",
            extra={"num_documents": len(results)},
        )
        return results
