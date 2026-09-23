# src/data_processing/document_classifier.py
"""
Document classifier for M&A due diligence documents.

Classifies documents into one of 7 categories based on filename patterns,
content keywords, and structural heuristics:
    financial | legal | board | audit | regulatory | operational | other

The classification is stored as document_category in the Qdrant payload
and used as a filter parameter during retrieval.
"""

from __future__ import annotations

import re
from typing import Literal

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DocumentCategory = Literal[
    "financial", "legal", "board", "audit", "regulatory", "operational", "other"
]

# ─── Filename-based classification patterns ──────────────────────────────────
FILENAME_PATTERNS: dict[DocumentCategory, list[str]] = {
    "financial": [
        r'(?i)(income.?statement|balance.?sheet|cash.?flow|p\s*&?\s*l|profit.?loss)',
        r'(?i)(financial.?statement|cap.?table|capitalization|revenue|budget)',
        r'(?i)(forecast|projection|valuation|dcf|model|ebitda)',
        r'(?i)(10-?[kq]|annual.?report|quarterly.?report)',
    ],
    "legal": [
        r'(?i)(contract|agreement|amendment|addendum|mou|memorandum)',
        r'(?i)(merger|acquisition|purchase|sale|asset.?purchase)',
        r'(?i)(nda|non.?disclosure|confidential|indemnif)',
        r'(?i)(term.?sheet|loi|letter.?of.?intent|definitive)',
        r'(?i)(license|lease|employment.?agreement|ip.?assign)',
        r'(?i)(representation|warrant|covenant|escrow)',
    ],
    "board": [
        r'(?i)(board|director|presentation|deck|slide|pptx)',
        r'(?i)(committee|governance|meeting.?minute|resolution)',
        r'(?i)(strategy|overview|executive.?summary)',
    ],
    "audit": [
        r'(?i)(audit|auditor|sox|internal.?control)',
        r'(?i)(compliance|accounting|gaap|ifrs)',
        r'(?i)(review|assessment|finding|observation)',
    ],
    "regulatory": [
        r'(?i)(regulatory|regulation|filing|permit|license)',
        r'(?i)(sec|fda|epa|osha|ftc|doj|antitrust|hsr)',
        r'(?i)(compliance.?report|consent|decree|enforcement)',
    ],
    "operational": [
        r'(?i)(operational|operation|process|procedure|workflow)',
        r'(?i)(hr|human.?resource|employee|headcount|org.?chart)',
        r'(?i)(it|technology|system|infrastructure|cybersecurity)',
        r'(?i)(supply.?chain|vendor|customer|inventory)',
        r'(?i)(insurance|real.?estate|property|facility)',
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

    def _classify_by_filename(self, file_name: str) -> DocumentCategory:
        """
        Classify by matching filename against known patterns.

        Args:
            file_name: Document filename.

        Returns:
            Matched category or "other".
        """
        scores: dict[DocumentCategory, int] = {}

        for category, patterns in self._filename_patterns.items():
            score = sum(1 for p in patterns if p.search(file_name))
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
