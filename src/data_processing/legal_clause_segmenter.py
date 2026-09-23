# src/data_processing/legal_clause_segmenter.py
"""
Pattern-based clause segmenter for legal contract PDFs.

Legal PDFs use numbered clauses in body text — NOT Word heading styles.
Font-size heuristics (used for layout PDFs) do not work here.

Each numbered or lettered clause becomes one structural unit
before semantic chunking is applied within it.

Assigns clause_id from the detected numbering (e.g., "4.3(a)(i)")
for direct citation by the answer synthesizer.
"""

import re
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Clause boundary patterns (from p4.md spec) ──────────────────────────────
CLAUSE_PATTERNS = [
    # Article headers: "ARTICLE I", "Article 1.", "ARTICLE IV:"
    r'^(ARTICLE\s+[IVXLCDM]+|Article\s+\d+)[\.:\s]',
    # Numbered sections: "1.", "4.3", "12.1.2"
    r'^\s*\d+(\.\d+)*\.?\s+[A-Z]',
    # Lettered subsections: "(a)", "(iv)", "(A)"
    r'^\s*\([a-zA-Z]+\)\s',
    # Roman numeral subsections: "(i)", "(iv)"
    r'^\s*\([ivxlcdmIVXLCDM]+\)\s',
    # Common contract headers
    r'^(WHEREAS|NOW,?\s+THEREFORE|IN WITNESS WHEREOF|RECITALS?|DEFINITIONS?)',
    # Signature block
    r'^(EXECUTED|SIGNED|IN WITNESS)',
]

COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in CLAUSE_PATTERNS]

# Pattern to extract clause identifiers from text
CLAUSE_ID_PATTERN = re.compile(
    r'^(?:'
    r'(ARTICLE\s+[IVXLCDM]+|Article\s+\d+)'           # Article headers
    r'|(\d+(?:\.\d+)*)'                                 # Numbered sections
    r'|(\([a-zA-Z]+\))'                                 # Lettered subsections
    r'|(\([ivxlcdmIVXLCDM]+\))'                         # Roman numeral subs
    r'|(WHEREAS|NOW,?\s+THEREFORE|IN WITNESS WHEREOF'
    r'|RECITALS?|DEFINITIONS?|EXECUTED|SIGNED|IN WITNESS)'  # Named headers
    r')\s*',
    re.MULTILINE,
)


class SegmentationError(Exception):
    """Raised when no clause boundaries are detected in the input text."""
    pass


@dataclass
class ClauseSegment:
    """A single segmented clause from a legal document."""
    text: str
    clause_id: str
    page_number: int
    start_offset: int = 0
    end_offset: int = 0


class LegalClauseSegmenter:
    """
    Pattern-based clause segmenter for legal contract PDFs.
    Legal PDFs use numbered clauses in body text — NOT Word heading styles.
    Font-size heuristics (used for layout PDFs) do not work here.

    Each numbered or lettered clause becomes one structural unit
    before semantic chunking is applied within it.

    Assigns clause_id from the detected numbering (e.g., "4.3(a)(i)")
    for direct citation by the answer synthesizer.
    """

    def __init__(self) -> None:
        """Initialize the segmenter with compiled patterns."""
        self._patterns = COMPILED_PATTERNS
        self._clause_id_pattern = CLAUSE_ID_PATTERN

    def segment(self, text: str, page_number: int) -> list[dict]:
        """
        Segments legal contract text into clause-based structural units.

        Args:
            text: Full extracted text from a legal PDF page or section.
            page_number: Source page number for citation metadata.

        Returns:
            List of dicts with keys: text, clause_id, page_number.

        Raises:
            SegmentationError: If no clause boundaries detected (falls back to
                               paragraph-level segmentation).
        """
        logger.info(
            "Segmenting legal text",
            extra={"page_number": page_number, "text_length": len(text)},
        )

        if not text or not text.strip():
            logger.warning("Empty text provided for segmentation")
            return []

        # Find all clause boundary positions
        boundaries = self._find_boundaries(text)

        if not boundaries:
            logger.warning(
                "No clause boundaries detected, falling back to paragraph segmentation",
                extra={"page_number": page_number},
            )
            return self._paragraph_fallback(text, page_number)

        # Sort boundaries by position
        boundaries.sort(key=lambda b: b[0])

        # Build segments from boundaries
        segments = self._build_segments(text, boundaries, page_number)

        logger.info(
            "Segmentation complete",
            extra={
                "page_number": page_number,
                "num_segments": len(segments),
                "clause_ids": [s["clause_id"] for s in segments],
            },
        )

        return segments

    def _find_boundaries(self, text: str) -> list[tuple[int, str]]:
        """
        Finds all clause boundary positions in the text.

        Args:
            text: Full text to scan for clause boundaries.

        Returns:
            List of (position, matched_text) tuples for each boundary.
        """
        boundaries: list[tuple[int, str]] = []
        seen_positions: set[int] = set()

        for pattern in self._patterns:
            for match in pattern.finditer(text):
                pos = match.start()
                if pos not in seen_positions:
                    seen_positions.add(pos)
                    boundaries.append((pos, match.group(0).strip()))

        return boundaries

    def _extract_clause_id(self, boundary_text: str, text_after: str) -> str:
        """
        Extracts a structured clause_id from the boundary text.

        Builds hierarchical IDs like "4.3(a)(i)" by combining the primary
        numbering with any inline sub-numbering found in the clause start.

        Args:
            boundary_text: The matched boundary text.
            text_after: Text following the boundary for context.

        Returns:
            Extracted clause_id string (e.g., "ARTICLE_I", "4.3", "(a)").
        """
        combined = boundary_text + " " + text_after[:200]

        match = self._clause_id_pattern.match(combined.strip())
        if not match:
            return boundary_text.strip()[:50]

        # Extract the first matched group
        groups = match.groups()
        clause_id_parts: list[str] = []

        if groups[0]:  # Article header
            clause_id_parts.append(groups[0].replace(" ", "_"))
        elif groups[1]:  # Numbered section
            clause_id_parts.append(groups[1])
        elif groups[2]:  # Lettered subsection
            clause_id_parts.append(groups[2])
        elif groups[3]:  # Roman numeral subsection
            clause_id_parts.append(groups[3])
        elif groups[4]:  # Named header
            clause_id_parts.append(groups[4].replace(",", "").replace(" ", "_"))

        # Look for inline sub-numbering after the primary ID
        remaining = combined[match.end():]
        sub_patterns = [
            re.compile(r'^\s*(\([a-zA-Z]+\))'),
            re.compile(r'^\s*(\([ivxlcdmIVXLCDM]+\))'),
        ]
        for sub_pat in sub_patterns:
            sub_match = sub_pat.match(remaining)
            if sub_match:
                clause_id_parts.append(sub_match.group(1))
                remaining = remaining[sub_match.end():]

        return "".join(clause_id_parts) if clause_id_parts else boundary_text.strip()[:50]

    def _build_segments(
        self,
        text: str,
        boundaries: list[tuple[int, str]],
        page_number: int,
    ) -> list[dict]:
        """
        Builds segment dicts from identified boundary positions.

        Args:
            text: Full text.
            boundaries: Sorted list of (position, matched_text) tuples.
            page_number: Source page number.

        Returns:
            List of segment dicts with text, clause_id, page_number.
        """
        segments: list[dict] = []

        # If the first boundary doesn't start at position 0, capture preamble
        if boundaries[0][0] > 0:
            preamble_text = text[: boundaries[0][0]].strip()
            if preamble_text:
                segments.append({
                    "text": preamble_text,
                    "clause_id": "PREAMBLE",
                    "page_number": page_number,
                })

        for i, (pos, boundary_text) in enumerate(boundaries):
            # Determine end of this segment
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(text)

            segment_text = text[pos:end_pos].strip()
            if not segment_text:
                continue

            # Extract clause ID
            text_after = text[pos + len(boundary_text): min(pos + len(boundary_text) + 200, len(text))]
            clause_id = self._extract_clause_id(boundary_text, text_after)

            segments.append({
                "text": segment_text,
                "clause_id": clause_id,
                "page_number": page_number,
            })

        return segments

    def _paragraph_fallback(self, text: str, page_number: int) -> list[dict]:
        """
        Fallback segmentation when no clause boundaries are detected.
        Splits on double newlines to create paragraph-level segments.

        Args:
            text: Full text to segment.
            page_number: Source page number.

        Returns:
            List of paragraph-level segment dicts.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        segments: list[dict] = []

        for idx, para in enumerate(paragraphs):
            stripped = para.strip()
            if not stripped:
                continue
            segments.append({
                "text": stripped,
                "clause_id": f"PARA_{idx + 1}",
                "page_number": page_number,
            })

        logger.info(
            "Paragraph fallback produced segments",
            extra={"page_number": page_number, "num_segments": len(segments)},
        )
        return segments

    def segment_multipage(
        self,
        pages: list[tuple[int, str]],
    ) -> list[dict]:
        """
        Segments text across multiple pages, preserving page attribution.

        Args:
            pages: List of (page_number, page_text) tuples.

        Returns:
            Flat list of clause segments across all pages.
        """
        logger.info(
            "Segmenting multi-page legal document",
            extra={"num_pages": len(pages)},
        )
        all_segments: list[dict] = []
        for page_number, page_text in pages:
            page_segments = self.segment(page_text, page_number)
            all_segments.extend(page_segments)

        logger.info(
            "Multi-page segmentation complete",
            extra={"total_segments": len(all_segments)},
        )
        return all_segments
