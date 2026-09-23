"""
Structural chunker — heading/section-based splitting.

First tier of the 3-tier chunking pipeline:
Structural → Semantic → Financial Special Handling.

Splits documents at heading/section boundaries before semantic chunking
refines within each section.
"""

from dataclasses import dataclass, field

from src.utils.token_counter import count_tokens
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Section keys the chunker consumes itself. Every other top-level key on a
# section (sheet_name, currency, table_id, content_type, slide_number, ...) is
# carried through in StructuralChunk.metadata — processors emit them at the top
# level, and dropping them here is what used to strip table and slide metadata
# before it ever reached the payload.
_CORE_SECTION_KEYS = {
    "text",
    "section_heading",
    "page_number",
    "page_range",
    "section_type",
    "clause_id",
    "is_table",
    "metadata",
}


@dataclass
class StructuralChunk:
    """A chunk produced by structural splitting."""
    text: str
    section_heading: str = ""
    page_number: int | None = None  # None when the format has no pages
    page_range: list[int] = field(default_factory=list)
    chunk_type: str = "text"  # text, table, clause
    clause_id: str | None = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


def _section_metadata(section: dict) -> dict:
    """
    Collects a section's metadata: its explicit `metadata` dict plus any
    non-core top-level keys.

    Args:
        section: Section dict from a document processor.

    Returns:
        Merged metadata dict (top-level keys win).
    """
    meta = dict(section.get("metadata") or {})
    for key, value in section.items():
        if key not in _CORE_SECTION_KEYS:
            meta[key] = value
    if section.get("is_table"):
        meta["is_table"] = 1
    return meta


class StructuralChunker:
    """
    Splits document content at structural boundaries (headings, sections,
    clauses, page breaks) before semantic chunking refines within sections.

    Preserves section_heading metadata for citation.
    Does NOT split tables — they are handled separately by FinancialTableConverter.

    Small sections are merged with a neighbour only when both share the same
    heading, page, clause and metadata — otherwise the merged chunk would cite
    the first section's heading/page for text that belongs to another.

    Args:
        max_tokens: Maximum chunk size (default 2048 — larger than semantic max
                    because semantic chunker will sub-split).
        min_tokens: Minimum chunk size (default 64 — merge very small sections).
    """

    def __init__(self, max_tokens: int = 2048, min_tokens: int = 64):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk(self, sections: list[dict]) -> list[StructuralChunk]:
        """
        Splits sections into structural chunks.

        Args:
            sections: List of dicts from document processors. Expected keys:
                      text, section_heading, page_number, section_type,
                      clause_id (optional), is_table (optional). Any other
                      keys are preserved in StructuralChunk.metadata.

        Returns:
            List of StructuralChunk objects.
        """
        logger.info(
            "Structural chunking starting",
            extra={"num_sections": len(sections)},
        )

        chunks: list[StructuralChunk] = []
        current: StructuralChunk | None = None

        def flush() -> None:
            nonlocal current
            if current is not None:
                chunks.append(current)
                current = None

        for section in sections:
            text = (section.get("text") or "").strip()
            if not text:
                continue

            heading = section.get("section_heading", "") or ""
            page = section.get("page_number")
            clause_id = section.get("clause_id")
            meta = _section_metadata(section)

            # Tables pass through as-is (not split by structural chunker)
            if section.get("is_table") or section.get("section_type") == "table":
                flush()
                chunks.append(StructuralChunk(
                    text=text,
                    section_heading=heading,
                    page_number=page,
                    page_range=section.get("page_range", []),
                    chunk_type="table",
                    clause_id=clause_id,
                    token_count=count_tokens(text),
                    metadata=meta,
                ))
                continue

            tokens = count_tokens(text)
            section_type = section.get("section_type", "text")

            compatible = (
                current is not None
                and current.section_heading == heading
                and current.page_number == page
                and current.clause_id == clause_id
                and current.metadata == meta
            )
            if (
                compatible
                and current.token_count + tokens <= self.max_tokens
                and (current.token_count < self.min_tokens or tokens < self.min_tokens)
            ):
                current.text = current.text + "\n\n" + text
                current.token_count = count_tokens(current.text)
                if current.chunk_type != section_type:
                    current.chunk_type = "text"
                continue

            flush()

            # Section within limits — becomes the open chunk so a small
            # neighbour can still merge into it.
            if tokens <= self.max_tokens:
                current = StructuralChunk(
                    text=text,
                    section_heading=heading,
                    page_number=page,
                    clause_id=clause_id,
                    chunk_type=section_type,
                    token_count=tokens,
                    metadata=meta,
                )
                continue

            # Section too large — split at paragraph boundaries
            for part_text in self._split_paragraphs(text):
                chunks.append(StructuralChunk(
                    text=part_text,
                    section_heading=heading,
                    page_number=page,
                    clause_id=clause_id,
                    chunk_type="text",
                    token_count=count_tokens(part_text),
                    metadata=dict(meta),
                ))

        flush()

        logger.info(
            "Structural chunking complete",
            extra={
                "input_sections": len(sections),
                "output_chunks": len(chunks),
                "table_chunks": sum(1 for c in chunks if c.chunk_type == "table"),
            },
        )

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Packs the paragraphs of an oversize section into pieces of at most
        max_tokens. A single paragraph larger than that is kept whole here; the
        semantic chunker enforces the hard retrieval cap one tier down.

        Args:
            text: Section text.

        Returns:
            List of piece texts.
        """
        pieces: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > self.max_tokens and current_parts:
                pieces.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            pieces.append("\n\n".join(current_parts))
        return pieces
