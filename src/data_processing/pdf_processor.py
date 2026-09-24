"""
PDF layout processor using PyMuPDF (fitz).

Extracts text with layout awareness: headings detected by font size,
paragraphs, tables, and page numbers. Integrates with LegalClauseSegmenter
for legal PDFs and MultiPageTableStitcher for financial PDFs.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field

from src.data_processing.legal_clause_segmenter import LegalClauseSegmenter
from src.data_processing.multi_page_table_stitcher import MultiPageTableStitcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PDFSection:
    """A structural section extracted from a PDF."""
    text: str
    page_number: int
    page_range: list[int] = field(default_factory=list)
    section_heading: str = ""
    section_type: str = "body"  # heading, body, table, footer, header
    font_size: float = 0.0
    is_table: bool = False
    clause_id: str | None = None
    # Header row first, then data rows — lets the ingestion pipeline build the
    # FinancialTableConverter representations instead of indexing flat text.
    table_rows: list[list[str]] | None = None


class PDFProcessor:
    """
    PDF layout processor that extracts structured sections from PDF documents.

    Strategy:
    1. Extract page-by-page text with font metadata via PyMuPDF
    2. Detect headings by font size (> median * 1.2)
    3. Group text into structural sections under headings
    4. For legal PDFs: delegate to LegalClauseSegmenter
    5. For financial PDFs: delegate to MultiPageTableStitcher for tables

    Args:
        legal_mode: If True, use clause segmentation instead of font-based headings.
    """

    def __init__(self, legal_mode: bool = False):
        self.legal_mode = legal_mode
        self._clause_segmenter = LegalClauseSegmenter() if legal_mode else None
        self._table_stitcher = MultiPageTableStitcher()

    def process(self, pdf_path: str, doc_id: str) -> list[PDFSection]:
        """
        Processes a PDF file into structured sections.

        Args:
            pdf_path: Absolute path to the PDF file.
            doc_id: Document identifier for metadata.

        Returns:
            List of PDFSection objects with text, page numbers, and headings.

        Raises:
            FileNotFoundError: If pdf_path does not exist.
            fitz.FileDataError: If PDF is corrupted or encrypted.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(
            "Processing PDF",
            extra={"pdf_path": pdf_path, "doc_id": doc_id, "legal_mode": self.legal_mode},
        )

        doc = fitz.open(pdf_path)
        sections: list[PDFSection] = []
        # Read before close(): PyMuPDF raises ValueError("document closed") on
        # any attribute access afterwards, which used to fail every PDF ingest.
        total_pages = doc.page_count

        try:
            if self.legal_mode:
                sections = self._process_legal(doc)
            else:
                sections = self._process_layout(doc)

            # Extract tables via pdfplumber integration
            table_sections = self._extract_tables(pdf_path)
            sections.extend(table_sections)

        finally:
            doc.close()

        logger.info(
            "PDF processing complete",
            extra={
                "doc_id": doc_id,
                "total_sections": len(sections),
                "total_pages": total_pages,
            },
        )

        return sections

    def _process_layout(self, doc: fitz.Document) -> list[PDFSection]:
        """
        Extract text with font-size-based heading detection.

        Args:
            doc: Open PyMuPDF document.

        Returns:
            List of PDFSection with headings and body text.
        """
        sections = []
        current_heading = ""
        current_text_parts: list[str] = []
        current_page = 1

        for page_num in range(len(doc)):
            # Close the running section at every page break so each body
            # section is cited with the page its text is actually on, not the
            # page its heading started on.
            if current_text_parts:
                sections.append(PDFSection(
                    text="\n".join(current_text_parts),
                    page_number=current_page,
                    section_heading=current_heading,
                    section_type="body",
                ))
                current_text_parts = []
            current_page = page_num + 1

            page = doc[page_num]
            blocks = page.get_text("dict", sort=True)["blocks"]

            # Collect font sizes to determine median
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(span["size"])

            median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 12.0
            heading_threshold = median_size * 1.2

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_text = ""
                    max_font = 0.0
                    for span in line["spans"]:
                        line_text += span["text"]
                        max_font = max(max_font, span["size"])

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    # Heading detection
                    if max_font >= heading_threshold and len(line_text) < 200:
                        # Flush previous section
                        if current_text_parts:
                            sections.append(PDFSection(
                                text="\n".join(current_text_parts),
                                page_number=current_page,
                                section_heading=current_heading,
                                section_type="body",
                            ))
                            current_text_parts = []

                        current_heading = line_text
                        current_page = page_num + 1
                        sections.append(PDFSection(
                            text=line_text,
                            page_number=page_num + 1,
                            section_heading=line_text,
                            section_type="heading",
                            font_size=max_font,
                        ))
                    else:
                        current_text_parts.append(line_text)

        # Flush final section
        if current_text_parts:
            sections.append(PDFSection(
                text="\n".join(current_text_parts),
                page_number=current_page,
                section_heading=current_heading,
                section_type="body",
            ))

        return sections

    def _process_legal(self, doc: fitz.Document) -> list[PDFSection]:
        """
        Process legal PDF using clause segmentation instead of font-based headings.

        Args:
            doc: Open PyMuPDF document.

        Returns:
            List of PDFSection with clause_ids.
        """
        sections = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if not text.strip():
                continue

            clauses = self._clause_segmenter.segment(text, page_num + 1)
            for clause in clauses:
                sections.append(PDFSection(
                    text=clause["text"],
                    page_number=clause["page_number"],
                    section_heading=clause.get("clause_id", ""),
                    section_type="clause",
                    clause_id=clause.get("clause_id"),
                ))

        return sections

    def _extract_tables(self, pdf_path: str) -> list[PDFSection]:
        """
        Extract tables using MultiPageTableStitcher.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of PDFSection for each extracted table.
        """
        try:
            tables = self._table_stitcher.stitch_document_tables(pdf_path)
        except Exception as e:
            logger.warning(
                "Table extraction failed, continuing without tables",
                extra={"error": str(e)},
            )
            return []

        sections = []
        for table in tables:
            # Convert table rows to text
            rows_text = []
            if table.get("headers"):
                rows_text.append(" | ".join(str(h) for h in table["headers"]))
                rows_text.append("-" * 40)
            for row in table.get("rows", []):
                rows_text.append(" | ".join(str(c) for c in row))

            text = "\n".join(rows_text)
            page_range = table.get("page_range", [1])
            table_rows = (
                [list(table["headers"])] + [list(r) for r in table.get("rows", [])]
                if table.get("headers") else None
            )

            sections.append(PDFSection(
                text=text,
                page_number=page_range[0],
                page_range=page_range,
                section_heading=table.get("sheet_name_equivalent", "Table"),
                section_type="table",
                is_table=True,
                table_rows=table_rows,
            ))

        return sections
