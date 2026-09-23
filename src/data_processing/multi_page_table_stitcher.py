# src/data_processing/multi_page_table_stitcher.py
"""
Detects and stitches financial tables that span page breaks in PDFs.

Financial statements and cap tables routinely span multiple pages —
naive page-by-page processing silently splits them mid-row or mid-section.

Strategy:
1. Extract tables from each page using pdfplumber
2. Fingerprint column structure (column count, header text if present)
3. If adjacent page has a table with matching column structure but no header row,
   treat it as a continuation and stitch the rows together
4. Store page_range=[start, end] in payload for multi-page tables
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import pdfplumber

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TableStitchError(Exception):
    """Raised on column structure mismatch mid-stitch.

    Signals two distinct adjacent tables, not one continuation.
    """
    pass


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""
    rows: list[list[str | None]]
    headers: list[str]
    page_number: int
    col_count: int
    has_header: bool = True


@dataclass
class StitchedTable:
    """A fully stitched table potentially spanning multiple pages."""
    rows: list[list[str | None]]
    headers: list[str]
    page_range: list[int]
    sheet_name_equivalent: str = ""


class MultiPageTableStitcher:
    """
    Detects and stitches financial tables that span page breaks in PDFs.
    Financial statements and cap tables routinely span multiple pages —
    naive page-by-page processing silently splits them mid-row or mid-section.

    Strategy:
    1. Extract tables from each page using pdfplumber
    2. Fingerprint column structure (column count, header text if present)
    3. If adjacent page has a table with matching column structure but no header row,
       treat it as a continuation and stitch the rows together
    4. Store page_range=[start, end] in payload for multi-page tables

    Args:
        similarity_threshold: Column header similarity required to merge (default 0.85)
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """
        Initialize the table stitcher.

        Args:
            similarity_threshold: Minimum similarity ratio between column
                structures to consider tables as continuations.
        """
        self.similarity_threshold = similarity_threshold

    def stitch_document_tables(self, pdf_path: str) -> list[dict]:
        """
        Process entire PDF, returning fully stitched tables with page_range metadata.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            List of table dicts: {rows, headers, page_range, sheet_name_equivalent}

        Raises:
            TableStitchError: On column structure mismatch mid-stitch (signals
                              two distinct adjacent tables, not one continuation).
            FileNotFoundError: If pdf_path does not exist.
            ValueError: If the PDF contains no extractable tables.
        """
        logger.info("Starting table stitching", extra={"pdf_path": pdf_path})

        page_tables = self._extract_all_page_tables(pdf_path)

        if not page_tables:
            logger.info("No tables found in PDF", extra={"pdf_path": pdf_path})
            return []

        stitched = self._stitch_tables(page_tables)

        result = [
            {
                "rows": table.rows,
                "headers": table.headers,
                "page_range": table.page_range,
                "sheet_name_equivalent": table.sheet_name_equivalent,
            }
            for table in stitched
        ]

        logger.info(
            "Table stitching complete",
            extra={
                "pdf_path": pdf_path,
                "num_stitched_tables": len(result),
                "page_ranges": [t["page_range"] for t in result],
            },
        )
        return result

    def _extract_all_page_tables(self, pdf_path: str) -> list[ExtractedTable]:
        """
        Extract tables from every page of a PDF using pdfplumber.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Flat list of ExtractedTable instances with page attribution.
        """
        all_tables: list[ExtractedTable] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_number = page_idx + 1
                    tables = page.extract_tables()

                    if not tables:
                        continue

                    for table_data in tables:
                        if not table_data or len(table_data) < 1:
                            continue

                        extracted = self._parse_raw_table(table_data, page_number)
                        if extracted is not None:
                            all_tables.append(extracted)

        except FileNotFoundError:
            logger.error("PDF file not found", extra={"pdf_path": pdf_path})
            raise
        except Exception as e:
            logger.error(
                "Error extracting tables from PDF",
                extra={"pdf_path": pdf_path, "error": str(e)},
            )
            raise

        logger.info(
            "Page-level table extraction complete",
            extra={"pdf_path": pdf_path, "num_tables": len(all_tables)},
        )
        return all_tables

    def _parse_raw_table(
        self,
        raw_table: list[list[str | None]],
        page_number: int,
    ) -> ExtractedTable | None:
        """
        Parse raw pdfplumber table data into an ExtractedTable.

        Determines if the first row is a header row by checking whether it
        contains mostly non-numeric text.

        Args:
            raw_table: Raw table rows from pdfplumber.
            page_number: Source page number.

        Returns:
            ExtractedTable instance or None if table is trivially small.
        """
        if len(raw_table) < 2:
            return None

        # Clean cells: replace None with empty strings for comparison
        cleaned = [
            [cell.strip() if cell else "" for cell in row]
            for row in raw_table
        ]

        # Determine header
        first_row = cleaned[0]
        has_header = self._is_header_row(first_row)

        if has_header:
            headers = first_row
            data_rows = cleaned[1:]
        else:
            headers = []
            data_rows = cleaned

        col_count = max(len(row) for row in cleaned) if cleaned else 0

        return ExtractedTable(
            rows=data_rows,
            headers=headers,
            page_number=page_number,
            col_count=col_count,
            has_header=has_header,
        )

    def _is_header_row(self, row: list[str]) -> bool:
        """
        Heuristic: a row is a header if most cells are non-empty and
        predominantly non-numeric text.

        Args:
            row: List of cell values.

        Returns:
            True if the row appears to be a header row.
        """
        if not row:
            return False

        non_empty = [cell for cell in row if cell.strip()]
        if not non_empty:
            return False

        # Count cells that are purely numeric (possibly with commas, decimals, $)
        numeric_pattern = r'^[\s$€£¥₹]*[-+]?[\d,]+\.?\d*%?\s*$'
        numeric_count = sum(
            1 for cell in non_empty
            if __import__("re").match(numeric_pattern, cell)
        )

        # If less than half are numeric, it's likely a header
        return numeric_count < len(non_empty) / 2

    def _stitch_tables(self, tables: list[ExtractedTable]) -> list[StitchedTable]:
        """
        Stitch consecutive tables with matching column structures.

        Args:
            tables: List of extracted tables sorted by page order.

        Returns:
            List of StitchedTable instances.
        """
        if not tables:
            return []

        stitched_tables: list[StitchedTable] = []
        current: StitchedTable | None = None

        for table in tables:
            if current is None:
                # Start a new stitched table
                current = StitchedTable(
                    rows=list(table.rows),
                    headers=list(table.headers),
                    page_range=[table.page_number, table.page_number],
                    sheet_name_equivalent=self._infer_table_name(table.headers),
                )
                continue

            # Check if this table is a continuation of the current one
            if self._is_continuation(current, table):
                # Stitch: append rows and extend page range
                current.rows.extend(table.rows)
                current.page_range[1] = table.page_number
                logger.debug(
                    "Stitched continuation table",
                    extra={
                        "page": table.page_number,
                        "page_range": current.page_range,
                    },
                )
            else:
                # Save current and start a new one
                stitched_tables.append(current)
                current = StitchedTable(
                    rows=list(table.rows),
                    headers=list(table.headers),
                    page_range=[table.page_number, table.page_number],
                    sheet_name_equivalent=self._infer_table_name(table.headers),
                )

        # Don't forget the last one
        if current is not None:
            stitched_tables.append(current)

        return stitched_tables

    def _is_continuation(
        self,
        current: StitchedTable,
        candidate: ExtractedTable,
    ) -> bool:
        """
        Determines if a candidate table is a continuation of the current stitched table.

        Criteria:
        - Candidate is on the page immediately after the current table's last page
        - Candidate has no header row (continuation tables typically lack headers)
        - Column count matches
        - If candidate has headers, they must be very similar to current headers

        Args:
            current: The current in-progress stitched table.
            candidate: The candidate table to potentially merge.

        Returns:
            True if the candidate is a continuation.
        """
        # A table can only continue across a page break. Two tables on the same
        # page, or pages apart, are distinct tables even when their column
        # counts match — merging them fabricates rows the source never had.
        if candidate.page_number != current.page_range[1] + 1:
            return False

        # If candidate has a header, it's likely a new table
        # Unless the headers match very closely (repeated header across pages)
        if candidate.has_header and candidate.headers:
            if current.headers:
                similarity = self._header_similarity(current.headers, candidate.headers)
                if similarity >= self.similarity_threshold:
                    # Same headers repeated — it's a continuation with repeated header
                    return True
            return False

        # No header on candidate — check column count compatibility
        current_col_count = len(current.headers) if current.headers else (
            max(len(r) for r in current.rows) if current.rows else 0
        )
        candidate_col_count = candidate.col_count

        if current_col_count == 0 or candidate_col_count == 0:
            return False

        # Allow minor column count differences (some cells may span)
        if abs(current_col_count - candidate_col_count) <= 1:
            return True

        return False

    def _header_similarity(
        self,
        headers_a: list[str],
        headers_b: list[str],
    ) -> float:
        """
        Compute similarity between two header rows using SequenceMatcher.

        Args:
            headers_a: First header row.
            headers_b: Second header row.

        Returns:
            Similarity ratio between 0.0 and 1.0.
        """
        if not headers_a or not headers_b:
            return 0.0

        # Normalize headers for comparison
        norm_a = " | ".join(h.lower().strip() for h in headers_a)
        norm_b = " | ".join(h.lower().strip() for h in headers_b)

        return SequenceMatcher(None, norm_a, norm_b).ratio()

    def _infer_table_name(self, headers: list[str]) -> str:
        """
        Infer a descriptive name for the table from its headers.

        Args:
            headers: Column headers of the table.

        Returns:
            Inferred table name or "Unknown Table".
        """
        if not headers:
            return "Unknown Table"

        # Use the first non-empty header as a base name
        first_header = next(
            (h for h in headers if h and h.strip()), "Unknown Table"
        )

        # Common financial table header patterns
        financial_keywords = {
            "revenue": "Income Statement",
            "income": "Income Statement",
            "balance": "Balance Sheet",
            "assets": "Balance Sheet",
            "cash flow": "Cash Flow Statement",
            "equity": "Equity Statement",
            "cap table": "Cap Table",
            "capitalization": "Cap Table",
        }

        combined = " ".join(h.lower() for h in headers if h)
        for keyword, name in financial_keywords.items():
            if keyword in combined:
                return name

        return first_header.strip()[:100]
