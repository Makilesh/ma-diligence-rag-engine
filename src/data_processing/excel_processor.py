"""
Excel processor using pandas + openpyxl.

Extracts sheets, detects tables, normalizes with ExcelNormalizer,
and generates 4 representations via FinancialTableConverter.

Only .xlsx is supported: openpyxl cannot read the legacy binary .xls format,
and advertising it meant every .xls upload failed.
"""

from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

from src.data_processing.excel_normalizer import ExcelNormalizer, TableNormalizationMeta
from src.data_processing.financial_table_converter import (
    FinancialTableConverter,
    frame_from_rows,
    representation_content_type,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExcelProcessingError(Exception):
    """Raised when no sheet of a workbook could be processed."""


@dataclass
class ExcelSheet:
    """Extracted content from a single Excel sheet."""
    sheet_name: str
    dataframe: pd.DataFrame = None
    normalization_meta: TableNormalizationMeta = None
    table_id: str = ""
    # FinancialTableConverter output: one dict per representation, all sharing table_id
    representations: list[dict] = field(default_factory=list)
    header_rows: list[str] = field(default_factory=list)


class ExcelProcessor:
    """
    Processes Excel files into structured, normalized table data.

    Pipeline:
    1. Read each sheet with pandas
    2. Detect and extract header rows
    3. Normalize scale/currency via ExcelNormalizer
    4. Generate 4 representations via FinancialTableConverter

    Sheet-level failures are recorded in `failed_sheets` and logged at error.
    If every non-empty sheet fails, process() raises instead of returning an
    empty list — an upload that silently yields zero chunks looks like success.
    """

    def __init__(self):
        self._normalizer = ExcelNormalizer()
        self._converter = FinancialTableConverter()
        self.failed_sheets: list[dict] = []

    def process(
        self,
        excel_path: str,
        doc_id: str,
        table_id_prefix: str | None = None,
    ) -> list[ExcelSheet]:
        """
        Processes an Excel file into structured sheet objects.

        Args:
            excel_path: Absolute path to the Excel file.
            doc_id: Document identifier for metadata.
            table_id_prefix: Prefix for per-sheet table_ids (default: doc_id).
                The ingestion pipeline passes "{deal_id}_{doc_id}".

        Returns:
            List of ExcelSheet objects with normalized data and representations.

        Raises:
            FileNotFoundError: If excel_path does not exist.
            ExcelProcessingError: If sheets had content but none could be processed.
        """
        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        logger.info(
            "Processing Excel file",
            extra={"path": excel_path, "doc_id": doc_id},
        )

        prefix = table_id_prefix or doc_id
        self.failed_sheets = []

        # Read all sheets
        sheets: list[ExcelSheet] = []
        with pd.ExcelFile(excel_path, engine="openpyxl") as xl:
            for seq, sheet_name in enumerate(xl.sheet_names):
                try:
                    sheet = self._process_sheet(
                        xl, sheet_name, doc_id, table_id=f"{prefix}_t{seq:03d}"
                    )
                    if sheet is not None:
                        sheets.append(sheet)
                except Exception as e:
                    logger.error(
                        f"Failed to process sheet '{sheet_name}'",
                        extra={"error": str(e), "doc_id": doc_id},
                        exc_info=True,
                    )
                    self.failed_sheets.append({"sheet_name": sheet_name, "error": str(e)})

        if not sheets and self.failed_sheets:
            raise ExcelProcessingError(
                f"None of {len(self.failed_sheets)} sheet(s) could be processed: "
                + "; ".join(f"{f['sheet_name']}: {f['error']}" for f in self.failed_sheets)
            )

        logger.info(
            "Excel processing complete",
            extra={
                "doc_id": doc_id,
                "total_sheets": len(sheets),
                "failed_sheets": [f["sheet_name"] for f in self.failed_sheets],
                "sheet_names": [s.sheet_name for s in sheets],
            },
        )

        return sheets

    def _process_sheet(
        self,
        xl: pd.ExcelFile,
        sheet_name: str,
        doc_id: str,
        table_id: str,
    ) -> ExcelSheet | None:
        """
        Processes a single sheet.

        Args:
            xl: Open ExcelFile object.
            sheet_name: Name of the sheet to process.
            doc_id: Document identifier.
            table_id: Identifier shared by all representations of this sheet.

        Returns:
            ExcelSheet object or None if sheet is empty.
        """
        # Read with all data as strings first to detect headers
        df_raw = pd.read_excel(xl, sheet_name=sheet_name, header=None, dtype=str)

        if df_raw.empty:
            return None

        # Find header row — first row with > 50% non-null string values
        header_row_idx = 0
        for idx in range(min(10, len(df_raw))):
            row = df_raw.iloc[idx]
            non_null = row.notna().sum()
            if non_null > len(row) * 0.5:
                # Check if values look like headers (non-numeric)
                str_count = sum(
                    1 for v in row if pd.notna(v) and not self._is_numeric(str(v))
                )
                if str_count > non_null * 0.5:
                    header_row_idx = idx
                    break

        # Re-read with detected header row
        df = pd.read_excel(xl, sheet_name=sheet_name, header=header_row_idx)

        # Drop completely empty rows/columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            return None

        # Extract header cells for normalization
        header_cells = [str(c) for c in df.columns.tolist()]

        # Also check first few rows above header for scale indicators
        pre_header_text = []
        if header_row_idx > 0:
            for idx in range(header_row_idx):
                row_text = " ".join(str(v) for v in df_raw.iloc[idx] if pd.notna(v))
                if row_text.strip():
                    pre_header_text.append(row_text)

        all_header_context = header_cells + pre_header_text

        # Normalize scale and currency
        norm_meta = self._normalizer.detect_scale(all_header_context)

        # The converter expects row labels as the index and periods as columns;
        # the first sheet column holds the line-item labels.
        table_df = frame_from_rows(header_cells, df.values.tolist())
        if table_df is not None:
            representations = self._converter.generate_all_representations(
                df=table_df,
                meta=norm_meta,
                table_id=table_id,
                source_metadata={"sheet_name": sheet_name},
            )
        else:
            # No numeric table (a notes or assumptions sheet): index the cells
            # verbatim rather than inventing financial representations.
            lines = [" | ".join(header_cells)] + [
                " | ".join("" if pd.isna(v) else str(v) for v in row)
                for row in df.values.tolist()
            ]
            representations = [{
                "text": "\n".join(lines),
                "table_representation": "verbatim",
                "table_id": table_id,
                "currency": norm_meta.currency,
                "scale_factor": norm_meta.scale_factor,
                "scale_label": norm_meta.scale_label,
            }]

        return ExcelSheet(
            sheet_name=sheet_name,
            dataframe=table_df,
            normalization_meta=norm_meta,
            table_id=table_id,
            representations=representations,
            header_rows=header_cells,
        )

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Check if a string value looks numeric."""
        try:
            cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
            if cleaned in ("", "-", "—", "–", "N/A", "n/a"):
                return False
            float(cleaned)
            return True
        except (ValueError, TypeError):
            return False

    def to_chunks(self, sheets: list[ExcelSheet]) -> list[dict]:
        """
        Converts ExcelSheet objects to sections for the ingestion pipeline.
        Each representation becomes a separate section sharing the sheet's
        table_id, so sibling retrieval can pull every representation back.

        Args:
            sheets: List of ExcelSheet from process().

        Returns:
            List of section dicts ready for chunking.
        """
        chunks = []
        for sheet in sheets:
            for rep in sheet.representations:
                rep_type = rep["table_representation"]
                text = rep.get("text", "")
                # A metrics summary with no computable metric is only its title.
                if rep_type == "metrics_summary" and not rep.get("metrics"):
                    continue
                if not text.strip():
                    continue
                chunk = {
                    "text": f"{sheet.sheet_name}\n{text}",
                    "section_heading": sheet.sheet_name,
                    "page_number": None,
                    "section_type": "table",
                    "is_table": 1,
                    "sheet_name": sheet.sheet_name,
                    "table_id": sheet.table_id,
                    "table_representation": rep_type,
                    "content_type": (
                        "table_text" if rep_type == "verbatim"
                        else representation_content_type(rep_type)
                    ),
                    "currency": rep.get("currency", "UNKNOWN"),
                    "scale_factor": rep.get("scale_factor", 1.0),
                    "scale_label": rep.get("scale_label", "units"),
                }
                if rep.get("metrics"):
                    chunk["metrics"] = rep["metrics"]
                chunks.append(chunk)

        return chunks
