# src/data_processing/financial_table_converter.py
"""
Generates 4 chunk representations per financial table, all sharing a table_id.

CRITICAL RULE: All derived metrics (margins, CAGR, YoY growth) are computed
DETERMINISTICALLY in pandas — NEVER via LLM. LLMs get arithmetic wrong silently.

Every computed value carries a citation_chain showing its inputs:
content_type="computed_metric" distinguishes these from verbatim source numbers.
The synthesis and validation agents must treat computed_metric chunks differently:
they cannot be verified against source text, only against their citation_chain.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd

from src.data_processing.excel_normalizer import TableNormalizationMeta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RepresentationType = Literal["narrative", "row_by_row", "metrics_summary", "markdown"]

# Plain amounts only: "1,234", "(8.2)", "$452.8", "-3". Percentages, multiples
# ("2.4x") and share counts ("12.0M") stay strings on purpose — scaling them by
# the table's "in millions" factor would turn 60.0% into 60,000,000.
_PLAIN_AMOUNT = re.compile(r"^\(?-?[$€£¥₹]?\s*-?[\d,]*\.?\d+\)?$")


def representation_content_type(representation: str) -> str:
    """
    Maps a table representation to the payload content_type.

    metrics_summary maps to "computed_metric" because the synthesizer and the
    financial verifier key on that value to label derived (non-verbatim)
    numbers; the others are "table_<representation>".

    Args:
        representation: narrative | row_by_row | metrics_summary | markdown.

    Returns:
        content_type string for the Qdrant payload.
    """
    if representation == "metrics_summary":
        return "computed_metric"
    return f"table_{representation}"


def _coerce_cell(value: Any) -> Any:
    """
    Converts a plain numeric cell to float; leaves everything else as text.

    Args:
        value: Raw cell value.

    Returns:
        float, stripped string, or None for empty cells.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "—", "–", "n/a"}:
        return None
    if _PLAIN_AMOUNT.match(text):
        negative = text.startswith("(") and text.endswith(")")
        cleaned = re.sub(r"[()$€£¥₹,\s]", "", text)
        try:
            number = float(cleaned)
        except ValueError:
            return text
        return -abs(number) if negative else number
    return text


def _dedupe(labels: list[str]) -> list[str]:
    """Suffixes repeated labels so DataFrame .loc lookups stay scalar."""
    seen: dict[str, int] = {}
    result = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        result.append(label if count == 0 else f"{label} ({count + 1})")
    return result


def frame_from_rows(header: list[Any], rows: list[list[Any]]) -> pd.DataFrame | None:
    """
    Builds the DataFrame shape FinancialTableConverter expects from a header
    row and data rows: row labels as the index, period labels as columns.

    Returns None when the table cannot be interpreted that way (fewer than two
    columns, no data rows, or no numeric cell at all) — the caller then indexes
    the table as verbatim text instead of fabricating representations.

    Args:
        header: Header cells; the first cell labels the row-label column.
        rows: Data rows.

    Returns:
        DataFrame with object dtype (floats for amounts, strings otherwise), or None.
    """
    if not header or len(header) < 2 or not rows:
        return None

    width = len(header)
    columns = []
    for j, cell in enumerate(header[1:], start=2):
        name = "" if cell is None else str(cell).strip()
        if not name or name.lower() == "nan" or name.startswith("Unnamed:"):
            name = f"Column {j}"
        columns.append(name)
    columns = _dedupe(columns)

    labels: list[str] = []
    data: list[list[Any]] = []
    for i, row in enumerate(rows, start=1):
        cells = list(row)[:width] + [None] * max(0, width - len(row))
        values = [_coerce_cell(c) for c in cells[1:]]
        raw_label = cells[0]
        label = "" if raw_label is None or (
            isinstance(raw_label, float) and pd.isna(raw_label)
        ) else str(raw_label).strip()
        if not label and all(v is None for v in values):
            continue
        labels.append(label or f"Row {i}")
        data.append(values)

    if not data:
        return None
    if not any(isinstance(v, float) for row in data for v in row):
        return None

    return pd.DataFrame(data, index=_dedupe(labels), columns=columns, dtype=object)


class FinancialTableConverter:
    """
    Generates 4 chunk representations per financial table, all sharing a table_id.

    CRITICAL RULE: All derived metrics (margins, CAGR, YoY growth) are computed
    DETERMINISTICALLY in pandas — NEVER via LLM. LLMs get arithmetic wrong silently.

    Every computed value carries a citation_chain showing its inputs:
    content_type="computed_metric" distinguishes these from verbatim source numbers.
    The synthesis and validation agents must treat computed_metric chunks differently:
    they cannot be verified against source text, only against their citation_chain.

    Representations:
        1. narrative — Natural language summary of the table
        2. row_by_row — Each row as a structured dict for precise lookup
        3. metrics_summary — Computed derived metrics (margins, CAGR, YoY)
        4. markdown — Full table in markdown format for display
    """

    def generate_all_representations(
        self,
        df: pd.DataFrame,
        meta: TableNormalizationMeta,
        table_id: str,
        source_metadata: dict,
    ) -> list[dict]:
        """
        Generate all 4 representations for a financial table.

        Args:
            df: DataFrame containing the table data. Index should be row labels,
                columns should be period labels (e.g., "FY2022", "FY2023").
            meta: Normalization metadata (currency, scale_factor, scale_label).
            table_id: Unique identifier for this table (format: "{deal_id}_{doc_id}_{table_seq}").
            source_metadata: Dict with source file info (doc_id, file_name, page_number, etc.).

        Returns:
            List of 4 representation dicts, each with table_representation type,
            table_id, and source metadata.
        """
        logger.info(
            "Generating 4 representations for financial table",
            extra={
                "table_id": table_id,
                "shape": f"{df.shape[0]}x{df.shape[1]}",
                "currency": meta.currency,
                "scale_label": meta.scale_label,
            },
        )

        representations = [
            self._narrative(df, meta, table_id, source_metadata),
            self._row_by_row(df, meta, table_id, source_metadata),
            self._metrics_summary(df, meta, table_id, source_metadata),
            self._markdown(df, meta, table_id, source_metadata),
        ]

        logger.info(
            "All 4 representations generated",
            extra={"table_id": table_id},
        )

        return representations

    def _narrative(
        self,
        df: pd.DataFrame,
        meta: TableNormalizationMeta,
        table_id: str,
        source_meta: dict,
    ) -> dict:
        """
        Generate a natural language narrative summary of the table.

        Describes the table structure, key figures, and any notable patterns
        in plain English suitable for semantic search matching.

        Args:
            df: Table DataFrame.
            meta: Normalization metadata.
            table_id: Unique table identifier.
            source_meta: Source file metadata.

        Returns:
            Dict with table_representation="narrative" and narrative text.
        """
        lines: list[str] = []

        # Table overview
        columns_str = ", ".join(str(c) for c in df.columns)
        lines.append(
            f"Financial table with {len(df)} rows and {len(df.columns)} columns "
            f"({columns_str}). Values are in {meta.currency} {meta.scale_label}."
        )

        # Summarize key rows
        for row_label in df.index:
            row_data = df.loc[row_label]
            non_null = row_data.dropna()
            if non_null.empty:
                continue

            values_parts: list[str] = []
            for col, val in non_null.items():
                try:
                    numeric_val = float(val) * meta.scale_factor
                    values_parts.append(f"{col}: {numeric_val:,.0f}")
                except (ValueError, TypeError):
                    values_parts.append(f"{col}: {val}")

            if values_parts:
                lines.append(f"{row_label}: {'; '.join(values_parts)}")

        narrative_text = "\n".join(lines)

        return {
            "text": narrative_text,
            "table_representation": "narrative",
            "table_id": table_id,
            "is_table": 1,
            "content_type": "clean",
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
            **source_meta,
        }

    def _row_by_row(
        self,
        df: pd.DataFrame,
        meta: TableNormalizationMeta,
        table_id: str,
        source_meta: dict,
    ) -> dict:
        """
        Generate a row-by-row structured representation.

        Each row becomes a structured entry with raw_value and normalized_value
        for every cell, enabling precise lookup of specific financial figures.

        Args:
            df: Table DataFrame.
            meta: Normalization metadata.
            table_id: Unique table identifier.
            source_meta: Source file metadata.

        Returns:
            Dict with table_representation="row_by_row" and structured rows.
        """
        rows_data: list[dict] = []
        text_parts: list[str] = []

        for row_label in df.index:
            row_entry: dict[str, Any] = {"line_item": str(row_label), "values": {}}
            row_text_parts: list[str] = [f"{row_label}:"]

            for col in df.columns:
                raw_val = df.loc[row_label, col]
                if pd.isna(raw_val):
                    continue

                try:
                    numeric_raw = float(raw_val)
                    normalized = numeric_raw * meta.scale_factor
                    row_entry["values"][str(col)] = {
                        "raw_value": numeric_raw,
                        "normalized_value": normalized,
                        "currency": meta.currency,
                        "scale_factor": meta.scale_factor,
                    }
                    row_text_parts.append(f"{col}={normalized:,.0f}")
                except (ValueError, TypeError):
                    row_entry["values"][str(col)] = {"raw_value": str(raw_val)}
                    row_text_parts.append(f"{col}={raw_val}")

            rows_data.append(row_entry)
            text_parts.append(" ".join(row_text_parts))

        return {
            "text": "\n".join(text_parts),
            "table_representation": "row_by_row",
            "table_id": table_id,
            "is_table": 1,
            "content_type": "clean",
            "structured_rows": rows_data,
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
            **source_meta,
        }

    def _metrics_summary(
        self,
        df: pd.DataFrame,
        meta: TableNormalizationMeta,
        table_id: str,
        source_meta: dict,
    ) -> dict:
        """
        Computes CAGR, margins, YoY growth via pandas. Each metric gets citation_chain.

        ALL derived metrics are computed DETERMINISTICALLY in pandas — NEVER via LLM.

        Args:
            df: Table DataFrame with row labels as index and period columns.
            meta: Normalization metadata.
            table_id: Unique table identifier.
            source_meta: Source file metadata.

        Returns:
            Dict with table_representation="metrics_summary", content_type="computed_metric",
            and computed metrics with citation_chains.
        """
        metrics: dict[str, dict] = {}
        text_parts: list[str] = ["Computed Metrics Summary:"]

        # ─── Revenue CAGR ─────────────────────────────────────────────────
        if "Revenue" in df.index:
            revenue_row = df.loc["Revenue"].apply(pd.to_numeric, errors="coerce") * meta.scale_factor
            years = sorted([c for c in revenue_row.index if str(c).startswith("FY")])

            if len(years) >= 2:
                first_val = revenue_row[years[0]]
                last_val = revenue_row[years[-1]]

                if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
                    n = len(years) - 1
                    cagr = (last_val / first_val) ** (1 / n) - 1
                    metrics["Revenue CAGR"] = {
                        "value": round(cagr * 100, 2),
                        "unit": "%",
                        "citation_chain": (
                            f"CAGR(Revenue[{years[0]}={first_val:,.0f} → "
                            f"{years[-1]}={last_val:,.0f}], n={n})"
                        ),
                        "content_type": "computed_metric",
                    }
                    text_parts.append(
                        f"Revenue CAGR: {cagr * 100:.2f}% "
                        f"({years[0]}: {first_val:,.0f} → {years[-1]}: {last_val:,.0f})"
                    )

                # YoY Revenue growth
                for i in range(1, len(years)):
                    prev_val = revenue_row[years[i - 1]]
                    curr_val = revenue_row[years[i]]
                    if pd.notna(prev_val) and pd.notna(curr_val) and prev_val != 0:
                        yoy = (curr_val - prev_val) / prev_val
                        key = f"Revenue YoY {years[i]}"
                        metrics[key] = {
                            "value": round(yoy * 100, 2),
                            "unit": "%",
                            "citation_chain": (
                                f"YoY(Revenue[{years[i - 1]}={prev_val:,.0f} → "
                                f"{years[i]}={curr_val:,.0f}])"
                            ),
                            "content_type": "computed_metric",
                        }
                        text_parts.append(
                            f"Revenue YoY {years[i]}: {yoy * 100:.2f}%"
                        )

        # ─── Gross Margin ─────────────────────────────────────────────────
        self._compute_margin(
            df, meta, metrics, text_parts,
            numerator_labels=["Gross Profit", "Gross_Profit"],
            denominator_labels=["Revenue"],
            metric_name="Gross Margin",
        )

        # ─── Operating Margin ────────────────────────────────────────────
        self._compute_margin(
            df, meta, metrics, text_parts,
            numerator_labels=["Operating Income", "Operating_Income", "EBIT"],
            denominator_labels=["Revenue"],
            metric_name="Operating Margin",
        )

        # ─── Net Margin ──────────────────────────────────────────────────
        self._compute_margin(
            df, meta, metrics, text_parts,
            numerator_labels=["Net Income", "Net_Income", "Net Profit"],
            denominator_labels=["Revenue"],
            metric_name="Net Margin",
        )

        # ─── EBITDA Margin ───────────────────────────────────────────────
        self._compute_margin(
            df, meta, metrics, text_parts,
            numerator_labels=["EBITDA"],
            denominator_labels=["Revenue"],
            metric_name="EBITDA Margin",
        )

        return {
            "text": "\n".join(text_parts),
            "table_representation": "metrics_summary",
            "metrics": metrics,
            "table_id": table_id,
            "is_table": 1,
            "content_type": "computed_metric",
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
            **source_meta,
        }

    def _compute_margin(
        self,
        df: pd.DataFrame,
        meta: TableNormalizationMeta,
        metrics: dict[str, dict],
        text_parts: list[str],
        numerator_labels: list[str],
        denominator_labels: list[str],
        metric_name: str,
    ) -> None:
        """
        Compute margin metrics (numerator/denominator) for each period column.

        Args:
            df: Table DataFrame.
            meta: Normalization metadata.
            metrics: Dict to append computed metrics to.
            text_parts: List to append text summaries to.
            numerator_labels: Possible row labels for the numerator.
            denominator_labels: Possible row labels for the denominator.
            metric_name: Name for this margin metric.
        """
        numerator_label = next(
            (label for label in numerator_labels if label in df.index), None
        )
        denominator_label = next(
            (label for label in denominator_labels if label in df.index), None
        )

        if numerator_label is None or denominator_label is None:
            return

        num_row = df.loc[numerator_label].apply(pd.to_numeric, errors="coerce") * meta.scale_factor
        den_row = df.loc[denominator_label].apply(pd.to_numeric, errors="coerce") * meta.scale_factor

        fy_cols = sorted([c for c in df.columns if str(c).startswith("FY")])
        if not fy_cols:
            fy_cols = list(df.columns)

        for col in fy_cols:
            num_val = num_row.get(col)
            den_val = den_row.get(col)

            if pd.notna(num_val) and pd.notna(den_val) and den_val != 0:
                margin = num_val / den_val
                key = f"{metric_name} {col}"
                metrics[key] = {
                    "value": round(margin * 100, 2),
                    "unit": "%",
                    "citation_chain": (
                        f"{metric_name}({numerator_label}[{col}]={num_val:,.0f} / "
                        f"{denominator_label}[{col}]={den_val:,.0f})"
                    ),
                    "content_type": "computed_metric",
                }
                text_parts.append(f"{metric_name} {col}: {margin * 100:.2f}%")

    def _markdown(
        self,
        df: pd.DataFrame,
        meta: TableNormalizationMeta,
        table_id: str,
        source_meta: dict,
    ) -> dict:
        """
        Generate a full markdown table representation.

        This representation preserves the exact table layout for display
        and for LLM consumption during answer synthesis.

        Args:
            df: Table DataFrame.
            meta: Normalization metadata.
            table_id: Unique table identifier.
            source_meta: Source file metadata.

        Returns:
            Dict with table_representation="markdown" and markdown table text.
        """
        header_note = (
            f"*Values in {meta.currency} {meta.scale_label} "
            f"(scale factor: {meta.scale_factor:,.0f})*\n\n"
        )

        # Build markdown table
        columns = [str(c) for c in df.columns]
        header_row = "| Line Item | " + " | ".join(columns) + " |"
        separator = "|---|" + "|".join(["---"] * len(columns)) + "|"

        rows: list[str] = []
        for row_label in df.index:
            cells: list[str] = [str(row_label)]
            for col in df.columns:
                val = df.loc[row_label, col]
                if pd.isna(val):
                    cells.append("-")
                else:
                    try:
                        numeric_val = float(val)
                        cells.append(f"{numeric_val:,.2f}")
                    except (ValueError, TypeError):
                        cells.append(str(val))
            rows.append("| " + " | ".join(cells) + " |")

        markdown_text = header_note + header_row + "\n" + separator + "\n" + "\n".join(rows)

        return {
            "text": markdown_text,
            "table_representation": "markdown",
            "table_id": table_id,
            "is_table": 1,
            "content_type": "clean",
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
            **source_meta,
        }
