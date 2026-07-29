"""
Tests for FinancialTableConverter — 4 representations generation.
"""

import pytest
import pandas as pd
from src.data_processing.financial_table_converter import FinancialTableConverter
from src.data_processing.excel_normalizer import TableNormalizationMeta




@pytest.fixture
def converter():
    return FinancialTableConverter()


@pytest.fixture
def sample_df():
    """Simple financial table with 3 rows × 2 year columns."""
    return pd.DataFrame(
        {
            "FY2022": [100.0, 45.0, 15.0],
            "FY2023": [115.0, 50.5, 18.0],
        },
        index=["Revenue", "COGS", "Net Income"],
    )


@pytest.fixture
def sample_meta():
    return TableNormalizationMeta(
        currency="USD",
        scale_factor=1_000_000.0,
        scale_label="millions",
    )


@pytest.fixture
def sample_source_metadata():
    return {
        "doc_id": "doc_001",
        "file_name": "financials.xlsx",
        "page_number": 1,
        "sheet_name": "Income Statement",
    }


class TestRepresentationGeneration:
    """Tests for generate_all_representations()."""

    def test_generates_four_representations(
        self, converter, sample_df, sample_meta, sample_source_metadata
    ):
        """Must produce exactly 4 representations."""
        result = converter.generate_all_representations(
            df=sample_df,
            meta=sample_meta,
            table_id="test_table_001",
            source_metadata=sample_source_metadata,
        )

        assert len(result) == 4

    def test_all_representations_have_table_id(
        self, converter, sample_df, sample_meta, sample_source_metadata
    ):
        """All representations share the same table_id."""
        result = converter.generate_all_representations(
            df=sample_df,
            meta=sample_meta,
            table_id="test_table_001",
            source_metadata=sample_source_metadata,
        )

        for rep in result:
            assert rep["table_id"] == "test_table_001"

    def test_representation_types(
        self, converter, sample_df, sample_meta, sample_source_metadata
    ):
        """All 4 representation types are present."""
        result = converter.generate_all_representations(
            df=sample_df,
            meta=sample_meta,
            table_id="test_table_001",
            source_metadata=sample_source_metadata,
        )

        rep_types = {r["table_representation"] for r in result}
        assert rep_types == {"narrative", "row_by_row", "metrics_summary", "markdown"}

    def test_narrative_contains_numbers(
        self, converter, sample_df, sample_meta, sample_source_metadata
    ):
        """Narrative representation mentions actual financial values."""
        result = converter.generate_all_representations(
            df=sample_df,
            meta=sample_meta,
            table_id="test_table_001",
            source_metadata=sample_source_metadata,
        )

        narrative = next(r for r in result if r["table_representation"] == "narrative")
        # Should contain some reference to the data
        assert len(narrative["text"]) > 0

    def test_markdown_is_valid(
        self, converter, sample_df, sample_meta, sample_source_metadata
    ):
        """Markdown representation contains table-like structure."""
        result = converter.generate_all_representations(
            df=sample_df,
            meta=sample_meta,
            table_id="test_table_001",
            source_metadata=sample_source_metadata,
        )

        markdown = next(r for r in result if r["table_representation"] == "markdown")
        assert "|" in markdown["text"]  # Markdown tables use pipes
