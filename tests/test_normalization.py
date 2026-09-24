"""
Tests for ExcelNormalizer scale/currency detection.
"""

import pytest
from src.data_processing.excel_normalizer import ExcelNormalizer


@pytest.fixture
def normalizer():
    return ExcelNormalizer()


class TestScaleDetection:
    """Tests for detect_scale()."""

    def test_thousands(self, normalizer):
        """Detect 'in thousands' scale indicator."""
        meta = normalizer.detect_scale(["Revenue", "in thousands", "FY2023"])
        assert meta.scale_factor == 1_000.0
        assert meta.scale_label == "thousands"

    def test_millions(self, normalizer):
        """Detect '$M' and 'in millions' scale indicators."""
        meta = normalizer.detect_scale(["Revenue ($M)", "FY2023"])
        assert meta.scale_factor == 1_000_000.0
        assert meta.scale_label == "millions"

    def test_millions_mm(self, normalizer):
        """Detect 'MM' (common in finance for millions)."""
        meta = normalizer.detect_scale(["EBITDA MM", "2023"])
        assert meta.scale_factor == 1_000_000.0

    def test_billions(self, normalizer):
        """Detect '$B' and 'in billions' scale indicators."""
        meta = normalizer.detect_scale(["Assets ($B)", "Total"])
        assert meta.scale_factor == 1_000_000_000.0
        assert meta.scale_label == "billions"

    def test_no_scale_defaults_to_units(self, normalizer):
        """No scale indicator defaults to 1.0 (units)."""
        meta = normalizer.detect_scale(["Revenue", "FY2023", "FY2022"])
        assert meta.scale_factor == 1.0
        assert meta.scale_label == "units"


class TestCurrencyDetection:
    """Tests for _detect_currency()."""

    def test_usd_dollar_sign(self, normalizer):
        """Detect USD from $ sign."""
        result = normalizer._detect_currency("Revenue ($)")
        assert result == "USD"

    def test_eur_symbol(self, normalizer):
        """Detect EUR from € sign."""
        result = normalizer._detect_currency("Revenue (€)")
        assert result == "EUR"

    def test_gbp_symbol(self, normalizer):
        """Detect GBP from £ sign."""
        result = normalizer._detect_currency("Revenue (£)")
        assert result == "GBP"

    def test_unknown_currency(self, normalizer):
        """No currency indicator returns UNKNOWN."""
        result = normalizer._detect_currency("Revenue")
        assert result == "UNKNOWN"

    def test_inr_symbol(self, normalizer):
        """Detect INR from ₹ sign."""
        result = normalizer._detect_currency("Revenue (₹)")
        assert result == "INR"


class TestCombinedDetection:
    """Tests for combined scale + currency in same header."""

    def test_usd_millions(self, normalizer):
        """Combined 'in millions of USD' detection."""
        meta = normalizer.detect_scale(["Revenue in millions of $"])
        assert meta.scale_factor == 1_000_000.0
        assert meta.currency == "USD"

    def test_eur_thousands(self, normalizer):
        """Combined '€ thousands' detection."""
        meta = normalizer.detect_scale(["Amount (€ in thousands)"])
        assert meta.scale_factor == 1_000.0
        assert meta.currency == "EUR"
