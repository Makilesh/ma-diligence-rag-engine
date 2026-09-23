"""
Tests for deterministic numeric grounding (src/verification/numeric_grounding.py)
and the financial verifier's registry (src/utils/numerical_registry.py).

Strings are taken from the sample data room (data/sample_deal/*.txt) and from
answers recorded in RESULTS.md, so the tests exercise the formats the pipeline
actually produces — "$452.8" in a millions table, "$452.8 million USD" in an
answer, "+$65.7M (+17.0%)" in a synthesized table row.
"""

from pathlib import Path

import pytest

from src.verification.numeric_grounding import (
    ContextIndex,
    extract_numbers,
    ground_answer_numbers,
    ground_texts,
    strip_citations,
)

SAMPLE_DEAL = Path(__file__).resolve().parent.parent / "data" / "sample_deal"

FINANCIALS_TABLE = """CONSOLIDATED INCOME STATEMENT
(in millions of USD)

                                          FY2023          FY2022
                                          ------          ------
Revenue                                   $452.8          $387.1
  Software Licenses                       $198.4          $172.3
Gross Profit                              $271.7          $228.8
  Gross Margin                             60.0%           59.1%
Operating Income (EBIT)                    $68.0           $51.4
Interest Expense                           ($8.2)          ($7.4)
  Net Debt/EBITDA                           0.2x            0.7x

Revenue Growth: 17.0% YoY
Total Customers: 2,847 (up from 2,412 in FY2022)
Enterprise Customers (>$100K ACV): 284 (up from 231 in FY2022)
"""

CONTEXT = [
    {"text": FINANCIALS_TABLE, "source_file": "aurora_financials_fy2023.txt"},
    {
        "text": "Section 8.2 — Indemnification Cap. The aggregate liability shall not "
        "exceed $69.6 million (10% of the Merger Consideration). Commitment fee of "
        "35 basis points on the undrawn portion.",
        "source_file": "merger_agreement_v2_final.txt",
    },
]


def _kinds(text: str) -> list[tuple[str, float, str]]:
    return [(m.raw, m.value, m.kind) for m in extract_numbers(text)]


class TestExtraction:
    """Every figure shape is found and normalised; references are not figures."""

    def test_currency_scales_normalise_to_the_same_value(self):
        for text in (
            "$452.8 million",
            "$452.8M",
            "$452.8 MM",
            "USD 452.8mn",
            "$452,800,000",
            "452.8 million USD",
        ):
            values = [m.value for m in extract_numbers(text)]
            assert values and values[0] == pytest.approx(452_800_000), text

    def test_billions_and_thousands(self):
        assert extract_numbers("USD 1.2bn")[0].value == pytest.approx(1.2e9)
        assert extract_numbers("$1.2 billion")[0].value == pytest.approx(1.2e9)
        assert extract_numbers("$120K")[0].value == pytest.approx(120_000)
        assert extract_numbers("EUR 14.2 million")[0].kind == "currency"

    def test_percentages_multiples_and_points(self):
        found = _kinds("margin 17.0%, leverage 0.2x vs 8.5×, up 0.9 percentage points or 90 bps, 12 percent")
        kinds = {raw: kind for raw, _, kind in found}
        assert kinds["17.0%"] == "percent"
        assert kinds["0.2x"] == "multiple"
        assert kinds["8.5×"] == "multiple"
        assert kinds["0.9 percentage points"] == "pp"
        assert kinds["90 bps"] == "bps"
        assert kinds["12 percent"] == "percent"

    def test_years_ordinals_sections_and_pages_are_ignored(self):
        text = (
            "In FY2023 and 2022, under Section 280G and Section 4.3(a), the 1st lien "
            "(p. 3, Note 5, Article 7) was repaid on December 31, 2023 and 12/31/2023."
        )
        assert extract_numbers(text) == []

    def test_plain_integers_are_not_claims_but_separated_ones_are(self):
        assert extract_numbers("284 enterprise customers across 12 regions") == []
        found = extract_numbers("Total customers grew to 2,847")
        assert [m.value for m in found] == [2847]

    def test_negative_parentheses_compare_by_magnitude(self):
        assert extract_numbers("Interest Expense ($8.2)")[0].value == pytest.approx(8.2)

    def test_citation_markers_are_stripped_before_extraction(self):
        answer = (
            "ARR reached $210.5 million [📄 aurora_financials_fy2023.txt | FY2023 | p.3 | "
            "Revenue Growth: 17.0% YoY]."
        )
        found = extract_numbers(strip_citations(answer))
        assert [m.raw for m in found] == ["$210.5 million"]

    def test_row_by_row_table_representation(self):
        """The ingest row_by_row format writes FY2023=452,800,000."""
        found = extract_numbers("Revenue: FY2023=452,800,000 FY2022=387,100,000")
        assert [m.value for m in found] == [452_800_000, 387_100_000]


class TestGrounding:
    """Figures are matched against the context after normalisation."""

    @pytest.fixture
    def index(self):
        return ContextIndex(CONTEXT)

    def _statuses(self, text, index):
        return [(c.raw, c.status) for c in ground_texts([text], index)[0]]

    def test_answer_units_ground_against_a_millions_table(self, index):
        text = "Revenue was $452.8 million in FY2023 ($452,800,000) versus $387.1M."
        assert all(s == "grounded" for _, s in self._statuses(text, index))

    def test_rounding_within_printed_precision_is_accepted(self, index):
        assert self._statuses("revenue of roughly $453 million", index) == [
            ("$453 million", "grounded")
        ]

    def test_fabricated_figure_is_unsupported(self, index):
        assert self._statuses("Revenue was $512.4 million in FY2023.", index) == [
            ("$512.4 million", "unsupported")
        ]

    def test_transposed_digits_are_not_grounded(self, index):
        assert self._statuses("Revenue was $425.8 million.", index) == [
            ("$425.8 million", "unsupported")
        ]

    def test_percent_does_not_ground_against_a_currency_figure(self, index):
        # 68.0 exists as $68.0 (EBIT); "68.0%" must not borrow it.
        assert self._statuses("a margin of 68.0%", index) == [("68.0%", "unsupported")]

    def test_bare_integer_in_context_does_not_ground_a_scaled_claim(self):
        index = ContextIndex([{"text": "Consent is required 30 days before closing.", "source_file": "x"}])
        assert self._statuses("a $30 million escrow", index) == [("$30 million", "unsupported")]

    def test_difference_in_the_same_row_is_derived(self, index):
        checks = ground_texts(
            ["Total Revenue: FY2023 $452.8M; FY2022 $387.1M; YoY Change +$65.7M (+17.0%)"],
            index,
        )[0]
        by_raw = {c.raw: c for c in checks}
        assert by_raw["$65.7M"].status == "derived"
        assert "452.8M" in by_raw["$65.7M"].formula
        assert by_raw["17.0%"].status == "grounded"

    def test_percentage_change_is_derived(self, index):
        checks = ground_texts(["Gross margin moved from 59.1% to 60.0%, up 0.9 percentage points."], index)[0]
        assert {c.raw: c.status for c in checks}["0.9 percentage points"] == "derived"

    def test_margin_ratio_is_derived(self, index):
        checks = ground_texts(
            ["FY2022 operating margin: $51.4 million / $387.1 million = 13.28%."], index
        )[0]
        assert {c.raw: c.status for c in checks}["13.28%"] == "derived"

    def test_bps_to_percent_conversion_is_derived(self, index):
        checks = ground_texts(["Commitment fee: 35 basis points (0.35%) on undrawn amounts."], index)[0]
        assert {c.raw: c.status for c in checks}["0.35%"] == "derived"

    def test_operands_from_another_sentence_do_not_derive(self, index):
        """Derivation is same-claim only; a stray pair elsewhere must not count."""
        checks = ground_texts(
            ["Revenue was $452.8M and prior-year revenue $387.1M.", "Headroom is $65.7M."],
            index,
        )
        assert checks[1][0].status == "unsupported"

    def test_unreproducible_calculation_is_flagged_not_failed(self, index):
        checks = ground_texts(["Implied EV/EBITDA of approximately 7.50x."], index)[0]
        assert checks[0].status == "derived_unverified"


class TestSampleDataRoom:
    """Figures from real RESULTS.md answers against the real sample documents."""

    @pytest.fixture(scope="class")
    def chunks(self):
        if not SAMPLE_DEAL.exists():
            pytest.skip("sample data room not present")
        return [
            {"text": p.read_text(encoding="utf-8"), "source_file": p.name}
            for p in sorted(SAMPLE_DEAL.glob("*.txt"))
        ]

    def test_recorded_revenue_answer_is_fully_grounded(self, chunks):
        answer = (
            "In **FY2023**, Aurora Technologies Inc. generated total revenue of "
            "**$452.8 million USD**, representing a **17.0% year-over-year growth** "
            "compared to total revenue of **$387.1 million USD** in **FY2022** "
            "[📄 aurora_financials_fy2023.txt | FY2023 | p.1 | Income Statement].\n\n"
            "| Segment | FY2023 | FY2022 | YoY Change ($) |\n"
            "|---|---|---|---|\n"
            "| **Software Licenses** | $198.4M | $172.3M | +$26.1M |\n"
            "| **Professional Services** | $64.8M | $66.6M | -$1.8M |\n"
            "- **Net Revenue Retention (NRR):** **118%** in FY2023 (up from 114% in FY2022).\n"
            "- **Customer Count:** Total customers grew to **2,847** in FY2023 (up from 2,412)."
        )
        checks = ground_answer_numbers(answer, chunks)
        statuses = {c["raw"]: c["status"] for c in checks}
        assert statuses["$452.8 million USD"] == "grounded"
        assert statuses["$26.1M"] == "derived"
        # $1.8M also appears verbatim in the financials (FY2022 other income),
        # so it may ground directly rather than as the segment difference.
        assert statuses["$1.8M"] in ("grounded", "derived")
        assert statuses["118%"] == "grounded"
        assert statuses["2,847"] == "grounded"
        assert not [c for c in checks if c["status"] == "unsupported"]

    def test_ebitda_answer_grounds_both_adjusted_figures(self, chunks):
        answer = (
            "Company Adjusted EBITDA is $97.3 million (21.5% margin), while the QoE "
            "report presents $99.0 million (21.9% margin); QoE normalization adds "
            "$6.2 million to reported EBITDA of $92.8 million."
        )
        checks = ground_answer_numbers(answer, chunks)
        assert all(c["status"] in ("grounded", "derived") for c in checks), checks

    def test_a_planted_wrong_figure_is_caught(self, chunks):
        answer = "Net Debt / EBITDA fell to 0.2x and the revolver balance was $142.0 million."
        statuses = {c["raw"]: c["status"] for c in ground_answer_numbers(answer, chunks)}
        assert statuses["0.2x"] == "grounded"
        assert statuses["$142.0 million"] == "unsupported"


class TestNumericalRegistry:
    """Agent 4's deterministic cross-document check."""

    def test_detects_the_adjusted_ebitda_disagreement(self):
        from src.utils.numerical_registry import NumericalRegistry

        chunks = [
            {
                "text": "(in millions of USD)\n            FY2023     FY2022\n"
                "  Adjusted EBITDA   $97.3   $74.0\n  Adjusted EBITDA Margin  21.5%  19.1%\n",
                "source_file": "aurora_financials_fy2023.txt",
            },
            {
                "text": "(in millions of USD)\n            FY2023     FY2022\n"
                "Adjusted EBITDA     $99.0   $73.8\n  Adjusted EBITDA Margin  21.9%  19.1%\n",
                "source_file": "quality_of_earnings_report_fy2023.txt",
            },
        ]
        registry = NumericalRegistry.from_chunks(chunks)
        found = {(i["metric"].strip(), i["fiscal_year"]) for i in registry.find_inconsistencies()}
        assert ("Adjusted EBITDA", "FY2023") in found
        assert ("Adjusted EBITDA", "FY2022") in found
        assert ("Adjusted EBITDA Margin", "FY2023") in found
        # Same margin in both sources for FY2022 — not an inconsistency.
        assert ("Adjusted EBITDA Margin", "FY2022") not in found
        assert all(i["method"] == "deterministic" for i in registry.find_inconsistencies())

    def test_reported_prefix_and_notes_are_normalised_but_qualifiers_are_not(self):
        from src.utils.numerical_registry import normalize_metric_label

        assert normalize_metric_label("Reported EBITDA") == normalize_metric_label("EBITDA")
        assert normalize_metric_label("Add: Restructuring Charges (Note 1)") == "restructuring charges"
        assert normalize_metric_label("Deferred Revenue (Non-Current)") != normalize_metric_label(
            "Deferred Revenue"
        )

    def test_scale_differences_are_not_inconsistencies(self):
        from src.utils.numerical_registry import NumericalRegistry

        chunks = [
            {
                "text": "(in millions of USD)\n      FY2023\nRevenue   $452.8\n",
                "source_file": "a.txt",
            },
            {
                "source_file": "model.xlsx",
                "text": "Revenue: FY2023=452,800,000",
                "structured_rows": [
                    {
                        "line_item": "Revenue",
                        "values": {
                            "FY2023": {
                                "raw_value": 452800.0,
                                "normalized_value": 452_800_000.0,
                                "scale_factor": 1000.0,
                            }
                        },
                    }
                ],
            },
        ]
        registry = NumericalRegistry.from_chunks(chunks)
        assert registry.cross_checked_count() >= 1
        assert registry.find_inconsistencies() == []

    def test_different_periods_never_conflict(self):
        from src.utils.numerical_registry import NumericalRegistry

        chunks = [
            {"text": "      FY2023\nRevenue   $452.8\n", "source_file": "a.txt"},
            {"text": "      FY2022\nRevenue   $387.1\n", "source_file": "b.txt"},
        ]
        assert NumericalRegistry.from_chunks(chunks).find_inconsistencies() == []

    def test_prose_lines_are_not_parsed_as_table_rows(self):
        from src.utils.numerical_registry import NumericalRegistry

        chunks = [
            {
                "text": "      FY2023\nThe purchase price was increased to $58.00 per share by the Board\n",
                "source_file": "a.txt",
            }
        ]
        assert len(NumericalRegistry.from_chunks(chunks)) == 0

    def test_long_table_chunks_keep_their_text(self):
        """
        Regression: the old verifier dropped the text of any table chunk of 500+
        characters before the check. The registry reads the full text.
        """
        from src.utils.numerical_registry import NumericalRegistry

        long_table = "(in millions of USD)\n                FY2023     FY2022\n" + "".join(
            f"Line item {i:02d}          ${100 + i}.0     ${90 + i}.0\n" for i in range(40)
        )
        assert len(long_table) > 500
        registry = NumericalRegistry.from_chunks([{"text": long_table, "source_file": "t.txt", "is_table": 1}])
        assert len(registry) == 80
