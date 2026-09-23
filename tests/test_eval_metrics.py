"""
Metric functions of the retrieval eval harness (eval/metrics.py).

Tiny synthetic inputs only — no models, no Qdrant, no LLM — so these run in
the normal offline CI suite. The harness itself (eval/run_retrieval_eval.py)
needs the real models and is run on demand.
"""

from __future__ import annotations

import math

import pytest

from eval import metrics as m

# ==============================================================================
# Fact normalisation and matching
# ==============================================================================


class TestFactMatching:
    def test_currency_symbol_and_table_cell_are_the_same_figure(self):
        assert m.fact_present("$452.8", "Total revenue   452.8   387.1")
        assert m.fact_present("452.8", "revenue of $452.8 million")

    def test_thousands_separators_are_ignored(self):
        assert m.fact_present("$9,065,750", "a total of 9065750 dollars")
        assert m.fact_present("$9065750", "a total of $9,065,750")

    def test_numeric_edges_do_not_match_inside_other_numbers(self):
        assert not m.fact_present("$85", "a facility of $850 million")
        assert not m.fact_present("$85", "ratio of 1.85x")
        assert not m.fact_present("$85", "$85.5 million")
        assert not m.fact_present("$150", "a fee of $150,000")
        assert not m.fact_present("0.2x", "leverage of 10.2x")

    def test_trailing_zero_decimals_still_match(self):
        assert m.fact_present("$47", "DCF range $47.00 to $63.00")
        assert m.fact_present("$85", "repay $85 million at close")
        assert m.fact_present("$4.5", "a $4.5M restructuring charge")

    def test_case_whitespace_and_markdown_are_normalised(self):
        assert m.fact_present("Superior Proposal", "a **superior\n   proposal** received")
        assert m.fact_present("HIGH", "certainty: high")

    def test_unicode_dashes_match_ascii_hyphen(self):
        assert m.fact_present("90-day", "the 90–day VWAP")

    def test_list_fact_matches_any_variant(self):
        fact = ["thirty-six", "36 months", "3 years"]
        assert m.fact_present(fact, "survive for 36 months after closing")
        assert not m.fact_present(fact, "survive for 24 months")

    def test_facts_in_text_returns_indices(self):
        facts = ["$92.8", "$97.3", "restructuring", ["add back", "add-back"]]
        assert m.facts_in_text(facts, "EBITDA $92.8M plus restructuring add-back") == {0, 2, 3}

    def test_empty_variant_never_matches(self):
        assert not m.fact_present("", "anything")


# ==============================================================================
# Relevance labels
# ==============================================================================


FACTS = ["$58.00", "42%", "$40.85"]
PATTERNS = ["merger_agreement"]


class TestRelevance:
    def test_grade_counts_distinct_facts_from_an_expected_source(self):
        text = "Merger consideration of $58.00 per share, a 42% premium; $58.00 again"
        assert m.relevance_grade(text, "merger_agreement_v2_final.txt", FACTS, PATTERNS) == 2

    def test_wrong_source_is_not_relevant_even_with_facts(self):
        text = "Merger consideration of $58.00 per share, a 42% premium"
        assert m.relevance_grade(text, "board_deck.txt", FACTS, PATTERNS) == 0

    def test_no_facts_means_no_relevance(self):
        assert m.relevance_grade("some text", "merger_agreement.txt", [], PATTERNS) == 0

    def test_source_pattern_match_is_case_insensitive(self):
        assert m.source_matches("Merger_Agreement_V2.txt", ["merger_agreement"])

    def test_build_qrels_keeps_only_relevant_chunks(self):
        chunks = [
            {"chunk_id": "a", "text": "$58.00 and 42%", "source_file": "merger_agreement.txt"},
            {"chunk_id": "b", "text": "nothing here", "source_file": "merger_agreement.txt"},
            {"chunk_id": "c", "text": "$40.85", "source_file": "board_deck.txt"},
        ]
        assert m.build_qrels(chunks, FACTS, PATTERNS) == {"a": 2}


# ==============================================================================
# Ranked-list metrics
# ==============================================================================


QRELS = {"r1": 2, "r2": 1, "r3": 1}


class TestRankedMetrics:
    def test_recall_at_k(self):
        ranked = ["x", "r1", "y", "r2", "z"]
        assert m.recall_at_k(ranked, QRELS, 2) == pytest.approx(1 / 3)
        assert m.recall_at_k(ranked, QRELS, 5) == pytest.approx(2 / 3)

    def test_recall_counts_a_repeated_id_once(self):
        assert m.recall_at_k(["r1", "r1", "r1"], QRELS, 3) == pytest.approx(1 / 3)

    def test_mrr(self):
        assert m.mrr_at_k(["x", "y", "r2"], QRELS, 10) == pytest.approx(1 / 3)
        assert m.mrr_at_k(["r1"], QRELS, 10) == 1.0
        assert m.mrr_at_k(["x", "y", "r2"], QRELS, 2) == 0.0

    def test_metrics_are_undefined_without_relevant_chunks(self):
        assert m.recall_at_k(["a"], {}, 5) is None
        assert m.mrr_at_k(["a"], {}, 5) is None
        assert m.ndcg_at_k(["a"], {}, 5) is None

    def test_ndcg_is_one_for_the_ideal_order(self):
        assert m.ndcg_at_k(["r1", "r2", "r3"], QRELS, 10) == pytest.approx(1.0)

    def test_ndcg_linear_gain_value(self):
        # DCG = 1/log2(2) + 2/log2(3); IDCG = 2/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1 + 2 / math.log2(3)
        idcg = 2 + 1 / math.log2(3) + 1 / math.log2(4)
        assert m.ndcg_at_k(["r2", "r1", "x"], QRELS, 10) == pytest.approx(dcg / idcg)

    def test_ndcg_ignores_duplicates_and_irrelevant(self):
        assert m.ndcg_at_k(["x", "y"], QRELS, 10) == 0.0
        once = m.ndcg_at_k(["r1"], QRELS, 10)
        assert m.ndcg_at_k(["r1", "r1"], QRELS, 10) == pytest.approx(once)


class TestCoverage:
    def test_fact_coverage_across_texts(self):
        facts = ["$452.8", "$387.1", "17.0%", "growth"]
        coverage, found = m.fact_coverage(["Revenue $452.8M", "vs 387.1 prior, 17.0%"], facts)
        assert coverage == pytest.approx(0.75)
        assert found == [0, 1, 2]

    def test_fact_coverage_is_undefined_without_facts(self):
        assert m.fact_coverage(["text"], []) == (None, [])

    def test_coverage_does_not_need_the_expected_source(self):
        # Coverage is the README metric: facts anywhere in what was retrieved.
        coverage, _ = m.fact_coverage(["$58.00"], ["$58.00"])
        assert coverage == 1.0

    def test_source_hit_at_k(self):
        sources = ["a_financials.txt", "x.txt", "board_deck.txt"]
        assert m.source_hit_at_k(sources, ["financials", "board_deck"], 1) == 0.5
        assert m.source_hit_at_k(sources, ["financials", "board_deck"], 3) == 1.0
        assert m.source_hit_at_k(sources, [], 3) is None


# ==============================================================================
# Aggregation and the regression gate
# ==============================================================================


class TestAggregation:
    def test_mean_skips_none(self):
        assert m.mean([1.0, None, 0.0]) == 0.5
        assert m.mean([None]) is None

    def test_percentile(self):
        assert m.percentile([10, 20, 30, 40], 50) == pytest.approx(25)
        assert m.percentile([5], 95) == 5
        assert m.percentile([], 50) is None

    def test_gate_passes_within_tolerance(self):
        baseline = {"production": {"fact_coverage@10": 0.80}}
        current = {"production": {"fact_coverage@10": 0.785}}
        assert m.compare_to_baseline(current, baseline, 0.02) == []

    def test_gate_fails_beyond_tolerance(self):
        baseline = {"production": {"fact_coverage@10": 0.80, "mrr@10": 0.6}}
        current = {"production": {"fact_coverage@10": 0.77, "mrr@10": 0.9}}
        failures = m.compare_to_baseline(current, baseline, 0.02)
        assert len(failures) == 1 and "fact_coverage@10" in failures[0]

    def test_gate_fails_when_a_gated_metric_was_not_measured(self):
        baseline = {"production_decomp": {"fact_coverage@10": 0.8}}
        failures = m.compare_to_baseline({}, baseline, 0.02)
        assert failures and "not measured" in failures[0]
