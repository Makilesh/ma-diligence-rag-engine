"""
Tests for RRF fusion: flatten_deduplicate and reciprocal_rank_fusion.
"""

import pytest
from tests.conftest import make_scored_point


class TestFlattenDeduplicate:
    """Tests for flatten_deduplicate()."""

    def test_dedup_keeps_highest_score(self):
        """Duplicate chunk_id across lists — highest score wins."""
        from src.vector_db.rrf_fusion import flatten_deduplicate

        list1 = [make_scored_point("chunk_a", 0.9), make_scored_point("chunk_b", 0.7)]
        list2 = [make_scored_point("chunk_a", 0.95), make_scored_point("chunk_c", 0.6)]

        result = flatten_deduplicate([list1, list2])
        ids = [p.payload["chunk_id"] for p in result]

        assert len(result) == 3
        assert ids[0] == "chunk_a"  # highest score
        assert result[0].score == 0.95  # kept the higher score

    def test_dedup_single_list(self):
        """Single list passes through unchanged."""
        from src.vector_db.rrf_fusion import flatten_deduplicate

        single = [make_scored_point("c1", 0.8), make_scored_point("c2", 0.6)]
        result = flatten_deduplicate([single])

        assert len(result) == 2

    def test_dedup_empty_lists(self):
        """Empty input lists return empty result."""
        from src.vector_db.rrf_fusion import flatten_deduplicate

        result = flatten_deduplicate([[], []])
        assert result == []

    def test_dedup_sorted_descending(self):
        """Output is sorted by score descending."""
        from src.vector_db.rrf_fusion import flatten_deduplicate

        list1 = [make_scored_point("c1", 0.3), make_scored_point("c2", 0.9)]
        result = flatten_deduplicate([list1])

        assert result[0].score >= result[1].score


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion()."""

    def test_rrf_equal_weights(self, sample_dense_results, sample_sparse_results):
        """RRF with equal weights treats both channels equally."""
        from src.vector_db.rrf_fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion(
            sample_dense_results, sample_sparse_results,
            k=60, dense_weight=0.5, sparse_weight=0.5,
        )

        assert len(result) > 0
        # All results are (chunk_id, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
        # Chunks in both lists should rank higher
        ids = [r[0] for r in result]
        # chunk_a and chunk_c appear in both — should be top-ranked
        assert "chunk_a" in ids[:4]
        assert "chunk_c" in ids[:4]

    def test_rrf_asymmetric_weights(self, sample_dense_results, sample_sparse_results):
        """Dense-heavy weights favor dense-only results."""
        from src.vector_db.rrf_fusion import reciprocal_rank_fusion

        result_dense_heavy = reciprocal_rank_fusion(
            sample_dense_results, sample_sparse_results,
            k=60, dense_weight=0.9, sparse_weight=0.1,
        )
        result_sparse_heavy = reciprocal_rank_fusion(
            sample_dense_results, sample_sparse_results,
            k=60, dense_weight=0.1, sparse_weight=0.9,
        )

        # Different weights should produce different rankings
        ids_dense = [r[0] for r in result_dense_heavy[:3]]
        ids_sparse = [r[0] for r in result_sparse_heavy[:3]]
        assert ids_dense != ids_sparse or len(sample_dense_results) == len(sample_sparse_results)

    def test_rrf_empty_returns_empty(self):
        """
        Both lists empty returns [] rather than raising.

        "Nothing matched" is a legitimate retrieval outcome — a well-formed query
        whose answer is not in the data room, or a metadata filter that emptied
        the candidate set. Raising turned that into a 500; returning empty lets
        the Quality Assessor see zero chunks and take the refusal path the
        pipeline is designed around.
        """
        from src.vector_db.rrf_fusion import reciprocal_rank_fusion

        assert reciprocal_rank_fusion([], []) == []

    def test_rrf_one_empty_list(self, sample_dense_results):
        """One empty list still works — results from non-empty list only."""
        from src.vector_db.rrf_fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion(sample_dense_results, [])
        assert len(result) == len(sample_dense_results)

    def test_rrf_scores_are_positive(self, sample_dense_results, sample_sparse_results):
        """All RRF scores should be positive."""
        from src.vector_db.rrf_fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion(sample_dense_results, sample_sparse_results)
        assert all(score > 0 for _, score in result)

    def test_rrf_sorted_descending(self, sample_dense_results, sample_sparse_results):
        """Output sorted by RRF score descending."""
        from src.vector_db.rrf_fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion(sample_dense_results, sample_sparse_results)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
