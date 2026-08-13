"""
Tests for agent nodes with mocked LLM calls.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestRetrievalStrategy:
    """Tests for Agent 2 — deterministic retrieval config."""

    def test_financial_config(self):
        """Financial query returns financial-optimized config."""
        from src.agents.retrieval_strategy import get_retrieval_config

        config = get_retrieval_config("financial", {})
        assert config["dense_weight"] == 0.5
        assert config["sparse_weight"] == 0.5
        assert config["use_parent_expansion"] is True
        assert config["use_sibling_expansion"] is True

    def test_legal_config(self):
        """Legal query weights sparse higher for keyword matching."""
        from src.agents.retrieval_strategy import get_retrieval_config

        config = get_retrieval_config("legal", {})
        assert config["sparse_weight"] > config["dense_weight"]

    def test_unknown_type_defaults_to_summary(self):
        """Unknown query type falls back to summary config."""
        from src.agents.retrieval_strategy import get_retrieval_config

        config = get_retrieval_config("nonexistent_type", {})
        # Should get summary config (the fallback)
        assert config is not None
        assert "dense_weight" in config

    def test_numerical_precision_raises_threshold(self):
        """requires_numerical_precision raises reranker threshold."""
        from src.agents.retrieval_strategy import get_retrieval_config

        config_normal = get_retrieval_config("financial", {})
        config_precise = get_retrieval_config(
            "financial", {"requires_numerical_precision": True}
        )

        assert config_precise["reranker_threshold"] >= config_normal["reranker_threshold"]

    def test_cross_document_increases_top_k(self):
        """requires_cross_document increases top_k values."""
        from src.agents.retrieval_strategy import get_retrieval_config

        config_normal = get_retrieval_config("summary", {})
        config_cross = get_retrieval_config(
            "summary", {"requires_cross_document": True}
        )

        assert config_cross["top_k_dense"] >= config_normal["top_k_dense"]

    def test_all_query_types_have_config(self):
        """All 5 query types return valid configs."""
        from src.agents.retrieval_strategy import get_retrieval_config

        for qt in ["financial", "legal", "comparative", "summary", "multi_hop"]:
            config = get_retrieval_config(qt, {})
            assert "dense_weight" in config
            assert "sparse_weight" in config
            assert "reranker_threshold" in config


class TestQualityAssessorHeuristic:
    """Tests for Agent 5 heuristic path (no LLM needed)."""

    def test_no_chunks_forces_refusal(self, sample_agent_state):
        """Empty results force refusal."""
        from src.agents.quality_assessor import _heuristic_assessment

        state = {**sample_agent_state, "reranked_results": []}
        result = _heuristic_assessment(state)

        assert result is not None
        assert result["context_quality_score"] == 0.0
        assert result["force_refusal"] is True

    def test_high_scores_pass_heuristic(self, sample_agent_state):
        """High reranker scores pass without LLM."""
        from src.agents.quality_assessor import _heuristic_assessment

        chunks = [
            {"reranker_score": 0.9, "source_file": "f1.pdf"},
            {"reranker_score": 0.85, "source_file": "f2.pdf"},
            {"reranker_score": 0.8, "source_file": "f3.pdf"},
        ]
        state = {**sample_agent_state, "reranked_results": chunks}
        result = _heuristic_assessment(state)

        assert result is not None
        assert result["context_quality_score"] > 0.5
        assert result["force_refusal"] is False
        assert result["quality_method"] == "heuristic"

    def test_low_scores_detected(self, sample_agent_state):
        """Low reranker scores detected by heuristic."""
        from src.agents.quality_assessor import _heuristic_assessment

        chunks = [
            {"reranker_score": 0.03, "source_file": "f1.pdf"},
            {"reranker_score": 0.01, "source_file": "f2.pdf"},
        ]
        state = {**sample_agent_state, "reranked_results": chunks}
        result = _heuristic_assessment(state)

        assert result is not None
        assert result["context_quality_score"] < 0.3
        assert result["force_refusal"] is True

    def test_ambiguous_falls_through_to_llm(self, sample_agent_state):
        """Scores in the ambiguous band return None (triggering LLM fallback)."""
        from src.agents.quality_assessor import _heuristic_assessment

        # Best chunk sits between CONFIDENT_REFUSAL_CEILING (0.05) and
        # CONFIDENT_PASS_FLOOR (0.30) — too weak to trust, too strong to reject.
        chunks = [
            {"reranker_score": 0.20, "source_file": "f1.pdf"},
            {"reranker_score": 0.08, "source_file": "f2.pdf"},
        ]
        state = {**sample_agent_state, "reranked_results": chunks}
        result = _heuristic_assessment(state)

        assert result is None

    def test_one_decisive_chunk_is_enough(self, sample_agent_state):
        """
        A single strongly-relevant chunk passes even when surrounded by noise.

        This is the regression guard for the metric-design fix: under the old
        mean/min-of-top-5 scoring, the noise tail dragged quality below the gate
        and a pointed question with a decisive answer chunk was refused.
        """
        from src.agents.quality_assessor import _heuristic_assessment
        from src.workflow.conditional_edges import _meets_type_threshold

        chunks = [
            {"reranker_score": 0.95, "source_file": "f1.pdf"},
            {"reranker_score": 0.0004, "source_file": "f1.pdf"},
            {"reranker_score": 0.0001, "source_file": "f1.pdf"},
        ]
        state = {**sample_agent_state, "reranked_results": chunks,
                 "query_type": "legal"}
        result = _heuristic_assessment(state)

        assert result is not None
        assert result["force_refusal"] is False

        state.update(
            context_quality_score=result["context_quality_score"],
            quality_breakdown=result["quality_breakdown"],
        )
        assert _meets_type_threshold(state) is True

    def test_retrieving_more_noise_does_not_lower_quality(self, sample_agent_state):
        """
        Adding irrelevant candidates must not degrade the quality verdict.

        The original scoring averaged over the whole top-k, so improving recall
        made context look worse — the defect this design corrects.
        """
        from src.agents.quality_assessor import _heuristic_assessment

        good = [{"reranker_score": 0.9, "source_file": "f1.pdf"},
                {"reranker_score": 0.7, "source_file": "f2.pdf"}]
        noisy = good + [{"reranker_score": 0.001, "source_file": "f3.pdf"}
                        for _ in range(8)]

        a = _heuristic_assessment({**sample_agent_state, "reranked_results": good})
        b = _heuristic_assessment({**sample_agent_state, "reranked_results": noisy})

        assert a is not None and b is not None
        assert b["context_quality_score"] >= a["context_quality_score"]


class TestProgressiveFilterRelaxation:
    """
    Agent 3 must relax the document_category filter once retrieval has failed.

    document_category encodes Agent 1's guess at which document holds the answer,
    applied as a hard Qdrant `must`. When the guess is wrong the answer is outside
    the search space and the rewrite loop — which only changes query text — can
    never recover it. Isolation (deal_id) and compliance (PII, version) filters
    must survive relaxation.
    """

    @staticmethod
    async def _capture_filters(monkeypatch, rewrite_iteration: int) -> dict:
        """Runs the node with retrieval stubbed out, returning the filters it sent."""
        import numpy as np
        import src.agents.retrieval_executor as rx

        captured: dict = {}

        async def fake_hybrid_search(**kwargs):
            captured.update(kwargs["metadata_filters"])
            return [], []

        async def fake_embed(texts):
            return [np.zeros(1024)]

        async def fake_expand(**kwargs):
            return []

        monkeypatch.setattr(rx, "hybrid_search", fake_hybrid_search)
        monkeypatch.setattr(rx, "embed_texts_async", fake_embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: None)
        monkeypatch.setattr(rx, "expand_context", fake_expand)
        monkeypatch.setattr(rx, "reciprocal_rank_fusion", lambda **kw: [])

        await rx.retrieval_executor_node({
            "current_query": "what is the merger consideration per share?",
            "query_type": "legal",
            "parsed_intent": {},
            "deal_id": "deal-1",
            "include_pii": False,
            "rewrite_iteration": rewrite_iteration,
            "retrieval_config": {},
            "extracted_filters": {"document_category": "financial", "doc_id": "d1"},
        })
        return captured

    @pytest.mark.asyncio
    async def test_category_kept_on_first_attempt(self, monkeypatch):
        """First attempt keeps the category filter — it buys precision."""
        filters = await self._capture_filters(monkeypatch, rewrite_iteration=0)
        assert filters["document_category"] == "financial"

    @pytest.mark.asyncio
    async def test_category_dropped_after_rewrite(self, monkeypatch):
        """After a failed attempt, trade precision for recall."""
        filters = await self._capture_filters(monkeypatch, rewrite_iteration=1)
        assert "document_category" not in filters

    @pytest.mark.asyncio
    async def test_isolation_and_compliance_filters_survive(self, monkeypatch):
        """deal_id scoping and PII exclusion are never relaxed."""
        import src.agents.retrieval_executor as rx

        captured = {}

        async def fake_hybrid_search(**kwargs):
            captured.update(kwargs)
            return [], []

        import numpy as np
        monkeypatch.setattr(rx, "hybrid_search", fake_hybrid_search)
        monkeypatch.setattr(rx, "embed_texts_async", lambda t: _coro([np.zeros(1024)]))
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: None)
        monkeypatch.setattr(rx, "expand_context", lambda **kw: _coro([]))
        monkeypatch.setattr(rx, "reciprocal_rank_fusion", lambda **kw: [])

        await rx.retrieval_executor_node({
            "current_query": "q", "query_type": "legal", "parsed_intent": {},
            "deal_id": "deal-1", "include_pii": False, "rewrite_iteration": 2,
            "retrieval_config": {}, "extracted_filters": {"document_category": "legal"},
        })

        assert captured["deal_id"] == "deal-1"
        assert captured["metadata_filters"]["include_pii"] is False


async def _coro(value):
    return value


class TestCitationSelection:
    """
    Citations must name the passages the answer used, not every chunk from a
    document the answer happened to mention.
    """

    @staticmethod
    def _chunks():
        return [
            {"chunk_id": "a", "source_file": "merger.txt", "page_number": 1},
            {"chunk_id": "b", "source_file": "merger.txt", "page_number": 7},
            {"chunk_id": "c", "source_file": "financials.txt", "page_number": 3},
        ]

    def test_page_in_marker_narrows_to_that_passage(self):
        """A marker naming p.7 must not drag in p.1 of the same document."""
        from src.agents.answer_synthesizer import _select_cited_chunks

        answer = "The cap is $174M [📄 merger.txt | FY2024 | p.7 | ARTICLE VIII]."
        got = [c["chunk_id"] for c in _select_cited_chunks(answer, self._chunks())]
        assert got == ["b"]

    def test_multiple_markers_select_each_passage(self):
        from src.agents.answer_synthesizer import _select_cited_chunks

        answer = (
            "Revenue was $452.8M [📄 financials.txt | FY2023 | p.3 | INCOME STATEMENT] "
            "and the cap is $174M [📄 merger.txt | p.7 | ARTICLE VIII]."
        )
        got = {c["chunk_id"] for c in _select_cited_chunks(answer, self._chunks())}
        assert got == {"b", "c"}

    def test_marker_without_page_keeps_document_match(self):
        """Excel/slide citations carry no page; the document match must stand."""
        from src.agents.answer_synthesizer import _select_cited_chunks

        answer = "See [📊 merger.txt | Sheet \"Cap\" | COMPUTED: indemnity cap]."
        got = {c["chunk_id"] for c in _select_cited_chunks(answer, self._chunks())}
        assert got == {"a", "b"}

    def test_falls_back_when_no_marker_parsed(self):
        """
        A bare filename mention with no marker must still produce citations.

        An over-broad citation list is recoverable; an empty one loses the
        provenance the whole system exists to provide.
        """
        from src.agents.answer_synthesizer import _select_cited_chunks

        answer = "According to merger.txt the cap is $174M."
        got = {c["chunk_id"] for c in _select_cited_chunks(answer, self._chunks())}
        assert got == {"a", "b"}

    def test_uncited_documents_are_excluded(self):
        from src.agents.answer_synthesizer import _select_cited_chunks

        answer = "Revenue was $452.8M [📄 financials.txt | p.3 | INCOME STATEMENT]."
        got = [c["chunk_id"] for c in _select_cited_chunks(answer, self._chunks())]
        assert got == ["c"]

    def test_no_citations_when_nothing_referenced(self):
        from src.agents.answer_synthesizer import _select_cited_chunks

        assert _select_cited_chunks("I could not find that.", self._chunks()) == []

    def test_handles_empty_answer(self):
        from src.agents.answer_synthesizer import _select_cited_chunks

        assert _select_cited_chunks("", self._chunks()) == []
        assert _select_cited_chunks(None, self._chunks()) == []
