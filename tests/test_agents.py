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


class TestSubQuestionDecomposition:
    """
    Multi-hop queries must retrieve each facet separately and keep all of them.

    The failure this addresses: asked for an implied EV/EBITDA multiple, the
    engine retrieved EBITDA passages but not the share price, because the
    reranker scores every candidate against the whole question and a passage
    holding only a price scores poorly against "what multiple is implied". Query
    expansion cannot fix it — rephrasing the question five ways still never asks
    for the price.
    """

    @staticmethod
    def _chunk(cid, score, source="d.txt"):
        return {"chunk_id": cid, "reranker_score": score, "source_file": source,
                "text": cid}

    def test_weaker_facet_survives_a_dominant_one(self):
        """
        The core guarantee.

        Facet A's passages all outscore facet B's. A global top-3 would return
        only A — which is precisely the original bug. The quota merge must keep
        B represented.
        """
        from src.agents.retrieval_executor import _merge_by_quota

        ebitda = [self._chunk("e1", 0.99), self._chunk("e2", 0.97),
                  self._chunk("e3", 0.95)]
        price = [self._chunk("p1", 0.42), self._chunk("p2", 0.31)]

        merged = _merge_by_quota([ebitda, price], final_top_k=3)
        ids = {c["chunk_id"] for c in merged}

        assert len(merged) == 3
        assert ids & {"p1", "p2"}, "the weaker facet must not be crowded out"
        assert ids & {"e1", "e2", "e3"}

    def test_global_top_k_would_have_dropped_it(self):
        """Documents why a plain union is not sufficient."""
        from src.agents.retrieval_executor import _merge_by_quota

        ebitda = [self._chunk("e1", 0.99), self._chunk("e2", 0.97),
                  self._chunk("e3", 0.95)]
        price = [self._chunk("p1", 0.42)]

        naive = sorted(ebitda + price,
                       key=lambda c: c["reranker_score"], reverse=True)[:3]
        assert "p1" not in {c["chunk_id"] for c in naive}

        merged = _merge_by_quota([ebitda, price], final_top_k=3)
        assert "p1" in {c["chunk_id"] for c in merged}

    def test_duplicates_keep_their_best_score(self):
        """A passage found by two sub-questions is ranked by its strongest match."""
        from src.agents.retrieval_executor import _merge_by_quota

        a = [self._chunk("shared", 0.40)]
        b = [self._chunk("shared", 0.90)]

        merged = _merge_by_quota([a, b], final_top_k=5)
        assert len(merged) == 1
        assert merged[0]["reranker_score"] == 0.90

    def test_respects_the_context_budget(self):
        from src.agents.retrieval_executor import _merge_by_quota

        lists = [[self._chunk(f"{p}{i}", 0.5) for i in range(10)]
                 for p in ("a", "b", "c")]
        assert len(_merge_by_quota(lists, final_top_k=6)) == 6

    def test_handles_empty_and_uneven_passes(self):
        """A sub-question that retrieved nothing must not break the merge."""
        from src.agents.retrieval_executor import _merge_by_quota

        merged = _merge_by_quota([[], [self._chunk("x", 0.5)], []], final_top_k=4)
        assert [c["chunk_id"] for c in merged] == ["x"]
        assert _merge_by_quota([], final_top_k=4) == []
        assert _merge_by_quota([[], []], final_top_k=4) == []

    def test_single_pass_is_unchanged(self):
        """Pointed queries must behave exactly as before decomposition existed."""
        from src.agents.retrieval_executor import _merge_by_quota

        only = [self._chunk("a", 0.9), self._chunk("b", 0.8), self._chunk("c", 0.7)]
        merged = _merge_by_quota([only], final_top_k=2)
        assert [c["chunk_id"] for c in merged] == ["a", "b"]

    def test_prompt_contract_names_the_decomposable_types(self):
        """
        The prompt instructs decomposition for these types; the constant records it.

        This is a contract check only. It deliberately does NOT assert that other
        types are refused — see the next test for why that veto was removed.
        """
        from src.agents.query_intelligence import DECOMPOSABLE_QUERY_TYPES

        assert "multi_hop" in DECOMPOSABLE_QUERY_TYPES
        assert "comparative" in DECOMPOSABLE_QUERY_TYPES

    @pytest.mark.asyncio
    async def test_sub_questions_survive_an_unexpected_query_type(self, monkeypatch):
        """
        Sub-questions must not be discarded because the classifier said 'financial'.

        Regression test for a live failure. Asked for the implied EV/EBITDA
        multiple, Agent 1 decomposed the question correctly *and* classified it
        `financial`; a `query_type not in DECOMPOSABLE_QUERY_TYPES` veto then
        threw the decomposition away, and the engine answered about the offer
        price instead of computing the multiple. Emitting sub-questions is the
        model's judgment that the answer spans several facts; classification is a
        separate judgment made for routing. Requiring both to agree made the
        feature fail silently whenever they disagreed.
        """
        import src.agents.query_intelligence as qi

        async def fake_call(*args, **kwargs):
            return {
                "query_type": "financial",
                "primary_intent": "compute the implied multiple",
                "extracted_entities": {},
                "metadata_filters": {},
                "query_expansions": [],
                "sub_questions": [
                    "What is the per-share merger consideration?",
                    "How many fully diluted shares are outstanding?",
                    "What is FY2023 Adjusted EBITDA?",
                ],
                "reformulated_query": "implied EV/EBITDA multiple on adjusted EBITDA",
            }

        class _Choice:
            model = "gemini/test"
            api_key = "k"
            key_index = 0

        class _Tracker:
            async def get_model_for_agent(self):
                return _Choice()

        async def fake_instance(*args, **kwargs):
            return _Tracker()

        monkeypatch.setattr(qi, "call_structured_agent", fake_call)
        monkeypatch.setattr(qi.BudgetTracker, "get_instance", fake_instance)

        out = await qi.query_intelligence_node(
            {"original_query": "implied EV/EBITDA multiple?", "deal_id": "d1"}
        )

        assert out["query_type"] == "financial"
        assert len(out["sub_questions"]) == 3, (
            "sub-questions were discarded because the classifier disagreed with "
            "the decomposer"
        )

    @pytest.mark.asyncio
    async def test_sub_questions_are_capped(self, monkeypatch):
        """Cost is bounded by MAX_SUB_QUESTIONS, not by the query type veto."""
        import src.agents.query_intelligence as qi

        async def fake_call(*args, **kwargs):
            return {
                "query_type": "multi_hop",
                "metadata_filters": {},
                "sub_questions": [f"question {i}?" for i in range(12)],
                "reformulated_query": "q",
            }

        class _Choice:
            model = "gemini/test"
            api_key = "k"
            key_index = 0

        class _Tracker:
            async def get_model_for_agent(self):
                return _Choice()

        async def fake_instance(*args, **kwargs):
            return _Tracker()

        monkeypatch.setattr(qi, "call_structured_agent", fake_call)
        monkeypatch.setattr(qi.BudgetTracker, "get_instance", fake_instance)

        out = await qi.query_intelligence_node(
            {"original_query": "q", "deal_id": "d1"}
        )
        assert len(out["sub_questions"]) == qi.MAX_SUB_QUESTIONS

    @pytest.mark.asyncio
    async def test_sub_questions_do_not_inherit_the_category_filter(self, monkeypatch):
        """
        A sub-question targets a different fact, usually in a different document.

        Applying the parent's document_category guess to it re-creates the very
        problem decomposition solves. Measured: an EV/EBITDA query decomposed
        correctly into price, share count and EBITDA, but all three passes
        inherited one category filter and returned chunks from a single document.
        """
        import numpy as np
        import src.agents.retrieval_executor as rx

        seen: list[dict] = []

        async def fake_retrieve(query, config, deal_id, metadata_filters):
            seen.append({"query": query, "filters": dict(metadata_filters)})
            return []

        monkeypatch.setattr(rx, "_retrieve_for_query", fake_retrieve)
        monkeypatch.setattr(rx, "expand_context",
                            lambda **kw: _coro([]))

        await rx.retrieval_executor_node({
            "current_query": "implied EV/EBITDA multiple",
            "query_type": "multi_hop",
            "parsed_intent": {},
            "deal_id": "d1",
            "include_pii": False,
            "rewrite_iteration": 0,
            "retrieval_config": {},
            "extracted_filters": {"document_category": "financial"},
            "sub_questions": ["What is the per-share price?", "What is FY2023 EBITDA?"],
        })

        assert len(seen) == 3, "parent query plus one pass per sub-question"

        # EVERY pass drops the category once the query has been decomposed —
        # including the parent. Decomposing is Agent 1 saying the answer spans
        # several facts, and those facts sit in documents of different
        # categories; constraining the parent search to one contradicts the
        # judgment that produced the sub-questions.
        #
        # Measured on the real corpus: the $696M aggregate merger consideration
        # is stated in the regulatory memo and the merger agreement. Filtering
        # to `financial` or `legal` removes the chunk that states it, so the
        # context comes back full of plausible high-scoring chunks with the one
        # required number missing, and the engine reports the multiple cannot be
        # computed. Progressive relaxation does not save it — that fires on low
        # context quality, and this context scores well. It is incomplete, not
        # weak, and nothing downstream can tell those apart.
        for pass_ in seen:
            assert "document_category" not in pass_["filters"]
            # Isolation and compliance filters still apply to every pass.
            assert pass_["filters"]["include_pii"] is False

    @pytest.mark.asyncio
    async def test_undecomposed_query_keeps_the_category_filter(self, monkeypatch):
        """
        Without sub-questions the category filter stays — it buys precision.

        The relaxation above is scoped to decomposed queries deliberately. A
        pointed single-fact lookup is exactly the case the filter helps, and
        dropping it everywhere would trade precision for nothing.
        """
        import src.agents.retrieval_executor as rx

        seen: list[dict] = []

        async def fake_retrieve(query, config, deal_id, metadata_filters):
            seen.append({"query": query, "filters": dict(metadata_filters)})
            return []

        monkeypatch.setattr(rx, "_retrieve_for_query", fake_retrieve)
        monkeypatch.setattr(rx, "expand_context", lambda **kw: _coro([]))

        await rx.retrieval_executor_node({
            "current_query": "What was FY2023 revenue?",
            "query_type": "financial",
            "parsed_intent": {},
            "deal_id": "d1",
            "include_pii": False,
            "rewrite_iteration": 0,
            "retrieval_config": {},
            "extracted_filters": {"document_category": "financial"},
            "sub_questions": [],
        })

        assert len(seen) == 1
        assert seen[0]["filters"].get("document_category") == "financial"

    @pytest.mark.asyncio
    async def test_decomposition_never_shrinks_the_parent_budget(self, monkeypatch):
        """
        Decomposition must add evidence, never displace it.

        Splitting a fixed final_top_k across passes cost the parent query most of
        its slots. Measured on comp_05: the one chunk holding the full severance
        table was squeezed out and fact coverage fell 20% -> 0%. The parent now
        keeps its whole budget and sub-questions add on top.
        """
        import src.agents.retrieval_executor as rx

        captured = {}

        async def fake_retrieve(query, config, deal_id, metadata_filters):
            captured.setdefault("k", config["final_top_k"])
            # Distinct ids per query — identical ids would dedupe and mask the
            # very allocation this test is checking.
            slug = query.replace(" ", "_")
            return [{"chunk_id": f"{slug}-{i}", "reranker_score": 0.9 - i / 100,
                     "source_file": "d.txt", "text": "x"} for i in range(20)]

        monkeypatch.setattr(rx, "_retrieve_for_query", fake_retrieve)
        monkeypatch.setattr(rx, "expand_context", lambda **kw: _coro([]))

        out = await rx.retrieval_executor_node({
            "current_query": "parent", "query_type": "multi_hop",
            "parsed_intent": {}, "deal_id": "d1", "include_pii": False,
            "rewrite_iteration": 0,
            "retrieval_config": {"final_top_k": 10},
            "extracted_filters": {},
            "sub_questions": ["sub one", "sub two"],
        })

        chunks = out["reranked_results"]
        # Parent keeps 10; each of 2 sub-questions adds 2 => 14, not 10.
        assert len(chunks) == 14, f"expected additive budget, got {len(chunks)}"

        parent_ids = {c["chunk_id"] for c in chunks
                      if c["chunk_id"].startswith("parent-")}
        assert len(parent_ids) == 10, "parent must retain its full allocation"


class TestUsableAnswerGuard:
    """
    An answer that cites nothing is a failed generation, not a weak answer.

    During a provider incident gemini-3.6-flash returned answers at 10-15% of
    their usual length with zero citation markers, and the pipeline published
    them. One scored `validation=passed` with `confidence=1.0` and no sources at
    all; another opened mid-thought with "Let's double-check all details to
    ensure accuracy" — the model's own working notes — and shipped at 0.85.

    For a tool whose premise is that every claim is traceable, an unsourced
    figure presented confidently is the exact failure it exists to prevent.
    """

    @staticmethod
    def _chunks(n=3):
        return [{"text": f"chunk {i}", "source_file": "f.txt"} for i in range(n)]

    def test_cited_answer_is_accepted(self):
        from src.agents.answer_synthesizer import _is_usable_answer

        answer = (
            "### 1. Direct Answer\n\nFY2023 revenue was $452.8 million, up 17.0% "
            "from $387.1 million in FY2022 "
            "[aurora_financials_fy2023.txt | p.1 | CONSOLIDATED INCOME STATEMENT]. "
            "The increase was driven by SaaS subscriptions, which grew 27.9% "
            "[aurora_financials_fy2023.txt | p.1 | CONSOLIDATED INCOME STATEMENT]."
        )
        assert _is_usable_answer(answer, self._chunks())

    def test_uncited_answer_is_rejected(self):
        """The real sum_03 failure: a full-looking answer with zero sources."""
        from src.agents.answer_synthesizer import _is_usable_answer

        answer = (
            "### 1. Direct Answer\n\nUnder Article VI of the Merger Agreement, "
            "closing is subject to regulatory conditions and government "
            "clearances, including antitrust clearance, CFIUS review, and "
            "certain foreign competition filings that must be completed before "
            "the transaction may be consummated by either party hereto."
        )
        assert not _is_usable_answer(answer, self._chunks())

    def test_scratchpad_leak_is_rejected(self):
        """The real sum_04 failure: internal reasoning shipped as the answer."""
        from src.agents.answer_synthesizer import _is_usable_answer

        answer = (
            "Wu's patents: [ip_portfolio_and_litigation_schedule.txt | p.2 | "
            "SECTION 2] Let's double-check all details to ensure accuracy and "
            "strict adherence to rules. Everything looks precise and complete "
            "so the final response can now be assembled for the reviewer."
        )
        assert not _is_usable_answer(answer, self._chunks())

    def test_truncated_fragment_is_rejected(self):
        from src.agents.answer_synthesizer import _is_usable_answer

        assert not _is_usable_answer("### 1. Direct Answer\n\nThe", self._chunks())
        assert not _is_usable_answer("", self._chunks())
        assert not _is_usable_answer(None, self._chunks())

    def test_uncited_decline_is_allowed_through(self):
        """
        Declining with evidence present must not be retried.

        This is the counterpart to the uncited-assertion case, and getting it
        wrong was expensive. Asked for a multiple the corpus cannot support,
        three successive reasoning models correctly replied that the transaction
        value was not stated — short, and with nothing to cite. The guard scored
        each as a failed generation, so the ladder burned three scarce 20-RPD
        rungs in a row, spent 183 seconds, and finally accepted the *weakest*
        model's answer. A guard that spends the good models to reach the worst
        one is worse than no guard.
        """
        from src.agents.answer_synthesizer import _is_usable_answer

        decline = (
            "### 1. Direct Answer. Based on the provided context, the "
            "information is insufficient to calculate an exact Enterprise Value "
            "to EBITDA multiple for the Vertex acquisition, because the total "
            "transaction purchase price is not stated in the retrieved documents."
        )
        assert _is_usable_answer(decline, self._chunks())

    def test_uncited_assertion_is_still_rejected(self):
        """The exception must not swallow the case the guard was built for."""
        from src.agents.answer_synthesizer import _is_usable_answer

        assertion = (
            "### 1. Direct Answer. Under Article VI of the Merger Agreement, "
            "closing is subject to antitrust clearance, CFIUS review, and certain "
            "foreign competition filings, each of which must be completed before "
            "the transaction may be consummated by either party hereto."
        )
        assert not _is_usable_answer(assertion, self._chunks())

    def test_refusal_without_context_is_allowed_through(self):
        """
        A genuine refusal has nothing to cite and must not be retried.

        This is the guard's most important limit: the refusal path is a designed
        outcome, and rejecting uncited text unconditionally would send the engine
        into a retry loop every time it correctly declined to answer.
        """
        from src.agents.answer_synthesizer import _is_usable_answer

        refusal = (
            "I was unable to find sufficient relevant information in the data "
            "room to answer this question, even after refining the search."
        )
        assert _is_usable_answer(refusal, [])
