"""
Tests for retrieval pipeline components.
"""



class TestConditionalEdges:
    """Tests for LangGraph conditional edge functions."""

    def test_route_financial_to_verifier(self, sample_agent_state):
        """Financial queries route to financial_verifier."""
        from src.workflow.conditional_edges import route_to_financial_verifier

        state = {**sample_agent_state, "query_type": "financial"}
        result = route_to_financial_verifier(state)
        assert result == "financial_verifier"

    def test_route_legal_to_quality(self, sample_agent_state):
        """Legal queries skip financial verifier."""
        from src.workflow.conditional_edges import route_to_financial_verifier

        state = {
            **sample_agent_state,
            "query_type": "legal",
            "parsed_intent": {},
        }
        result = route_to_financial_verifier(state)
        assert result == "quality_assessor"

    def test_numerical_precision_routes_to_verifier(self, sample_agent_state):
        """requires_numerical_precision routes to financial_verifier regardless of type."""
        from src.workflow.conditional_edges import route_to_financial_verifier

        state = {
            **sample_agent_state,
            "query_type": "summary",
            "parsed_intent": {"requires_numerical_precision": True},
        }
        result = route_to_financial_verifier(state)
        assert result == "financial_verifier"

    def test_quality_pass_routes_to_synthesizer(self, sample_agent_state):
        """Good quality score routes to answer_synthesizer."""
        from src.workflow.conditional_edges import route_after_quality_check

        state = {
            **sample_agent_state,
            "context_quality_score": 0.8,
            "quality_breakdown": {"relevance": 0.9, "completeness": 0.8, "precision": 0.85},
            "rewrite_iteration": 0,
        }
        result = route_after_quality_check(state)
        assert result == "answer_synthesizer"

    def test_quality_fail_routes_to_rewriter(self, sample_agent_state):
        """Low quality with rewrites remaining routes to rewriter."""
        from src.workflow.conditional_edges import route_after_quality_check

        state = {
            **sample_agent_state,
            "context_quality_score": 0.15,
            "quality_breakdown": {"relevance": 0.2, "completeness": 0.1, "precision": 0.15},
            "rewrite_iteration": 0,
        }
        result = route_after_quality_check(state)
        assert result == "query_rewriter"

    def test_max_rewrites_routes_to_insufficient(self, sample_agent_state):
        """Exhausted rewrites route to insufficient_context."""
        from src.workflow.conditional_edges import route_after_quality_check

        state = {
            **sample_agent_state,
            "context_quality_score": 0.15,
            "quality_breakdown": {"relevance": 0.2, "completeness": 0.1, "precision": 0.15},
            "rewrite_iteration": 2,
        }
        result = route_after_quality_check(state)
        assert result == "insufficient_context"

    def test_validation_pass_routes_to_end(self, sample_agent_state):
        """Passed validation routes to end."""
        from src.workflow.conditional_edges import route_after_validation

        state = {**sample_agent_state, "validation_status": "passed", "validation_attempt": 1}
        result = route_after_validation(state)
        assert result == "end"

    def test_validation_fail_retries(self, sample_agent_state):
        """Failed validation with retries remaining routes to retry."""
        from src.workflow.conditional_edges import route_after_validation

        state = {**sample_agent_state, "validation_status": "failed", "validation_attempt": 0}
        result = route_after_validation(state)
        assert result == "retry_synthesis"


class TestStateDefinitions:
    """Tests for AgentState TypedDict structure."""

    def test_state_has_required_fields(self, sample_agent_state):
        """Verify all required fields exist."""
        required_fields = [
            "original_query", "current_query", "query_type",
            "parsed_intent", "extracted_filters", "retrieval_config",
            "dense_results", "sparse_results", "fused_results",
            "reranked_results", "expanded_context",
            "context_quality_score", "quality_breakdown",
            "rewrite_iteration", "rewrite_history", "agent_trace",
            "generated_answer", "citations",
            "confidence_score", "hallucination_flags", "validation_status",
            "deal_id", "session_id",
        ]
        for field in required_fields:
            assert field in sample_agent_state, f"Missing field: {field}"
