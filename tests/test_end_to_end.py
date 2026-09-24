"""
Integration-style tests with all external calls mocked.
"""

import pytest
from unittest.mock import patch


class TestQueryPipelineIntegration:
    """End-to-end pipeline tests with mocked externals."""

    def test_graph_topology(self):
        """The graph wires all nine nodes with the expected fixed edges, and compiles."""
        from src.workflow.orchestrator import build_graph

        graph = build_graph()

        assert set(graph.nodes) == {
            "query_intelligence",
            "retrieval_executor",
            "financial_verifier",
            "quality_assessor",
            "query_rewriter",
            "answer_synthesizer",
            "hallucination_validator",
            "insufficient_context",
            "retry_synthesis",
        }
        # Unconditional edges; the routing decisions are conditional edges
        # covered by the conditional_edges tests in test_agents.py.
        assert {
            ("__start__", "query_intelligence"),
            ("query_intelligence", "retrieval_executor"),
            ("financial_verifier", "quality_assessor"),
            ("query_rewriter", "retrieval_executor"),
            ("answer_synthesizer", "hallucination_validator"),
            ("retry_synthesis", "hallucination_validator"),
            ("insufficient_context", "__end__"),
        } <= set(graph.edges)
        # Compiling validates that every edge targets a registered node.
        graph.compile()

    def test_initial_state_covers_every_agent_state_field(self):
        """
        The run's seed state populates exactly the AgentState schema.

        A field missing from the seed is not a crash at build time — TypedDict
        access on an absent key only fails where some agent reads it — so this
        pins the seed to the schema.
        """
        from src.workflow.orchestrator import _build_initial_state
        from src.workflow.state_definitions import AgentState

        state = _build_initial_state(
            query="What was the revenue in FY2023?",
            deal_id="test_deal_001",
            session_id="test_session_001",
            include_pii=False,
        )

        assert set(state) == set(AgentState.__annotations__)
        assert state["original_query"] == state["current_query"] == "What was the revenue in FY2023?"
        assert state["deal_id"] == "test_deal_001"
        assert state["session_id"] == "test_session_001"
        assert state["include_pii"] is False
        assert state["rewrite_iteration"] == 0
        assert state["force_refusal"] is False

    @pytest.mark.asyncio
    async def test_forced_refusal_answer(self, sample_agent_state):
        """With force_refusal set, the synthesizer refuses without calling an LLM."""
        from src.agents.answer_synthesizer import answer_synthesizer_node

        state = {**sample_agent_state, "force_refusal": True}
        with patch(
            "src.agents.answer_synthesizer.BudgetTracker.get_instance",
            side_effect=AssertionError("forced refusal must not reach the model ladder"),
        ):
            result = await answer_synthesizer_node(state)

        assert "sufficient information" in result["generated_answer"]
        assert result["citations"] == []
        assert result["numerical_claims"] == []
        assert result["confidence_score"] == 0.0
        assert result["agent_trace"] == [
            {"agent": "answer_synthesizer", "forced_refusal": True}
        ]


class TestInsufficientContextPath:
    """Tests for the forced refusal flow."""

    @pytest.mark.asyncio
    async def test_insufficient_context_node(self, sample_agent_state):
        """insufficient_context_node sets force_refusal and generates message."""
        from src.workflow.orchestrator import insufficient_context_node

        state = {**sample_agent_state, "rewrite_iteration": 2, "context_quality_score": 0.1}
        result = await insufficient_context_node(state)

        assert result["force_refusal"] is True
        assert result["confidence_score"] == 0.0
        assert result["validation_status"] == "passed"
        assert "unable to find" in result["generated_answer"].lower()


class TestPromptTemplates:
    """Verify prompt templates are well-formed."""

    def test_query_intelligence_prompt(self):
        """Query intelligence template has required format variables."""
        from src.llm.prompt_templates.query_intelligence import (
            QUERY_INTELLIGENCE_SYSTEM_PROMPT,
            QUERY_INTELLIGENCE_USER_TEMPLATE,
        )

        assert "query_type" in QUERY_INTELLIGENCE_SYSTEM_PROMPT
        assert "{query}" in QUERY_INTELLIGENCE_USER_TEMPLATE
        assert "{deal_id}" in QUERY_INTELLIGENCE_USER_TEMPLATE

    def test_answer_synthesizer_prompt(self):
        """Answer synthesizer template has required format variables."""
        from src.llm.prompt_templates.answer_synthesizer import (
            ANSWER_SYNTHESIZER_SYSTEM_PROMPT,
            ANSWER_SYNTHESIZER_USER_TEMPLATE,
        )

        assert "citation" in ANSWER_SYNTHESIZER_SYSTEM_PROMPT.lower()
        assert "{query}" in ANSWER_SYNTHESIZER_USER_TEMPLATE
        assert "{context}" in ANSWER_SYNTHESIZER_USER_TEMPLATE

    def test_query_rewriter_prompt(self):
        """Query rewriter template has required format variables."""
        from src.llm.prompt_templates.query_rewriter import (
            QUERY_REWRITER_SYSTEM_PROMPT,
            QUERY_REWRITER_USER_TEMPLATE,
        )

        assert "include_pii" in QUERY_REWRITER_SYSTEM_PROMPT.lower()
        assert "{original_query}" in QUERY_REWRITER_USER_TEMPLATE
        assert "{missing_aspects}" in QUERY_REWRITER_USER_TEMPLATE

    def test_no_pii_in_query_intelligence(self):
        """Query intelligence prompt forbids include_pii."""
        from src.llm.prompt_templates.query_intelligence import (
            QUERY_INTELLIGENCE_SYSTEM_PROMPT,
        )

        assert "include_pii" in QUERY_INTELLIGENCE_SYSTEM_PROMPT.lower()
        assert "never" in QUERY_INTELLIGENCE_SYSTEM_PROMPT.lower() or \
               "forbidden" in QUERY_INTELLIGENCE_SYSTEM_PROMPT.lower()


class TestLiteLLMWrapper:
    """Tests for LiteLLM wrapper functions."""

    def test_call_structured_agent_is_async(self):
        """call_structured_agent is a coroutine function."""
        import asyncio
        from src.llm.litellm_wrapper import call_structured_agent

        assert asyncio.iscoroutinefunction(call_structured_agent)

    def test_call_prose_agent_is_async(self):
        """call_prose_agent is a coroutine function."""
        import asyncio
        from src.llm.litellm_wrapper import call_prose_agent

        assert asyncio.iscoroutinefunction(call_prose_agent)

    def test_call_local_agent_is_async(self):
        """call_local_agent is a coroutine function."""
        import asyncio
        from src.llm.litellm_wrapper import call_local_agent

        assert asyncio.iscoroutinefunction(call_local_agent)
