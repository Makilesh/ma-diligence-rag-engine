"""
Streamlit UI — M&A Due Diligence Intelligence Engine.

Sidebar: deal selector, live budget status, document uploader.
Main: query interface plus a tabbed result surface (answer, citations,
agent trace, risk signals, version history).

This module is a pure HTTP client. It holds no pipeline logic — every decision
(refusal, validation status, confidence) is made server-side and rendered here.
"""

import os
import sys
from pathlib import Path

import streamlit as st
import requests

# `streamlit run app/streamlit_app.py` puts only app/ on sys.path — not the
# project root — so the absolute `app.components.*` imports below would fail.
# Prepend the project root explicitly. Works identically locally and in Docker.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.styles import inject_styles, pill
from app.components.deal_manager import render_deal_manager
from app.components.document_uploader import render_document_uploader
from app.components.query_interface import render_query_interface
from app.components.answer_display import render_answer, render_refusal
from app.components.citation_viewer import render_citations
from app.components.agent_trace_viewer import render_agent_trace
from app.components.risk_dashboard import render_risk_dashboard
from app.components.version_browser import render_version_browser

# API base URL — docker-compose injects API_URL=http://api:8000 (service DNS name);
# falls back to localhost for a bare-metal `streamlit run`.
API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE}/api/v1"

# The query pipeline runs 8 agents including two local LLM passes; ~70s is typical,
# so the client timeout is deliberately generous.
QUERY_TIMEOUT_S = 300.0

st.set_page_config(
    page_title="M&A Due Diligence Intelligence Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_json(path: str, timeout: int = 10):
    """
    GETs a JSON endpoint, returning None on any failure.

    The dashboard must stay usable when a panel's backing endpoint is down, so
    transport errors degrade to an empty panel rather than a crashed script.
    """
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def render_budget_panel() -> None:
    """Renders the live API budget gauges in the sidebar."""
    st.subheader("💰 API Budget")

    budget = _get_json("/budget")
    if budget is None:
        st.caption("⚠ Budget status unavailable")
        return

    for model_key, status in budget.items():
        remaining = status.get("remaining", 0)
        limit = status.get("limit", 0)
        used = status.get("used", 0)
        pct = (used / limit) if limit > 0 else 0.0

        # Budget exhaustion silently degrades synthesis to the local model,
        # so surfacing the pressure early is worth the sidebar real estate.
        variant = "ok" if pct < 0.7 else ("warn" if pct < 0.95 else "bad")
        st.markdown(
            f"<span class='dd-cite-meta'>{model_key.replace('_', ' ').title()}</span> "
            + pill(f"{remaining}/{limit} left", variant),
            unsafe_allow_html=True,
        )
        st.progress(min(pct, 1.0))


def run_query(query: str, deal_id: str, include_pii: bool) -> dict | None:
    """
    Posts a query to the pipeline and returns the parsed response.

    Args:
        query: Natural language question.
        deal_id: Deal scope for retrieval isolation.
        include_pii: Compliance override, forwarded to the API.

    Returns:
        Response dict, or None if the request failed (error already surfaced).
    """
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"query": query, "deal_id": deal_id, "include_pii": include_pii},
            timeout=QUERY_TIMEOUT_S,
        )
    except requests.ConnectionError:
        st.error("❌ Cannot connect to the API server. Is it running?")
        return None
    except requests.Timeout:
        st.error("⏱ The query exceeded the timeout. Try narrowing the question.")
        return None
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

    if resp.status_code != 200:
        st.error(f"Query failed ({resp.status_code}): {resp.text}")
        return None

    return resp.json()


def render_results(result: dict, deal_id: str) -> None:
    """
    Renders a completed query result across the tabbed surface.

    Args:
        result: Parsed QueryResponse.
        deal_id: Current deal, used to fetch deal-scoped risk and version panels.
    """
    citations = result.get("citations", [])
    trace = result.get("agent_trace", [])

    tab_answer, tab_cites, tab_trace, tab_risk, tab_versions = st.tabs(
        [
            "💬 Answer",
            f"📚 Citations ({len(citations)})",
            f"🧭 Agent Trace ({len(trace)})",
            "🚨 Risk Signals",
            "🗂 Versions",
        ]
    )

    with tab_answer:
        if result.get("is_refusal"):
            # A refusal is a designed outcome, not an error — it gets its own
            # component with the quality score and remediation hints.
            render_refusal(
                quality_score=result.get("context_quality_score", 0.0),
                rewrite_count=result.get("rewrite_iterations", 0),
            )
        else:
            render_answer(
                answer=result.get("answer", ""),
                confidence_score=result.get("confidence_score", 0.0),
                validation_status=result.get("validation_status", "passed"),
                hallucination_flags=result.get("hallucination_flags", []),
                query_type=result.get("query_type", "summary"),
                latency_ms=result.get("total_latency_ms", 0.0),
                rewrite_iterations=result.get("rewrite_iterations", 0),
            )

    with tab_cites:
        render_citations(citations)

    with tab_trace:
        render_agent_trace(trace)

    with tab_risk:
        render_risk_dashboard(_get_json(f"/deals/{deal_id}/risk-signals") or [])

    with tab_versions:
        render_version_browser(_get_json(f"/deals/{deal_id}/documents") or [])


def main():
    """Main Streamlit application."""
    inject_styles()

    # ==================== Sidebar ====================
    with st.sidebar:
        st.title("⚙️ Control Panel")

        deal_id = render_deal_manager(API_URL)
        st.divider()

        render_budget_panel()
        st.divider()

        if deal_id:
            render_document_uploader(API_URL, deal_id)
        else:
            st.info("👈 Select or create a Deal ID to enable document upload.")

    # ==================== Header ====================
    st.markdown(
        "<div class='dd-header'><h1>🔍 M&A Due Diligence Intelligence Engine</h1></div>"
        "<p class='dd-sub'>Agentic RAG over the data room — every claim traced to a "
        "source, every unsupported claim refused.</p>"
        "<hr class='dd-rule'/>",
        unsafe_allow_html=True,
    )

    if not deal_id:
        st.info("👈 Enter or select a Deal ID in the sidebar to get started.")
        return

    query_result = render_query_interface()

    if query_result:
        query, include_pii = query_result
        with st.spinner("Running the agentic pipeline — retrieval, verification, synthesis…"):
            result = run_query(query, deal_id, include_pii)
        if result is not None:
            # Cached in session state so switching tabs (which reruns the script)
            # does not discard a result that took ~70s to produce.
            st.session_state["last_result"] = result
            st.session_state["last_result_deal"] = deal_id

    cached = st.session_state.get("last_result")
    if cached and st.session_state.get("last_result_deal") == deal_id:
        render_results(cached, deal_id)


if __name__ == "__main__":
    main()
