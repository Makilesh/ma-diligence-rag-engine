"""
Streamlit UI — M&A Due Diligence Intelligence Engine.

Sidebar: live budget status display, deal selector, document uploader.
Main: query interface, answer display, citation viewer, agent execution trace.
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

from app.components.deal_manager import render_deal_manager
from app.components.document_uploader import render_document_uploader
from app.components.query_interface import render_query_interface
from app.components.answer_display import render_answer, render_refusal
from app.components.citation_viewer import render_citations
from app.components.agent_trace_viewer import render_agent_trace

# API base URL — docker-compose injects API_URL=http://api:8000 (service DNS name);
# falls back to localhost for a bare-metal `streamlit run`.
API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE}/api/v1"

st.set_page_config(
    page_title="M&A Due Diligence Intelligence Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main Streamlit application."""

    # ==================== Sidebar ====================
    with st.sidebar:
        st.title("⚙️ Control Panel")

        # 1. Deal selector
        deal_id = render_deal_manager(API_URL)

        st.divider()

        # 2. Budget status
        st.subheader("💰 API Budget Status")
        try:
            resp = requests.get(f"{API_URL}/budget", timeout=10)
            if resp.status_code == 200:
                budget = resp.json()
                for model_key, status in budget.items():
                    remaining = status.get("remaining", 0)
                    limit = status.get("limit", 0)
                    used = status.get("used", 0)
                    pct = (used / limit * 100) if limit > 0 else 0

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.caption(model_key.replace("_", " ").title())
                    with col2:
                        st.caption(f"{remaining}/{limit}")

                    st.progress(min(pct / 100, 1.0))
        except Exception:
            st.warning("⚠ Could not fetch budget status")

        st.divider()

        # 3. Document upload
        if deal_id:
            render_document_uploader(API_URL, deal_id)
        else:
            st.info("👈 Select or create a Deal ID to enable document upload.")

    # ==================== Main Content ====================
    st.title("🔍 M&A Due Diligence Intelligence Engine")
    st.caption("Ask questions about your deal documents with traceable citations")

    if not deal_id:
        st.info("👈 Enter or select a Deal ID in the sidebar to get started.")
        return

    # Render query interface
    query_result = render_query_interface()

    if query_result:
        query, include_pii = query_result
        with st.spinner("Running query pipeline..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "deal_id": deal_id,
                        "include_pii": include_pii,
                    },
                    timeout=300.0,
                )

                if resp.status_code == 200:
                    result = resp.json()

                    # Render answer
                    # If confidence is below threshold and validation failed, or answer is refused, render refusal
                    # But the synthesizer has its own refusal logic, and answer_display has render_refusal.
                    # We will render the answer using render_answer component.
                    render_answer(
                        answer=result.get("answer", ""),
                        confidence_score=result.get("confidence_score", 0.0),
                        validation_status=result.get("validation_status", "passed"),
                        hallucination_flags=result.get("hallucination_flags", []),
                        query_type=result.get("query_type", "summary"),
                        latency_ms=result.get("total_latency_ms", 0.0),
                        rewrite_iterations=result.get("rewrite_iterations", 0),
                    )

                    # Render citations
                    if result.get("citations"):
                        st.markdown("---")
                        render_citations(result.get("citations", []))

                    # Render agent trace
                    if result.get("agent_trace"):
                        st.markdown("---")
                        render_agent_trace(result.get("agent_trace", []))

                else:
                    st.error(f"Query failed: {resp.text}")

            except requests.ConnectionError:
                st.error("❌ Cannot connect to API server. Is it running?")
            except Exception as e:
                st.error(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
