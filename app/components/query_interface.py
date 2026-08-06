"""
Query interface component — query input with type hints and controls.
"""

import streamlit as st

# One example per query type the classifier recognises, so the examples double
# as documentation of what the engine routes differently.
EXAMPLE_QUERIES = {
    "Financial": "What was the total revenue in FY2023 and how does it compare to FY2022?",
    "Legal": "What are the key change of control provisions in the merger agreement?",
    "Comparative": "Compare the EBITDA margins across the last three fiscal years.",
    "Summary": "Summarize the key findings from the most recent board minutes.",
    "Multi-hop": "What indemnification caps are tied to the representations that reference FY2023 financials?",
}


def render_query_interface() -> tuple[str, bool] | None:
    """
    Renders query input area with type hints, example queries,
    PII checkbox, and submit button.

    Returns:
        Tuple of (query_text, include_pii) if submitted, None otherwise.
    """
    # Example buttons write straight into the text area's own session_state key.
    #
    # The obvious approach — keeping a separate "query_text" key and passing it as
    # `value=` — silently does not work: once a keyed widget exists, Streamlit
    # serves its stored state and ignores `value=` on later reruns. The box
    # appeared to fill in while the widget still returned "", so `disabled=not
    # query` kept the Search button greyed out and clicking an example did
    # nothing. Assigning to the widget's own key before it is instantiated is the
    # supported way to set it.
    with st.expander("💡 Example queries — one per routed query type", expanded=False):
        for category, example in EXAMPLE_QUERIES.items():
            col1, col2 = st.columns([1, 6])
            with col1:
                st.caption(f"**{category}**")
            with col2:
                if st.button(example, key=f"example_{category}", use_container_width=True):
                    st.session_state["query_input"] = example
                    st.rerun()

    query = st.text_area(
        "Ask a question about the deal",
        placeholder="e.g. What are the termination rights under the supply agreement?",
        height=100,
        key="query_input",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1.4, 1.6, 5])
    with col1:
        submit = st.button(
            "🔎 Run Analysis",
            type="primary",
            disabled=not query,
            use_container_width=True,
        )
    with col2:
        include_pii = st.checkbox(
            "Include PII",
            value=False,
            help=(
                "Include PII-flagged content (HR records, salary data). Excluded by "
                "default; every authorized use is written to the audit log."
            ),
        )
    with col3:
        st.caption(
            "Answers cite their sources. Superseded document versions are flagged, "
            "and the engine refuses rather than guessing when context is thin."
        )

    if submit and query:
        return query.strip(), include_pii

    return None
