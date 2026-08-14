"""
Answer display component — renders answer with confidence badge and validation status.
"""

import streamlit as st

from app.styles import escape_currency, pill, stat_row


def render_answer(
    answer: str,
    confidence_score: float,
    validation_status: str,
    hallucination_flags: list[str] | None = None,
    query_type: str = "",
    latency_ms: float = 0.0,
    rewrite_iterations: int = 0,
) -> None:
    """
    Renders the main answer with confidence badge, validation status,
    and hallucination warnings.

    Args:
        answer: Generated answer text with citations.
        confidence_score: Answer confidence 0.0–1.0.
        validation_status: "passed", "warning", or "failed".
        hallucination_flags: List of unsupported claims.
        query_type: Detected query type.
        latency_ms: Total pipeline latency.
        rewrite_iterations: Number of query rewrites performed.
    """
    # Header metrics. A compact strip rather than st.metric cards: these are
    # context for the answer below, not the subject of the page, and four full
    # cards pushed the answer itself below the fold.
    st.markdown(
        stat_row(
            [
                ("Confidence", f"{confidence_score:.0%}"),
                ("Query Type", (query_type or "—").replace("_", " ").title()),
                ("Latency", f"{latency_ms / 1000:.1f}s"),
                ("Rewrites", str(rewrite_iterations)),
            ]
        ),
        unsafe_allow_html=True,
    )

    # Validation status banner — the pipeline's own verdict on its own answer
    st.write("")
    if validation_status == "passed":
        st.markdown(
            pill("✓ VALIDATED", "ok")
            + "<span class='dd-cite-meta'>Every claim traced to a source document</span>",
            unsafe_allow_html=True,
        )
    elif validation_status == "warning":
        st.markdown(
            pill("⚠ VALIDATED WITH WARNINGS", "warn")
            + "<span class='dd-cite-meta'>Some claims need reviewer confirmation</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            pill("✕ VALIDATION FAILED", "bad")
            + "<span class='dd-cite-meta'>May contain unsupported claims — verify before relying on it</span>",
            unsafe_allow_html=True,
        )

    # Answer body — a bordered container rather than raw HTML, so the model's
    # markdown (lists, tables, emphasis) still renders correctly.
    #
    # The empty marker span is how the stylesheet finds this container. Wrapping
    # the answer in an opening <div> and closing it in a later st.markdown call
    # does not work: Streamlit renders every markdown call into its own DOM
    # container, so the parser closes the div immediately and the answer ends up
    # outside it. A marker plus `:has()` targets the real container instead.
    st.write("")
    with st.container(border=True):
        st.markdown("<span class='dd-answer-marker'></span>", unsafe_allow_html=True)
        st.markdown(escape_currency(answer))

    if hallucination_flags:
        st.write("")
        st.markdown(pill("UNSUPPORTED CLAIMS", "bad"), unsafe_allow_html=True)
        for flag in hallucination_flags:
            st.markdown(f"- 🔴 {escape_currency(flag)}")


def render_refusal(quality_score: float, rewrite_count: int) -> None:
    """
    Renders a styled refusal message when context is insufficient.

    A refusal is a designed terminal state, not an error: the engine prefers a
    traceable "I don't know" over a confident guess on a deal-critical number.

    Args:
        quality_score: Best achieved quality score.
        rewrite_count: Total search attempts.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Quality Score", f"{quality_score:.0%}")
    with col2:
        st.metric("Search Attempts", str(rewrite_count + 1))

    st.write("")
    st.markdown(pill("⊘ INSUFFICIENT CONTEXT", "bad"), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            "The engine could not find sufficient relevant information in the data "
            "room to answer this question accurately, so it declined to answer "
            "rather than produce an unsupported one."
        )

    st.write("")
    st.markdown(
        "**Try this:**\n"
        "- Confirm the relevant documents have been uploaded to this deal\n"
        "- Rephrase using the terminology the documents themselves use\n"
        "- Narrow the scope (a specific fiscal year, or a single document type)"
    )
