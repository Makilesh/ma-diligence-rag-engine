"""
Risk dashboard component — summarizes risk signals detected during ingestion.
"""

import streamlit as st

from app.styles import pill

# Display metadata per signal type. Keys mirror RISK_PATTERNS in
# src/data_processing/risk_signal_extractor.py — keep the two in sync when a
# new pattern family is added there, or its signals fall through to "other".
SIGNAL_CATEGORIES: dict[str, dict] = {
    "change_of_control": {"icon": "🔄", "label": "Change of Control"},
    "material_adverse_change": {"icon": "⚠️", "label": "Material Adverse Change"},
    "litigation": {"icon": "⚖️", "label": "Litigation / Legal Risk"},
    "regulatory_risk": {"icon": "🏛️", "label": "Regulatory Risk"},
    "financial_distress": {"icon": "📉", "label": "Financial Distress"},
    "environmental_liability": {"icon": "🌱", "label": "Environmental Liability"},
    "key_person": {"icon": "👤", "label": "Key Person Dependency"},
    "ip_risk": {"icon": "💡", "label": "IP Risk"},
    "customer_concentration": {"icon": "🎯", "label": "Customer Concentration"},
    "indemnification": {"icon": "🛡️", "label": "Indemnification"},
    "other": {"icon": "📋", "label": "Other Signals"},
}

SEVERITY_PILL = {"high": "bad", "medium": "warn", "low": "ok"}
SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def render_risk_dashboard(risk_signals: list[dict]) -> None:
    """
    Renders a dashboard of risk signals detected across deal documents.

    Signals are produced by a deterministic regex pass at ingestion time, so this
    view costs nothing at query time and stays consistent across queries.

    Args:
        risk_signals: List of risk signal dicts from GET /deals/{id}/risk-signals.
            Each dict has: signal_type, severity, source_file, description, page_number.
    """
    if not risk_signals:
        st.success(
            "No risk signals detected in this deal's documents — "
            "or no documents have been ingested yet."
        )
        return

    # Summary metrics
    counts = {"high": 0, "medium": 0, "low": 0}
    for s in risk_signals:
        sev = s.get("severity", "low")
        if sev in counts:
            counts[sev] += 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", len(risk_signals))
    with col2:
        st.metric("🔴 High", counts["high"])
    with col3:
        st.metric("🟡 Medium", counts["medium"])
    with col4:
        st.metric("🟢 Low", counts["low"])

    st.write("")

    # Group by signal type
    grouped: dict[str, list[dict]] = {}
    for signal in risk_signals:
        sig_type = signal.get("signal_type", "other")
        if sig_type not in SIGNAL_CATEGORIES:
            sig_type = "other"
        grouped.setdefault(sig_type, []).append(signal)

    # Render in the declared category order so the layout is stable across deals
    for cat_key, cat_info in SIGNAL_CATEGORIES.items():
        items = grouped.get(cat_key)
        if not items:
            continue

        has_high = any(i.get("severity") == "high" for i in items)
        with st.expander(
            f"{cat_info['icon']}  {cat_info['label']}  ({len(items)})",
            expanded=has_high,
        ):
            for item in items:
                severity = item.get("severity", "low")
                source = item.get("source_file", "Unknown")
                page = item.get("page_number")
                desc = item.get("description", "No description")

                st.markdown(
                    pill(severity.upper(), SEVERITY_PILL.get(severity, "muted"))
                    + f"<span class='dd-cite-src'>{source}</span>"
                    + (f"<span class='dd-cite-meta'> · page {page}</span>" if page else ""),
                    unsafe_allow_html=True,
                )
                st.caption(f"{SEVERITY_ICON.get(severity, '⚪')} {desc}")
