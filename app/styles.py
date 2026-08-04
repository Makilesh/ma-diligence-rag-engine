"""
Shared UI styling for the Streamlit dashboard.

Keeps all CSS in one place so components stay logic-only. The palette mirrors
.streamlit/config.toml — change both together or the injected CSS will drift
from the base theme.
"""

import streamlit as st

# Palette (kept in sync with .streamlit/config.toml)
ACCENT = "#D4A574"
SURFACE = "#171B26"
BORDER = "#2A2F3D"
MUTED = "#8B93A7"

SEVERITY_COLORS = {
    "high": "#E5534B",
    "medium": "#D4A574",
    "low": "#4FB477",
}

_CSS = f"""
<style>
/* ---------- Layout ---------- */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* ---------- Page header ---------- */
.dd-header {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0 0 0.35rem 0;
}}
.dd-header h1 {{
    font-size: 1.75rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin: 0;
}}
.dd-sub {{
    color: {MUTED};
    font-size: 0.9rem;
    margin: 0 0 1.4rem 0;
}}
.dd-rule {{
    height: 1px;
    background: linear-gradient(90deg, {ACCENT}55, transparent);
    border: 0;
    margin: 0 0 1.5rem 0;
}}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.85rem 1rem;
}}
div[data-testid="stMetric"] label {{
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.45rem !important;
    font-weight: 600;
}}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {{
    font-size: 0.92rem;
    font-weight: 550;
    padding: 0.55rem 0.15rem;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {ACCENT};
}}

/* ---------- Expanders / cards ---------- */
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {SURFACE};
}}

/* ---------- Answer surface ---------- */
.dd-answer {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 10px;
    padding: 1.15rem 1.35rem;
    line-height: 1.65;
}}

/* ---------- Pills ---------- */
.dd-pill {{
    display: inline-block;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid {BORDER};
    margin-right: 0.35rem;
}}
.dd-pill-accent {{ background: {ACCENT}22; color: {ACCENT}; border-color: {ACCENT}55; }}
.dd-pill-ok     {{ background: #4FB47722; color: #4FB477; border-color: #4FB47755; }}
.dd-pill-warn   {{ background: #D4A57422; color: #D4A574; border-color: #D4A57455; }}
.dd-pill-bad    {{ background: #E5534B22; color: #E5534B; border-color: #E5534B55; }}
.dd-pill-muted  {{ background: transparent;  color: {MUTED}; }}

/* ---------- Citation rows ---------- */
.dd-cite {{
    border-left: 2px solid {BORDER};
    padding: 0.5rem 0 0.5rem 0.9rem;
    margin-bottom: 0.5rem;
}}
.dd-cite-stale {{ border-left-color: #E5534B; }}
.dd-cite-src {{ font-weight: 600; }}
.dd-cite-meta {{ color: {MUTED}; font-size: 0.82rem; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stProgress > div > div > div {{
    background-color: {ACCENT};
}}

/* ---------- Misc ---------- */
div[data-testid="stStatusWidget"] {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_styles() -> None:
    """Injects the dashboard stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def pill(text: str, variant: str = "muted") -> str:
    """
    Returns an inline HTML pill badge.

    Args:
        text: Label text.
        variant: One of accent|ok|warn|bad|muted.

    Returns:
        HTML string for use inside st.markdown(..., unsafe_allow_html=True).
    """
    return f'<span class="dd-pill dd-pill-{variant}">{text}</span>'
