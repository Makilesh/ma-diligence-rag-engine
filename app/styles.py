"""
Shared UI styling for the Streamlit dashboard.

All CSS lives here so components stay logic-only and emit semantic class names
(`dd-*`) rather than inline style. The palette below is the single source of
truth; `.streamlit/config.toml` repeats four of these values because Streamlit
reads its base theme before any Python runs, so those two must be changed
together or the injected CSS will drift from the widgets Streamlit paints
itself.

Design intent: an institutional finance surface — deep ink canvas, quiet layered
elevation, one warm accent used sparingly for actions and provenance. Colour
carries meaning here (validated / warning / failed / refused), so it is spent on
state rather than decoration.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Canvas and elevation. Surfaces get lighter as they come forward; borders are
# a separate ramp so a raised card reads as raised without a heavy outline.
CANVAS = "#0B0E14"
SURFACE = "#12151F"
SURFACE_2 = "#171B27"
SURFACE_3 = "#1E2331"
BORDER = "#252B3A"
BORDER_STRONG = "#333A4D"

# Type
TEXT = "#E6E9EF"
MUTED = "#939CB0"
FAINT = "#6B7488"

# Accent — warm gold. Reserved for actions, focus, and source provenance.
ACCENT = "#E0B36A"
ACCENT_BRIGHT = "#F2CB86"
ACCENT_DIM = "#8A6E3F"

# Semantic state
OK = "#45B87C"
WARN = "#E0A458"
BAD = "#E5645B"
INFO = "#6FA8DC"
VIOLET = "#8B7FE8"

SEVERITY_COLORS = {"high": BAD, "medium": WARN, "low": OK}

_CSS = f"""
<style>
:root {{
    --dd-canvas: {CANVAS};
    --dd-surface: {SURFACE};
    --dd-surface-2: {SURFACE_2};
    --dd-surface-3: {SURFACE_3};
    --dd-border: {BORDER};
    --dd-border-strong: {BORDER_STRONG};
    --dd-text: {TEXT};
    --dd-muted: {MUTED};
    --dd-faint: {FAINT};
    --dd-accent: {ACCENT};
    --dd-accent-bright: {ACCENT_BRIGHT};
    --dd-ok: {OK};
    --dd-warn: {WARN};
    --dd-bad: {BAD};
    --dd-violet: {VIOLET};
    --dd-radius: 12px;
    --dd-radius-sm: 8px;
    --dd-shadow: 0 1px 2px rgba(0,0,0,.28), 0 8px 24px -12px rgba(0,0,0,.55);
}}

/* ---------- Layout ---------- */
.stApp {{ background: var(--dd-canvas); }}
.block-container {{
    padding-top: 2.4rem;
    padding-bottom: 4rem;
    max-width: 1360px;
}}

/* A very subtle top-of-page glow keeps the near-black canvas from reading flat
   on large monitors, without becoming a decorative gradient. */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 320px;
    background: radial-gradient(1200px 320px at 18% -60px, {ACCENT}0E, transparent 70%);
    pointer-events: none;
    z-index: 0;
}}

/* ---------- Page header ---------- */
.dd-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
}}
.dd-header h1 {{
    font-size: 1.72rem;
    font-weight: 680;
    letter-spacing: -0.025em;
    margin: 0;
    background: linear-gradient(92deg, {TEXT} 12%, {ACCENT_BRIGHT} 92%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.dd-sub {{
    color: var(--dd-muted);
    font-size: 0.92rem;
    line-height: 1.5;
    margin: 0.45rem 0 1.15rem 0;
    max-width: 68ch;
}}
.dd-rule {{
    height: 1px;
    border: 0;
    background: linear-gradient(90deg, {ACCENT}66, {BORDER} 38%, transparent);
    margin: 0 0 1.6rem 0;
}}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {{
    position: relative;
    background: linear-gradient(180deg, var(--dd-surface-2), var(--dd-surface));
    border: 1px solid var(--dd-border);
    border-radius: var(--dd-radius);
    padding: 0.95rem 1.1rem 0.9rem;
    box-shadow: var(--dd-shadow);
    overflow: hidden;
    transition: border-color .16s ease, transform .16s ease;
}}
/* Hairline of accent along the top edge — enough to group the row visually
   without adding another border weight. */
div[data-testid="stMetric"]::after {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT}CC, {ACCENT}22 60%, transparent);
}}
div[data-testid="stMetric"]:hover {{
    border-color: var(--dd-border-strong);
    transform: translateY(-1px);
}}
div[data-testid="stMetric"] label {{
    color: var(--dd-faint) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.085em;
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.55rem !important;
    font-weight: 640;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}}

/* ---------- Tabs: segmented control ---------- */
div[data-baseweb="tab-list"] {{
    gap: 0.3rem;
    background: var(--dd-surface);
    border: 1px solid var(--dd-border);
    border-radius: 11px;
    padding: 0.32rem;
    margin-bottom: 1.15rem;
}}
button[data-baseweb="tab"] {{
    font-size: 0.87rem;
    font-weight: 560;
    padding: 0.5rem 0.95rem;
    border-radius: var(--dd-radius-sm);
    color: var(--dd-muted);
    transition: background .16s ease, color .16s ease;
}}
button[data-baseweb="tab"]:hover {{
    background: var(--dd-surface-3);
    color: var(--dd-text);
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: var(--dd-surface-3);
    color: var(--dd-accent-bright);
    box-shadow: inset 0 0 0 1px {ACCENT}3D;
}}
/* The sliding underline is redundant once tabs are pills. */
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{
    display: none;
}}

/* ---------- Containers, expanders ---------- */
div[data-testid="stExpander"] {{
    border: 1px solid var(--dd-border);
    border-radius: var(--dd-radius);
    background: var(--dd-surface);
    overflow: hidden;
}}
div[data-testid="stExpander"] summary:hover {{ color: var(--dd-accent); }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: var(--dd-radius);
}}

/* ---------- Answer surface ---------- */
.dd-answer {{
    background: var(--dd-surface);
    border: 1px solid var(--dd-border);
    border-left: 3px solid var(--dd-accent);
    border-radius: var(--dd-radius);
    padding: 1.2rem 1.4rem;
    line-height: 1.68;
    box-shadow: var(--dd-shadow);
}}

/* Answer bodies are model markdown — tighten the rhythm so long answers with
   headings and lists stay readable rather than sprawling.
   `:has(.dd-answer-marker)` selects the bordered container that answer_display
   drops an empty marker span into; see the comment there for why the obvious
   wrapper-div approach cannot work under Streamlit. */
#dd-answer, div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) {{
    border-color: var(--dd-border) !important;
    border-left: 3px solid var(--dd-accent) !important;
    border-radius: var(--dd-radius) !important;
    background: var(--dd-surface) !important;
    box-shadow: var(--dd-shadow);
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) h1,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) h2,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) h3,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) h4 {{
    font-size: 0.98rem;
    font-weight: 640;
    letter-spacing: 0.01em;
    color: var(--dd-accent-bright);
    margin: 1.2rem 0 0.5rem;
    padding: 0;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) li {{
    margin-bottom: 0.3rem;
    line-height: 1.62;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) p {{
    line-height: 1.68;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.87rem;
    font-variant-numeric: tabular-nums;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) th {{
    background: var(--dd-surface-3);
    color: var(--dd-muted);
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) th,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.dd-answer-marker) td {{
    border: 1px solid var(--dd-border);
    padding: 0.45rem 0.7rem;
    text-align: left;
}}

/* ---------- Pills ---------- */
.dd-pill {{
    display: inline-block;
    padding: 0.2rem 0.62rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 680;
    letter-spacing: 0.045em;
    border: 1px solid var(--dd-border);
    margin-right: 0.45rem;
    vertical-align: middle;
    white-space: nowrap;
}}
.dd-pill-accent {{ background: {ACCENT}1F; color: {ACCENT_BRIGHT}; border-color: {ACCENT}59; }}
.dd-pill-ok     {{ background: {OK}1F;     color: {OK};     border-color: {OK}59; }}
.dd-pill-warn   {{ background: {WARN}1F;   color: {WARN};   border-color: {WARN}59; }}
.dd-pill-bad    {{ background: {BAD}1F;    color: {BAD};    border-color: {BAD}59; }}
.dd-pill-info   {{ background: {VIOLET}1F; color: {VIOLET}; border-color: {VIOLET}59; }}
.dd-pill-muted  {{ background: transparent; color: var(--dd-muted); }}

/* ---------- Inline stat strip ---------- */
.dd-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.2rem 0 1rem 0;
}}
.dd-stat {{
    flex: 1 1 140px;
    background: linear-gradient(180deg, var(--dd-surface-2), var(--dd-surface));
    border: 1px solid var(--dd-border);
    border-radius: var(--dd-radius-sm);
    padding: 0.6rem 0.8rem;
}}
.dd-stat-l {{
    color: var(--dd-faint);
    font-size: 0.65rem;
    font-weight: 640;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.2rem;
}}
.dd-stat-v {{
    font-size: 1.02rem;
    font-weight: 620;
    color: var(--dd-text);
    font-variant-numeric: tabular-nums;
}}

/* ---------- Citation rows ---------- */
.dd-cite {{
    border-left: 2px solid var(--dd-border-strong);
    background: linear-gradient(90deg, {ACCENT}08, transparent 55%);
    border-radius: 0 var(--dd-radius-sm) var(--dd-radius-sm) 0;
    padding: 0.6rem 0.85rem 0.6rem 0.95rem;
    margin-bottom: 0.55rem;
    transition: border-color .16s ease;
}}
.dd-cite:hover {{ border-left-color: var(--dd-accent); }}
.dd-cite-stale {{
    border-left-color: {BAD};
    background: linear-gradient(90deg, {BAD}0F, transparent 55%);
}}
.dd-cite-src {{ font-weight: 620; color: var(--dd-text); }}
.dd-cite-meta {{ color: var(--dd-muted); font-size: 0.81rem; }}

/* ---------- Buttons ---------- */
.stButton > button {{
    border-radius: var(--dd-radius-sm);
    font-weight: 580;
    border: 1px solid var(--dd-border-strong);
    transition: transform .12s ease, border-color .16s ease, background .16s ease;
}}
.stButton > button:hover {{
    border-color: var(--dd-accent);
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(180deg, {ACCENT_BRIGHT}, {ACCENT});
    color: #241B08;
    border: none;
    font-weight: 660;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 18px -8px {ACCENT}CC;
}}
.stButton > button:disabled, .stButton > button:disabled:hover {{
    opacity: .42;
    transform: none;
    box-shadow: none;
}}

/* ---------- Inputs ---------- */
.stTextArea textarea, .stTextInput input {{
    background: var(--dd-surface) !important;
    border-radius: var(--dd-radius-sm) !important;
    border: 1px solid var(--dd-border) !important;
    font-size: 0.94rem !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {ACCENT}99 !important;
    box-shadow: 0 0 0 3px {ACCENT}1F !important;
}}

/* Visible keyboard focus. Streamlit's default outline is nearly invisible on a
   dark canvas, which makes the app unusable without a mouse. */
*:focus-visible {{
    outline: 2px solid {ACCENT}CC !important;
    outline-offset: 2px;
}}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SURFACE}, {CANVAS} 62%);
    border-right: 1px solid var(--dd-border);
}}
section[data-testid="stSidebar"] .stProgress > div > div > div {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_BRIGHT});
}}
section[data-testid="stSidebar"] .stProgress > div > div {{
    background: var(--dd-surface-3);
    height: 5px;
    border-radius: 999px;
}}
section[data-testid="stSidebar"] h1 {{
    font-size: 1.08rem;
    letter-spacing: -0.01em;
}}
section[data-testid="stSidebar"] h3 {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--dd-faint);
}}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {{
    border-radius: var(--dd-radius);
    border: 1px solid var(--dd-border);
}}

/* ---------- Scrollbars ---------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: {CANVAS}; }}
::-webkit-scrollbar-thumb {{
    background: {BORDER_STRONG};
    border-radius: 999px;
    border: 2px solid {CANVAS};
}}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_DIM}; }}

/* ---------- Misc chrome ---------- */
div[data-testid="stStatusWidget"] {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
div[data-testid="stDecoration"] {{ display: none; }}
code {{
    background: var(--dd-surface-3);
    border: 1px solid var(--dd-border);
    border-radius: 5px;
    padding: 0.08rem 0.34rem;
    font-size: 0.86em;
    color: {ACCENT_BRIGHT};
}}
</style>
"""


def inject_styles() -> None:
    """Injects the dashboard stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def escape_currency(text: str) -> str:
    """
    Escapes dollar signs so Streamlit does not parse them as LaTeX math.

    Streamlit's markdown treats `$...$` as inline math. An answer containing two
    currency figures on one line — "revenue was $184.2M, up from $151.7M" — has
    the span between them rendered as math, and BOTH dollar signs plus any
    formatting inside are silently swallowed. For a tool whose premise is that a
    wrong number is a hard failure, a figure disappearing from the rendered
    answer is unacceptable, so every LLM-produced string is escaped before display.

    Args:
        text: Raw text that may contain currency figures.

    Returns:
        Text with dollar signs escaped for Streamlit markdown.
    """
    if not text:
        return text
    # Normalise first so already-escaped input is not double-escaped.
    return text.replace("\\$", "$").replace("$", r"\$")


def pill(text: str, variant: str = "muted") -> str:
    """
    Returns an inline HTML pill badge.

    Args:
        text: Label text.
        variant: One of accent|ok|warn|bad|info|muted.

    Returns:
        HTML string for use inside st.markdown(..., unsafe_allow_html=True).
    """
    return f'<span class="dd-pill dd-pill-{variant}">{text}</span>'


def stat_row(items: list[tuple[str, str]]) -> str:
    """
    Returns a compact inline stat strip: label/value pairs on one line.

    Used where `st.metric` cards would be too heavy — a run summary above an
    answer, for instance, where the numbers are context rather than the subject.

    Args:
        items: (label, value) pairs, rendered left to right.

    Returns:
        HTML string for st.markdown(..., unsafe_allow_html=True).
    """
    cells = "".join(
        f'<div class="dd-stat"><div class="dd-stat-l">{label}</div>'
        f'<div class="dd-stat-v">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="dd-stats">{cells}</div>'
