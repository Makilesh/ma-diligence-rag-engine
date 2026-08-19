"""
Citation viewer component — renders citations with version warnings and computed flags.
"""

import re

import streamlit as st

from app.styles import pill

# A chunk's `section_heading` is whatever line the structural chunker judged to
# be the heading, and on plain-text documents that is sometimes not a heading at
# all — a rule of "=" characters used as a visual divider, or the first sentence
# of a paragraph. Displaying those verbatim makes the citation list look broken
# in exactly the place the product is asking to be trusted, so they are filtered
# out and the citation falls back to file and page.
#
# This is presentation-only triage. The underlying imprecision is real and
# documented (DECISIONS_LOG Decision 16); hiding the worst artefacts is not the
# same as fixing heading granularity in the chunker.
_SEPARATOR_RUN = re.compile(r"^[\s=_\-*~#.·•]+$")
# Terminal punctuation on a long string means a complete sentence, which a
# heading is not. Set at 70 by inspection of every heading in the corpus: it
# drops "Customer concentration: the top ten customers represent 47.8% of FY2023
# revenue." and three like it, while keeping the table rows that legitimately
# end in a period ("SOC 2 Type II  FY2023 examination issued with NO
# exceptions.", "ISO 27001  Certified, valid through October 2025.").
#
# Corpus-tuned, and openly so. Thresholds picked against one document set do not
# generalise; the durable fix is per-heading granularity in the chunker
# (DECISIONS_LOG Decision 16), not more numbers here.
_MAX_HEADING_CHARS = 70

# A long "heading" that stops on a preposition, article, conjunction or dangling
# punctuation is a sentence the chunker cut, not a heading. Measured against
# every distinct section_heading in the sample corpus (109 of them), this drops
# 24 — including "VERTEX CAPITAL PARTNERS LLC, a Delaware limited liability
# company (" and "The aggregate Merger Consideration is approximately $696
# million, subject to" — without catching a single legitimate one. Real headings
# here end on a noun, a number or a closing bracket: "Note 1 - Restructuring
# Charges ($4.5M, FY2023)", "Section 9.2 — Effect on Material Contracts".
_CONTINUATION_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "at", "is", "are", "was", "were", "be", "been", "that", "which",
    "as", "its", "their", "this", "these", "those", "not", "no", "any", "all",
    "such", "shall", "may", "will", "under",
}
# Low enough to catch short contract clauses used as headings — "(a) amend its
# certificate of incorporation or bylaws;" is 53 characters — while leaving
# genuine short labels ("Net Revenue Retention", "Section 3.5") untouched, since
# those end on a noun rather than dangling punctuation.
_MIN_PROSE_CHARS = 45
_DANGLING_END = (",", "(", ":", "-", ";")


def _clean_heading(section: str) -> str:
    """
    Returns a section heading worth showing, or "" if it is an artefact.

    Args:
        section: Raw `section_heading` from the chunk payload.

    Returns:
        Trimmed heading, or empty string when it should not be displayed.
    """
    if not section:
        return ""

    heading = section.strip()
    if not heading or _SEPARATOR_RUN.match(heading):
        return ""

    # Long strings ending in sentence punctuation are prose that got captured as
    # a heading, not a heading. Real headings here are short and label-like
    # ("Section 8.2 — Indemnification Cap").
    if len(heading) > _MAX_HEADING_CHARS and heading.endswith((".", ";", ",")):
        return ""

    # Trim decorative rules that trail a genuine heading.
    heading = heading.strip("=_-*~ ").strip()

    if _looks_truncated(heading):
        return ""

    return heading


def _looks_truncated(heading: str) -> bool:
    """
    True when a heading is really a sentence the chunker cut mid-flow.

    Deliberately high-precision rather than exhaustive: it only fires on long
    strings that end somewhere a heading never would. Some prose still gets
    through — a fragment ending on an ordinary noun is indistinguishable from a
    terse heading without parsing it — and that is the honest limit of fixing
    this in the viewer. The real repair is per-heading granularity in the
    chunker (see DECISIONS_LOG Decision 16); this only stops the citation panel
    showing the worst of it.
    """
    if len(heading) < _MIN_PROSE_CHARS:
        return False

    trimmed = heading.rstrip()
    if trimmed.endswith(_DANGLING_END):
        return True

    words = trimmed.split()
    if not words:
        return False
    last = re.sub(r"[^A-Za-z]", "", words[-1]).lower()
    return last in _CONTINUATION_WORDS


def render_citations(citations: list[dict]) -> None:
    """
    Renders citations with version warnings (⚠ NOT CURRENT VERSION),
    computed metric flags, and source links.

    Args:
        citations: List of citation dicts from QueryResponse.
    """
    if not citations:
        st.caption("No citations available for this answer.")
        return

    stale = sum(1 for c in citations if not c.get("is_current_version", True))
    computed = sum(
        1
        for c in citations
        if c.get("content_type") in ("computed_metric", "table_metrics_summary")
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sources Cited", len(citations))
    with col2:
        st.metric("Superseded", stale)
    with col3:
        st.metric("Computed Metrics", computed)

    st.write("")

    for i, cite in enumerate(citations, 1):
        source = cite.get("source_file", "Unknown")
        page = cite.get("page_number")
        section = _clean_heading(cite.get("section_heading", ""))
        is_current = cite.get("is_current_version", True)
        content_type = cite.get("content_type", "text")

        # Flags — each maps to a specific provenance risk the reviewer must see
        flags = ""
        if not is_current:
            flags += pill("⚠ NOT CURRENT VERSION", "bad")
        if content_type in ("computed_metric", "table_metrics_summary"):
            # Computed metrics come from deterministic pandas arithmetic during
            # ingestion, never from LLM arithmetic — worth signalling explicitly.
            flags += pill("🔢 COMPUTED", "accent")
        if cite.get("is_redline"):
            flags += pill("📝 REDLINE", "warn")

        meta_parts = []
        if section:
            meta_parts.append(section)
        if page:
            meta_parts.append(f"page {page}")
        meta = " · ".join(meta_parts) or "—"

        css_class = "dd-cite dd-cite-stale" if not is_current else "dd-cite"
        st.markdown(
            f"<div class='{css_class}'>"
            f"<div class='dd-cite-src'>{i}. {source}</div>"
            f"<div class='dd-cite-meta'>{meta}</div>"
            f"<div style='margin-top:0.35rem'>{flags}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if not is_current:
            superseded_by = cite.get("superseded_by") or "a newer version"
            st.caption(
                f"↳ Superseded by {superseded_by}. This information may be outdated."
            )
