"""
Version browser component — document version history viewer.
"""

import streamlit as st

from app.styles import pill


def render_version_browser(version_chain: list[dict]) -> None:
    """
    Renders the deal's document inventory and version chain, showing supersedes
    relationships and redline availability.

    Superseded documents are not deleted — their chunks are flipped to
    is_current_version=0, which removes them from default retrieval while
    keeping them auditable here.

    Args:
        version_chain: Document records from GET /deals/{id}/documents,
            ordered newest-first. Each dict has: doc_id, filename,
            document_category, chunks_created, version_label, upload_date,
            is_current_version, supersedes_doc_id, superseded_by, has_redline.
    """
    if not version_chain:
        st.info("No documents ingested for this deal yet.")
        return

    current = sum(1 for v in version_chain if v.get("is_current_version"))
    total_chunks = sum(v.get("chunks_created", 0) for v in version_chain)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", len(version_chain))
    with col2:
        st.metric("Current Versions", current)
    with col3:
        st.metric("Indexed Chunks", total_chunks)

    st.write("")

    for i, version in enumerate(version_chain):
        is_current = version.get("is_current_version", False)
        doc_id = version.get("doc_id", "?")
        filename = version.get("filename", "Unknown")
        label = version.get("version_label", f"v{len(version_chain) - i}")
        date = version.get("upload_date", "")
        category = version.get("document_category", "")
        chunks = version.get("chunks_created", 0)
        supersedes = version.get("supersedes_doc_id", "")
        superseded_by = version.get("superseded_by", "")
        has_redline = version.get("has_redline", False)

        badge = "🟢 CURRENT" if is_current else "🔴 SUPERSEDED"

        with st.expander(f"{badge}  |  {filename}  ({label})", expanded=is_current):
            flags = pill(badge, "ok" if is_current else "bad")
            if category:
                flags += pill(category.upper(), "muted")
            if has_redline:
                flags += pill("📝 REDLINE", "warn")
            st.markdown(flags, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**Doc ID:** `{doc_id[:12]}…`")
            with col2:
                st.caption(f"**Uploaded:** {date or '—'}")
            with col3:
                st.caption(f"**Chunks:** {chunks}")

            if supersedes:
                st.caption(f"↳ Supersedes `{supersedes[:12]}…`")
            if superseded_by:
                st.caption(f"↳ Superseded by `{superseded_by[:12]}…`")

            if not is_current:
                st.warning(
                    "Excluded from retrieval by default. Any citation that does "
                    "surface it will carry a version warning."
                )

        # Connector between entries
        if i < len(version_chain) - 1:
            st.markdown(
                "<div style='text-align:center;color:#8B93A7;font-size:16px;margin:-0.35rem 0'>↓</div>",
                unsafe_allow_html=True,
            )
