"""
Document uploader component — file upload with category override and version controls.
"""

import streamlit as st
import requests

from src.utils.admin_client import admin_headers


SUPPORTED_TYPES = ["pdf", "docx", "pptx", "xlsx", "txt"]
CATEGORIES = [
    "auto-detect",
    "financial",
    "legal",
    "board",
    "audit",
    "regulatory",
    "operational",
    "other",
]


def render_document_uploader(api_url: str, deal_id: str) -> None:
    """
    Renders document upload widget with category override,
    version controls, and progress tracking.

    Args:
        api_url: Base API URL.
        deal_id: Current deal identifier.
    """
    st.subheader("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, PPTX, XLSX",
    )

    if not uploaded_files:
        return

    # Upload options
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Document Category", CATEGORIES)
    with col2:
        is_current = st.checkbox("Current Version", value=True)

    supersedes_id = st.text_input(
        "Supersedes Document ID (optional)",
        placeholder="Enter doc_id of the version this replaces",
    )

    if st.button("📤 Upload All", type="primary"):
        progress = st.progress(0.0)
        results = []

        for i, file in enumerate(uploaded_files):
            with st.spinner(f"Ingesting {file.name}..."):
                try:
                    data = {"deal_id": deal_id, "is_current_version": str(is_current)}
                    if category != "auto-detect":
                        data["document_category"] = category
                    if supersedes_id:
                        data["supersedes_doc_id"] = supersedes_id

                    resp = requests.post(
                        f"{api_url}/ingest",
                        files={"file": (file.name, file.getvalue())},
                        data=data,
                        headers=admin_headers(),
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        results.append(
                            f"✅ **{file.name}**: {result['chunks_created']} chunks "
                            f"({result['document_category']})"
                        )
                    else:
                        results.append(f"❌ **{file.name}**: {resp.text}")

                except Exception as e:
                    results.append(f"❌ **{file.name}**: {str(e)}")

            progress.progress((i + 1) / len(uploaded_files))

        # Show results
        st.markdown("### Upload Results")
        for r in results:
            st.markdown(r)
