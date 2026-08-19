"""
Deal manager component — deal CRUD UI.
"""

import streamlit as st
import requests


def render_deal_manager(api_url: str) -> str | None:
    """
    Renders deal management UI: create, list, and select deals.

    Args:
        api_url: Base API URL (e.g., "http://localhost:8000/api/v1").

    Returns:
        Selected deal_id or None if no deal selected.
    """
    st.subheader("📁 Deal Management")

    # Create new deal
    with st.expander("➕ Create New Deal", expanded=False):
        deal_name = st.text_input("Deal Name", placeholder="e.g., Acme Corp Acquisition")
        description = st.text_area(
            "Description", placeholder="Brief description of the deal", height=80
        )
        if st.button("Create Deal", type="primary"):
            if deal_name:
                try:
                    resp = requests.post(
                        f"{api_url}/deals",
                        json={"deal_name": deal_name, "description": description},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"✅ Deal created: `{result['deal_id']}`")
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to API server")
            else:
                st.warning("Please enter a deal name")

    # List and select deals
    try:
        resp = requests.get(f"{api_url}/deals", timeout=10)
        if resp.status_code == 200:
            deals = resp.json()
            if deals:
                deal_options = {
                    f"{d['deal_name']} ({d['deal_id'][:8]}...)": d["deal_id"]
                    for d in deals
                }
                selected = st.selectbox("Select Deal", options=list(deal_options.keys()))
                if selected:
                    deal_id = deal_options[selected]

                    # Show deal info
                    deal = next(d for d in deals if d["deal_id"] == deal_id)
                    st.caption(f"Documents: {deal.get('document_count', 0)}")
                    st.caption(f"Status: {deal.get('status', 'active')}")

                    return deal_id
            else:
                st.info("No deals found. Create one above.")
    except requests.ConnectionError:
        st.warning("Cannot connect to API server")

    # Manual deal ID entry as fallback
    manual_id = st.text_input("Or enter Deal ID directly", placeholder="UUID")
    if manual_id:
        return manual_id

    # `?deal_id=...` deep link, last.
    #
    # Two reasons this exists. It makes a deal shareable — a link that opens the
    # dashboard already scoped to one data room, rather than "open it and then
    # pick Aurora from the dropdown". And it gives the UI tests a way to select a
    # deal without a running API: every other path here depends on GET /deals, so
    # those tests silently required the backend and passed or failed on whether
    # an unrelated server happened to be up.
    return st.query_params.get("deal_id") or None
