"""
Pydantic request models for the API.
"""

import re
from typing import Annotated

from fastapi import Path
from pydantic import BaseModel, Field

# One character class for every id a client can send. deal_id reaches Qdrant
# filters, log lines and URLs; restricting it to URL-safe word characters rules
# out path tricks and log injection at the boundary instead of at each use.
# Covers every existing id shape: uuid4 strings, "aurora_vertex_2024", and the
# "sbx-<hex>-<hex>" sandbox ids.
DEAL_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
DEAL_ID_REGEX = re.compile(DEAL_ID_PATTERN)
SESSION_ID_PATTERN = DEAL_ID_PATTERN

# Path-parameter form, for routes that take the deal id in the URL.
DealIdPath = Annotated[str, Path(pattern=DEAL_ID_PATTERN, description="Deal identifier")]


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    deal_id: str = Field(
        ..., description="Deal identifier for data isolation", pattern=DEAL_ID_PATTERN
    )
    session_id: str | None = Field(
        None, description="Optional session ID for resuming", pattern=SESSION_ID_PATTERN
    )
    include_pii: bool = Field(
        False,
        description=(
            "Include PII-flagged content. Honoured for admin callers only; "
            "forced to false for public callers."
        ),
    )


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    deal_id: str = Field(..., description="Deal identifier", pattern=DEAL_ID_PATTERN)
    document_category: str | None = Field(
        None,
        description="Override category: financial|legal|board|audit|regulatory|operational|other",
    )
    is_current_version: bool = Field(True, description="Whether this is the current version")
    supersedes_doc_id: str | None = Field(
        None, description="Doc ID this version supersedes", max_length=64
    )


class DealCreateRequest(BaseModel):
    """Request model for creating a new deal."""
    deal_name: str = Field(..., description="Human-readable deal name", min_length=1, max_length=200)
    description: str = Field("", description="Deal description", max_length=1000)
    is_sandbox: bool = Field(
        False,
        description=(
            "Mark the deal as ephemeral. Sandbox deals are purged when the "
            "visitor's tab closes, and swept on a TTL if that never arrives. "
            "Creating a non-sandbox deal requires the admin key."
        ),
    )
