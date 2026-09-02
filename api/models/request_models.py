"""
Pydantic request models for the API.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    deal_id: str = Field(..., description="Deal identifier for data isolation")
    session_id: str | None = Field(None, description="Optional session ID for resuming")
    include_pii: bool = Field(False, description="Include PII-flagged content (authorized users only)")


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    deal_id: str = Field(..., description="Deal identifier")
    document_category: str | None = Field(
        None,
        description="Override category: financial|legal|board|audit|regulatory|operational|other",
    )
    is_current_version: bool = Field(True, description="Whether this is the current version")
    supersedes_doc_id: str | None = Field(None, description="Doc ID this version supersedes")


class DealCreateRequest(BaseModel):
    """Request model for creating a new deal."""
    deal_name: str = Field(..., description="Human-readable deal name")
    description: str = Field("", description="Deal description")
    is_sandbox: bool = Field(
        False,
        description=(
            "Mark the deal as ephemeral. Sandbox deals are purged when the "
            "visitor's tab closes, and swept on a TTL if that never arrives."
        ),
    )
