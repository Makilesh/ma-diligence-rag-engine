"""
Pydantic response models for the API.
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation reference in the answer."""
    chunk_id: str = ""
    source_file: str = ""
    page_number: int | None = None
    section_heading: str = ""
    is_current_version: bool = True
    content_type: str = "text"
    is_redline: bool = False
    superseded_by: str = ""


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    answer: str = Field(..., description="Generated answer with citations")
    query_type: str = Field(..., description="Detected query type")
    confidence_score: float = Field(..., description="Answer confidence 0.0-1.0")
    validation_status: str = Field(..., description="passed|warning|failed")
    citations: list[Citation] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    total_latency_ms: float = Field(..., description="Total pipeline latency")
    session_id: str = Field(..., description="Session ID for this query")
    rewrite_iterations: int = Field(0, description="Number of query rewrites performed")
    agent_trace: list[dict] = Field(default_factory=list, description="Agent execution trace")
    is_refusal: bool = Field(
        False,
        description="True when the pipeline refused to answer for lack of usable context",
    )
    context_quality_score: float = Field(
        0.0, description="Best context quality score achieved across retrieval attempts"
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    doc_id: str = Field(..., description="Assigned document ID")
    deal_id: str = Field(..., description="Deal identifier")
    document_category: str = Field(..., description="Detected/assigned category")
    chunks_created: int = Field(..., description="Number of chunks indexed")
    status: str = Field("success", description="Ingestion status")


class DealResponse(BaseModel):
    """Response model for deal operations."""
    deal_id: str
    deal_name: str
    description: str = ""
    document_count: int = 0
    status: str = "active"
    is_sandbox: bool = False
    expires_at: str = Field(
        "",
        description="ISO-8601 TTL deadline for sandbox deals; empty for permanent ones",
    )


class DocumentRecord(BaseModel):
    """A single ingested document and its place in the version chain."""
    doc_id: str
    filename: str
    document_category: str = ""
    chunks_created: int = 0
    version_label: str = ""
    upload_date: str = ""
    is_current_version: bool = True
    supersedes_doc_id: str = ""
    superseded_by: str = ""
    has_redline: bool = False


class RiskSignal(BaseModel):
    """A risk signal detected in a deal document during ingestion."""
    signal_type: str
    severity: str = "low"
    source_file: str = ""
    description: str = ""
    page_number: int | None = None


class BudgetStatusResponse(BaseModel):
    """Response model for budget status."""
    synthesis_primary: dict = Field(default_factory=dict)
    agent_workhorse: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "healthy"
    service: str = "manda-rag"
