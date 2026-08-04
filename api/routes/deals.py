"""
Deal management routes.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models.request_models import DealCreateRequest
from api.models.response_models import DealResponse, DocumentRecord, RiskSignal
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# In-memory deal store (in production, use Postgres)
_deals: dict[str, dict] = {}

# In-memory document registry, keyed by deal_id -> list of document records.
# Mirrors the _deals pattern: it is process-local and lost on restart, which is
# the same single-worker constraint the rest of the API already carries. Qdrant
# remains the source of truth for chunk content; this only tracks document-level
# provenance (version chain, risk signals) that has no home in the vector store.
_documents: dict[str, list[dict]] = {}

# Severity ranking for the risk signal types emitted by RiskSignalExtractor.
# Defined here rather than in the extractor because severity is a presentation
# concern — the extractor reports what it matched, the dashboard ranks it.
_RISK_SEVERITY: dict[str, str] = {
    "change_of_control": "high",
    "material_adverse_change": "high",
    "financial_distress": "high",
    "litigation": "medium",
    "regulatory_risk": "medium",
    "environmental_liability": "medium",
    "ip_risk": "medium",
    "customer_concentration": "low",
    "key_person": "low",
    "indemnification": "low",
}


@router.post("/deals", response_model=DealResponse)
async def create_deal(request: DealCreateRequest):
    """Creates a new deal."""
    deal_id = str(uuid.uuid4())
    _deals[deal_id] = {
        "deal_id": deal_id,
        "deal_name": request.deal_name,
        "description": request.description,
        "document_count": 0,
        "status": "active",
    }

    logger.info(
        "Deal created",
        extra={"deal_id": deal_id, "deal_name": request.deal_name},
    )

    return DealResponse(**_deals[deal_id])


@router.get("/deals", response_model=list[DealResponse])
async def list_deals():
    """Lists all deals."""
    return [DealResponse(**d) for d in _deals.values()]


@router.get("/deals/{deal_id}", response_model=DealResponse)
async def get_deal(deal_id: str):
    """Gets a specific deal by ID."""
    if deal_id not in _deals:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")
    return DealResponse(**_deals[deal_id])
