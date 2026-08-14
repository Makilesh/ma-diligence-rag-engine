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


async def _discover_indexed_deals() -> dict[str, int]:
    """
    Finds deals that exist in the vector store, with their document counts.

    `_deals` only knows about deals created through `POST /deals` in *this*
    process. That leaves two ways for real, queryable data to be invisible in
    the UI: anything ingested directly against a deal_id (which is how the
    evaluation harness loads the corpus), and everything at all after a restart,
    since the registry is in-memory while Qdrant is on disk. Both produced the
    same dead end — 131 indexed chunks, and a sidebar reading "No deals found."

    Qdrant is the source of truth for what is actually searchable, so ask it.

    Returns:
        Mapping of deal_id to distinct document count. Empty on any failure —
        the endpoint still returns the in-memory deals.
    """
    from src.vector_db.qdrant_client import get_qdrant_client
    from src.vector_db.constants import COLLECTION_NAME

    client = get_qdrant_client()
    try:
        deal_facet = await client.facet(
            collection_name=COLLECTION_NAME, key="deal_id", limit=1000
        )
    except Exception as e:
        logger.warning(f"Could not enumerate deals from the vector store: {e}")
        return {}

    discovered: dict[str, int] = {}
    for hit in deal_facet.hits:
        deal_id = str(hit.value)
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            file_facet = await client.facet(
                collection_name=COLLECTION_NAME,
                key="source_file",
                facet_filter=Filter(
                    must=[FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]
                ),
                limit=1000,
            )
            discovered[deal_id] = len(file_facet.hits)
        except Exception:
            # Chunk count is a poor stand-in for document count, but a deal that
            # is listed with the wrong count is far better than one that is
            # missing entirely.
            discovered[deal_id] = 0
    return discovered


@router.get("/deals", response_model=list[DealResponse])
async def list_deals():
    """
    Lists every deal that is queryable — registered in this process or indexed.

    Deals created via `POST /deals` keep their name and description; deals found
    only in the vector store are listed under their deal_id so they can still be
    selected.
    """
    deals = {d["deal_id"]: dict(d) for d in _deals.values()}

    for deal_id, doc_count in (await _discover_indexed_deals()).items():
        if deal_id in deals:
            # Prefer the live index count over the registry's, which drifts on
            # restart while the vector store does not.
            if doc_count:
                deals[deal_id]["document_count"] = doc_count
            continue
        deals[deal_id] = {
            "deal_id": deal_id,
            "deal_name": deal_id,
            "description": "Discovered in the vector store",
            "document_count": doc_count,
            "status": "active",
        }

    return [DealResponse(**d) for d in deals.values()]


@router.get("/deals/{deal_id}", response_model=DealResponse)
async def get_deal(deal_id: str):
    """Gets a specific deal by ID."""
    if deal_id not in _deals:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")
    return DealResponse(**_deals[deal_id])


# ==============================================================================
# Document registry — populated by the ingestion route
# ==============================================================================


def register_document(
    deal_id: str,
    doc_id: str,
    filename: str,
    document_category: str,
    chunks_created: int,
    is_current_version: bool,
    supersedes_doc_id: str | None,
    risk_signals: list[dict] | None = None,
) -> None:
    """
    Records an ingested document against its deal.

    Also maintains the version chain: when this document supersedes another,
    the superseded record is flipped to is_current_version=False and stamped
    with superseded_by, so the version browser and citation version warnings
    have a consistent view without re-reading Qdrant.

    Args:
        deal_id: Owning deal.
        doc_id: Newly assigned document ID.
        filename: Original uploaded filename.
        document_category: Detected or overridden category.
        chunks_created: Number of chunks indexed for this document.
        is_current_version: Whether this upload is the current version.
        supersedes_doc_id: Doc ID this version replaces, if any.
        risk_signals: Risk signal dicts detected during ingestion.
    """
    records = _documents.setdefault(deal_id, [])

    records.append(
        {
            "doc_id": doc_id,
            "deal_id": deal_id,
            "filename": filename,
            "document_category": document_category,
            "chunks_created": chunks_created,
            "is_current_version": is_current_version,
            "supersedes_doc_id": supersedes_doc_id or "",
            "superseded_by": "",
            "upload_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "risk_signals": risk_signals or [],
            "has_redline": False,  # set once redline indexing is wired in ingest
        }
    )

    if supersedes_doc_id:
        for prior in records:
            if prior["doc_id"] == supersedes_doc_id:
                prior["is_current_version"] = False
                prior["superseded_by"] = doc_id
                break

    # document_count previously never moved off its initial 0
    if deal_id in _deals:
        _deals[deal_id]["document_count"] = len(records)


@router.get("/deals/{deal_id}/documents", response_model=list[DocumentRecord])
async def list_deal_documents(deal_id: str):
    """
    Lists ingested documents for a deal, newest first.

    Feeds the version browser: each record carries its position in the version
    chain (supersedes_doc_id / superseded_by / is_current_version).
    """
    records = _documents.get(deal_id, [])
    ordered = sorted(records, key=lambda r: r["upload_date"], reverse=True)
    return [
        DocumentRecord(
            doc_id=r["doc_id"],
            filename=r["filename"],
            document_category=r["document_category"],
            chunks_created=r["chunks_created"],
            version_label=f"v{len(records) - i}",
            upload_date=r["upload_date"],
            is_current_version=r["is_current_version"],
            supersedes_doc_id=r["supersedes_doc_id"],
            superseded_by=r["superseded_by"],
            has_redline=r["has_redline"],
        )
        for i, r in enumerate(ordered)
    ]


@router.get("/deals/{deal_id}/risk-signals", response_model=list[RiskSignal])
async def list_deal_risk_signals(deal_id: str):
    """
    Returns risk signals detected across all documents in a deal.

    Signals are produced by RiskSignalExtractor at ingestion time; severity is
    assigned here from _RISK_SEVERITY. Sorted high → low so the dashboard's
    most important categories expand first.
    """
    signals: list[RiskSignal] = []

    for record in _documents.get(deal_id, []):
        for signal in record.get("risk_signals", []):
            signal_type = signal.get("signal_type", "other")
            match_count = signal.get("match_count", 0)
            samples = signal.get("sample_matches", [])
            sample_str = ", ".join(str(s) for s in samples if s)

            description = f"{match_count} match(es)"
            if sample_str:
                description += f" — e.g. \"{sample_str[:120]}\""

            signals.append(
                RiskSignal(
                    signal_type=signal_type,
                    severity=_RISK_SEVERITY.get(signal_type, "low"),
                    source_file=record["filename"],
                    description=description,
                    page_number=signal.get("page_number"),
                )
            )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    signals.sort(key=lambda s: severity_rank.get(s.severity, 3))
    return signals
